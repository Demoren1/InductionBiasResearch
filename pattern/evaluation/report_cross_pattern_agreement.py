"""Post-hoc summary of the cross-pattern decoder-agreement experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import ideal_mask  # noqa: E402
from evaluation.decoder_agreement import align_columns  # noqa: E402
from evaluation.run_cross_pattern_agreement import (  # noqa: E402
    PAIRS, ROOT, load_model, path_for,
)
from evaluation.train_agreement_vaes import sha256_file  # noqa: E402


def summary(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": None, "sample_sd": None, "ci95_t": None}
    sd = float(x.std(ddof=1)) if x.size > 1 else 0.
    margin = float(student_t.ppf(.975, x.size - 1) * sd / np.sqrt(x.size)) if x.size > 1 else 0.
    return {"n": int(x.size), "mean": float(x.mean()), "sample_sd": sd,
            "ci95_t": [float(x.mean() - margin), float(x.mean() + margin)],
            "values": x.tolist()}


def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aligned = align_columns(a.float(), b.float())
    intersection = (a.bool() & aligned.bool()).sum((1, 2)).float()
    union = (a.bool() | aligned.bool()).sum((1, 2)).float()
    return intersection / union.clamp_min(1)


def exact(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a == align_columns(a.float(), b.float())).all((1, 2)).float().mean())


def gold(mask: torch.Tensor) -> float:
    target = ideal_mask().float().unsqueeze(0).expand_as(mask)
    return float(iou(target, mask).mean())


def validate_search(folder: Path) -> None:
    record = json.loads((folder / "search_provenance.json").read_text())
    if record["protocol_sha256"] != sha256_file(folder / "protocol.json"):
        raise ValueError(f"changed search protocol at {folder}")
    for name, digest in record["artifact_sha256"].items():
        if sha256_file(folder / name) != digest:
            raise ValueError(f"changed search artifact at {folder / name}")
    protocol = json.loads((folder / "protocol.json").read_text())
    for model in protocol["models"]:
        if sha256_file(Path(model["checkpoint"])) != model["checkpoint_sha256"]:
            raise ValueError(f"changed VAE checkpoint: {model['checkpoint']}")


def mask_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    return {
        "hard_iou": float(iou(a, b).mean()),
        "exact_fraction": exact(a, b),
        "gold_iou": float(np.mean([gold(a), gold(b)])),
        "unique_fraction": float(np.mean([
            len(torch.unique(a.flatten(1), dim=0)) / len(a),
            len(torch.unique(b.flatten(1), dim=0)) / len(b),
        ])),
    }


def read_search(folder: Path) -> dict:
    validate_search(folder)
    optimization = torch.load(folder / "optimization.pt", map_location="cpu", weights_only=True)
    protocol = json.loads((folder / "protocol.json").read_text())
    result = {
        "models": [{"pattern": m["pattern"], "seed": m["seed"],
                    "validation_loss": m["validation_loss"],
                    "decoder_std": m["noncollapse"]["decoder_probability_feature_std_mean"]}
                   for m in protocol["models"]],
        "soft_mse_initial": float(optimization["initial_loss"].mean()),
        "soft_mse_optimized": float(optimization["final_loss"].mean()),
        "initial": mask_metrics(optimization["initial_masks1"], optimization["initial_masks2"]),
        "optimized": mask_metrics(optimization["final_masks1"], optimization["final_masks2"]),
    }
    final_norms = torch.cat([optimization["final_z1"], optimization["final_z2"]]).norm(dim=1)
    final_soft = torch.cat([optimization["final_soft1"], optimization["final_soft2"]])
    result["final_latents"] = {
        "mean_norm": float(final_norms.mean()),
        "at_radius_fraction": float((final_norms >= protocol["radius"] - 1e-5).float().mean()),
        "softness": float((final_soft * (1 - final_soft)).mean()),
    }
    random_path = folder / "random_search.pt"
    if random_path.exists():
        random = torch.load(random_path, map_location="cpu", weights_only=True)
        result["soft_mse_random"] = float(random["loss"].mean())
        result["random"] = mask_metrics(random["masks1"], random["masks2"])
    task_files = sorted(folder.glob("task_*.json"))
    if task_files:
        if len(task_files) != 2:
            raise ValueError(f"incomplete downstream evaluation in {folder}")
        scores = [json.loads(path.read_text()) for path in task_files]
        result["downstream"] = {
            method: {
                "accuracy": float(np.mean([s[method]["accuracy"]["mean"] for s in scores])),
                "bce": float(np.mean([s[method]["bce"]["mean"] for s in scores])),
            } for method in scores[0]
        }
    return result


def reconstruction_specialization(root: Path, pair_index: int,
                                  val_cache: dict[str, torch.Tensor]) -> dict:
    patterns = PAIRS[pair_index]
    for pattern in patterns:
        if pattern not in val_cache:
            bank_dir = root / "bank" / f"pattern_{pattern}"
            protocol = json.loads((bank_dir / "protocol.json").read_text())
            original = Path(__file__).resolve().parents[1] / "outputs" / "checkpoints" / f"pattern_{pattern}" / "importance.pt"
            chunks = [bank_dir / f"chunk_{i:03d}.pt" for i in range(
                (protocol["n_extra"] + protocol["chunk_size"] - 1) // protocol["chunk_size"]) ]
            parts = [torch.load(path, map_location="cpu", weights_only=True)
                     for path in (original, *chunks)]
            maps = torch.cat([part["importance"] for part in parts])
            losses = torch.cat([part["val_loss"] for part in parts])
            # Ranks 801–1000 are held out from all VAE training and checkpoint
            # choices, while remaining close to the selected top-800 quality.
            indices = torch.argsort(losses, stable=True)[800:1000]
            val_cache[pattern] = maps[indices].reshape(len(indices), -1).float()

    @torch.no_grad()
    def recon(model: torch.nn.Module, x: torch.Tensor) -> float:
        condition = x.new_zeros(len(x), 0)
        mu, _ = model.encode(x, condition)
        logits = model.decode(mu, condition)
        return float(F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1).mean())

    replicates = []
    for rep in range(4):
        models = [load_model(root, pair_index, rep, side, torch.device("cpu"))[0]
                  for side in (0, 1)]
        losses = [[recon(model, val_cache[pattern]) for pattern in patterns]
                  for model in models]
        # For each dataset, compare partner reconstruction to the VAE trained
        # on that dataset. Positive values indicate pattern specialization.
        effect_a = losses[1][0] - losses[0][0]
        effect_b = losses[0][1] - losses[1][1]
        replicates.append({"losses": losses, "partner_minus_own_a": effect_a,
                           "partner_minus_own_b": effect_b,
                           "mean_effect": (effect_a + effect_b) / 2})
    return {"replicates": replicates,
            "mean_partner_minus_own": float(np.mean([r["mean_effect"] for r in replicates]))}


def pair_record(root: Path, pair_index: int,
                val_cache: dict[str, torch.Tensor]) -> dict:
    folder = path_for(root, pair_index)
    cross = [read_search(folder / f"cross_{rep}") for rep in range(4)]
    within = [read_search(folder / f"within_{side}_{a}{b}")
              for side in "ab" for a, b in ((0, 1), (2, 3))]
    def mean(rows: list[dict], *path: str) -> float:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value[key]
            values.append(value)
        return float(np.mean(values))
    record = {
        "pair_index": pair_index, "patterns": list(PAIRS[pair_index]),
        "stratum": "hamming_1" if pair_index < 4 else "complement",
        "cross_replicates": cross, "within_controls": within,
        "reconstruction_specialization": reconstruction_specialization(root, pair_index, val_cache),
        "summary": {
            "cross_initial_iou": mean(cross, "initial", "hard_iou"),
            "cross_optimized_iou": mean(cross, "optimized", "hard_iou"),
            "cross_random_iou": mean(cross, "random", "hard_iou"),
            "cross_exact": mean(cross, "optimized", "exact_fraction"),
            "cross_initial_gold": mean(cross, "initial", "gold_iou"),
            "cross_optimized_gold": mean(cross, "optimized", "gold_iou"),
            "cross_random_gold": mean(cross, "random", "gold_iou"),
            "cross_unique": mean(cross, "optimized", "unique_fraction"),
            "cross_soft_mse": mean(cross, "soft_mse_optimized"),
            "cross_at_radius": mean(cross, "final_latents", "at_radius_fraction"),
            "cross_softness": mean(cross, "final_latents", "softness"),
            "within_optimized_iou": mean(within, "optimized", "hard_iou"),
            "within_exact": mean(within, "optimized", "exact_fraction"),
            "within_optimized_gold": mean(within, "optimized", "gold_iou"),
            "within_soft_mse": mean(within, "soft_mse_optimized"),
            "within_at_radius": mean(within, "final_latents", "at_radius_fraction"),
            "within_softness": mean(within, "final_latents", "softness"),
        },
    }
    s = record["summary"]
    s["reconstruction_partner_minus_own"] = record["reconstruction_specialization"]["mean_partner_minus_own"]
    s["cross_minus_random_iou"] = s["cross_optimized_iou"] - s["cross_random_iou"]
    s["cross_minus_within_iou"] = s["cross_optimized_iou"] - s["within_optimized_iou"]
    s["cross_minus_within_exact"] = s["cross_exact"] - s["within_exact"]
    s["cross_minus_within_gold"] = s["cross_optimized_gold"] - s["within_optimized_gold"]
    s["cross_minus_within_radius"] = s["cross_at_radius"] - s["within_at_radius"]
    s["cross_gold_gain"] = s["cross_optimized_gold"] - s["cross_initial_gold"]
    if all("downstream" in row for row in cross):
        methods = cross[0]["downstream"]
        record["downstream"] = {
            method: {metric: mean(cross, "downstream", method, metric)
                     for metric in ("accuracy", "bce")}
            for method in methods
        }
        s["own_accuracy_gain"] = (record["downstream"]["own_optimized"]["accuracy"]
                                  - record["downstream"]["own_initial"]["accuracy"])
        s["own_minus_partner_accuracy"] = (
            record["downstream"]["own_optimized"]["accuracy"]
            - record["downstream"]["partner_optimized"]["accuracy"])
        s["own_minus_random_accuracy"] = (
            record["downstream"]["own_optimized"]["accuracy"]
            - record["downstream"]["random_search"]["accuracy"])
        s["own_bce_gain"] = (record["downstream"]["own_initial"]["bce"]
                             - record["downstream"]["own_optimized"]["bce"])
    return record


def fmt(record: dict, digits: int = 4) -> str:
    return f"{record['mean']:.{digits}f} [{record['ci95_t'][0]:.{digits}f}; {record['ci95_t'][1]:.{digits}f}]"


def write_report(root: Path, records: list[dict], aggregate: dict) -> None:
    lines = [
        "# Agreement отдельных VAE для разных паттернов",
        "",
        "Дата: 19 сентября 2026 года.",
        "",
        "Восемь непересекающихся пар покрывают все 16 паттернов длины 4: четыре пары различаются одним битом, четыре — всеми битами. На каждую пару обучены четыре независимые пары VAE с разными инициализациями. Каждый VAE видит importance maps только своего паттерна. Для каждого cross-сравнения использованы 64 latent-старта и 2000 шагов оптимизации agreement. Внутрипаттернные сравнения используют те же гиперпараметры и независимые VAE.",
        "",
        "Банк каждого паттерна содержит 2000 прежних и 6000 новых обученных masked MLP. В VAE поступают 800 лучших нормированных importance maps по validation BCE; 15% этих карт выделены для выбора checkpoint VAE. Архитектура и loss VAE соответствуют прежнему протоколу: latent 32, hidden 256, BCE-sum + 0.1 KL, максимум 160 эпох.",
        "",
        "Проверка специализации VAE использует следующие 200 карт по quality rank (801–1000), не использованные ни при обучении, ни при выборе checkpoint. Для каждой такой выборки сравнивается BCE-sum реконструкции posterior mean у своего и чужого VAE; положительная разность «чужой − свой» означает специализацию.",
        "",
        "Аналитическая бинарная поддержка локальных окон одинакова для всех 16 паттернов: различаются требуемые значения весов. Поэтому совпадение двух масок само по себе не доказывает восстановление правила. Проверки здесь — сравнение cross/within agreement, post-hoc Gold IoU, вариативность масок и качество свежей MLP на каждом паттерне.",
        "",
        "Gold IoU сравнивает маску с одной канонической аналитической поддержкой после перестановки скрытых нейронов. Иные поддержки могут реализовывать ту же функцию на выбранном распределении; IoU ниже единицы сам по себе не доказывает функциональную ошибку.",
        "",
        "Побитовое дополнение паттерна является симметрией задачи при одновременной замене битов входа. Для unsigned importance maps такие пары могут иметь очень похожие распределения, несмотря на расстояние Хэмминга 4; сравнение двух типов пар описательное и не является экспериментом по причинному эффекту расстояния Хэмминга.",
        "",
        "Единица сравнения в общих описательных 95% t-интервалах — выбранная пара паттернов (n=8); четыре seed-повтора вложены в неё. Пары покрывают все 16 паттернов, но не являются случайной выборкой из более широкого семейства. Исходные MLP обучались с метками своих задач; их importance maps служили данными VAE. Аналитическая маска и сами метки не поступали на вход VAE или в objective поиска agreement. Gold IoU и downstream accuracy вычислены после сохранения масок.",
        "",
        "## Основные результаты",
        "",
        "| Метрика | Cross-pattern | Within-pattern / контроль |",
        "|---|---:|---:|",
    ]
    rows = [
        ("Hard IoU после agreement", "cross_optimized_iou", "within_optimized_iou"),
        ("Доля точного совпадения", "cross_exact", "within_exact"),
        ("Gold IoU после agreement", "cross_optimized_gold", "within_optimized_gold"),
        ("Soft agreement MSE", "cross_soft_mse", "within_soft_mse"),
        ("Доля latent-кодов на границе радиуса", "cross_at_radius", "within_at_radius"),
        ("Мягкость S(1−S)", "cross_softness", "within_softness"),
    ]
    for label, cross_key, within_key in rows:
        lines.append(f"| {label} | {fmt(aggregate[cross_key])} | {fmt(aggregate[within_key])} |")
    lines += ["", "| Cross-pattern контроль | Значение, 95% t-CI по 8 парам |",
              "|---|---:|"]
    for label, key in [
        ("Initial hard IoU", "cross_initial_iou"),
        ("Random-search hard IoU", "cross_random_iou"),
        ("Agreement − random-search hard IoU", "cross_minus_random_iou"),
        ("Random-search Gold IoU", "cross_random_gold"),
        ("Изменение Gold IoU после agreement", "cross_gold_gain"),
        ("Cross − within hard IoU", "cross_minus_within_iou"),
        ("Cross − within доля точного совпадения", "cross_minus_within_exact"),
        ("Cross − within Gold IoU", "cross_minus_within_gold"),
        ("Cross − within доля кодов на границе", "cross_minus_within_radius"),
        ("Доля уникальных final масок", "cross_unique"),
        ("Реконструкция чужих карт − своих, BCE-sum", "reconstruction_partner_minus_own"),
    ]:
        lines.append(f"| {label} | {fmt(aggregate[key])} |")
    lines += ["", "## По парам", "",
              "| Пара | Тип | Cross hard IoU | Within hard IoU | Cross exact | Cross Gold IoU |",
              "|---|---|---:|---:|---:|---:|"]
    for record in records:
        s = record["summary"]
        lines.append(f"| {' / '.join(record['patterns'])} | {record['stratum']} | "
                     f"{s['cross_optimized_iou']:.4f} | {s['within_optimized_iou']:.4f} | "
                     f"{s['cross_exact']:.4f} | {s['cross_optimized_gold']:.4f} |")
    lines += ["", "## Близкие и противоположные паттерны", "",
              "| Тип (4 пары) | Cross hard IoU | Cross Gold IoU | Реконструкция чужих − своих карт |",
              "|---|---:|---:|---:|"]
    for group in ("hamming_1", "complement"):
        subset = [r for r in records if r["stratum"] == group]
        means = {key: float(np.mean([r["summary"][key] for r in subset]))
                 for key in ("cross_optimized_iou", "cross_optimized_gold",
                             "reconstruction_partner_minus_own")}
        lines.append(f"| {group} | {means['cross_optimized_iou']:.4f} | "
                     f"{means['cross_optimized_gold']:.4f} | "
                     f"{means['reconstruction_partner_minus_own']:.4f} |")
    if all("downstream" in record for record in records):
        lines += ["", "## Свежие MLP на обоих паттернах пары", "",
                  "Для каждого повторения взяты первые 16 latent-стартов, без отбора по качеству. Веса MLP обучены с нуля при одинаковой инициализации и потоке данных для всех масок. Каждая маска оценена на обеих задачах пары. Оценка использует новую сбалансированную выборку, но пространство входов содержит лишь 256 строк: совпадения входов с обучением возможны, поэтому это функциональный контроль, а не тест переноса на уникальные строки.", "",
                  "| Маска | Accuracy, 95% t-CI | BCE, 95% t-CI |",
                  "|---|---:|---:|"]
        for method in records[0]["downstream"]:
            accuracy = summary([r["downstream"][method]["accuracy"] for r in records])
            bce = summary([r["downstream"][method]["bce"] for r in records])
            lines.append(f"| {method} | {fmt(accuracy)} | {fmt(bce)} |")
        lines += ["", "| Парное сравнение по 8 парам | Средний эффект, 95% t-CI |",
                  "|---|---:|"]
        for label, key in (
            ("Accuracy: own optimized − own initial", "own_accuracy_gain"),
            ("Accuracy: own optimized − partner optimized", "own_minus_partner_accuracy"),
            ("Accuracy: own optimized − random search", "own_minus_random_accuracy"),
            ("BCE: own initial − own optimized", "own_bce_gain"),
        ):
            lines.append(f"| {label} | {fmt(aggregate[key])} |")
    lines += ["", "## Вывод", "",
              f"- Разные паттерны дают почти совпадающие маски после agreement: hard IoU {aggregate['cross_optimized_iou']['mean']:.4f}, exact {aggregate['cross_exact']['mean']:.4f}; budget-matched random search даёт hard IoU {aggregate['cross_random_iou']['mean']:.4f}.",
              f"- Cross-pattern поиск немного повышает Gold IoU относительно within-pattern контроля: разность {fmt(aggregate['cross_minus_within_gold'])}. До аналитического IoU 1.0 остаётся большой разрыв.",
              f"- Устойчивого по восьми парам функционального выигрыша над собственным prior не наблюдалось: парная разность accuracy {fmt(aggregate['own_accuracy_gain'])}; разность с random-search {fmt(aggregate['own_minus_random_accuracy'])}.",
              f"- Латенты чаще оказываются на границе радиуса в cross-pattern поиске: разность долей {fmt(aggregate['cross_minus_within_radius'])}. Это диагностический признак выхода к краю разрешённой области, не доказательство выхода из распределения VAE.",
              ""]
    lines += ["", "## Воспроизведение", "",
              "Код: `pattern/evaluation/extend_pattern_importance.py`, `pattern/evaluation/run_cross_pattern_agreement.py`, `pattern/evaluation/report_cross_pattern_agreement.py`. Данные и веса: `pattern/outputs/cross_pattern_agreement_20260919/`. Структурная сводка: `summary.json`.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))


def plot(root: Path, records: list[dict]) -> None:
    labels = ["/".join(r["patterns"]) for r in records]
    x = np.arange(len(records))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    series = [
        ("hard IoU", "cross_optimized_iou", "within_optimized_iou"),
        ("exact agreement", "cross_exact", "within_exact"),
        ("Gold IoU", "cross_optimized_gold", "within_optimized_gold"),
    ]
    for ax, (title, cross_key, within_key) in zip(axes, series):
        ax.bar(x-.2, [r["summary"][cross_key] for r in records], .4,
               color="#3177aa", label="different patterns")
        ax.bar(x+.2, [r["summary"][within_key] for r in records], .4,
               color="#d69b41", label="same pattern")
        ax.set(title=title, ylim=(0, 1.03), xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelrotation=55)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("VAE decoder agreement: pattern-specific training, four seeds per pair")
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), layout="constrained")
    for group, pair_index in enumerate((0, 4)):
        folder = path_for(root, pair_index) / "cross_0"
        result = torch.load(folder / "optimization.pt", map_location="cpu", weights_only=True)
        matched = align_columns(result["final_masks1"], result["final_masks2"])
        ideal = ideal_mask().float()
        for row in range(2):
            entries = (result["initial_masks1"][row], result["final_masks1"][row],
                       matched[row], ideal)
            for col, mask in enumerate(entries):
                ax = axes[group * 2 + row, col]
                ax.imshow(mask, cmap="Greys", vmin=0, vmax=1)
                ax.set(xticks=[], yticks=[])
    for col, title in enumerate(("prior A", "optimized A", "optimized B aligned", "analytic")):
        axes[0, col].set_title(title, fontsize=10)
    axes[0, 0].set_ylabel("0000 / 0001\nstart 0")
    axes[1, 0].set_ylabel("start 1")
    axes[2, 0].set_ylabel("0100 / 1011\nstart 0")
    axes[3, 0].set_ylabel("start 1")
    fig.suptitle("Cross-pattern masks: first two starts, fixed before quality assessment")
    fig.savefig(root / "mask_examples.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    val_cache: dict[str, torch.Tensor] = {}
    records = [pair_record(args.root, i, val_cache) for i in range(len(PAIRS))]
    keys = records[0]["summary"].keys()
    aggregate = {key: summary([record["summary"][key] for record in records])
                 for key in keys}
    strata = {group: {key: summary([r["summary"][key] for r in records
                                   if r["stratum"] == group]) for key in keys}
              for group in ("hamming_1", "complement")}
    payload = {"pairs": records, "aggregate_over_pattern_pairs": aggregate,
               "strata": strata,
               "analysis_unit": "8 pattern pairs; 4 VAE-seed replicates nested per pair"}
    (args.root / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    write_report(args.root, records, aggregate)
    plot(args.root, records)
    print(f"cross hard IoU {fmt(aggregate['cross_optimized_iou'])}; "
          f"within {fmt(aggregate['within_optimized_iou'])}; "
          f"cross Gold {fmt(aggregate['cross_optimized_gold'])}", flush=True)


if __name__ == "__main__":
    main()

"""Test whether distributing fixed VAE mask edges across pixels improves MNIST8m."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

import pattern.mnist8m_raw_mlp_bce as exp
from pattern.mnist8m_importance_bce import save


CAPS = (None, 16, 8, 4)
SOURCES = ("vae0", "vae1", "consensus", "shared_consensus", "mean0", "mean1",
           "mean_consensus")


def capped_topk(scores: torch.Tensor, cap: int | None) -> torch.Tensor:
    """Select exactly K highest scoring edges, with at most cap per input pixel."""
    if cap is None:
        return exp.topk(scores)
    if cap * exp.FEATURES < exp.K:
        raise ValueError("cap cannot accommodate K edges")
    eligible = torch.zeros_like(scores, dtype=torch.bool)
    eligible.scatter_(-1, scores.topk(cap, dim=-1).indices, True)
    allowed = scores.masked_fill(~eligible, -torch.inf)
    mask = exp.topk(allowed)
    assert mask.sum().item() == exp.K
    assert mask.sum(-1).max().item() <= cap
    return mask


def load_sources(root: Path) -> dict[str, torch.Tensor]:
    own0 = torch.load(root / "search_lambda0.pt", map_location="cpu",
                      weights_only=True)
    own1 = torch.load(root / "search_lambda1.pt", map_location="cpu",
                      weights_only=True)
    shared1 = torch.load(root / "search_shared_lambda1.pt", map_location="cpu",
                         weights_only=True)
    raw0 = own0["logits"][:, own0["chosen_start"]]
    raw1 = own1["logits"][:, own1["chosen_start"]]
    raw_shared = shared1["logits"][:, shared1["chosen_start"]]
    sources = {"vae0": raw0[0], "vae1": raw0[1],
               "consensus": raw1.mean(0),
               "shared_consensus": raw_shared.mean(0)}
    for task in range(2):
        bank = torch.load(root / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        sources[f"mean{task}"] = bank["importance"].mean(0)
    sources["mean_consensus"] = (sources["mean0"] + sources["mean1"]) / 2
    return sources


def make_masks(root: Path) -> tuple[list[str], torch.Tensor]:
    names, masks = [], []
    for source, scores in load_sources(root).items():
        for cap in CAPS:
            names.append(f"{source}_cap{cap or 0}")
            masks.append(capped_topk(scores, cap))
    for task in range(2):
        bank = torch.load(root / f"bank_task{task}.pt", map_location="cpu",
                          weights_only=True)
        names.append(f"direct{task}")
        masks.append(bank["masks"][0].float())
    names.extend(f"random{i}" for i in range(4))
    masks.extend(exp.exact_masks(4, 20261099, torch.device("cpu")))
    generator = torch.Generator().manual_seed(20261099)
    random_scores = torch.rand(4, exp.FEATURES, exp.HIDDEN,
                               generator=generator)
    names.extend(f"random_cap4_{i}" for i in range(4))
    masks.extend(capped_topk(scores, 4) for scores in random_scores)
    return names, torch.stack(masks)


def evaluate(root: Path, output: Path, device: torch.device, steps: int) -> dict:
    path = output / "evaluation.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    data = torch.load(root / "features.pt", map_location="cpu", weights_only=True)
    train = tuple(t.to(device) for t in data["train"])
    validation = tuple(t.to(device) for t in data["validation"])
    test = tuple(t.to(device) for t in data["test"])
    names, masks_cpu = make_masks(root)
    masks = masks_cpu.to(device)
    repeats = 4
    expanded = masks.repeat_interleave(repeats, dim=0)
    count = len(expanded)
    # All variants begin with the same weights for each task and repeat.
    generator = torch.Generator(device=device).manual_seed(20261121)
    initial_weight = torch.randn(2, repeats, exp.FEATURES, exp.HIDDEN,
                                 generator=generator, device=device) * .08
    initial_readout = torch.randn(2, repeats, exp.HIDDEN,
                                  generator=generator, device=device) * .08
    weight = torch.nn.Parameter(initial_weight[:, None].expand(
        2, len(names), repeats, exp.FEATURES, exp.HIDDEN).clone().reshape(
            2, count, exp.FEATURES, exp.HIDDEN))
    bias = torch.nn.Parameter(torch.zeros(2, count, exp.HIDDEN, device=device))
    readout = torch.nn.Parameter(initial_readout[:, None].expand(
        2, len(names), repeats, exp.HIDDEN).clone().reshape(2, count, exp.HIDDEN))
    offset = torch.nn.Parameter(torch.zeros(2, count, device=device))
    optimizer = torch.optim.Adam((weight, bias, readout, offset), lr=.003)
    train_gen = torch.Generator(device=device).manual_seed(20261122)
    val_gen = torch.Generator(device=device).manual_seed(20261123)
    val_batches = [exp.balanced_batch(validation, digit, 512, val_gen)
                   for digit in data["pair"]]
    best = torch.full((2, count), float("inf"), device=device)
    best_step = torch.zeros(2, count, dtype=torch.int32, device=device)
    best_values = None
    stale = 0
    for step in tqdm(range(1, steps + 1), desc="coverage diagnostic", unit="step",
                     mininterval=2):
        losses = []
        for task, digit in enumerate(data["pair"]):
            x, y = exp.balanced_batch(train, digit, 128, train_gen)
            logits = exp.predict(x, expanded, weight[task], bias[task],
                                 readout[task], offset[task])
            losses.append(F.binary_cross_entropy_with_logits(
                logits, y.expand_as(logits), reduction="none").mean(-1))
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).sum().backward()
        optimizer.step()
        if step % 100:
            continue
        with torch.no_grad():
            val = torch.stack([
                exp.balanced_bce(exp.predict(x, expanded, weight[t], bias[t],
                                              readout[t], offset[t]), y)
                for t, (x, y) in enumerate(val_batches)])
            improved = val < best - .0001
            best = torch.where(improved, val, best)
            best_step = torch.where(improved, step, best_step)
            values = (weight.detach(), bias.detach(), readout.detach(),
                      offset.detach())
            if best_values is None:
                best_values = tuple(value.clone() for value in values)
            else:
                best_values = tuple(torch.where(
                    improved.reshape(*improved.shape,
                                     *((1,) * (value.ndim - improved.ndim))),
                    value, previous)
                    for value, previous in zip(values, best_values))
            stale = 0 if improved.any() else stale + 1
            if stale >= 15:
                break
    assert best_values is not None
    metrics = {"bce": [], "balanced_accuracy": []}
    for task, digit in enumerate(data["pair"]):
        x, labels = test
        target = (labels == digit).float()
        with torch.no_grad():
            logits = exp.predict(x, expanded,
                                 *(value[task] for value in best_values))
            metrics["bce"].append(exp.balanced_bce(logits, target).cpu().reshape(
                len(names), repeats))
            metrics["balanced_accuracy"].append(
                exp.balanced_accuracy(logits, target).cpu().reshape(
                    len(names), repeats))
    result = {"names": names, "masks": masks_cpu, "pair": data["pair"],
              "metrics": {key: torch.stack(value) for key, value in metrics.items()},
              "validation_bce": best.cpu().reshape(2, len(names), repeats),
              "best_step": best_step.cpu().reshape(2, len(names), repeats),
              "settings": {"steps": steps, "actual_steps": step, "repeats": repeats,
                           "plateau": stale >= 15, "caps": [0, 16, 8, 4]}}
    save(result, path)
    return result


def write_report(output: Path, result: dict) -> None:
    names = result["names"]
    metrics = result["metrics"]
    caps = (0, 16, 8, 4)
    labels = {"vae0": "VAE цифры 3", "vae1": "VAE цифры 8",
              "consensus": "Agreement, своя BCE, λ=1",
              "shared_consensus": "Agreement, обе BCE, λ=1",
              "mean0": "Средняя карта банка 3", "mean1": "Средняя карта банка 8",
              "mean_consensus": "Средняя карта двух банков"}

    def average(method: str, key: str) -> float:
        if method.startswith(("vae0_", "mean0_")) or method == "direct0":
            tasks = (0,)
        elif method.startswith(("vae1_", "mean1_")) or method == "direct1":
            tasks = (1,)
        else:
            tasks = (0, 1)
        return float(torch.stack([metrics[key][task, names.index(method)].mean()
                                  for task in tasks]).mean())

    rows = []
    for source in SOURCES:
        for cap in caps:
            name = f"{source}_cap{cap}"
            mask = result["masks"][names.index(name)]
            rows.append((labels[source], "top-K" if cap == 0 else str(cap),
                         int(mask.bool().any(-1).sum()),
                         average(name, "balanced_accuracy"),
                         average(name, "bce")))
    for task in range(2):
        name = f"direct{task}"
        mask = result["masks"][names.index(name)]
        rows.append((f"Прямая карта MLP, {result['pair'][task]}", "—",
                     int(mask.bool().any(-1).sum()),
                     average(name, "balanced_accuracy"), average(name, "bce")))
    random_names = [f"random{i}" for i in range(4)]
    random_acc = float(metrics["balanced_accuracy"][:, [names.index(n) for n in random_names]].mean())
    random_bce = float(metrics["bce"][:, [names.index(n) for n in random_names]].mean())
    random_coverage = float(torch.stack([
        result["masks"][names.index(n)].bool().any(-1).sum() for n in random_names
    ]).float().mean())
    rows.append(("Случайные маски", "—", round(random_coverage),
                 random_acc, random_bce))
    capped_random_names = [f"random_cap4_{i}" for i in range(4)]
    capped_random_acc = float(metrics["balanced_accuracy"][:,
        [names.index(n) for n in capped_random_names]].mean())
    capped_random_bce = float(metrics["bce"][:,
        [names.index(n) for n in capped_random_names]].mean())
    capped_random_coverage = float(torch.stack([
        result["masks"][names.index(n)].bool().any(-1).sum()
        for n in capped_random_names]).float().mean())
    rows.append(("Случайные маски с ограничением", "4",
                 round(capped_random_coverage), capped_random_acc,
                 capped_random_bce))

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for axis, source, title in ((axes[0], "vae0", "Цифра 3"),
                                (axes[1], "vae1", "Цифра 8")):
        task = 0 if source == "vae0" else 1
        xs = range(len(caps))
        ys = [float(metrics["balanced_accuracy"][task,
                            names.index(f"{source}_cap{cap}")].mean()) for cap in caps]
        axis.plot(xs, ys, "o-", color="tab:blue", label="VAE")
        axis.axhline(float(metrics["balanced_accuracy"][task,
                   names.index(f"direct{task}")].mean()), color="tab:green",
                   linestyle="--", label="Прямая карта")
        axis.axhline(float(metrics["balanced_accuracy"][task,
                   [names.index(n) for n in random_names]].mean()),
                   color="tab:gray", linestyle=":", label="Случайная")
        axis.axhline(float(metrics["balanced_accuracy"][task,
                   [names.index(n) for n in capped_random_names]].mean()),
                   color="tab:orange", linestyle="-.",
                   label="Случайная, cap 4")
        axis.set_xticks(list(xs), ("top-K", "16", "8", "4"))
        axis.set_xlabel("Максимум связей на пиксель")
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Сбалансированная точность")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "quality_by_cap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    for axis, source, title in ((axes[0], "consensus", "BCE своей задачи"),
                                (axes[1], "shared_consensus", "BCE обеих задач")):
        xs = range(len(caps))
        ys = [average(f"{source}_cap{cap}", "balanced_accuracy")
              for cap in caps]
        mean_ys = [average(f"mean_consensus_cap{cap}", "balanced_accuracy")
                   for cap in caps]
        axis.plot(xs, ys, "o-", label="Agreement")
        axis.plot(xs, mean_ys, "s--", label="Среднее карт банка")
        axis.axhline(random_acc, color="tab:gray", linestyle=":",
                     label="Случайная")
        axis.axhline(capped_random_acc, color="tab:orange", linestyle="-.",
                     label="Случайная, cap 4")
        axis.set_xticks(list(xs), ("top-K", "16", "8", "4"))
        axis.set_xlabel("Максимум связей на пиксель")
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Сбалансированная точность")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "agreement_quality_by_cap.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(9.8, 5.2))
    image = None
    for task, source in enumerate(("vae0", "vae1")):
        for column, cap in enumerate(caps):
            mask = result["masks"][names.index(f"{source}_cap{cap}")]
            projection = mask.float().mean(-1).reshape(28, 28)
            image = axes[task, column].imshow(projection, cmap="viridis",
                                              vmin=0, vmax=.25)
            axes[task, column].set_xticks([])
            axes[task, column].set_yticks([])
            if task == 0:
                axes[task, column].set_title("top-K" if cap == 0 else f"cap {cap}")
            if column == 0:
                axes[task, column].set_ylabel(f"Цифра {result['pair'][task]}")
    fig.subplots_adjust(left=.05, right=.87, top=.89, bottom=.04, wspace=.1, hspace=.12)
    colorbar = fig.add_axes((.90, .16, .018, .66))
    fig.colorbar(image, cax=colorbar, label="Доля связей на пиксель")
    fig.savefig(output / "coverage_maps.png", dpi=180)
    plt.close(fig)

    lines = ["# MNIST8m: проверка покрытия пикселей при фиксированном VAE", "",
             "Использованы уже обученные VAE и найденные z для цифр 3 и 8 против "
             "остальных. Из тех же логитов декодера взяты ровно 2509 связей "
             "(5%). Для каждого пикселя разрешено не более 16, 8 или 4 связей; "
             "обычный top-K ограничений не имеет. Такой же предел 4 применён "
             "к случайным маскам. Также проверены общие маски "
             "agreement (λ=1), средние карты каждого банка и их среднее. "
             "Для каждой маски обучены "
             "четыре новых MLP с одинаковой инициализацией и пакетами данных. "
             "На тесте измерены точность и BCE; лучшая эпоха выбрана по "
             "валидации.", "",
             "| Маска | Предел связей/пиксель | Пикселей со связью | "
             "Точность ↑ | BCE ↓ |", "|---|---:|---:|---:|---:|"]
    for label, cap, coverage, acc, bce in rows:
        lines.append(f"| {label} | {cap} | {coverage} | {acc:.4f} | {bce:.4f} |")
    lines.extend(["", "Показатели VAE и карты отдельного банка даны на своей задаче; "
                  "для agreement и случайной маски — среднее по двум задачам. "
                  "Во всех строках ровно 5% связей.", "",
                  "![Качество отдельных VAE при разных ограничениях](quality_by_cap.png)", "",
                  "![Качество общей маски при разных ограничениях]"
                  "(agreement_quality_by_cap.png)", "",
                  "![Распределение связей VAE по пикселям](coverage_maps.png)", "",
                  f"Обучение: {result['settings']['actual_steps']} шагов из "
                  f"{result['settings']['steps']}; "
                  f"{'остановка по плато' if result['settings']['plateau'] else 'лимит шагов'}. "
                  "Индивидуальные маски и численные результаты сохранены в "
                  "`evaluation.pt`.", "",
                  "**Вывод.** При том же VAE и тех же z распределение связей "
                  "по пикселям повысило качество отдельных масок и общей "
                  "маски. Случайной маске такое ограничение почти не помогло. "
                  "Значит, существенная часть потери качества возникала при "
                  "глобальном top-K: он выбирал слишком много связей у "
                  "одних и тех же пикселей. При cap=4 общий agreement по обеим "
                  "задачам достигает близкой к среднему карт банка точности, "
                  "но преимущество перед этим простым контролем невелико. "
                  "Проверка проведена на двух цифрах и четырёх повторениях; "
                  "она не показывает, что VAE всегда будет лучше среднего карт.", ""])
    (output / "report.md").write_text("\n".join(lines))
    assets = Path("mds/assets/2026-09-26")
    assets.mkdir(parents=True, exist_ok=True)
    published = "\n".join(lines)
    for figure in ("quality_by_cap.png", "agreement_quality_by_cap.png",
                   "coverage_maps.png"):
        target = f"mnist8m_coverage_{figure}"
        shutil.copy2(output / figure, assets / target)
        published = published.replace(f"({figure})",
                                      f"(assets/2026-09-26/{target})")
    published += ("\n[Код эксперимента](../pattern/mnist8m_coverage_diagnostic.py) "
                  "· [сырые результаты](../pattern/outputs/mnist8m_raw_mlp_bce/coverage_diagnostic/)\n")
    Path("mds/MNIST8M_COVERAGE_DIAGNOSTIC_2026-09-26.md").write_text(published)
    print(output / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("pattern/outputs/mnist8m_raw_mlp_bce/pair38_5pct"))
    parser.add_argument("--out", type=Path,
                        default=Path("pattern/outputs/mnist8m_raw_mlp_bce/coverage_diagnostic"))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--steps", type=int, default=15000)
    args = parser.parse_args()
    exp.K = round(.05 * exp.FEATURES * exp.HIDDEN)
    args.out.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.root, args.out, torch.device(args.device), args.steps)
    write_report(args.out, result)


if __name__ == "__main__":
    main()

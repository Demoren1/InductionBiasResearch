"""Post-hoc report of the fixed decoder-agreement experiment; never selects masks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/decoder_agreement/seed_20260906")
    args = parser.parse_args()
    root = args.out_dir
    summary = json.loads((root / "summary.json").read_text())
    opt = torch.load(root / "optimization.pt", weights_only=True, map_location="cpu")
    random = torch.load(root / "random_search.pt", weights_only=True, map_location="cpu")
    training = json.loads((root / "training.json").read_text())
    diagnostics = {"optimization": {}, "comparisons": {}}
    masks = torch.load(root / "masks.pt", weights_only=True, map_location="cpu")
    # Column-bit encoding gives an exact orbit key without using any gold support.
    bit_weights = (2 ** torch.arange(8))[None, :, None]
    diagnostics["unique_masks_modulo_columns"] = {
        name: int(torch.unique((value.long() * bit_weights).sum(1).sort(1).values, dim=0).size(0))
        for name, value in masks.items()}
    for stage in ("initial", "final"):
        soft = torch.cat([opt[f"{stage}_soft1"], opt[f"{stage}_soft2"]])
        norm = torch.cat([opt[f"{stage}_z1"], opt[f"{stage}_z2"]]).norm(dim=1)
        diagnostics["optimization"][stage] = {
            "mean_soft_agreement_loss": float(opt[f"{stage}_loss"].mean()),
            "mean_softness_s_times_one_minus_s": float((soft * (1 - soft)).mean()),
            "latent_norm_mean": float(norm.mean()), "latent_norm_max": float(norm.max()),
            "latents_at_radius_bound": int((norm >= 8 - 1e-5).sum())}
    diagnostics["optimization"]["random_search_mean_loss"] = float(random["loss"].mean())
    initial_iou = np.mean([summary["structure"][f"initial_vae{i}"]["iou"]["values"] for i in (1, 2)], axis=0)
    final_iou = np.mean([summary["structure"][f"optimized_vae{i}"]["iou"]["values"] for i in (1, 2)], axis=0)
    diagnostics["comparisons"]["optimized_minus_initial_iou"] = {
        "mean": float((final_iou - initial_iou).mean()),
        "starts_improved": int((final_iou > initial_iou).sum()),
        "starts_equal": int((final_iou == initial_iou).sum()), "total_starts": len(final_iou)}
    for reference in ("initial", "random_search"):
        before, after = [], []
        for task in summary["tasks"].values():
            for i in (1, 2):
                before.append(task[f"{reference}_vae{i}"]["accuracy"]["values"])
                after.append(task[f"optimized_vae{i}"]["accuracy"]["values"])
        delta = np.asarray(after) - np.asarray(before)
        diagnostics["comparisons"][f"optimized_minus_{reference}_accuracy"] = {
            "mean": float(delta.mean()), "per_task_decoder_start_differences": delta.tolist()}
    (root / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    history = np.asarray(opt["history"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    axes[0].semilogy(history.mean(axis=1), label="Gradient search: mean over 64 starts")
    axes[0].axhline(float(random["loss"].mean()), color="#b39162", linestyle="--", label="Random search: final mean")
    axes[0].set(xlabel="Adam updates", ylabel="Soft agreement MSE", title="Agreement-only optimization")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    for stage, losses, iou, color in (("Initial", opt["initial_loss"], initial_iou, "#7295b6"),
                                     ("Optimized", opt["final_loss"], final_iou, "#328369")):
        axes[1].scatter(losses.numpy(), iou, alpha=.7, s=22, label=stage, color=color)
    axes[1].set(xscale="log", xlabel="Pair agreement MSE", ylabel="Mean ideal-support IoU of pair", title="Gold structure measured only after search")
    axes[1].legend()
    axes[1].grid(alpha=.2)
    for extension in ("png", "pdf"):
        fig.savefig(root / f"agreement_diagnostics.{extension}", dpi=160)
    plt.close(fig)

    names = {"initial_vae1": "VAE 1: до поиска", "initial_vae2": "VAE 2: до поиска",
             "optimized_vae1": "VAE 1: оптимизация z", "optimized_vae2": "VAE 2: оптимизация z",
             "random_search_vae1": "VAE 1: случайный поиск", "random_search_vae2": "VAE 2: случайный поиск",
             "random_exact32": "Случайная exact-32", "ideal": "Идеальная поддержка"}
    lines = ["# Два замороженных VAE: поиск согласованных масок", "", "Дата: 2026-09-06.", "",
             "Полный запуск на GPU: два VAE seeds 42/43, один OOD split seed 42, 64 независимых старта. "
             "Оптимизация использует только расстояние мягких top-32 масок после Hungarian-перестановки колонок. "
             "Gold и held-out метки не участвуют в поиске, выборе checkpoint или лучшей итерации.", "",
             "## Согласие двух генераторов", "",
             "| Состояние | Soft MSE | IoU между бинарными масками | Среднее число различающихся битов | Полные совпадения |",
             "|---|---:|---:|---:|---:|"]
    for name, softloss in (("initial", opt["initial_loss"]), ("optimized", opt["final_loss"]), ("random_search", random["loss"])):
        record = summary["agreement"][name]
        lines.append(f"| {name} | {float(softloss.mean()):.6f} | {record['iou']['mean']:.4f} | {record['hamming']['mean']:.2f} | {record['exact_matches']}/64 |")
    lines += ["", "## Независимая проверка структуры и качества", "",
              "Accuracy — среднее по четырём held-out задачам, 64 маски на метод; свежие MLP, "
              "2000 шагов, одинаковые начальные веса и minibatches между методами. "
              "IoU — с идеальной поддержкой после оптимального сопоставления hidden columns.", "",
              "| Источник маски | Gold IoU | Accuracy | BCE | Различных масок | Покрыто точных окон из 5 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, title in names.items():
        s, d = summary["structure"][name], summary["downstream"][name]
        lines.append(f"| {title} | {s['iou']['mean']:.4f} | {d['accuracy']['mean']:.4f} | {d['bce']['mean']:.4f} | {s['unique_masks']} | {s['mean_distinct_exact_windows']:.2f} |")
    delta = diagnostics["comparisons"]
    lines += ["", f"Среднее изменение Gold IoU после оптимизации: **{delta['optimized_minus_initial_iou']['mean']:+.4f}**; "
              f"рост у {delta['optimized_minus_initial_iou']['starts_improved']}/64 пар стартов.", "",
              f"Среднее изменение accuracy, усредняя оба decoder: **{100*delta['optimized_minus_initial_accuracy']['mean']:+.2f} п.п.** "
              f"относительно исходных сэмплов и **{100*delta['optimized_minus_random_search_accuracy']['mean']:+.2f} п.п.** относительно случайного поиска.", "",
              "## Контроль качества VAE и ограничений поиска", ""]
    lines.append("Число разных масок с точностью до перестановки колонок: " + "; ".join(
        f"{name}: {count}/64" for name, count in diagnostics["unique_masks_modulo_columns"].items()) + ".")
    lines.append("")
    for result in training["results"]:
        lines.append(f"- VAE seed {result['seed']}: best validation loss {result['best_val']:.4f}; "
                     f"средний std decoder probabilities по prior-сэмплам {result['noncollapse']['decoder_probability_feature_std_mean']:.4f}.")
    for stage in ("initial", "final"):
        d = diagnostics["optimization"][stage]
        lines.append(f"- {stage}: средняя норма latent {d['latent_norm_mean']:.3f}, "
                     f"на границе радиуса 8 — {d['latents_at_radius_bound']}/128; "
                     f"среднее S(1−S) = {d['mean_softness_s_times_one_minus_s']:.4f} (0 для бинарных значений, 0.25 для 0.5).")
    lines += ["", "Это один pair seeds и одно разбиение задач. 64 старта не заменяют повторение с другими VAE. "
              "Случайный поиск сопоставлен по числу независимых пар-кандидатов (1001 на старт), а не по времени вычислений. "
              "Радиус latent ограничен, но это не гарантирует нахождение на типичной области распределения декодируемых карт.", "",
              "## Артефакты", "",
              "- `protocol.json`, `training.json`, `search_provenance.json`: настройки и контроль происхождения.",
              "- `optimization.pt`, `random_search.pt`, `masks.pt`: латентные векторы, история и фиксированные маски.",
              "- `summary.json`, `diagnostics.json`, `eval_<pattern>.json`: структурные и функциональные метрики.",
              "- `summary.png`, `agreement_diagnostics.png`, `mask_examples.png`: графики и первые четыре старта без отбора по качеству.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"optimization": diagnostics["optimization"], "comparisons": {k: {kk: vv for kk, vv in v.items() if kk != "per_task_decoder_start_differences"} for k, v in delta.items()}}, indent=2))


if __name__ == "__main__":
    main()

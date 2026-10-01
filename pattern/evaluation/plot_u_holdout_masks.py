"""Plot binary support and sharing codes of U from the 12/4 pattern experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import torch

from pattern.evaluation.u_reparameterization_pilot import SharedUModel, analytic_assignment


def load_codes(root: Path, method: str, fold: int, seed: int) -> tuple[np.ndarray, float]:
    directory = root / f"{method}_fold{fold}_seed{seed}"
    audit = json.loads((directory / "toeplitz_audit.json").read_text())
    saved = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model = SharedUModel(method, seed, 11)
    model.load_state_dict(saved["state_dict"])
    with torch.no_grad():
        raw = model.assignment_matrix().argmax(1).reshape(11, 8).numpy()
    codes = raw[:, audit["hidden_permutation_new_to_old"]]
    gold = analytic_assignment(11).argmax(1).reshape(11, 8).numpy() != 0
    support = codes != 0
    iou = (support & gold).sum() / (support | gold).sum()
    assert support.sum() == 32 and abs(iou - audit["toeplitz_support_iou"]) < 1e-9
    return codes, iou


def draw_support(ax: plt.Axes, codes: np.ndarray, title: str,
                 color: str = "#2463a5", labels: bool = False) -> None:
    ax.imshow(codes != 0, cmap=ListedColormap(["#ffffff", color]),
              vmin=0, vmax=1, interpolation="none")
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(11))
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which="minor", color="#bcc6d0", linewidth=.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    if not labels:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.tick_params(length=0, labelsize=8)
    ax.set_title(title, fontsize=11)


def plot_seed42(root: Path, out: Path) -> None:
    gold = analytic_assignment(11).argmax(1).reshape(11, 8).numpy()
    fig, axes = plt.subplots(4, 3, figsize=(10, 14), constrained_layout=True)
    for fold in range(4):
        draw_support(axes[fold, 0], gold, "Идеальная полоса", "#36865b", True)
        axes[fold, 0].set_ylabel(f"Группа {fold}\nВход i", fontsize=10)
        for column, method, label in [(1, "learned_binary", "4 кода"),
                                      (2, "learned_binary_40", "40 кодов")]:
            codes, iou = load_codes(root, method, fold, 42)
            draw_support(axes[fold, column], codes,
                         f"{label} · IoU {iou:.3f}", labels=True)
        for ax in axes[fold]:
            ax.set_xlabel("Скрытый нейрон j", fontsize=9)
    fig.suptitle("Маски общей U после обучения на 12 паттернах · seed 42\n"
                 "Цвет = связь есть; белый = связи нет. Каждая маска: ровно 32 из 88.",
                 fontsize=13)
    fig.savefig(out / "u_masks_seed42_all_folds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_all(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(4, 6, figsize=(17, 14), constrained_layout=True)
    for fold in range(4):
        for column, method in enumerate(("learned_binary", "learned_binary_40")):
            for seed_offset, seed in enumerate((42, 43, 44)):
                ax = axes[fold, column * 3 + seed_offset]
                codes, iou = load_codes(root, method, fold, seed)
                draw_support(ax, codes,
                             f"{'4' if column == 0 else '40'} кодов · s{seed}\n"
                             f"IoU {iou:.3f}")
                if seed_offset == 0 and column == 0:
                    ax.set_ylabel(f"Группа {fold}", fontsize=10)
    fig.suptitle("Все 24 выученные маски U · 4 группы × 3 начальных состояния × 2 размера кода\n"
                 "Столбцы скрытого слоя выровнены только для сравнения; 32 связи в каждой маске.",
                 fontsize=14)
    fig.savefig(out / "u_masks_all_24.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_codes(ax: plt.Axes, codes: np.ndarray, title: str) -> None:
    unique = np.unique(codes[codes != 0])
    ranks = np.zeros_like(codes)
    for rank, code in enumerate(unique, start=1):
        ranks[codes == code] = rank
    palette = ["#ffffff", *plt.cm.hsv(np.linspace(.02, .96, len(unique)))]
    ax.imshow(ranks, cmap=ListedColormap(palette), vmin=0,
              vmax=len(unique), interpolation="none")
    ax.set_xticks(np.arange(8))
    ax.set_yticks(np.arange(11))
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True)
    ax.grid(which="minor", color="#9daab9", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0, labelsize=9)
    for i, j in np.argwhere(codes != 0):
        ax.text(j, i, str(codes[i, j]), ha="center", va="center",
                color="#101820", fontsize=8, fontweight="bold",
                bbox={"facecolor": "white", "alpha": .65, "edgecolor": "none", "pad": .4})
    ax.set_title(f"{title} · {len(unique)} разных ненулевых кодов", fontsize=12)
    ax.set_xlabel("Скрытый нейрон j")
    ax.set_ylabel("Вход i")


def plot_codes(root: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
    for ax, method, label in zip(axes,
                                 ("learned_binary", "learned_binary_40"),
                                 ("4 кода", "40 кодов")):
        codes, _ = load_codes(root, method, 0, 42)
        draw_codes(ax, codes, label)
    fig.suptitle("Коды полной бинарной U · группа 0, seed 42\n"
                 "Одинаковые числа означают один общий коэффициент v внутри задачи; 0 скрыт.",
                 fontsize=12)
    fig.savefig(out / "u_codes_fold0_seed42.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plot_seed42(args.root, args.out)
    plot_all(args.root, args.out)
    plot_codes(args.root, args.out)
    print("Saved three mask figures to", args.out)


if __name__ == "__main__":
    main()

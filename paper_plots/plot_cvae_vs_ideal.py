"""Plot gap-OOD CVAE samples against their ideal Toeplitz supports."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch


ROOT = Path(__file__).resolve().parents[1]
MOTIF_ROOT = ROOT / "motif_pair"
sys.path.insert(0, str(MOTIF_ROOT))

import config  # noqa: E402
from evaluation.baselines import gold_mask  # noqa: E402
from evaluation.structural import best_permutation_iou  # noqa: E402
from models.cvae import CVAE, checkpoint_condition_encoding  # noqa: E402


DEFAULT_INTERPOLATION_CHECKPOINT = (
    MOTIF_ROOT
    / "outputs/ood/gap_interp_g05_g08_seed_42/generative/cvae/best.pt"
)
DEFAULT_EXTRAPOLATION_CHECKPOINT = (
    MOTIF_ROOT
    / "outputs/ood/gap_extrap_g03_g04_seed_42/generative/cvae/best.pt"
)
DEFAULT_OUTPUT = ROOT / "paper/plots/gap_ood_cvae_vs_ideal.pdf"
DISPLAY_COLUMNS = (
    ("Interpolation", 5),
    ("Interpolation", 8),
    ("Extrapolation", 3),
    ("Extrapolation", 4),
)
SAMPLE_SEED = 20260824


def load_cvae(path: Path) -> CVAE:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    encoding = checkpoint_condition_encoding(payload)
    model = CVAE(
        payload.get("mask_dim", config.MASK_DIM),
        payload.get("latent_dim", config.LATENT_DIM),
        payload.get("hidden", config.CVAE_HIDDEN),
        payload.get("cond_dim", config.COND_DIM),
        condition_encoding=encoding,
    )
    model.load_state_dict(payload["state_dict"])
    return model.eval()


@torch.no_grad()
def fixed_seed_samples(model: CVAE) -> dict[int, torch.Tensor]:
    """Reproduce the fixed latent draws used by the full diagnostic figure."""
    all_tasks = [f"A000_B001_G{gap:02d}" for gap in config.GAPS]
    generator = torch.Generator().manual_seed(SAMPLE_SEED)
    z = torch.randn(len(all_tasks), model.latent_dim, generator=generator)
    conditions = model.condition(all_tasks)
    scores = torch.sigmoid(model.decode(z, conditions))
    indices = scores.topk(config.K_ACTIVE, dim=-1).indices
    masks = torch.zeros_like(scores).scatter_(1, indices, 1.0)
    return {
        gap: mask.reshape(config.SEQ_LEN, config.H)
        for gap, mask in zip(config.GAPS, masks)
    }


def align_to_ideal(mask: torch.Tensor, ideal: torch.Tensor):
    diagnostic = best_permutation_iou(mask, ideal)
    permutation = torch.as_tensor(diagnostic["permutation"], dtype=torch.long)
    aligned = mask[:, torch.argsort(permutation)]
    return aligned, float(diagnostic["iou"])


def plot(interpolation_checkpoint: Path, extrapolation_checkpoint: Path,
         output: Path) -> None:
    models = {
        "Interpolation": load_cvae(interpolation_checkpoint),
        "Extrapolation": load_cvae(extrapolation_checkpoint),
    }
    samples = {name: fixed_seed_samples(model) for name, model in models.items()}
    cmap = ListedColormap(("#ffffff", "#111827"))

    fig, axes = plt.subplots(
        2,
        len(DISPLAY_COLUMNS),
        figsize=(7.2, 3.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    ticks = (0, 5, 10, 15)
    for column, (profile, gap) in enumerate(DISPLAY_COLUMNS):
        task = f"A000_B001_G{gap:02d}"
        ideal = gold_mask(task).reshape(config.SEQ_LEN, config.H)
        aligned, iou = align_to_ideal(samples[profile][gap], ideal)
        for row, matrix in enumerate((aligned, ideal)):
            axis = axes[row, column]
            axis.imshow(
                matrix,
                origin="lower",
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
                rasterized=False,
            )
            axis.set_xticks(ticks)
            axis.set_yticks(ticks)
            axis.tick_params(labelsize=7, length=2)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
        axes[0, column].set_title(
            f"{profile}\nheld-out gap $g={gap}$",
            fontsize=8.5,
        )
        axes[0, column].text(
            0.04,
            0.05,
            f"IoU = {iou:.2f}",
            transform=axes[0, column].transAxes,
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
                "linewidth": 0.5,
            },
        )

    axes[0, 0].set_ylabel("CVAE sample\n(aligned)", fontsize=9)
    axes[1, 0].set_ylabel("Ideal support", fontsize=9)
    fig.supxlabel("hidden unit", fontsize=9)
    fig.supylabel("input position", fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", metadata={"Creator": __file__})
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interpolation-checkpoint",
        type=Path,
        default=DEFAULT_INTERPOLATION_CHECKPOINT,
    )
    parser.add_argument(
        "--extrapolation-checkpoint",
        type=Path,
        default=DEFAULT_EXTRAPOLATION_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    plot(args.interpolation_checkpoint, args.extrapolation_checkpoint, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()

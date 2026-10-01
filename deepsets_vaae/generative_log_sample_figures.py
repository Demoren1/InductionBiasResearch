"""Auxiliary logarithmic view of fixed generated/source functional maps."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs/deepsets_vaae/20261001_other_generators_corrected"
DEFAULT_CANONICAL = ROOT / "outputs/deepsets_vaae/20261001_converged_functional_vae"
METHODS = ("diffusion", "flow_matching", "gnn_flow", "set_transformer_flow", "gan")


def build(out: Path, canonical: Path) -> None:
    out, canonical = Path(out), Path(canonical)
    target = out / "sample_figures"
    if not target.is_dir():
        raise FileNotFoundError(f"Run generative_sample_figures first: {target}")
    source = canonical / "seed_4100" / "functional" / "functional_vae_arrays.npz"
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source) as arrays:
        train = torch.as_tensor(arrays["function_train_aligned"][0, 0], dtype=torch.float32)
        heldout = torch.as_tensor(arrays["function_validation_aligned"][0, 0], dtype=torch.float32)
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(10.8, 13), constrained_layout=True)
    norm = LogNorm(vmin=1e-5, vmax=1.)
    for row, method in enumerate(METHODS):
        payload = torch.load(out / "seed_4100" / method / "samples.pt", map_location="cpu", weights_only=True)
        generated = torch.as_tensor(payload["samples"], dtype=torch.float32)[0, 0]
        for ax, value, title in zip(axes[row], (train, heldout, generated),
                                    ("source train[task0,map0]", "source held-out[task0,map0]", f"{method} sample[task0,0]")):
            if tuple(value.shape) != (784, 32) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"invalid fixed map for {method}")
            image = ax.imshow(value.clamp_min(1e-5).T.numpy(), aspect="auto", cmap="viridis", norm=norm,
                              interpolation="nearest")
            ax.set_title(title, fontsize=8); ax.set_xlabel("pixel (784)"); ax.set_ylabel("hidden (32)")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=.72, label="log colour; values clamped to [1e-5, 1]")
    fig.savefig(target / "fixed_generated_maps_vs_source_log.png", dpi=180, bbox_inches="tight")
    fig.savefig(target / "fixed_generated_maps_vs_source_log.pdf", bbox_inches="tight")
    plt.close(fig)
    with (target / "CAPTIONS.md").open("a", encoding="utf-8") as stream:
        stream.write("\n## Вспомогательная логарифмическая шкала\n\n")
        stream.write("`fixed_generated_maps_vs_source_log.png` повторяет те же фиксированные train/held-out/generated элементы, что и raw figure, без отбора. Цветовая шкала общая LogNorm `[1e-5, 1]`; значения ниже `1e-5` отображены как `1e-5`. Эта фигура нужна только для различимости малых значений и не заменяет raw `[0,1]` figure.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    args = parser.parse_args()
    build(args.out, args.canonical)


if __name__ == "__main__":
    main()

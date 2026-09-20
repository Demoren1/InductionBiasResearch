"""Measure a zero-label-free padding correction on already trained models.

The notebook trains with capacity ten and sums the encoded padding image.
Subtracting (test_length - 10) times its scalar contribution preserves every
training prediction while extending the same per-image contributions to longer
sets. No test target is used to determine the correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .run import ARMS, TEST_LENGTHS, ImageSum, batch, load_split


COLORS = {
    "paper_mlp": "#242424",
    "learned_z": "#d95f02",
    "fixed_z": "#7570b3",
    "no_z": "#1b9e77",
}
LABELS = {
    "paper_mlp": "Deep Sets MLP",
    "learned_z": "learned z",
    "fixed_z": "fixed z",
    "no_z": "no z",
}


@torch.no_grad()
def evaluate_pair(model: ImageSum, images: np.ndarray, split: dict[str, np.ndarray],
                  length: int, pad_score: float, device: torch.device,
                  pixel_scale: float, batch_size: int) -> dict:
    raw_errors = []
    corrected_errors = []
    raw_exact = []
    corrected_exact = []
    for start in range(0, len(split["targets"]), batch_size):
        ids = np.arange(start, min(start + batch_size, len(split["targets"])))
        x, mask, target = batch(images, split, ids, device, pixel_scale)
        raw = model(x, mask)
        corrected = raw - (length - 10) * pad_score
        raw_errors.append((raw - target).cpu().numpy())
        corrected_errors.append((corrected - target).cpu().numpy())
        raw_exact.append((raw.round() == target).cpu().numpy())
        corrected_exact.append((corrected.round() == target).cpu().numpy())

    def metrics(parts: list[np.ndarray], exact_parts: list[np.ndarray]) -> dict[str, float]:
        error = np.concatenate(parts).astype(np.float64)
        return {
            "mae": float(np.abs(error).mean()),
            "exact_round_accuracy": float(np.concatenate(exact_parts).mean()),
            "mean_error": float(error.mean()),
            "median_error": float(np.median(error)),
        }

    return {"raw": metrics(raw_errors, raw_exact),
            "corrected": metrics(corrected_errors, corrected_exact)}


def make_plot(results: dict, output: Path) -> None:
    for seed in sorted({int(seed) for arm in results["runs"].values() for seed in arm}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        for arm in ARMS:
            run = results["runs"].get(arm, {}).get(str(seed))
            if run is None:
                continue
            lengths = sorted(int(length) for length in run["lengths"])
            for variant, style, alpha in (("raw", "--", 0.55),
                                          ("corrected", "-", 1.0)):
                values = [run["lengths"][str(length)][variant] for length in lengths]
                label = f"{LABELS[arm]} ({variant})"
                axes[0].plot(lengths, [v["exact_round_accuracy"] * 100 for v in values],
                             style, color=COLORS[arm], alpha=alpha, label=label)
                axes[1].plot(lengths, [v["mae"] for v in values], style,
                             color=COLORS[arm], alpha=alpha, label=label)
        axes[0].set_ylabel("Exact sum accuracy, %")
        axes[0].set_ylim(0, 102)
        axes[1].set_ylabel("Test MAE")
        axes[1].set_yscale("log")
        for ax in axes:
            ax.set_xlabel("Set length")
            ax.set_xticks(TEST_LENGTHS)
            ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8, ncol=2, loc="lower left")
        fig.suptitle(f"Padding correction, seed {seed}; solid: corrected, dashed: raw")
        fig.savefig(output.with_name(f"{output.stem}_seed{seed}.png"), dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/results_notebook_padding"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--output", type=Path, default=Path(
        "deepsets_z/mnist8m/padding_correction.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    images = np.load(args.data_dir / "images.npy", mmap_mode="r")
    tests = {length: load_split(args.data_dir / "sets_authors" / f"test_{length}.npz")
             for length in TEST_LENGTHS}
    runs = {arm: {} for arm in ARMS}
    for result_file in sorted(args.results.glob("*.json")):
        record = json.loads(result_file.read_text())
        arm, seed = record["arm"], record["seed"]
        config = record["config"]
        if config["padding_mode"] != "notebook" or config["sets_subdir"] != "sets_authors":
            raise ValueError(f"Unexpected protocol: {result_file}")
        model = ImageSum(arm, seed, paper_initialization=config["paper_initialization"],
                         padding_mode="notebook").to(device).eval()
        saved = torch.load(args.results / f"{arm}_seed{seed}_last.pt",
                           map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        x0 = torch.from_numpy(np.asarray(images[0], dtype=np.float32).copy()).to(device)
        x0 = x0.reshape(1, 1, 784).div(config["pixel_scale"])
        with torch.no_grad():
            bias = float(model.readout.bias.item())
            pad_score = float(model(x0, torch.ones((1, 1), device=device, dtype=torch.bool)).item() - bias)
        by_length = {}
        for length, split in tests.items():
            measurements = evaluate_pair(model, images, split, length, pad_score,
                                         device, config["pixel_scale"], args.batch_size)
            reference = record["metrics"][f"test_{length}"]
            mae_drift = abs(measurements["raw"]["mae"] - reference["mae"])
            accuracy_drift = abs(measurements["raw"]["exact_round_accuracy"] -
                                 reference["exact_round_accuracy"])
            if mae_drift > 1e-4:
                raise AssertionError(f"Raw MAE does not match {result_file} at {length}")
            if accuracy_drift > 1e-4:
                raise AssertionError(f"Raw accuracy does not match {result_file} at {length}")
            measurements["raw_reference_drift"] = {"mae": mae_drift,
                                                    "exact_round_accuracy": accuracy_drift}
            if mae_drift > 1e-4 or accuracy_drift > 1e-4:
                print(f"NOTE {arm} seed={seed} length={length} "
                      f"raw_mae_drift={mae_drift:.5f} "
                      f"raw_accuracy_drift={accuracy_drift:.5f}", flush=True)
            by_length[str(length)] = measurements
        runs[arm][str(seed)] = {
            "pad_score": pad_score,
            "readout_bias": bias,
            "training_capacity": 10,
            "bias_plus_ten_pad_scores": bias + 10 * pad_score,
            "lengths": by_length,
        }
        last = by_length["50"]["corrected"]
        print(f"{arm} seed={seed} corrected_acc50={last['exact_round_accuracy']:.4f} "
              f"corrected_mae50={last['mae']:.4f}", flush=True)
    result = {"method": "pred_corrected = pred_raw - (test_length - 10) * pad_score",
              "test_targets_used_to_fit_correction": False,
              "runs": runs}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    make_plot(result, args.output)


if __name__ == "__main__":
    main()

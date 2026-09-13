"""Label-free agreement hyperparameter sweep on a disjoint VAE seed set.

This script deliberately reports only the agreement objective, hard agreement
between the two decoders, and latent-radius diagnostics.  It never imports the
gold mask or held-out task data, so the sweep cannot tune on evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.decoder_agreement import align_columns, optimize_agreement  # noqa: E402
from evaluation.run_decoder_agreement import load_models  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "outputs/decoder_agreement/hparam_tuning_20260912/vae_e240"
VARIANTS = (
    {"name": "baseline", "steps": 1000, "lr": 0.03, "temperature": 0.5, "radius": 8.0},
    {"name": "steps_2000", "steps": 2000, "lr": 0.03, "temperature": 0.5, "radius": 8.0},
    {"name": "radius_12", "steps": 1000, "lr": 0.03, "temperature": 0.5, "radius": 12.0},
    {"name": "steps_2000_radius_12", "steps": 2000, "lr": 0.03, "temperature": 0.5, "radius": 12.0},
    {"name": "temperature_025", "steps": 1000, "lr": 0.03, "temperature": 0.25, "radius": 8.0},
    {"name": "lr_001", "steps": 1000, "lr": 0.01, "temperature": 0.5, "radius": 8.0},
    {"name": "lr_01", "steps": 1000, "lr": 0.1, "temperature": 0.5, "radius": 8.0},
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hard_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    aligned = align_columns(first, second)
    intersection = (first * aligned).sum((1, 2))
    union = (first + aligned).clamp(max=1).sum((1, 2))
    exact = (first == aligned).all(2).all(1).float()
    return {"iou": float((intersection / union).mean()),
            "exact_fraction": float(exact.mean())}


def result_metrics(result: dict, radius: float) -> dict[str, float]:
    norm1 = result["final_z1"].norm(dim=1)
    norm2 = result["final_z2"].norm(dim=1)
    norms = torch.cat((norm1, norm2))
    history = torch.as_tensor(result["history"])
    return {
        "soft_mse": float(result["final_loss"].mean()),
        **hard_metrics(result["final_masks1"], result["final_masks2"]),
        "latent_norm_mean": float(norms.mean()),
        "latent_norm_max": float(norms.max()),
        "radius_boundary_fraction": float((norms >= radius - 1e-4).float().mean()),
        "last_100_soft_mse_mean": float(history[-min(100, len(history)):].mean()),
    }


def worker(pair_root: Path, output: Path, n_starts: int, seed: int) -> None:
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    models, provenance = load_models(pair_root, device)
    protocol_hash = sha256_file(pair_root / "protocol.json")
    payload = {"pair": pair_root.name, "protocol_sha256": protocol_hash,
               "model_checkpoints": [{"seed": item["seed"], "sha256": item["sha256"]}
                                     for item in provenance],
               "n_starts": n_starts, "seed": seed, "variants": {}}
    output.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        result_path = output / f"{variant['name']}.pt"
        if result_path.exists():
            saved = torch.load(result_path, map_location="cpu", weights_only=True)
            if (saved.get("settings") != variant
                    or saved.get("protocol_sha256") != protocol_hash
                    or saved.get("model_checkpoints") != payload["model_checkpoints"]):
                raise FileExistsError(f"incompatible existing result: {result_path}")
            result = saved["result"]
        else:
            print(f"[agreement-sweep] {pair_root.name} {variant['name']}", flush=True)
            result = optimize_agreement(
                *models, n_starts=n_starts, steps=variant["steps"], lr=variant["lr"],
                seed=seed, temperature=variant["temperature"], radius=variant["radius"],
                device=device,
            )
            torch.save({"settings": variant, "protocol_sha256": protocol_hash,
                        "model_checkpoints": payload["model_checkpoints"], "result": result}, result_path)
        payload["variants"][variant["name"]] = {
            "settings": variant,
            "metrics": result_metrics(result, variant["radius"]),
            "artifact": str(result_path),
            "artifact_sha256": sha256_file(result_path),
        }
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def describe(values: list[float], lower: float | None = None,
             upper: float | None = None) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) > 1:
        half = float(stats.t.ppf(0.975, len(array) - 1) * stats.sem(array))
    else:
        half = float("nan")
    low, high = mean - half, mean + half
    if lower is not None:
        low = max(lower, low)
    if upper is not None:
        high = min(upper, high)
    return {"mean": mean, "ci95_low": low, "ci95_high": high,
            "min": float(array.min()), "max": float(array.max())}


def aggregate(root: Path, pair_roots: list[Path], n_starts: int, seed: int) -> dict:
    records = []
    for pair_root in pair_roots:
        path = root / "agreement_sweep" / pair_root.name / "metrics.json"
        record = json.loads(path.read_text())
        if record["protocol_sha256"] != sha256_file(pair_root / "protocol.json"):
            raise ValueError(f"protocol hash mismatch: {pair_root}")
        if record["n_starts"] != n_starts or record["seed"] != seed:
            raise ValueError(f"sweep settings mismatch: {pair_root}")
        for model in record["model_checkpoints"]:
            checkpoint = pair_root / f"vae_{model['seed']}" / "cvae_best.pt"
            if sha256_file(checkpoint) != model["sha256"]:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        for name, entry in record["variants"].items():
            artifact = Path(entry["artifact"])
            if sha256_file(artifact) != entry["artifact_sha256"]:
                raise ValueError(f"artifact hash mismatch: {artifact}")
        records.append(record)
    summary = {"selection_data": "decoder agreement only; no gold masks or task labels",
               "pair_roots": [str(path) for path in pair_roots],
               "n_pairs": len(records), "n_starts_per_pair": n_starts, "seed": seed,
               "variants": {}}
    for variant in VARIANTS:
        name = variant["name"]
        metrics = records[0]["variants"][name]["metrics"]
        summary["variants"][name] = {
            "settings": variant,
            "metrics": {metric: describe(
                            [record["variants"][name]["metrics"][metric]
                             for record in records],
                            lower=0.0 if metric in {
                                "iou", "exact_fraction", "radius_boundary_fraction"} else None,
                            upper=1.0 if metric in {
                                "iou", "exact_fraction", "radius_boundary_fraction"} else None)
                        for metric in metrics},
        }
    # Predeclared label-free ordering: maximize exact hard agreement, then
    # minimize the differentiable objective.  Pair means are the units.
    ranking = sorted(
        VARIANTS,
        key=lambda variant: (-summary["variants"][variant["name"]]["metrics"]["exact_fraction"]["mean"],
                             summary["variants"][variant["name"]]["metrics"]["soft_mse"]["mean"]),
    )
    summary["label_free_ranking"] = [variant["name"] for variant in ranking]
    summary["selected"] = ranking[0]["name"]
    path = root / "agreement_sweep" / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    render(summary, path.parent)
    return summary


def render(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = summary["label_free_ranking"]
    labels = {
        "baseline": "baseline",
        "steps_2000": "2000 steps",
        "radius_12": "radius 12",
        "steps_2000_radius_12": "2000 steps + radius 12",
        "temperature_025": "temperature 0.25",
        "lr_001": "lr 0.01",
        "lr_01": "lr 0.1",
    }
    colors = ["#328369" if name == summary["selected"] else "#7295b6" for name in names]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for axis, metric, title in (
        (axes[0], "exact_fraction", "Exact hard-mask agreement"),
        (axes[1], "soft_mse", "Soft disagreement MSE"),
    ):
        values = [summary["variants"][name]["metrics"][metric] for name in names]
        means = np.array([value["mean"] for value in values])
        errors = np.array([[value["mean"] - value["ci95_low"] for value in values],
                           [value["ci95_high"] - value["mean"] for value in values]])
        for index, (mean, error, color) in enumerate(zip(means, errors.T, colors)):
            axis.errorbar(mean, index, xerr=error[:, None], fmt="o", color=color,
                          capsize=3, markersize=7)
        axis.set_yticks(y, [labels[name] for name in names])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.2)
        axis.set_title(title)
        axis.set_xlabel("pair mean and 95% t-CI (8 VAE pairs)")
    axes[0].set_xlim(0.89, 1.002)
    axes[1].set_xscale("log")
    fig.suptitle("Label-free agreement hyperparameter sweep")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"agreement_hparam_sweep.{extension}", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pairs", nargs="+", default=[f"pair_{seed}_{seed + 1}"
                                                        for seed in range(170, 186, 2)])
    parser.add_argument("--gpus", nargs="+", default=[str(index) for index in range(8)])
    parser.add_argument("--n_starts", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--worker_pair", type=Path)
    parser.add_argument("--worker_output", type=Path)
    parser.add_argument("--aggregate_only", action="store_true")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.worker_pair is not None:
        if args.worker_output is None:
            raise ValueError("--worker_output is required with --worker_pair")
        worker(args.worker_pair.resolve(), args.worker_output.resolve(), args.n_starts, args.seed)
        return
    pair_roots = [(args.root / name).resolve() for name in args.pairs]
    if args.aggregate_only:
        summary = aggregate(args.root, pair_roots, args.n_starts, args.seed)
        print(json.dumps({"selected": summary["selected"]}, indent=2))
        return
    if len(pair_roots) > len(args.gpus):
        raise ValueError("this sweep requires at least one GPU per pair")
    output_root = args.root / "agreement_sweep"
    output_root.mkdir(parents=True, exist_ok=True)

    def launch(pair_root: Path, gpu: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        command = [sys.executable, str(Path(__file__).resolve()),
                   "--root", str(args.root), "--worker_pair", str(pair_root),
                   "--worker_output", str(output_root / pair_root.name),
                   "--n_starts", str(args.n_starts), "--seed", str(args.seed)]
        subprocess.run(command, cwd=ROOT, env=env, check=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pair_roots)) as executor:
        futures = [executor.submit(launch, pair_root, gpu)
                   for pair_root, gpu in zip(pair_roots, args.gpus)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    summary = aggregate(args.root, pair_roots, args.n_starts, args.seed)
    compact = {name: {metric: values["mean"] for metric, values in entry["metrics"].items()}
               for name, entry in summary["variants"].items()}
    print(json.dumps({"selected": summary["selected"], "metrics": compact}, indent=2))


if __name__ == "__main__":
    main()

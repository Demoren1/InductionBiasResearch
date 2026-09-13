"""Finite-search radius sweep for oracle ideal-mask reachability."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
sys.path.insert(0, str(PATTERN_ROOT))

from data.generate import ideal_mask  # noqa: E402
from evaluation.oracle_ideal import optimize_ideal  # noqa: E402
from evaluation.z_star_noise import (ci, load_model, load_protocol, sha256_file,
                                     atomic_save, write_json)  # noqa: E402


RUN_RADII = (1.0, 2.0, 3.0, 4.0, 6.0)
ALL_RADII = (1.0, 2.0, 3.0, 4.0, 6.0, 12.0)


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    if seed not in settings.model_seeds:
        raise ValueError("seed absent from parent z-star protocol")
    root = out / f"seed_{seed}" / "radius_sweep"
    root.mkdir(parents=True, exist_ok=True)
    parent_oracle = torch.load(out / f"seed_{seed}/oracle.pt", map_location="cpu", weights_only=True)
    initial = parent_oracle["initial"]["z"]
    device = torch.device(device_name)
    model, provenance = load_model(Path(protocol["parent"]), seed, device)
    rows = {}
    for radius in RUN_RADII:
        destination = root / f"radius_{radius:g}.pt"
        if destination.exists():
            result = torch.load(destination, map_location="cpu", weights_only=True)
        else:
            result = optimize_ideal(
                model, initial.to(device), ideal_mask().float().to(device),
                steps=settings.oracle_steps, lr=settings.oracle_lr,
                temperature=settings.temperature, radius=radius)
            result.update({"model_seed": seed, "radius": radius,
                           "checkpoint_sha256": provenance["checkpoint_sha256"],
                           "initial_source": str(out / f"seed_{seed}/oracle.pt"),
                           "initial_source_sha256": sha256_file(out / f"seed_{seed}/oracle.pt")})
            atomic_save(destination, result)
        iou = result["best_soft"]["iou"]
        exact = iou == 1
        norms = result["best_soft"]["z"].norm(dim=1)
        rows[f"{radius:g}"] = {
            "radius": radius, "exact_count": int(exact.sum()),
            "exact_fraction": float(exact.float().mean()), "mean_iou": float(iou.mean()),
            "mean_z_norm": float(norms.mean()),
            "mean_exact_z_norm": float(norms[exact].mean()) if exact.any() else None,
            "at_radius_fraction": float((norms >= radius - 1e-5).float().mean()),
            "artifact": str(destination), "artifact_sha256": sha256_file(destination),
        }
    iou = parent_oracle["best_soft"]["iou"]
    exact = iou == 1
    norms = parent_oracle["best_soft"]["z"].norm(dim=1)
    rows["12"] = {
        "radius": 12.0, "exact_count": int(exact.sum()),
        "exact_fraction": float(exact.float().mean()), "mean_iou": float(iou.mean()),
        "mean_z_norm": float(norms.mean()), "mean_exact_z_norm": float(norms[exact].mean()),
        "at_radius_fraction": float((norms >= 12 - 1e-5).float().mean()),
        "artifact": str(out / f"seed_{seed}/oracle.pt"),
        "artifact_sha256": sha256_file(out / f"seed_{seed}/oracle.pt"),
    }
    write_json(root / "summary.json", {"model_seed": seed, "starts": len(initial),
                                        "steps": settings.oracle_steps, "rows": rows})


def report(out: Path) -> None:
    settings, _ = load_protocol(out)
    records = [json.loads((out / f"seed_{seed}/radius_sweep/summary.json").read_text())
               for seed in settings.model_seeds]
    summary = {"models": len(records), "starts_per_model": settings.oracle_starts,
               "steps": settings.oracle_steps, "radii": {}}
    for radius in ALL_RADII:
        key = f"{radius:g}"
        summary["radii"][key] = {}
        for metric in ("exact_fraction", "mean_iou", "mean_z_norm", "at_radius_fraction"):
            summary["radii"][key][metric] = ci([record["rows"][key][metric] for record in records])
        exact_norms = [record["rows"][key]["mean_exact_z_norm"] for record in records
                       if record["rows"][key]["mean_exact_z_norm"] is not None]
        summary["radii"][key]["mean_exact_z_norm"] = ci(exact_norms) if exact_norms else None
    write_json(out / "radius_sweep_summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    radii = list(ALL_RADII)
    exact = [summary["radii"][f"{radius:g}"]["exact_fraction"]["mean"] for radius in radii]
    exact_low = [max(0.0, summary["radii"][f"{radius:g}"]["exact_fraction"]["ci95"][0])
                 for radius in radii]
    exact_high = [min(1.0, summary["radii"][f"{radius:g}"]["exact_fraction"]["ci95"][1])
                  for radius in radii]
    iou = [summary["radii"][f"{radius:g}"]["mean_iou"]["mean"] for radius in radii]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout="constrained")
    axes[0].plot(radii, exact, marker="o", color="#2f7689")
    axes[0].fill_between(radii, exact_low, exact_high, color="#2f7689", alpha=0.2)
    axes[0].set(ylabel="Exact ideal fraction", ylim=(-0.02, 1.02))
    axes[1].plot(radii, iou, marker="o", color="#bd643c")
    axes[1].set(ylabel="Mean Gold IoU", ylim=(0.75, 1.01))
    for axis in axes:
        axis.set(xlabel="Oracle latent radius")
        axis.grid(alpha=0.25)
    fig.suptitle("Finite-search reachability of the ideal mask (8 VAEs, 128 starts each)")
    fig.savefig(out / "radius_reachability.png", dpi=170)
    fig.savefig(out / "radius_reachability.pdf")
    plt.close(fig)
    print(json.dumps({radius: {metric: value["mean"] if value else None
                               for metric, value in row.items()}
                      for radius, row in summary["radii"].items()}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("run", "report"), required=True)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "run":
        if args.model_seed is None:
            parser.error("--model-seed is required")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        run_seed(out, args.model_seed, args.device)
    else:
        report(out)


if __name__ == "__main__":
    main()

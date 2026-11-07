"""Instantiate the registered scale-aware projection on high-dimensional data."""

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from ..protocol.real_boundary import StudyBoundary
from .scalable_projection import assign


def mapping(reference, observed):
    positive = ((reference[:, :, None] - observed[:, None]) ** 2).sum(0)
    negative = ((reference[:, :, None] + observed[:, None]) ** 2).sum(0)
    left, right = linear_sum_assignment(np.minimum(positive, negative))
    return right[np.argsort(left)]


def run(root, device):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "source" / "current_proposals"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    d, h = protocol["inputs"], protocol["hidden"]
    dense = dict(np.load(root / "source" / "dense" / "models.npz"))
    weight = dense["parameters"][:, : d * h].reshape(-1, d, h)
    normal = weight / np.maximum(np.linalg.norm(weight, axis=1, keepdims=True), 1e-12)
    ensemble = dense["ensemble_parameters"][..., : d * h].reshape(len(normal), -1, d, h)
    deviation = []
    for task, choices in enumerate(ensemble):
        aligned = []
        for item in choices:
            unit = item / np.maximum(np.linalg.norm(item, axis=0), 1e-12)
            aligned.append(unit[:, mapping(normal[task], unit)])
        deviation.append(
            np.sqrt(np.mean((np.stack(aligned) - normal[task]) ** 2, axis=(0, 1)))
        )
    uncertainty = np.maximum(np.asarray(deviation), 0.0003)
    ranking = dense["numerical_cross_entropy"].min(0).argsort()[:2]
    save_json(
        folder / "registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "projection_sha256": digest(
                Path(__file__).with_name("scalable_projection.py")
            ),
            "dense_sha256": digest(root / "source" / "dense" / "models.npz"),
            "objective": "Same scale-weighted least squares and Student categorical projection as G34; code values eliminated by Schur complement before the small H-dimensional solve",
            "real_data_instantiation": "Two H0-ranked dense anchors; full, anchor, 1-vote and 3-vote masks; quantile and random initialization; 40 alternating steps; four geometric proposals plus unrefined anchor seeds per code budget. Finite real-data budget is disclosed separately from synthetic G34.",
            "uncertainty": "Direction dispersion among four independently fitted, H0-selected dense models after Hungarian matching; no teacher parameters",
            "code_grid": protocol["independent_code_grid"],
            "post_selection": "All final choices use functional source M, not geometric score or target data",
        },
    )
    for name in ("real_project.py", "scalable_projection.py"):
        shutil.copy2(Path(__file__).with_name(name), folder / name)
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    candidates, origins = {}, []
    status = "failed"
    try:
        with counter:
            rng = np.random.default_rng(765001)
            for codes in protocol["independent_code_grid"]:
                bank, seeds = [], []
                for anchor in ranking:
                    reference = normal[anchor]
                    permutations = [mapping(reference, observed) for observed in normal]
                    aligned = np.stack([a[:, m] for a, m in zip(normal, permutations)])
                    u = np.stack([a[m] for a, m in zip(uncertainty, permutations)])
                    evidence = np.abs(aligned) > np.maximum(3 * u[:, None], 0.002)
                    centers = np.quantile(
                        np.abs(reference), np.linspace(0.1, 0.95, codes)
                    )
                    distance = np.abs(np.abs(reference)[..., None] - centers)
                    initial = (
                        (distance.argmin(-1) + 1) * np.where(reference >= 0, 1, -1)
                    ).astype(np.int16)
                    masks = [
                        np.ones((d, h), dtype=bool),
                        evidence[anchor],
                        evidence.sum(0) >= 1,
                        evidence.sum(0) >= 3,
                    ]
                    for mask_number, mask in enumerate(masks):
                        seed = initial.copy()
                        seed[~mask] = 0
                        if mask.sum() < codes:
                            continue
                        seeds.append(
                            (
                                seed,
                                {
                                    "anchor": int(anchor),
                                    "mask": mask_number,
                                    "unrefined": True,
                                },
                            )
                        )
                        for restart in range(2):
                            proposal = (
                                seed
                                if restart == 0
                                else (
                                    rng.integers(1, codes + 1, size=(d, h))
                                    * np.where(reference >= 0, 1, -1)
                                ).astype(np.int16)
                            )
                            result = assign(
                                proposal,
                                mask,
                                torch.tensor(
                                    aligned, device=device, dtype=torch.float64
                                ),
                                torch.tensor(u, device=device, dtype=torch.float64),
                                codes,
                            )
                            if result is None:
                                continue
                            index, _, scales, score, iterations = result
                            for ref in (0,):
                                index = index * np.where(scales[ref] >= 0, 1, -1)[None]
                                bank.append(
                                    (
                                        score,
                                        index,
                                        {
                                            "anchor": int(anchor),
                                            "mask": mask_number,
                                            "restart": restart,
                                            "iterations": iterations,
                                            "geometric_score": score,
                                        },
                                    )
                                )
                    print(
                        json.dumps(
                            {
                                "projection_codes": codes,
                                "anchor": int(anchor),
                                "seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                ordered = [
                    (item[1], item[2])
                    for item in sorted(bank, key=lambda item: item[0])[:4]
                ] + seeds[:4]
                seen = set()
                for index, origin in ordered:
                    # Exact duplicates only: no factorial canonicalization at large K.
                    key = index.tobytes()
                    if key in seen:
                        continue
                    seen.add(key)
                    name = f"candidate_{len(origins):03d}"
                    candidates[name] = index
                    origins.append({"name": name, "codes": codes, **origin})
        np.savez_compressed(folder / "candidates.npz", **candidates)
        save_json(folder / "candidates.json", {"candidates": origins})
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()

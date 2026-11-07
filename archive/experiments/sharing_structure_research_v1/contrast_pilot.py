"""G3 pilot: equality-graph stopping and full-noise spectral controls."""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .contrasts import noise_whitened_basis, signed_graph
from .gaussian_stats import fit_stats, nested_observations
from .gradient_spectrum import BASE, decode, spectral
from .kernel_control import coordinate_partition


def register(root, collinear=False):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G3 exploratory pilot, not final competitive confirmation",
        "seed": 2611055917,
        "families": [
            {"id": 6101, "kind": "conv"},
            {"id": 6102, "kind": "conv"},
            {"id": 6201, "kind": "signed"},
            {"id": 6202, "kind": "signed"},
        ],
        "inputs": 16,
        "outputs": 14,
        "codes": 3,
        "tasks": 128,
        "observations": [512, 2048, 8192, 32768, 131072],
        "condition": [1.0, 100.0],
        "noise": [0.2, 1.0],
        "excitation": ["weak"],
        "additional": [
            {"condition": 100.0, "noise": 1.0, "excitation": "full"},
            {"condition": 1.0, "noise": 0.2, "excitation": "collision"},
        ],
        "alpha": 0.05,
        "alpha_allocation": "alpha/5 at each of the five prespecified nested looks, each divided among E^2 tests",
        "decoder_restarts": 8,
        "decoder_steps": 64,
        "methods": [
            "ols_profiles",
            "ols_pca",
            "ols_noise_debiased_pca",
            "ols_cross_pca",
            "noise_whitened_pca",
            "diagonal_noise_whitened_pca",
            "corrected_kernel_coordinate",
            "signed_contrast_graph",
        ],
        "data": "Exact Gaussian regression sufficient-statistic simulation: independent Bartlett sample Grams, conditional projected Gaussian noise, independent residual chi-square. Two independent halves, nested across n. Actual sample Grams, not population substitutes. All estimators receive observation statistics only.",
        "known": "Dimensions and exactly K nonempty signed groups. No support size, group count per code, true C, or coordinate roles supplied to learner.",
        "output": "Graph may abstain. K+1 or 2K+1 component stopping uses K as exact group count; not valid when K is merely an upper bound. Spectral algorithms remain known controls.",
        "checks": "Scientific numerical guards and lint only. No software tests.",
    }
    if collinear:
        plan.update(
            {
                "stage": "G5 rank diagnostic on already inspected geometries; new collinear but distinct code profiles, not independent confirmation",
                "condition": [100.0],
                "noise": [1.0],
                "excitation": ["collinear"],
                "additional": [],
                "hypothesis": "K=3 groups remain identifiable with rank(C)=1 if their profiles0/+-1/+-2/+-3 are distinct. Compare full profiles and PCA scores against forced top-K normalized subspaces.",
            }
        )
        plan["methods"].insert(2, "ols_pca_scores")
    jobs = []
    for family in plan["families"]:
        configurations = [
            {"condition": condition, "noise": noise, "excitation": excitation}
            for condition, noise, excitation in itertools.product(
                plan["condition"], plan["noise"], plan["excitation"]
            )
        ] + plan["additional"]
        for config in configurations:
            jobs.append({"family": family, **config})
    plan["jobs"] = jobs
    save_json(root / "protocol.json", plan)
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "theory_sha256": digest(BASE / "THEORY.md"),
            "code_sha256": {
                name: digest(BASE / name)
                for name in ("contrasts.py", "gaussian_stats.py", "contrast_pilot.py")
            },
        },
    )
    for name in ("cases", "statistics", "showcases"):
        (root / name).mkdir()


@torch.no_grad()
def run_job(root, job_id, plan, device):
    destination = root / "cases" / f"job_{job_id:03d}.json"
    if destination.exists():
        raise RuntimeError("Pilot job already completed")
    job = plan["jobs"][job_id]
    family, records = job["family"]["id"], []
    started = time.perf_counter()
    observed = nested_observations(
        family,
        job["family"]["kind"],
        job["condition"],
        job["excitation"],
        job["noise"],
        plan,
        device,
    )
    first_return = None
    for n, truth, stats, halves in observed:
        fitted, residual, inverse = fit_stats(stats)
        variance = residual.sum() / (
            plan["tasks"] * plan["outputs"] * (n - plan["inputs"])
        )
        profiles = fitted.flatten(1).T
        raw = profiles @ profiles.T / plan["tasks"]
        noise = torch.block_diag(*[inverse.mean(0) * variance] * plan["outputs"])
        corrected = raw - noise
        part_profiles = [fit_stats(part)[0].flatten(1).T for part in halves]
        cross = (
            part_profiles[0] @ part_profiles[1].T
            + part_profiles[1] @ part_profiles[0].T
        ) / (2 * plan["tasks"])
        values = {
            "ols_profiles": profiles,
            "ols_pca": spectral(raw, plan["codes"])[0],
            "ols_noise_debiased_pca": spectral(corrected, plan["codes"])[0],
            "ols_cross_pca": spectral(cross, plan["codes"])[0],
            "noise_whitened_pca": noise_whitened_basis(profiles, noise, plan["codes"])[
                0
            ],
            "diagonal_noise_whitened_pca": noise_whitened_basis(
                profiles, noise, plan["codes"], diagonal_only=True
            )[0],
        }
        if "ols_pca_scores" in plan["methods"]:
            vectors, eigenvalues, _ = spectral(raw, plan["codes"])
            values["ols_pca_scores"] = vectors * eigenvalues.clamp_min(0).sqrt()[None]
        candidates, arrays = {}, {"truth_audit_only": truth.cpu().numpy()}
        for method, embedding in values.items():
            index, _, _ = decode(embedding, plan, plan["seed"] + family + n)
            candidates[method] = index.reshape(truth.shape)
        kernel_results = [
            coordinate_partition(candidates[method], corrected, plan["codes"])
            for method in ("ols_profiles", "ols_noise_debiased_pca")
        ]
        candidates["corrected_kernel_coordinate"] = max(
            kernel_results, key=lambda result: result[2][-1]
        )[0]
        for method, index in candidates.items():
            metrics, aligned = alignment(
                index.cpu().numpy(), truth.cpu().numpy(), plan["codes"]
            )
            records.append({"n": n, "method": method, "recovery": metrics})
            arrays[f"{method}_indices"] = index.cpu().numpy()
            arrays[f"{method}_aligned_indices"] = aligned
        graph_index, graph, components = signed_graph(
            fitted,
            inverse,
            variance,
            plan["tasks"] * plan["outputs"] * (n - plan["inputs"]),
            plan["codes"],
            plan["alpha"] / len(plan["observations"]),
        )
        if graph_index is not None:
            graph["recovery"], aligned = alignment(
                graph_index, truth.cpu().numpy(), plan["codes"]
            )
            arrays["signed_contrast_graph_indices"] = graph_index
            arrays["signed_contrast_graph_aligned_indices"] = aligned
            if first_return is None:
                first_return = {"n": n, "recovery": graph["recovery"]}
        records.append(
            {
                "n": n,
                "method": "signed_contrast_graph",
                "graph": graph,
                "recovery": graph.get("recovery"),
            }
        )
        arrays["signed_contrast_graph_components"] = components
        np.savez_compressed(root / "showcases" / f"job_{job_id:03d}_n{n}.npz", **arrays)
        np.savez_compressed(
            root / "statistics" / f"job_{job_id:03d}_n{n}.npz",
            **{name: value.cpu().numpy() for name, value in stats.items()},
        )
        print(
            job_id,
            family,
            n,
            {
                r["method"]: None
                if r["recovery"] is None
                else r["recovery"]["categorical_errors"]
                for r in records
                if r["n"] == n
            },
            "components",
            graph["components"],
            flush=True,
        )
    save_json(
        destination,
        {
            "job": job,
            "records": records,
            "first_graph_return": first_return,
            "seconds": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collinear", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.collinear)
    else:
        plan = json.loads((args.root / "protocol.json").read_text())
        for job in range(args.shard, len(plan["jobs"]), args.shards):
            run_job(args.root, job, plan, args.device)


if __name__ == "__main__":
    main()

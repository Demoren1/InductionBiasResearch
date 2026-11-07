"""G4 source-only tuning of explicit multi-task competitor adaptations."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .competitor_models import configurations, train_quantized, train_relaxed
from .gaussian_stats import sample_index
from .gradient_spectrum import BASE

METHODS = ("yeh", "msr_exact", "lc", "lutq")


def regression_validation(family, kind, plan, device):
    """Observed extra source functions; never an oracle structural score."""
    index = sample_index(family, kind, plan, device)
    rng = np.random.default_rng(plan["seed"] + family * 5 + 8)
    centers = torch.tensor(
        rng.normal(size=(32, plan["codes"])), device=device, dtype=torch.float64
    )
    weights = centers[:, index.abs().clamp_min(1) - 1] * index.sign()[None]
    result = {"support_n": 4, "query_n": 128}
    for name, n in (("support", 4), ("query", 128)):
        x = torch.tensor(
            rng.normal(size=(32, n, plan["inputs"])), device=device, dtype=torch.float64
        )
        error = torch.tensor(
            rng.normal(size=(32, n, plan["outputs"])),
            device=device,
            dtype=torch.float64,
        )
        y = x @ weights.transpose(-1, -2) + 0.1 * error
        result[name] = {
            "gram": x.transpose(-1, -2) @ x,
            "cross": y.transpose(-1, -2) @ x,
            "square_y": y.square().sum(1),
        }
    return result


def register(root, source):
    source_plan = json.loads((source / "protocol.json").read_text())
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **{
            name: source_plan[name]
            for name in (
                "seed",
                "inputs",
                "outputs",
                "codes",
                "tasks",
                "decoder_restarts",
                "decoder_steps",
            )
        },
        "stage": "G4 competitor tuning only; confirmation geometries not opened",
        "source_root": str(source.resolve()),
        "source_protocol_sha256": digest(source / "protocol.json"),
        "pilot_jobs": [3, 9, 15, 21],
        "observations": [8192, 131072],
        "methods": list(METHODS),
        "steps": {"yeh": 3000, "msr_exact": 3000, "lutq": 3000, "lc": 120},
        "checkpoint_every": 100,
        "seed_pilot": 17,
        "candidates": {method: configurations(method) for method in METHODS},
        "selection": "One global configuration per representation by mean observed validation MSE across all four pilot families and both n budgets. Checkpoints selected using 32 separate source functions, full code variation, 4 support+128 query, isotropic inputs, noise0.1. These extra validation data are available equally but not used in spectral or contrast estimators.",
        "adaptations": {
            "yeh": "One categorical probability per position shared across tasks, differentiable exact ridge C_t fit on all source observations, source MSE plus entropy/nuclear terms; native MAP separate from continuous soft basis. Multi-task objective adaptation, not unchanged author code or a literal support/query bilevel reproduction.",
            "msr_exact": "Continuous common basis, exact ridge C_t support fit replaces original short inner SGD. A stronger numerical control for the shared representation; native continuous basis and additional signed-row clustering readout are separate.",
            "lc": "Common vector codebook assignment across task profiles. Analytic full-data L-step, warm Lloyd C-step, growing penalty, dual update. Native projection assignments, no external readout.",
            "lutq": "Common vector codebook assignment across task profiles. One Lloyd step per full-data STE Adam update, codebook updated by assigned profile means. Native assignments. Explicit multi-task/vector extension.",
        },
        "checks": "No software tests; exact sufficient-statistic objective, numerical finiteness, lint, scientific curves.",
    }
    jobs = []
    for job in plan["pilot_jobs"]:
        for n in plan["observations"]:
            for method in METHODS:
                jobs.append({"source_job": job, "n": n, "method": method})
    plan["jobs"] = jobs
    save_json(root / "protocol.json", plan)
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "executed_sha256": {
                name: digest(BASE / name)
                for name in (
                    "competitor_models.py",
                    "competitor_pilot.py",
                    "gaussian_stats.py",
                )
            },
        },
    )
    (root / "pilot").mkdir()


def save_results(folder, prefix, result):
    arrays, metadata = {}, {}
    for name, representation in result.items():
        if isinstance(representation, np.ndarray):
            arrays[name] = representation
        elif isinstance(representation, dict):
            metadata[name] = {}
            for key, value in representation.items():
                if isinstance(value, np.ndarray):
                    arrays[f"{name}__{key}"] = value
                else:
                    metadata[name][key] = value
        else:
            metadata[name] = representation
    path = folder / f"{prefix}.npz"
    np.savez_compressed(path, **arrays)
    save_json(path.with_suffix(".json"), {**metadata, "result_sha256": digest(path)})


def run_job(root, job_id, device):
    plan = json.loads((root / "protocol.json").read_text())
    job = plan["jobs"][job_id]
    prefix = f"job_{job_id:03d}"
    if (root / "pilot" / f"{prefix}.json").exists():
        raise RuntimeError("Pilot job already complete")
    source = Path(plan["source_root"])
    source_plan = json.loads((source / "protocol.json").read_text())
    family = source_plan["jobs"][job["source_job"]]["family"]
    with np.load(
        source / "statistics" / f"job_{job['source_job']:03d}_n{job['n']}.npz"
    ) as archive:
        data = {
            name: torch.tensor(archive[name], device=device) for name in archive.files
        }
    validation = regression_validation(family["id"], family["kind"], plan, device)
    local = {
        **plan,
        "n": job["n"],
        "checkpoint_every": 5 if job["method"] == "lc" else plan["checkpoint_every"],
    }
    function = (
        train_relaxed if job["method"] in ("yeh", "msr_exact") else train_quantized
    )
    print("starting", job_id, job, family, flush=True)
    result = function(
        job["method"],
        plan["candidates"][job["method"]],
        data,
        validation,
        local,
        plan["seed_pilot"],
        plan["steps"][job["method"]],
    )
    save_results(root / "pilot", prefix, result)
    print(
        "finished",
        job_id,
        job["method"],
        float(np.min(result["native_or_decoded"]["selection_mse"])),
        result["seconds"],
        flush=True,
    )


def freeze(root):
    plan = json.loads((root / "protocol.json").read_text())
    choices, hashes = {}, {}
    for method in METHODS:
        ids = [i for i, job in enumerate(plan["jobs"]) if job["method"] == method]
        for representation in ("native_or_decoded", "continuous"):
            scores = []
            for job_id in ids:
                path = root / "pilot" / f"job_{job_id:03d}.npz"
                with np.load(path) as archive:
                    key = f"{representation}__selection_mse"
                    if key in archive.files:
                        scores.append(archive[key])
                hashes[str(path.relative_to(root))] = digest(path)
            if scores:
                mean = np.mean(scores, axis=0)
                winner = int(mean.argmin())
                choices[f"{method}__{representation}"] = {
                    "row": winner,
                    "settings": plan["candidates"][method][winner],
                    "mean_validation_mse": float(mean[winner]),
                    "all_mean_mse": mean.tolist(),
                }
    destination = root / "hpo_frozen.json"
    if destination.exists():
        raise RuntimeError("HPO already frozen")
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "choices": choices,
            "input_sha256": hashes,
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )
    print(choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.source)
    elif args.action == "freeze":
        freeze(args.root)
    else:
        plan = json.loads((args.root / "protocol.json").read_text())
        for job_id in range(args.shard, len(plan["jobs"]), args.shards):
            run_job(args.root, job_id, args.device)


if __name__ == "__main__":
    main()

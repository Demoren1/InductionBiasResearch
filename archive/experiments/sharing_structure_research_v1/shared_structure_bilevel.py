"""Registered shared T/V/H bilevel comparison on G23's frozen observations."""

import argparse
import hashlib
import itertools
import json
import math
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from . import shared_structure_common as common
from .shared_structure_benchmark import freeze as freeze_extract
from .shared_structure_bilevel_optimization import (
    ScenarioAdam,
    code_mse,
    code_statistics,
    derivative_diagnostic,
    index_features,
    inverse_actions,
    mixed_gradient,
    polish_indices,
    regularizer_gradient,
    signed_basis,
    train_codes,
    validation_gradient,
)
from .shared_structure_data import load_source
from .shared_structure_native import gn_step, project
from .shared_structure_numeric import canonical_index
from .shared_structure_pipeline import check_code

BASE = Path(__file__).resolve().parent
METHODS = {"direct": "yeh_bilevel_direct", "literal": "yeh_bilevel_literal"}


def configurations():
    return [
        {"lr": lr, "initialization": initial, "restart": restart}
        for lr, initial, restart in itertools.product(
            (0.003, 0.01, 0.03), ("random", "source0_GN"), range(3)
        )
    ]


def register(root, parent):
    if (parent / "common/audit.json").exists():
        raise RuntimeError("This supplemental configuration precedes G23 teacher audit")
    inherited = json.loads((parent / "protocol.json").read_text())
    check_code(inherited)
    root.mkdir(parents=True, exist_ok=False)
    names = set(inherited["code_sha256"]) | {
        "shared_structure_bilevel.py",
        "shared_structure_bilevel_optimization.py",
        "shared_structure_bilevel_report.py",
        "shared_structure_bilevel_pipeline.py",
        "shared_structure_pipeline.py",
        "shared_structure_common.py",
    }
    plan = {
        **inherited,
        "registered_utc": datetime.now(UTC).isoformat(),
        "stage": "G23 supplementary shared signed T/V/H Yeh control, same geometries; not independent confirmation or our novel method",
        "parent": str(parent.resolve()),
        "parent_protocol_sha256": digest(parent / "protocol.json"),
        "parent_data_freeze_sha256": digest(parent / "data_frozen.json"),
        "methods": list(METHODS.values()),
        "evaluated_scopes": ["common4"],
        "bilevel_inner_steps": 250,
        "bilevel_inner_lr": 0.001,
        "bilevel_outer_steps": 500,
        "bilevel_checkpoint": 25,
        "bilevel_neumann_steps": 20,
        "bilevel_ridge": 1e-8,
        "bilevel_entropy": 0.025,
        "bilevel_nuclear": 0.05,
        "bilevel_hypergradient_clip": 25.0,
        "bilevel_bank_size": 16,
        "bilevel_configurations": configurations(),
        "bilevel_observations": "T=train65568, architectureV=prefix_validation8192, H=fit_validation2048. Uniform query loss / observed population-variance scale (floor1e-6). Inner minibatch1024 shared across paired trials, independent Adam moments. No new labels; metaY never used by extraction or HPO.",
        "bilevel_inverse": "Two matched controls: literal published Neumann20 sum without eta, or direct KxK local Hessian solve. Same C optimizer, objective and initializations; direct is the strong primary numerical control, literal an author-convention diagnostic. Record stationarity and inverse residual; finite inner optimization is approximate implicit differentiation.",
        "bilevel_selection": "Native MAP checkpoints every25updates, warm-C fixed-I polish on T/H only, no feedback of polished C or I into the bilevel trajectory. At most16uniquecanonicalI per inverse method, ranked on mean four-task H. Standard fresh32signedC fits on T/H, whole predictor freeze, meta selection, whole selected-I freeze, late teacher audit. Only common4 evaluated; source1 is not a failed trial.",
        "bilevel_budget": "24family/noise/inverse jobs x18pairedLR/init/restarts =432trajectories. Extra HPO compute and joint4task access disclosed. Runs after original G23 structure report. No independent-confirmation claim.",
        "author_repository_commit": "164fe57aba0939e7db62daf72a899b2f25ce93c7",
        "code_sha256": {name: digest(BASE / name) for name in sorted(names)},
        "theory_sha256": digest(BASE / "THEORY.md"),
        "extract_jobs": [
            {"family_id": family, "noise": noise, "inverse": inverse}
            for family, noise, inverse in itertools.product(
                range(len(inherited["families"])), inherited["noise"], METHODS
            )
        ],
        "gpu_allowlist": [0, 1, 2, 3],
        "reserved_gpus": [6, 7],
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in sorted(names):
        shutil.copy2(BASE / name, snapshot / name)
    (root / "data").symlink_to((parent / "data").resolve(), target_is_directory=True)
    shutil.copy2(parent / "data_frozen.json", root / "data_frozen.json")
    for name in ("extract", "logs"):
        (root / name).mkdir()


def observations(root, plan, family_id, noise, device):
    observed = [
        load_source(root, plan, family_id, task, noise, device)
        for task in range(plan["sources"])
    ]
    data = {}
    for panel in ("train", "prefix_validation", "fit_validation"):
        data[panel + "_x"] = observed[0][panel + "_x"]
        data[panel + "_y"] = torch.stack(
            [value[f"{panel}_y_{noise}"] for value in observed]
        )
    return data


def initialize(plan, family, data):
    x, y = data["train_x"], data["train_y"]
    starts, warmup = {}, []
    for restart in range(3):
        generator = torch.Generator(device=x.device).manual_seed(
            plan["seed"] + family * 71 + restart * 100003
        )
        weight = torch.randn(
            (plan["inputs"], plan["hidden"]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        ) / math.sqrt(plan["inputs"])
        random_logits = 0.1 * torch.randn(
            (plan["inputs"], plan["hidden"], 2 * plan["codes"] + 1),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        code = 0.5 * torch.randn(
            (plan["sources"], plan["codes"]),
            device=x.device,
            dtype=x.dtype,
            generator=generator,
        )
        starts[("random", restart)] = (random_logits, code.clone())
        solves = 0
        for _ in range(plan["gradient_warmup_solves"]):
            weight, result = gn_step(x, y[0], weight, plan, 1e-10)
            solves += 1
            if not result["accepted"]:
                break
        _, center, _ = project(weight, plan)
        dictionary = torch.cat((center[:1] * 0, center, -center))
        spread = weight.std().clamp_min(0.05)
        logits = (
            -(weight[..., None] - dictionary).square() / (0.05 * spread.square())
        ).clamp_min(-20)
        code[0] = center
        starts[("source0_GN", restart)] = (logits, code)
        warmup.append({"restart": restart, "GN_solves": solves})
    selected = [
        starts[(row["initialization"], row["restart"])]
        for row in plan["bilevel_configurations"]
    ]
    return (
        torch.stack([s[0] for s in selected]),
        torch.stack([s[1] for s in selected]),
        warmup,
    )


def extract_job(root, plan, job_id, device):
    destination = root / "extract" / f"job_{job_id:04d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Supplemental source job already completed")
    started = time.monotonic()
    job = plan["extract_jobs"][job_id]
    data = observations(root, plan, job["family_id"], job["noise"], device)
    family = plan["families"][job["family_id"]]["family"]
    logits, code, warmup = initialize(plan, family, data)
    x, y = data["train_x"], data["train_y"]
    vx, vy = data["prefix_validation_x"], data["prefix_validation_y"]
    hx, hy = data["fit_validation_x"], data["fit_validation_y"]
    train_scale, validation_scale = (
        y.var(unbiased=False).clamp_min(1e-6),
        vy.var(unbiased=False).clamp_min(1e-6),
    )
    rates = x.new_tensor([row["lr"] for row in plan["bilevel_configurations"]])
    modes = torch.full(
        (len(rates),), job["inverse"] == "direct", device=x.device, dtype=torch.bool
    )
    outer = ScenarioAdam(logits, rates)
    inner = ScenarioAdam(code, [plan["bilevel_inner_lr"]])
    generator = torch.Generator(device=x.device).manual_seed(
        plan["seed"] + family * 1709 + 41
    )
    trace = {
        name: []
        for name in (
            "step",
            "index",
            "soft_weights",
            "code",
            "polished_code",
            "H_mse",
            "polish_step",
            "soft_train_mse",
            "soft_H_mse",
            "raw_hard_H_mse",
        )
    }
    diagnostics, diagnostic_steps, numerical_failure = {}, [], None

    @torch.no_grad()
    def checkpoint(step):
        probability = logits.softmax(-1)
        label = probability.argmax(-1)
        index = torch.where(label > plan["codes"], plan["codes"] - label, label)
        polished, score, polish_step = polish_indices(index, code, x, y, hx, hy, plan)
        basis = signed_basis(probability, plan["codes"])
        feature = torch.einsum("nd,sdhk->snhk", x, basis)
        values = {
            "step": np.array(step),
            "index": index.cpu().numpy().astype(np.int16),
            "soft_weights": torch.einsum("sdhk,stk->stdh", basis, code).cpu().numpy(),
            "code": code.cpu().numpy(),
            "polished_code": polished.cpu().numpy(),
            "H_mse": score.cpu().numpy(),
            "polish_step": polish_step.cpu().numpy(),
            "soft_train_mse": code_mse(feature, code, y, plan).cpu().numpy(),
            "soft_H_mse": code_mse(
                torch.einsum("nd,sdhk->snhk", hx, basis), code, hy, plan
            )
            .cpu()
            .numpy(),
            "raw_hard_H_mse": code_mse(
                index_features(hx, index, plan["codes"]), code, hy, plan
            )
            .cpu()
            .numpy(),
        }
        for name, value in values.items():
            trace[name].append(value.copy())
        print(
            json.dumps(
                {
                    "job": job_id,
                    "inverse": job["inverse"],
                    "outer_updates": step,
                    "best_H_mean_mse": float(score.mean(1).min()),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )

    checkpoint(0)
    identity = torch.eye(plan["codes"], device=x.device, dtype=x.dtype)
    try:
        for step in range(1, plan["bilevel_outer_steps"] + 1):
            probability = logits.softmax(-1)
            feature = torch.einsum(
                "nd,sdhk->snhk", x, signed_basis(probability, plan["codes"])
            )
            train_codes(feature, code, y, train_scale, inner, generator, plan)
            if not torch.isfinite(code).all():
                raise FloatingPointError("Nonfinite inner Adam codebooks")
            gram, score, training_mse = code_statistics(feature, code, y, plan)
            hessian = gram / (plan["sources"] * train_scale)
            hessian += plan["bilevel_ridge"] / plan["sources"] * identity
            direct, code_gradient, validation_mse = validation_gradient(
                probability, code, vx, vy, validation_scale, plan
            )
            inverse, residuals = inverse_actions(
                hessian, code_gradient, rates, modes, plan["bilevel_neumann_steps"]
            )
            indirect = mixed_gradient(
                probability, code, feature, x, y, inverse, train_scale, plan
            )
            regularizer, entropy, nuclear = regularizer_gradient(logits, plan)
            gradient = direct - indirect + regularizer
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite implicit architecture gradient")
            stationarity = (
                (
                    score / (plan["sources"] * train_scale)
                    + plan["bilevel_ridge"] / plan["sources"] * code
                )
                .norm(dim=-1)
                .amax(1)
            )
            norm = gradient.flatten(1).norm(dim=1)
            measurements = {
                **residuals,
                "C_stationarity_max_task": stationarity,
                "H_min_eigenvalue": torch.linalg.eigvalsh(hessian)[..., 0].amin(1),
                "direct_gradient_norm": direct.flatten(1).norm(dim=1),
                "indirect_gradient_norm": indirect.flatten(1).norm(dim=1),
                "hypergradient_norm_before_clip": norm,
                "entropy": entropy,
                "nuclear": nuclear,
                "soft_train_mse": training_mse.mean(1),
                "architecture_V_mse": validation_mse.mean(1),
                "C_abs_max": code.abs().flatten(1).amax(1),
            }
            for name, value in measurements.items():
                diagnostics.setdefault(name, []).append(value.cpu().numpy())
            diagnostic_steps.append(step - 1)
            fraction = (
                plan["bilevel_hypergradient_clip"] / norm.clamp_min(1e-30)
            ).clamp_max(1)
            outer.step(logits, gradient * fraction[:, None, None, None])
            logits -= logits.mean(-1, keepdim=True)
            del feature
            if step % plan["bilevel_checkpoint"] == 0:
                checkpoint(step)
    except FloatingPointError as error:
        numerical_failure = {
            "outer_update": step,
            "reason": str(error),
            "policy": "Keep only finite earlier checkpoints; no silent restart or changed settings. Other inverse jobs remain independent.",
        }
    arrays = {f"trace_{name}": np.stack(values) for name, values in trace.items()}
    arrays.update(
        {f"diagnostic_{name}": np.stack(values) for name, values in diagnostics.items()}
    )
    arrays["diagnostic_outer_state"] = np.array(diagnostic_steps)
    candidates, seen = [], set()
    ordering = np.argsort(arrays["trace_H_mse"].mean(-1).flatten(), kind="stable")
    scenarios = len(rates)
    for position in ordering:
        checkpoint_id, scenario = divmod(int(position), scenarios)
        index = canonical_index(
            arrays["trace_index"][checkpoint_id, scenario], plan["codes"]
        )
        signature = hashlib.sha256(index.tobytes()).hexdigest()
        if signature in seen:
            continue
        seen.add(signature)
        arrays[f"candidate_{len(candidates)}__index"] = index
        candidates.append(
            {
                "signature": signature,
                "checkpoint": checkpoint_id,
                "outer_updates": int(arrays["trace_step"][checkpoint_id]),
                "scenario": scenario,
                "configuration": plan["bilevel_configurations"][scenario],
                "mean_H_mse": float(
                    arrays["trace_H_mse"][checkpoint_id, scenario].mean()
                ),
            }
        )
        if len(candidates) == plan["bilevel_bank_size"]:
            break
    np.savez_compressed(destination.with_suffix(".npz"), **arrays)
    save_json(
        destination.with_suffix(".json"),
        {
            "job": job,
            "method": METHODS[job["inverse"]],
            "candidates": candidates,
            "source_observation_sha256": digest(
                root / "data" / f"family_{job['family_id']:03d}_source.npz"
            ),
            "protocol_sha256": digest(root / "protocol.json"),
            "warmup": warmup,
            "train_variance_scale": float(train_scale),
            "architecture_V_variance_scale": float(validation_scale),
            "numerical_failure": numerical_failure,
            "seconds": time.monotonic() - started,
            "meta_labels_opened": False,
            "teacher_audit_opened": False,
        },
    )


def build_banks(root, plan):
    frozen = json.loads((root / "extract_frozen.json").read_text())
    for relative, expected in frozen["sha256"].items():
        if digest(root / relative) != expected:
            raise RuntimeError("Frozen supplemental extraction changed")
    common.register(root)
    phase_path = root / "common/protocol.json"
    phase = json.loads(phase_path.read_text())
    phase["evaluated_scopes"] = ["common4"]
    phase["selection"] = plan["bilevel_selection"]
    save_json(phase_path, phase)
    for family_id, noise_id in itertools.product(
        range(len(plan["families"])), range(len(plan["noise"]))
    ):
        noise = plan["noise"][noise_id]
        job_id = family_id * len(plan["noise"]) + noise_id
        path = root / "common/banks" / f"job_{job_id:03d}"
        indices, candidates, lookup = [], [], {}
        methods = {method: [] for method in plan["methods"]}
        for source_id, job in enumerate(plan["extract_jobs"]):
            if job["family_id"] != family_id or job["noise"] != noise:
                continue
            source = root / "extract" / f"job_{source_id:04d}"
            metadata = json.loads(source.with_suffix(".json").read_text())
            method = metadata["method"]
            with np.load(source.with_suffix(".npz")) as arrays:
                for number, candidate in enumerate(metadata["candidates"]):
                    signature = candidate["signature"]
                    if signature not in lookup:
                        lookup[signature] = len(indices)
                        indices.append(arrays[f"candidate_{number}__index"])
                        candidates.append({"signature": signature, "origins": []})
                    target = lookup[signature]
                    candidates[target]["origins"].append(
                        {
                            "method": method,
                            "task": -1,
                            "source_job": source_id,
                            "record": number,
                            "joint_source_tasks": list(range(plan["sources"])),
                            "fit_validation_mse": candidate["mean_H_mse"],
                        }
                    )
                    methods[method].append(target)
        np.savez_compressed(path.with_suffix(".npz"), indices=np.stack(indices))
        save_json(
            path.with_suffix(".json"),
            {
                "job_id": job_id,
                "family_id": family_id,
                "noise": noise,
                "candidates": candidates,
                "methods": methods,
                "meta_labels_opened": False,
                "source_extract_freeze_sha256": digest(root / "extract_frozen.json"),
            },
        )


def select_common_only(root, plan, job_id, device):
    common.select_job(root, plan, job_id, device)
    path = root / "common/selected" / f"job_{job_id:03d}.json"
    metadata = json.loads(path.read_text())
    unused = [row for row in metadata["records"] if row["scope"] == "source1"]
    if any(row["complete"] for row in unused):
        raise RuntimeError("Joint-source candidates unexpectedly entered source1")
    metadata["records"] = [
        row for row in metadata["records"] if row["scope"] == "common4"
    ]
    metadata["scopes_not_evaluated"] = ["source1"]
    save_json(path, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "register",
            "diagnose",
            "extract",
            "freeze_extract",
            "banks",
            "fit",
            "freeze_fit",
            "select",
            "freeze_selection",
            "audit",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.parent)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    check_code(plan)
    if args.action in ("extract", "fit", "select") and (
        os.environ.get("CUDA_VISIBLE_DEVICES")
        not in {str(g) for g in plan["gpu_allowlist"]}
    ):
        raise RuntimeError("Explicit single-GPU visibility from allowlist0--3 required")
    if args.action == "diagnose":
        data = observations(args.root, plan, 0, plan["noise"][0], "cpu")
        generator = torch.Generator().manual_seed(plan["seed"] + 17)
        logits = 0.1 * torch.randn(
            (1, plan["inputs"], plan["hidden"], 2 * plan["codes"] + 1),
            generator=generator,
            dtype=torch.float64,
        )
        code = torch.randn(
            (1, plan["sources"], plan["codes"]),
            generator=generator,
            dtype=torch.float64,
        )
        rows = torch.linspace(0, len(data["train_x"]) - 1, 128).long()
        result = derivative_diagnostic(
            logits,
            code,
            data["train_x"][rows],
            data["train_y"][:, rows],
            data["train_y"].var(unbiased=False).clamp_min(1e-6),
            plan,
        )
        result.update(
            {
                "protocol_sha256": digest(args.root / "protocol.json"),
                "source_observation_sha256": digest(
                    args.root / "data/family_000_source.npz"
                ),
                "source_rows": rows.tolist(),
            }
        )
        save_json(args.root / "local_derivative_diagnostic.json", result)
        print(json.dumps(result))
    elif args.action == "freeze_extract":
        freeze_extract(args.root, plan, "extract")
    elif args.action == "banks":
        build_banks(args.root, plan)
    elif args.action.startswith("freeze_"):
        common.freeze(args.root, plan, args.action.removeprefix("freeze_"))
    elif args.action == "audit":
        if (args.root / "common/audit.json").exists():
            raise RuntimeError("Supplemental structure audit already exists")
        common.audit(args.root, plan)
    else:
        length = (
            len(plan["extract_jobs"])
            if args.action == "extract"
            else len(plan["families"]) * len(plan["noise"])
        )
        function = {
            "extract": extract_job,
            "fit": common.fit_job,
            "select": select_common_only,
        }[args.action]
        for job_id in range(args.shard, length, args.shards):
            function(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()

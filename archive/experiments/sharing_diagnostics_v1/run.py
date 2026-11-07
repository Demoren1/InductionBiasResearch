"""Retrospective cause audit followed by prospectively frozen corrections."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.data import prepare_family
from experiments.sharing_benchmark_v1.evaluate import alignment, target_cohort, transfer
from experiments.sharing_benchmark_v1.hard import configurations
from experiments.sharing_benchmark_v1.linear import (
    adaptation_errors,
    adapted,
    index_basis,
    load_observations,
)

from .algorithms import coordinate_refine, joint_consensus, scalar_bank

BASE = Path(__file__).resolve().parent
BASE_METHODS = (
    "bank_dp",
    "bank_dp_late",
    "bank_dp_64",
    "bank_dp_long",
    "joint_constant",
    "joint_late",
    "bank_query_edits",
)
CONTROL_METHODS = ("joint_projection", "joint_8", "joint_projection_8")
METHODS = (*BASE_METHODS[:-1], *CONTROL_METHODS, BASE_METHODS[-1])


def stamp():
    return datetime.now(UTC).isoformat()


def register(root, previous):
    plan = json.loads((previous / "protocol.json").read_text())
    plan.update(
        seed=2611051907,
        pilot_families=[1001, 1002],
        pilot_seeds=[21, 22],
        confirmation_families=list(range(1101, 1107)),
        confirmation_seeds=[31, 32, 33],
        methods=list(BASE_METHODS),
    )
    plan["diagnostic_plan"] = {
        "previous": str(previous.resolve()),
        "retrospective_families": [201, 202, 203, 204, 205, 206],
        "diagnostics": [
            "All per-step original proposals, exact-I discovery versus selection, oracle insertion into selector.",
            "Initial sparsity, code usage, source rank, Euclidean projection objective against teacher projection.",
            "Oracle initialization is a retrospective diagnostic only and never enters prospective search.",
        ],
        "prospective": "New seed, new pilot and six new confirmation geometries. Twelve configurations per searched variant, independent coefficient validation only. Teacher/test closed until all prospective models freeze.",
        "single_changes": {
            "bank_dp": "Original scalar DP, 8 proposals, 80 rounds.",
            "bank_dp_late": "Only delay nonzero charge: zero through 30%, linear ramp to full charge by 70%.",
            "bank_dp_64": "Only proposal anchors 8→64, using the same source observations.",
            "bank_dp_long": "Only rounds 80→320.",
            "joint_constant": "Shared hard I with independent C_t for all 64 functions inside every consensus round; local vector projector, constant charge.",
            "joint_late": "Joint constant variant with the same delayed-charge policy.",
            "bank_query_edits": "Refine bank_dp I by up to 64 exact one-entry edits chosen by source-query after fresh exact support-fit C_t. All positions/zero/groups editable. Validation only selects checkpoint. No extra tuned hyperparameters.",
        },
        "projector_scope": "Multitask vector projection is local, not scalar exact DP. No known mask/count/labels or prelearned MSR/Yeh basis. It runs inside training, not as postprocessing of a competitor.",
        "checks": "No software tests; scientific diagnostics, numerical guards and minimal lint/figure inspection only.",
    }
    root.mkdir(parents=True, exist_ok=False)
    save_json(root / "protocol.json", plan)
    save_json(
        root / "candidates.json",
        {
            name: configurations("consensus_dp")
            for name in BASE_METHODS
            if name != "bank_query_edits"
        },
    )
    shutil.copytree(
        BASE,
        root / "registered_code",
        ignore=shutil.ignore_patterns("results", "__pycache__"),
    )
    save_json(
        root / "registration.json",
        {
            "registered_utc": stamp(),
            "previous_benchmark_freeze_sha256": digest(
                previous / "benchmark_frozen.json"
            ),
            "protocol_sha256": digest(root / "protocol.json"),
            "candidate_sha256": digest(root / "candidates.json"),
        },
    )
    for family in plan["pilot_families"]:
        prepare_family(root, plan, "hidden", family)


def save_result(path, result, inputs):
    path.parent.mkdir(exist_ok=True)
    arrays = {
        name: value for name, value in result.items() if isinstance(value, np.ndarray)
    }
    np.savez_compressed(path, **arrays)
    save_json(
        path.with_suffix(".json"),
        {
            "inputs": inputs,
            "completed_utc": stamp(),
            "result_sha256": digest(path),
            "code_sha256": {name.name: digest(name) for name in BASE.glob("*.py")},
            **{
                name: value
                for name, value in result.items()
                if not isinstance(value, np.ndarray)
            },
        },
    )


def run_one(root, family, phase, method, seed, rows, device):
    folder = root / "hidden" / f"family_{family}"
    plan = json.loads((root / "protocol.json").read_text())
    if not folder.exists():
        prepare_family(root, plan, "hidden", family)
    path = folder / phase / f"{method}_seed_{seed}.npz"
    inputs = {
        "method": method,
        "seed": seed,
        "rows": rows,
        "observation_sha256": digest(folder / "observations.npz"),
    }
    if path.exists():
        old = json.loads(path.with_suffix(".json").read_text())
        if inputs != old["inputs"] or old["result_sha256"] != digest(path):
            raise RuntimeError(f"Changed inputs/results: {path}")
        return
    spec = json.loads((folder / "specification.json").read_text())
    data = load_observations(folder, device)
    if method == "bank_query_edits":
        with np.load(folder / phase / f"bank_dp_seed_{seed}.npz") as archive:
            initial = archive["indices"][0]
        result = coordinate_refine(initial, spec, data)
    elif method.startswith("joint_"):
        result = joint_consensus(
            rows,
            spec,
            plan,
            data,
            seed,
            late=method.endswith("late"),
            task_limit=8 if method.endswith("_8") else None,
            projection_only="projection" in method,
        )
    else:
        result = scalar_bank(
            rows,
            spec,
            plan,
            data,
            seed,
            late=method.endswith("late"),
            anchors=64 if method.endswith("64") else None,
            rounds=320 if method.endswith("long") else None,
        )
    save_result(path, result, inputs)
    print(
        family,
        phase,
        method,
        seed,
        "val",
        result["selection_mse"].tolist(),
        "seconds",
        round(result["seconds"], 2),
        flush=True,
    )


def freeze_hpo(root):
    plan = json.loads((root / "protocol.json").read_text())
    candidates = json.loads((root / "candidates.json").read_text())
    choices = {}
    for method in candidates:
        errors = []
        for family, seed in itertools.product(
            plan["pilot_families"], plan["pilot_seeds"]
        ):
            with np.load(
                root
                / "hidden"
                / f"family_{family}"
                / "pilot"
                / f"{method}_seed_{seed}.npz"
            ) as archive:
                errors.append(archive["selection_mse"])
        mean = np.mean(errors, axis=0)
        chosen = int(mean.argmin())
        choices[method] = {
            "settings": candidates[method][chosen],
            "all_mean_selection_mse": mean.tolist(),
            "mean_selection_mse": float(mean[chosen]),
        }
    save_json(root / "hpo_frozen.json", {"frozen_utc": stamp(), "choices": choices})
    print(
        "frozen HPO",
        {name: value["settings"] for name, value in choices.items()},
        flush=True,
    )


def register_controls(root):
    path = root / "controls_frozen.json"
    source = root / "hpo_frozen.json"
    if path.exists():
        if json.loads(path.read_text())["source_hpo_sha256"] != digest(source):
            raise RuntimeError("Control settings refer to changed pilot choices")
        return
    chosen = json.loads(source.read_text())["choices"]["joint_constant"]["settings"]
    save_json(
        path,
        {
            "registered_utc": stamp(),
            "before_confirmation_and_test": True,
            "reason": "Causal controls separate using all source functions from consensus iterations. Inherit joint_constant pilot choice with no extra tuning.",
            "source_hpo_sha256": digest(source),
            "choices": {name: {"settings": chosen} for name in CONTROL_METHODS},
            "definitions": {
                "joint_projection": "One projection of all64 dense fits, no L/dual updates.",
                "joint_8": "Full joint search but only first8 functions in L/projection; source selection still uses all64.",
                "joint_projection_8": "One projection using first8 source functions.",
            },
        },
    )


def diagnose(root, previous, device):
    plan = json.loads((previous / "protocol.json").read_text())
    row = json.loads((previous / "hpo_frozen.json").read_text())["choices"][
        "consensus_dp"
    ]["settings"]
    output = root / "retrospective"
    output.mkdir(exist_ok=True)
    results = []
    for family in plan["confirmation_families"]:
        folder = previous / "hidden" / f"family_{family}"
        spec = json.loads((folder / "specification.json").read_text())
        data = load_observations(folder, device)
        with np.load(folder / "teacher_audit_only.npz") as archive:
            truth = archive["indices"]
        with np.load(folder / "confirmation/consensus_dp_seed_11.npz") as archive:
            old_index = archive["indices"][0]
        trace = scalar_bank([row], spec, plan, data, 11, trace=True)
        oracle_basis = index_basis(
            torch.tensor(truth, device=device)[None], spec["codes"]
        )
        oracle_source = float(
            adaptation_errors(
                oracle_basis,
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean()
        )
        oracle_validation = float(
            adaptation_errors(
                oracle_basis,
                data["selection_support_x"],
                data["selection_support_y"],
                data["selection_query_x"],
                data["selection_query_y"],
            ).mean()
        )
        chosen_basis = index_basis(
            torch.tensor(old_index, device=device)[None], spec["codes"]
        )
        chosen_source = float(
            adaptation_errors(
                chosen_basis,
                data["train_support_x"],
                data["train_support_y"],
                data["train_query_x"],
                data["train_query_y"],
            ).mean()
        )
        chosen_validation = float(
            adaptation_errors(
                chosen_basis,
                data["selection_support_x"],
                data["selection_support_y"],
                data["selection_query_x"],
                data["selection_query_y"],
            ).mean()
        )
        continuous_x = torch.cat((data["train_support_x"], data["train_query_x"]), 1)
        continuous_y = torch.cat((data["train_support_y"], data["train_query_y"]), 1)
        anchor_codes = adapted(
            oracle_basis.double(), continuous_x.double(), continuous_y.double()
        )[0]
        oracle_weights = torch.einsum(
            "oik,tk->toi", oracle_basis[0].double(), anchor_codes
        ).float()
        oracle_initialized = {}
        for late in (False, True):
            local = scalar_bank(
                [row], spec, plan, data, 11, late=late, oracle_weights=oracle_weights
            )
            recovery, _ = alignment(local["indices"][0], truth, spec["codes"])
            oracle_initialized["late" if late else "constant"] = {
                "recovery": recovery,
                "selection_mse": float(local["selection_mse"][0]),
            }
        path = output / f"family_{family}.npz"
        arrays = {
            name: np.stack([state[name] for state in trace["trace"]])
            for name in ("indices", "values", "quantized", "centers", "rho", "charge")
        }
        np.savez_compressed(path, **arrays)
        timeline = []
        truth_flat = oracle_basis[0].cpu().numpy().reshape(-1, spec["codes"])
        for state in trace["trace"]:
            metrics = [
                alignment(index, truth, spec["codes"])[0] for index in state["indices"]
            ]
            values = state["values"].reshape(len(metrics), -1).astype(np.float64)
            fitted = (values @ truth_flat) / np.square(truth_flat).sum(0)
            oracle_quantized = fitted @ truth_flat.T
            penalty = 2 * state["charge"] / state["rho"]
            dp_cost = np.square(values - state["quantized"].reshape(values.shape)).mean(
                -1
            ) + penalty * (state["indices"] != 0).mean((-1, -2))
            truth_cost = (
                np.square(values - oracle_quantized).mean(-1)
                + penalty * (truth != 0).mean()
            )
            timeline.append(
                {
                    "step": state["step"],
                    "exact_proposals": sum(item["exact_I"] for item in metrics),
                    "best_I_errors": min(
                        item["categorical_errors"] for item in metrics
                    ),
                    "active_mean": float((state["indices"] != 0).sum((-1, -2)).mean()),
                    "zero_proposals": int((state["indices"] == 0).all((-1, -2)).sum()),
                    "groups_mean": float(
                        np.mean(
                            [
                                len(np.unique(np.abs(index[index != 0])))
                                for index in state["indices"]
                            ]
                        )
                    ),
                    "min_teacher_minus_dp_projection_objective": float(
                        (truth_cost - dp_cost).min()
                    ),
                    "mean_teacher_minus_dp_projection_objective": float(
                        (truth_cost - dp_cost).mean()
                    ),
                }
            )
        original_recovery, _ = alignment(old_index, truth, spec["codes"])
        reproduced, _ = alignment(trace["indices"][0], old_index, spec["codes"])
        ranks = torch.linalg.matrix_rank(continuous_x).cpu().numpy()
        result = {
            "family": family,
            "timeline": timeline,
            "source_design_rank": ranks.tolist(),
            "source_design_nullity": (spec["inputs"] - ranks).tolist(),
            "selected_recovery": original_recovery,
            "reproduces_original_I": reproduced["exact_I"],
            "selected_source_mse": chosen_source,
            "selected_validation_mse": chosen_validation,
            "oracle_source_mse": oracle_source,
            "oracle_validation_mse": oracle_validation,
            "oracle_beats_selected_source": oracle_source < chosen_source - 1e-9,
            "exact_I_appeared_at_any_step": any(
                point["exact_proposals"] for point in timeline
            ),
            "oracle_initialized": oracle_initialized,
            "trace_sha256": digest(path),
        }
        save_json(output / f"family_{family}.json", result)
        results.append(result)
        print(
            "diagnosed",
            family,
            "original exact",
            original_recovery["exact_I"],
            "ever exact",
            result["exact_I_appeared_at_any_step"],
            "oracle source",
            oracle_source,
            "selected source",
            chosen_source,
            flush=True,
        )
    save_json(
        output / "summary.json",
        {"families": results, "diagnostic_only_teacher_use": True},
    )


def evaluate(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = root / "models_frozen.json"
    hashes = json.loads(frozen.read_text())["sha256"]
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Changed frozen state: {name}")
    records = []
    for family in plan["confirmation_families"]:
        folder = root / "hidden" / f"family_{family}"
        spec = json.loads((folder / "specification.json").read_text())
        with np.load(folder / "teacher_audit_only.npz") as archive:
            truth = dict(archive)
        cohort = target_cohort(
            folder,
            spec,
            truth,
            plan["seed"] + family * 700001 + 71000000,
            digest(frozen),
        )
        output = folder / "audit"
        output.mkdir(exist_ok=True)
        local_records = []
        for method in (*METHODS, "oracle"):
            for seed in [0] if method == "oracle" else plan["confirmation_seeds"]:
                if method == "oracle":
                    index = truth["indices"]
                    basis = index_basis(torch.tensor(index)[None], spec["codes"])[
                        0
                    ].numpy()
                else:
                    with np.load(
                        folder / "confirmation" / f"{method}_seed_{seed}.npz"
                    ) as archive:
                        index, basis = archive["indices"][0], archive["basis"][0]
                recovery, aligned = alignment(index, truth["indices"], spec["codes"])
                performance, fitted = transfer(
                    basis, cohort, spec["test_support"], device
                )
                record = {
                    "family": family,
                    "method": method,
                    "seed": seed,
                    "recovery": recovery,
                    "transfer": performance,
                }
                local_records.append(record)
                np.savez_compressed(
                    output / f"{method}_seed_{seed}.npz",
                    indices=index,
                    aligned_indices=aligned,
                    basis=basis,
                    **fitted,
                )
        save_json(output / "summary.json", {"records": local_records})
        records.extend(local_records)
        print(
            "audited new family",
            family,
            {
                method: sum(
                    row["recovery"]["exact_I"]
                    for row in local_records
                    if row["method"] == method
                )
                for method in METHODS
            },
            flush=True,
        )
    save_json(root / "audit_summary.json", {"records": records})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "register",
            "diagnose",
            "pilot",
            "freeze",
            "controls",
            "confirmation",
            "evaluate",
        ),
    )
    parser.add_argument(
        "--previous",
        type=Path,
        default=Path(
            "experiments/sharing_benchmark_v1/results/20261105/convolution_competitors"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if args.stage == "register":
        register(args.root, args.previous)
        return
    if args.stage == "diagnose":
        diagnose(args.root, args.previous, device)
        return
    if args.stage == "freeze":
        freeze_hpo(args.root)
        return
    if args.stage == "controls":
        register_controls(args.root)
        return
    if args.stage == "evaluate":
        evaluate(args.root, device)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.stage == "pilot":
        for family, seed, method in itertools.product(
            plan["pilot_families"], plan["pilot_seeds"], plan["methods"][:-1]
        ):
            run_one(
                args.root,
                family,
                "pilot",
                method,
                seed,
                configurations("consensus_dp"),
                device,
            )
        freeze_hpo(args.root)
    else:
        choices = json.loads((args.root / "hpo_frozen.json").read_text())["choices"]
        controls = json.loads((args.root / "controls_frozen.json").read_text())[
            "choices"
        ]
        for family, seed, method in itertools.product(
            plan["confirmation_families"], plan["confirmation_seeds"], METHODS
        ):
            rows = (
                []
                if method == "bank_query_edits"
                else [(controls if method in controls else choices)[method]["settings"]]
            )
            run_one(args.root, family, "confirmation", method, seed, rows, device)
        paths = list(args.root.glob("hidden/family_*/confirmation/*.npz"))
        expected = (
            len(plan["confirmation_families"])
            * len(plan["confirmation_seeds"])
            * len(METHODS)
        )
        if len(paths) != expected:
            raise RuntimeError(f"Expected {expected} states, got {len(paths)}")
        paths.extend(
            [
                args.root / "protocol.json",
                args.root / "hpo_frozen.json",
                args.root / "candidates.json",
                args.root / "controls_frozen.json",
            ]
        )
        save_json(
            args.root / "models_frozen.json",
            {
                "frozen_utc": stamp(),
                "sha256": {
                    str(path.relative_to(args.root)): digest(path) for path in paths
                },
            },
        )


if __name__ == "__main__":
    main()

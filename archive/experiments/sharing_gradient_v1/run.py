"""Registered pilot selection, four-GPU shards and frozen fresh-family audit."""

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
from experiments.sharing_benchmark_v1.linear import load_observations
from experiments.sharing_diagnostics_v1.algorithms import coordinate_refine

from .algorithms import (
    gradient_audit,
    meta_search,
    projected_meta_search,
    projection_search,
)

BASE = Path(__file__).resolve().parent
PROJECTIONS = ("z_euclidean", "z_hessian", "z_taylor")
METAS = ("meta_stop", "meta_full")
PROJECTED = ("projected_stop", "projected_full")
METHODS = (*PROJECTIONS, *METAS, *PROJECTED, "query_edits")


def stamp():
    return datetime.now(UTC).isoformat()


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "seed": 2611052301,
        "pilot_families": [2001, 2002],
        "pilot_seeds": [21, 22],
        "confirmation_families": list(range(2101, 2107)),
        "confirmation_seeds": [41, 42, 43],
        "task_budgets": [2, 4, 8, 64],
        "pilot_task_budget": 4,
        "restarts": 4,
        "projection_sweeps": 8,
        "projection_checkpoint": 10,
        "meta_steps": 800,
        "meta_checkpoint": 20,
        "inner_ridge": 1e-6,
        "latent_clip": 1.5,
        "hidden": {
            "inputs": 16,
            "outputs": 14,
            "codes": 3,
            "train_tasks": 64,
            "train_support": 4,
            "train_query": 4,
            "selection_tasks": 24,
            "selection_support": 4,
            "selection_query": 64,
            "noise": 0.1,
            "consensus_rounds": 80,
            "test_tasks": 100,
            "test_support": [1, 4, 16],
            "test_query": 256,
            "hidden_permutations": True,
        },
        "methods": list(METHODS),
        "definitions": {
            "z_euclidean": "Common I, separate Z_t/C_t, Euclidean projection. Same initialization, source observations, L-step and dual schedule as both other Z variants; no sparsity penalty.",
            "z_hessian": "Only replace projection by full empirical Hessian plus Euclidean proximal term. Omit the linear task-gradient term. Local conditional assignment/C sweeps; no exact multidimensional DP claim.",
            "z_taylor": "Same full metric as z_hessian, add the task gradient at Z+U. For squared loss this projector minimizes actual source support+query loss plus proximity. Its temporary C sees these source training observations; all evaluation C are reset and fit on support only.",
            "meta_stop": "Warm-start the pilot-selected z_euclidean I. Shared signed-one-hot/zero A, hard forward, support-fitted C, query gradient with C response stopped. Six Adam rates, four latent jitters, no per-entry query enumeration.",
            "meta_full": "Exactly meta_stop except differentiate through the damped exact support-fit C. The hard projection uses a surrogate straight-through backward, not an exact discrete gradient.",
            "query_edits": "Existing exact one-entry source-query refinement, at most64 accepted changes, initialized with the same z_euclidean I. Reference only, no HPO.",
        },
        "selection": {
            "checkpoint": "For each configuration, source-query chooses restart and improving source proposals; independent coefficient validation selects checkpoint. Exact unregularized support adaptation for all reported checkpoint scores.",
            "hpo": "Mean pilot validation over two new geometries and two seeds. First freeze projection settings, then meta pilots start their selected z_euclidean states. Meta settings frozen before new confirmation geometry preparation.",
            "budget_controls": "2/8/64-task controls use the same selected settings as4-task primary, with no extra HPO. These are nested subsets of64. Same independent validation tasks within family.",
            "pilot_calibration": "A prior8-task pilot at gradient_and_adaptation reached indistinguishable validation minima for every method/rate. No confirmation geometry/teacher/test had been opened. This final protocol uses4-task pilot for HPO, with2/8/64 only as frozen controls. Prior8-task artifacts remain unchanged.",
            "truth": "Only W shape and K enter search. Teacher mask, groups, active count and permutations are closed until every confirmation model is frozen.",
            "transfer": "100 new independent coefficient functions per new family, fresh exact support fit C, no source numeric state. Six independent geometries; seeds not counted as extra geometries.",
            "diagnostic": "Identical candidate pool from saved Z checkpoints and a fixed random set of one-entry perturbations. Compare Euclidean/refitted-C, empirical prediction metric, Taylor and actual post-support-fit query loss. Taylor equality is an algebraic guard, not evidence of a novel predictor.",
            "checks": "No software tests. One scientific gradient directional audit, exact Taylor identity guard, lint and figure inspection.",
        },
    }
    projections = [
        {"configuration": i, "rho_final": rho, "dual_step": dual}
        for i, (rho, dual) in enumerate(
            itertools.product((1.0, 10.0, 100.0), (0.25, 1.0))
        )
    ]
    metas = [
        {"configuration": i, "lr": rate}
        for i, rate in enumerate((0.001, 0.003, 0.01, 0.03, 0.1, 0.3))
    ]
    save_json(root / "protocol.json", plan)
    save_json(
        root / "candidates.json",
        {
            **{method: projections for method in PROJECTIONS},
            **{method: metas for method in METAS},
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
            "protocol_sha256": digest(root / "protocol.json"),
            "candidate_sha256": digest(root / "candidates.json"),
            "available_gpus_at_launch": [0, 1, 2, 3],
            "foreign_busy_gpus": [4, 5, 6, 7],
        },
    )
    for family in plan["pilot_families"]:
        prepare_family(root, plan, "hidden", family)


def cohort(folder, tasks, device):
    data = load_observations(folder, device)
    return {
        name: value[:tasks] if name.startswith("train_") else value
        for name, value in data.items()
    }


def result_path(folder, phase, tasks, method, seed):
    return folder / phase / f"tasks_{tasks}" / f"{method}_seed_{seed}.npz"


def save_result(path, result, inputs):
    path.parent.mkdir(parents=True, exist_ok=True)
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


def run_one(root, family, phase, tasks, method, seed, rows, device):
    folder = root / "hidden" / f"family_{family}"
    path = result_path(folder, phase, tasks, method, seed)
    inputs = {
        "method": method,
        "seed": seed,
        "tasks": tasks,
        "rows": rows,
        "observation_sha256": digest(folder / "observations.npz"),
    }
    if path.exists():
        stored = json.loads(path.with_suffix(".json").read_text())
        original = {
            name: value
            for name, value in stored["inputs"].items()
            if name
            not in (
                "warm_sha256",
                "meta_calibration_sha256",
                "gradient_controls_sha256",
            )
        }
        if original != inputs or stored["result_sha256"] != digest(path):
            raise RuntimeError(f"Changed inputs or result: {path}")
        return
    plan = json.loads((root / "protocol.json").read_text())
    controls = root / "gradient_controls.json"
    if controls.exists():
        plan["projected_gradient"] = json.loads(controls.read_text())[
            "projected_gradient"
        ]
    spec = json.loads((folder / "specification.json").read_text())
    data = cohort(folder, tasks, device)
    if method in PROJECTIONS:
        result = projection_search(method, rows, spec, plan, data, seed)
    else:
        warm_phase = "pilot_projection" if phase == "pilot_meta" else phase
        warm_path = result_path(folder, warm_phase, tasks, "z_euclidean", seed)
        warm_choice = json.loads((root / "projection_frozen.json").read_text())[
            "choices"
        ]["z_euclidean"]["settings"]["configuration"]
        with np.load(warm_path) as archive:
            initial = archive["indices"][warm_choice if phase == "pilot_meta" else 0]
        calibration = root / "meta_calibration.json"
        if phase == "pilot_meta" and calibration.exists():
            settings = json.loads(calibration.read_text())
            rng = np.random.default_rng(plan["seed"] + family * 100003 + seed)
            initial = initial.copy()
            positions = rng.choice(
                initial.size, settings["changed_entries"], replace=False
            )
            for position in positions:
                alternatives = np.arange(-spec["codes"], spec["codes"] + 1)
                alternatives = alternatives[alternatives != initial.flat[position]]
                initial.flat[position] = rng.choice(alternatives)
            inputs["meta_calibration_sha256"] = digest(calibration)
        if method in PROJECTED:
            inputs["gradient_controls_sha256"] = digest(controls)
            result = projected_meta_search(method, spec, plan, data, initial)
        elif method == "query_edits":
            result = coordinate_refine(initial, spec, data)
        else:
            if (
                family == plan["pilot_families"][0]
                and seed == plan["pilot_seeds"][0]
                and method == "meta_full"
            ):
                initial_tensor = torch.tensor(initial, device=device)
                audit = gradient_audit(
                    initial_tensor, data, spec["codes"], plan["inner_ridge"]
                )
                save_json(root / "hypergradient_audit.json", audit)
            result = meta_search(method, rows, spec, plan, data, seed, initial)
        inputs["warm_sha256"] = digest(warm_path)
    save_result(path, result, inputs)
    print(
        f"{family} {phase} T{tasks} {method} seed{seed} val={result['selection_mse'].tolist()} seconds={result['seconds']:.2f}",
        flush=True,
    )


def pilot(root, stage, shard, shards, device):
    plan = json.loads((root / "protocol.json").read_text())
    candidates = json.loads((root / "candidates.json").read_text())
    methods = (
        PROJECTIONS
        if stage == "pilot_projection"
        else (*METAS, *PROJECTED, "query_edits")
    )
    jobs = list(itertools.product(plan["pilot_families"], plan["pilot_seeds"]))
    for family, seed in jobs[shard::shards]:
        for method in methods:
            run_one(
                root,
                family,
                stage,
                plan["pilot_task_budget"],
                method,
                seed,
                candidates.get(method, [{}]),
                device,
            )


def freeze_hpo(root, projection_only):
    plan = json.loads((root / "protocol.json").read_text())
    candidates = json.loads((root / "candidates.json").read_text())
    choices = (
        {}
        if projection_only
        else json.loads((root / "projection_frozen.json").read_text())["choices"]
    )
    methods = PROJECTIONS if projection_only else METAS
    for method in methods:
        phase = "pilot_projection" if method in PROJECTIONS else "pilot_meta"
        errors = []
        for family, seed in itertools.product(
            plan["pilot_families"], plan["pilot_seeds"]
        ):
            path = result_path(
                root / "hidden" / f"family_{family}",
                phase,
                plan["pilot_task_budget"],
                method,
                seed,
            )
            with np.load(path) as archive:
                errors.append(archive["selection_mse"])
        mean = np.mean(errors, axis=0)
        selected = int(mean.argmin())
        choices[method] = {
            "settings": candidates[method][selected],
            "mean_selection_mse": float(mean[selected]),
            "all_mean_selection_mse": mean.tolist(),
        }
    output = "projection_frozen.json" if projection_only else "hpo_frozen.json"
    save_json(root / output, {"frozen_utc": stamp(), "choices": choices})
    print(
        output, {name: value["settings"] for name, value in choices.items()}, flush=True
    )
    if not projection_only:
        for family in plan["confirmation_families"]:
            prepare_family(root, plan, "hidden", family)
        shutil.copytree(
            BASE,
            root / "confirmation_code",
            ignore=shutil.ignore_patterns("results", "__pycache__"),
        )


def confirmation(root, shard, shards, device):
    plan = json.loads((root / "protocol.json").read_text())
    choices = json.loads((root / "hpo_frozen.json").read_text())["choices"]
    jobs = list(
        itertools.product(
            plan["confirmation_families"],
            plan["confirmation_seeds"],
            plan["task_budgets"],
        )
    )
    for family, seed, tasks in jobs[shard::shards]:
        for method in METHODS:
            rows = [choices[method]["settings"]] if method in choices else [{}]
            run_one(root, family, "confirmation", tasks, method, seed, rows, device)


def freeze_models(root):
    plan = json.loads((root / "protocol.json").read_text())
    hashes = {}
    for family, seed, tasks, method in itertools.product(
        plan["confirmation_families"],
        plan["confirmation_seeds"],
        plan["task_budgets"],
        METHODS,
    ):
        path = result_path(
            root / "hidden" / f"family_{family}", "confirmation", tasks, method, seed
        )
        hashes[str(path.relative_to(root))] = digest(path)
    save_json(root / "models_frozen.json", {"frozen_utc": stamp(), "sha256": hashes})
    print(
        "Frozen",
        len(hashes),
        "states before teacher and new target functions",
        flush=True,
    )


def evaluate(root, shard, shards, device):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = root / "models_frozen.json"
    manifest = json.loads(frozen.read_text())["sha256"]
    for name, expected in manifest.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Changed frozen model: {name}")
    for family in plan["confirmation_families"][shard::shards]:
        folder = root / "hidden" / f"family_{family}"
        spec = json.loads((folder / "specification.json").read_text())
        with np.load(folder / "teacher_audit_only.npz") as archive:
            truth = dict(archive)
        target = target_cohort(
            folder,
            spec,
            truth,
            plan["seed"] + family * 700001 + 81000000,
            digest(frozen),
        )
        records = []
        for tasks, method in itertools.product(
            plan["task_budgets"], (*METHODS, "oracle")
        ):
            for seed in [0] if method == "oracle" else plan["confirmation_seeds"]:
                if method == "oracle":
                    from experiments.sharing_benchmark_v1.linear import index_basis

                    index = truth["indices"]
                    basis = index_basis(torch.tensor(index[None]), spec["codes"])[
                        0
                    ].numpy()
                else:
                    with np.load(
                        result_path(folder, "confirmation", tasks, method, seed)
                    ) as archive:
                        index, basis = archive["indices"][0], archive["basis"][0]
                recovery, aligned = alignment(index, truth["indices"], spec["codes"])
                performance, states = transfer(
                    basis, target, spec["test_support"], device
                )
                records.append(
                    {
                        "family": family,
                        "tasks": tasks,
                        "method": method,
                        "seed": seed,
                        "recovery": recovery,
                        "transfer": performance,
                    }
                )
                output = folder / "audit" / f"tasks_{tasks}"
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    output / f"{method}_seed_{seed}.npz",
                    indices=index,
                    aligned_indices=aligned,
                    basis=basis,
                    **states,
                )
        save_json(folder / "audit" / "summary.json", {"records": records})
        print(
            "audited",
            family,
            {
                str(tasks): {
                    method: sum(
                        row["recovery"]["exact_I"]
                        for row in records
                        if row["tasks"] == tasks and row["method"] == method
                    )
                    for method in METHODS
                }
                for tasks in plan["task_budgets"]
            },
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "register",
            "pilot_projection",
            "freeze_projection",
            "pilot_meta",
            "freeze",
            "confirmation",
            "models_freeze",
            "evaluate",
        ),
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "register":
        register(args.root)
    elif args.stage == "freeze_projection":
        freeze_hpo(args.root, True)
    elif args.stage == "freeze":
        freeze_hpo(args.root, False)
    elif args.stage == "models_freeze":
        freeze_models(args.root)
    elif args.stage.startswith("pilot_"):
        pilot(args.root, args.stage, args.shard, args.shards, args.device)
    elif args.stage == "confirmation":
        confirmation(args.root, args.shard, args.shards, args.device)
    else:
        evaluate(args.root, args.shard, args.shards, args.device)


if __name__ == "__main__":
    main()

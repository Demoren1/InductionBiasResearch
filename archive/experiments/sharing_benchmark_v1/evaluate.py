"""Post-freeze structure audits and fresh coefficient-task transfer."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .data import prepare_family
from .decode import SOURCES as DECODED_SOURCES
from .decode import confirmation as decode_confirmation
from .linear import (
    adaptation_errors,
    adapted,
    index_basis,
    load_observations,
    prediction,
)
from .run import methods, run_one
from .soft import inner_adapt
from .yeh_shared import NAMES as SHARED_NAMES
from .yeh_shared import run_one as run_shared


def benchmark_freeze(root):
    core = root / "all_models_frozen.json"
    initial = json.loads(core.read_text())
    hashes = dict(initial["sha256"])
    hashes[core.name] = digest(core)
    for profile, expected in (("hidden", 36), ("native", 6)):
        path = root / f"{profile}_shared_models_frozen.json"
        state = json.loads(path.read_text())
        if len(state["sha256"]) != expected:
            raise RuntimeError(f"Incomplete shared-P confirmation: {profile}")
        hashes.update(state["sha256"])
        for name in (path.name, f"{profile}_shared_hpo_frozen.json"):
            hashes[name] = digest(root / name)
    hashes["shared_candidates.json"] = digest(root / "shared_candidates.json")
    decoded_path = root / "decoded_models_frozen.json"
    decoded = json.loads(decoded_path.read_text())
    hashes.update(decoded["sha256"])
    hashes[decoded_path.name] = digest(decoded_path)
    for name, expected in hashes.items():
        if digest(root / name) != expected:
            raise RuntimeError(f"Frozen model changed: {root / name}")
    path = root / "benchmark_frozen.json"
    if path.exists():
        if json.loads(path.read_text())["sha256"] != hashes:
            raise RuntimeError("The comprehensive benchmark freeze changed")
    else:
        save_json(path, {"frozen_utc": datetime.now(UTC).isoformat(), "sha256": hashes})
    return path


def dense_solution(x, y, ridge):
    if ridge == 0:
        return (torch.linalg.pinv(x, rtol=1e-10) @ y).transpose(-1, -2)
    identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    dual = torch.linalg.solve(
        x @ x.transpose(-1, -2) + ridge * x.shape[1] * identity, y
    )
    return (x.transpose(-1, -2) @ dual).transpose(-1, -2)


def freeze_dense(root, plan):
    path = root / "dense_hpo_frozen.json"
    if path.exists():
        return json.loads(path.read_text())
    candidates = np.geomspace(1e-6, 10, 12)
    chosen = {}
    for profile, families in (("hidden", plan["pilot_families"]), ("native", [301])):
        scores = []
        for family in families:
            data = load_observations(
                root / profile / f"family_{family}", torch.device("cpu")
            )
            x, y = (
                data["selection_support_x"].double(),
                data["selection_support_y"].double(),
            )
            qx, qy = (
                data["selection_query_x"].double(),
                data["selection_query_y"].double(),
            )
            values = []
            for ridge in candidates:
                weights = dense_solution(x, y, float(ridge))
                errors = (
                    (torch.einsum("tni,toi->tno", qx, weights) - qy).square().mean()
                )
                values.append(float(errors))
            scores.append(values)
        mean = np.mean(scores, axis=0)
        chosen[profile] = {
            "ridge": float(candidates[mean.argmin()]),
            "mean_selection_mse": mean.tolist(),
        }
    state = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "choices": chosen,
        "candidates": candidates.tolist(),
        "criterion": "Same independent coefficient-selection tasks, without teacher/test access; linear ridge numerical baseline.",
    }
    save_json(path, state)
    return state


def alignment(indices, truth, codes):
    scores = np.zeros((codes, codes), dtype=int)
    flips = np.ones((codes, codes), dtype=int)
    for left in range(1, codes + 1):
        for right in range(1, codes + 1):
            same = int(
                ((indices == left) & (truth == right)).sum()
                + ((indices == -left) & (truth == -right)).sum()
            )
            opposite = int(
                ((indices == left) & (truth == -right)).sum()
                + ((indices == -left) & (truth == right)).sum()
            )
            scores[left - 1, right - 1] = max(same, opposite)
            flips[left - 1, right - 1] = 1 if same >= opposite else -1
    left, right = linear_sum_assignment(-scores)
    aligned = np.zeros_like(indices)
    for source, target in zip(left, right):
        active = np.abs(indices) == source + 1
        aligned[active] = (
            np.sign(indices[active]) * flips[source, target] * (target + 1)
        )
    errors = int((aligned != truth).sum())
    predicted, actual = indices != 0, truth != 0
    true_positive = int((predicted & actual).sum())
    precision = true_positive / max(int(predicted.sum()), 1)
    recall = true_positive / max(int(actual.sum()), 1)
    return {
        "categorical_errors": errors,
        "exact_I": errors == 0,
        "mask_errors": int((predicted != actual).sum()),
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": 2 * precision * recall / max(precision + recall, 1e-15),
        "active": int(predicted.sum()),
    }, aligned


def subspace_error(basis, reference):
    matrix = basis.reshape(-1, basis.shape[-1]).astype(np.float64)
    target = reference.reshape(-1, reference.shape[-1]).astype(np.float64)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int((singular > max(singular.max(initial=0) * 1e-7, 1e-12)).sum())
    v, target_singular, _ = np.linalg.svd(target, full_matrices=False)
    target_rank = int((target_singular > 1e-10).sum())
    overlap = float((u[:, :rank].T @ v[:, :target_rank]).__pow__(2).sum())
    value = (rank + target_rank - 2 * overlap) / max(2 * target_rank, 1)
    return float(np.sqrt(max(value, 0))), rank


def target_cohort(folder, spec, truth, seed, frozen_sha256):
    path = folder / "target_after_freeze.npz"
    if path.exists():
        with np.load(path) as archive:
            return dict(archive)
    rng = np.random.default_rng(seed)
    codes = rng.normal(size=(spec["test_tasks"], spec["codes"]))
    index = truth["indices"]
    weights = np.where(index[None] != 0, codes[:, index - 1], 0)
    support = max(spec["test_support"])
    x = rng.normal(
        size=(spec["test_tasks"], support + spec["test_query"], spec["inputs"])
    )
    clean = np.einsum("tni,toi->tno", x, weights)
    observed = clean[:, :support] + spec["noise"] * rng.normal(
        size=clean[:, :support].shape
    )
    data = {
        "support_x": x[:, :support],
        "support_y": observed,
        "query_x": x[:, support:],
        "query_y": clean[:, support:],
        "teacher_weights_audit_only": weights,
    }
    np.savez_compressed(path, **data)
    save_json(
        folder / "target_manifest.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "all_models_frozen_sha256": frozen_sha256,
            "target_sha256": digest(path),
            "seed": seed,
        },
    )
    return data


def transfer(basis, cohort, sizes, device):
    basis_tensor = torch.tensor(basis[None], device=device, dtype=torch.float64)
    support_x = torch.tensor(cohort["support_x"], device=device, dtype=torch.float64)
    support_y = torch.tensor(cohort["support_y"], device=device, dtype=torch.float64)
    query_x = torch.tensor(cohort["query_x"], device=device, dtype=torch.float64)
    query_y = torch.tensor(cohort["query_y"], device=device, dtype=torch.float64)
    truth = torch.tensor(
        cohort["teacher_weights_audit_only"], device=device, dtype=torch.float64
    )
    records, states = {}, {}
    for size in sizes:
        coefficient = adapted(basis_tensor, support_x[:, :size], support_y[:, :size])
        weights = torch.einsum("soik,stk->stoi", basis_tensor, coefficient)[0]
        population = (weights - truth).square().sum((-1, -2)) / truth.shape[1]
        test = (
            (prediction(basis_tensor, coefficient, query_x)[0] - query_y)
            .square()
            .mean((-1, -2))
        )
        records[str(size)] = {
            "population_mse": float(population.mean()),
            "test_mse": float(test.mean()),
            "all_population_mse": population.cpu().tolist(),
            "all_test_mse": test.cpu().tolist(),
        }
        states[f"weights_support_{size}"] = weights[0].cpu().numpy()
        states[f"codes_support_{size}"] = coefficient[0, 0].cpu().numpy()
    matrix = basis.reshape(-1, basis.shape[-1]).astype(np.float64)
    teacher = cohort["teacher_weights_audit_only"].reshape(len(support_x), -1)
    projection = (teacher @ np.linalg.pinv(matrix).T) @ matrix.T
    records["representation_floor"] = float(
        np.mean(np.sum((projection - teacher) ** 2, axis=-1) / truth.shape[1])
    )
    return records, states


def dense_transfer(cohort, sizes, ridge, device):
    data = {
        key: torch.tensor(value, device=device, dtype=torch.float64)
        for key, value in cohort.items()
    }
    records, states = {}, {}
    truth = data["teacher_weights_audit_only"]
    for size in sizes:
        weights = dense_solution(
            data["support_x"][:, :size], data["support_y"][:, :size], ridge
        )
        population = (weights - truth).square().sum((-1, -2)) / truth.shape[1]
        test = (
            (torch.einsum("tni,toi->tno", data["query_x"], weights) - data["query_y"])
            .square()
            .mean((-1, -2))
        )
        records[str(size)] = {
            "population_mse": float(population.mean()),
            "test_mse": float(test.mean()),
            "all_population_mse": population.cpu().tolist(),
            "all_test_mse": test.cpu().tolist(),
        }
        states[f"weights_support_{size}"] = weights[0].cpu().numpy()
    return records, states


def evaluate_family(root, profile, family, seeds, dense_ridge, device, freeze_path):
    folder = root / profile / f"family_{family}"
    spec = json.loads((folder / "specification.json").read_text())
    plan = json.loads((root / "protocol.json").read_text())
    # All learned representations and source-derived negative controls are fixed
    # before generating any new target coefficient tasks.
    models = []
    for seed in seeds:
        for method in methods(plan):
            path = folder / "confirmation" / f"{method}_seed_{seed}.npz"
            with np.load(path) as archive:
                model = {key: archive[key][0].copy() for key in archive.files}
            models.append((method, seed, model))
        shared_choice = json.loads(
            (root / f"{profile}_shared_models_frozen.json").read_text()
        )
        for method in SHARED_NAMES:
            row = shared_choice["row_by_method"][method]
            path = folder / "shared_confirmation" / f"{method}_seed_{seed}.npz"
            with np.load(path) as archive:
                model = {key: archive[key][row].copy() for key in archive.files}
            models.append((method, seed, model))
        for source_method in DECODED_SOURCES:
            method = f"{source_method}_cluster"
            path = folder / "decoded_confirmation" / f"{method}_seed_{seed}.npz"
            with np.load(path) as archive:
                model = {key: archive[key][0].copy() for key in archive.files}
            models.append((method, seed, model))
        source = next(
            model
            for name, local_seed, model in models
            if name == "consensus_dp" and local_seed == seed
        )
        rng = np.random.default_rng(plan["seed"] + family * 1009 + seed)
        for method in ("shuffle_sharing", "shuffle_all"):
            index = source["indices"].copy()
            if method == "shuffle_sharing":
                index[index != 0] = rng.permutation(index[index != 0])
            else:
                index = rng.permutation(index.ravel()).reshape(index.shape)
            basis = index_basis(torch.tensor(index)[None], spec["codes"])[0].numpy()
            models.append((method, seed, {"indices": index, "basis": basis}))
    save_json(
        folder / "evaluation_choices.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "learned_and_shuffled_count": len(models),
            "models_frozen_sha256": digest(freeze_path),
            "controls": "Permute the already frozen learned assignments. No teacher mask or group counts used.",
        },
    )
    with np.load(folder / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    oracle = index_basis(torch.tensor(truth["indices"])[None], spec["codes"])[0].numpy()
    models.append(("oracle", 0, {"indices": truth["indices"], "basis": oracle}))
    cohort = target_cohort(
        folder,
        spec,
        truth,
        plan["seed"] + family * 100003 + 21100000,
        digest(freeze_path),
    )
    records = []
    output = folder / "audit"
    output.mkdir(exist_ok=True)
    for method, seed, model in models:
        record = {"method": method, "seed": seed, "profile": profile, "family": family}
        record["subspace_error"], record["rank"] = subspace_error(
            model["basis"], oracle
        )
        states = {"basis": model["basis"]}
        if "indices" in model:
            record["recovery"], aligned = alignment(
                model["indices"], truth["indices"], spec["codes"]
            )
            states.update(indices=model["indices"], aligned_indices=aligned)
        record["transfer"], fitted = transfer(
            model["basis"], cohort, spec["test_support"], device
        )
        states.update(fitted)
        if method == "msr":
            native_errors = {}
            basis = torch.tensor(model["basis"][None], device=device)
            for size in spec["test_support"]:
                x = torch.tensor(
                    cohort["support_x"][:, :size], device=device, dtype=torch.float32
                )
                y = torch.tensor(
                    cohort["support_y"][:, :size], device=device, dtype=torch.float32
                )
                coefficient = inner_adapt(
                    basis,
                    torch.tensor(model["initial"][None], device=device),
                    torch.tensor(model["inner_rate"][None], device=device),
                    x,
                    y,
                    9,
                )
                weights = torch.einsum("soik,stk->stoi", basis, coefficient)[0].double()
                teacher = torch.tensor(
                    cohort["teacher_weights_audit_only"], device=device
                )
                native_errors[str(size)] = float(
                    (weights - teacher).square().sum((-1, -2)).mean() / spec["outputs"]
                )
            record["msr_native_nine_steps"] = native_errors
        np.savez_compressed(output / f"{method}_seed_{seed}.npz", **states)
        records.append(record)
    for method, ridge in (("dense_ridge", dense_ridge), ("dense_ols", 0.0)):
        values, states = dense_transfer(cohort, spec["test_support"], ridge, device)
        records.append(
            {
                "method": method,
                "seed": 0,
                "family": family,
                "profile": profile,
                "ridge": ridge,
                "transfer": values,
            }
        )
        np.savez_compressed(output / f"{method}_seed_0.npz", **states)
    save_json(output / "summary.json", {"records": records})
    print(
        "audited",
        profile,
        family,
        {
            name: np.mean(
                [
                    row["transfer"]["4"]["population_mse"]
                    for row in records
                    if row["method"] == name
                ]
            )
            for name in {row["method"] for row in records}
        },
        flush=True,
    )


def tiny_benchmark(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    spec = {
        **plan["hidden"],
        "inputs": 4,
        "outputs": 2,
        "codes": 2,
        "train_tasks": 16,
        "train_support": 4,
        "train_query": 4,
        "selection_tasks": 16,
        "selection_support": 4,
        "selection_query": 32,
        "noise": 0.0,
        "test_tasks": 32,
        "test_support": [1, 4, 16],
        "test_query": 128,
    }
    plan["tiny"] = spec
    folder = root / "tiny" / "family_401"
    if not folder.exists():
        save_json(
            root / "tiny_registration.json",
            {
                "registered_utc": datetime.now(UTC).isoformat(),
                "specification": spec,
                "criterion": "Enumerate all canonical signed assignments with fixed zero and at most two groups. Fit numbers on selection-support and minimize mean selection-query. Tiny-model choices freeze before enumeration and teacher audit.",
            },
        )
        prepare_family(root, plan, "tiny", 401)
    frozen = json.loads((root / "hpo_frozen.json").read_text())
    for method in methods(plan):
        run_one(
            root,
            "tiny",
            401,
            "confirmation",
            method,
            11,
            [frozen["choices"][method]["settings"]],
            device,
        )
    shared = json.loads((root / "hidden_shared_hpo_frozen.json").read_text())["choices"]
    rows = list(
        {
            choice["settings"]["configuration"]: choice["settings"]
            for choice in shared.values()
        }.values()
    )
    run_shared(root, "tiny", 401, "shared_confirmation", 11, rows, device)
    decode_confirmation(root, "tiny", [401], [11], device)
    save_json(
        root / "tiny_shared_models_frozen.json",
        {
            "row_by_method": {
                name: next(
                    i
                    for i, row in enumerate(rows)
                    if row["configuration"] == choice["settings"]["configuration"]
                )
                for name, choice in shared.items()
            },
            "sha256": {
                str(path.relative_to(root)): digest(path)
                for path in (folder / "shared_confirmation").glob("*.npz")
            },
        },
    )
    paths = [
        *list((folder / "confirmation").glob("*.npz")),
        *list((folder / "shared_confirmation").glob("*.npz")),
        *list((folder / "decoded_confirmation").glob("*.npz")),
    ]
    save_json(
        folder / "tiny_models_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {str(path.relative_to(folder)): digest(path) for path in paths},
        },
    )
    data = load_observations(folder, device)
    assignments = []

    def extend(prefix, groups):
        if len(prefix) == 8:
            assignments.append(prefix)
            return
        extend([*prefix, 0], groups)
        for group in range(1, groups + 1):
            extend([*prefix, group], groups)
            extend([*prefix, -group], groups)
        if groups < 2:
            extend([*prefix, groups + 1], groups + 1)

    extend([], 0)
    indices = np.array(assignments, dtype=np.int16).reshape(-1, 2, 4)
    best, best_index = np.inf, None
    started = time.perf_counter()
    for start in range(0, len(indices), 512):
        batch = torch.tensor(indices[start : start + 512], device=device)
        error = adaptation_errors(
            index_basis(batch, 2),
            data["selection_support_x"],
            data["selection_support_y"],
            data["selection_query_x"],
            data["selection_query_y"],
            chunk=128,
        ).mean(-1)
        position = int(error.argmin())
        if float(error[position]) < best:
            best, best_index = float(error[position]), indices[start + position].copy()
    results = {}
    for method in methods(plan):
        path = folder / "confirmation" / f"{method}_seed_11.npz"
        with np.load(path) as archive:
            value = float(archive["selection_mse"][0])
            result = {
                "selection_mse": value,
                "gap_to_discrete_global": value - best,
                "within_1e_10_of_global": abs(value - best) <= 1e-10,
                "discrete": "indices" in archive.files,
            }
            if "indices" in archive.files:
                result["agreement_with_global"], _ = alignment(
                    archive["indices"][0], best_index, 2
                )
            results[method] = result
    for method in (*SHARED_NAMES, *(f"{name}_cluster" for name in DECODED_SOURCES)):
        is_shared = method in SHARED_NAMES
        path = (
            folder
            / ("shared_confirmation" if is_shared else "decoded_confirmation")
            / f"{method}_seed_11.npz"
        )
        if is_shared:
            choices = json.loads((root / "tiny_shared_models_frozen.json").read_text())
            row = choices["row_by_method"][method]
        else:
            row = 0
        with np.load(path) as archive:
            value = float(archive["selection_mse"][row])
            result = {
                "selection_mse": value,
                "gap_to_discrete_global": value - best,
                "within_1e_10_of_global": abs(value - best) <= 1e-10,
                "discrete": "indices" in archive.files,
            }
            if "indices" in archive.files:
                result["agreement_with_global"], _ = alignment(
                    archive["indices"][row], best_index, 2
                )
            results[method] = result
    save_json(
        folder / "global_enumeration.json",
        {
            "assignments": len(indices),
            "seconds": time.perf_counter() - started,
            "global_objective": best,
            "global_indices": best_index.tolist(),
            "results": results,
            "scope": "Global minimum over this finite tiny signed-indexator set, with exact numeric inner fit. Not a global-optimality certificate for the larger benchmark or continuous MSR U.",
        },
    )
    evaluate_family(
        root, "tiny", 401, [11], 0.0, device, folder / "tiny_models_frozen.json"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    frozen_path = args.root / "all_models_frozen.json"
    frozen = json.loads(frozen_path.read_text())
    for name, expected in frozen["sha256"].items():
        if digest(args.root / name) != expected:
            raise RuntimeError(f"Frozen learned state changed: {args.root / name}")
    frozen_path = benchmark_freeze(args.root)
    plan = json.loads((args.root / "protocol.json").read_text())
    device = torch.device(args.device)
    dense = freeze_dense(args.root, plan)
    if args.tiny:
        tiny_benchmark(args.root, device)
        return
    for profile, families in (
        ("hidden", plan["confirmation_families"]),
        ("native", [301]),
    ):
        for family in families:
            evaluate_family(
                args.root,
                profile,
                family,
                plan["confirmation_seeds"],
                dense["choices"][profile]["ridge"],
                device,
                frozen_path,
            )


if __name__ == "__main__":
    main()

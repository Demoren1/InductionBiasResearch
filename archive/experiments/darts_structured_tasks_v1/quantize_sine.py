"""Predeclared numeric clustering control for a soft sine search state."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from experiments.darts_indexator_v1.model import canonical_signed_probabilities

from .data import BASE, digest, save_json, teacher_predict
from .evaluate import align_indices, sine_population
from .model import score, soft_weights
from .train import fit, save_fit


def register():
    output = BASE / "results/20261104/sine_quantization_registration.json"
    if output.exists():
        raise FileExistsError(output)
    save_json(
        output,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "reason": "Synthetic DeepSets showed correct connectivity but dormant/merged categorical groups. Test whether clustering the canonical soft numerical matrix is a better discretizer than argmax P. No sine test result or teacher direction has been inspected.",
            "source_states": "For each of the three source-selected seeds of the winning confirmed configuration, use its already selected soft checkpoint and all four numerical replicas.",
            "zero_thresholds": [0.0, 0.05, 0.15],
            "threshold_scale": "max absolute soft weight, independently per vector",
            "quantizer": "KMeans of absolute active weights into the same m base groups, n_init32; signs become signed integer labels. No true roles, counts per group or teacher coefficients.",
            "fit": "Four Gaussian C starts at each radius .5/2/6/8 and each lr .003/.01/.03. Query chooses numerical configuration/start/checkpoint for each I; selection ranks I. Refit32 fresh C at the source-selected settings.",
            "evaluation": "Freeze before generating an independent16384 Gaussian test draw; also report exact population MSE after freezing. Keep direct DARTS and dense random/moment controls.",
            "claim": "This is an explicit numerical clustering postprocessor, not pure categorical argmax. It preserves the paired C/-C representation and canonicalizes P before constructing W.",
        },
    )
    print(output, flush=True)


def canonicalize(indices):
    labels, orientations = {}, {}
    result = np.zeros_like(indices)
    for position in np.ndindex(indices.shape):
        value = int(indices[position])
        if value == 0:
            continue
        key = abs(value)
        if key not in labels:
            labels[key] = len(labels) + 1
            orientations[key] = 1 if value > 0 else -1
        result[position] = labels[key] * orientations[key] * (1 if value > 0 else -1)
    return result


def run(parent, device):
    registration = BASE / "results/20261104/sine_quantization_registration.json"
    specification = json.loads((parent / "specification.json").read_text())
    plan = json.loads((parent / "protocol.json").read_text())
    selection = json.loads((parent / "architecture_selection.json").read_text())
    output = parent.parent / (parent.name + "_quantized")
    output.mkdir(exist_ok=False)
    shutil.copy2(registration, output / "registration.json")
    shutil.copy2(parent / "source.npz", output / "source.npz")
    save_json(output / "specification.json", specification)
    with np.load(parent / "confirmation/snapshots.npz") as archive:
        snapshots = dict(archive)
    proposals, provenance, seen = [], [], set()
    for chosen in selection["ranking"][0]["per_seed"]:
        number, root = chosen["snapshot"], chosen["root"]
        p = torch.tensor(snapshots["probabilities"][number, root : root + 1])
        numeric = torch.tensor(snapshots["numeric"][number, root : root + 1])
        weights = soft_weights(
            canonical_signed_probabilities(p), numeric, specification["codes"]
        ).numpy()[0, ..., 0]
        for restart, weight in enumerate(weights):
            magnitude = np.abs(weight)
            for threshold in (0.0, 0.05, 0.15):
                mask = magnitude >= threshold * magnitude.max()
                if mask.sum() < specification["codes"]:
                    continue
                clustering = KMeans(
                    n_clusters=specification["codes"],
                    n_init=32,
                    random_state=plan["seed"],
                ).fit(magnitude[mask, None])
                order = np.argsort(clustering.cluster_centers_[:, 0])
                labels = np.empty(specification["codes"], dtype=int)
                labels[order] = np.arange(1, specification["codes"] + 1)
                index = np.zeros((len(weight), 1), dtype=np.int16)
                index[mask, 0] = labels[clustering.labels_] * np.sign(
                    weight[mask]
                ).astype(int)
                index = canonicalize(index)
                key = index.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(index)
                provenance.append(
                    {
                        "source": chosen,
                        "numeric_restart": restart,
                        "zero_threshold": threshold,
                    }
                )
    rows, indices = [], []
    for candidate, index in enumerate(proposals):
        for radius, rate in itertools.product(
            (0.5, 2.0, 6.0, 8.0), (0.003, 0.01, 0.03)
        ):
            rows.append(
                {
                    "candidate": candidate,
                    "radius": radius,
                    "numeric_lr": rate,
                    "seed": 101,
                    "aligned_initial_codes": False,
                }
            )
            indices.append(index)
    result = fit(
        output, plan, specification, rows, device, indices=np.array(indices), restarts=4
    )
    result["indices"] = np.array(indices)
    save_fit(output / "candidate_fits", result, rows)
    eligible = []
    for candidate in range(len(proposals)):
        selected = min(
            (i for i, row in enumerate(rows) if row["candidate"] == candidate),
            key=lambda i: result["query_mse"][i].min(),
        )
        restart = int(result["query_mse"][selected].argmin())
        eligible.append(
            {
                "row": selected,
                "restart": restart,
                "selection_mse": float(result["selection_mse"][selected, restart]),
            }
        )
    selected = min(eligible, key=lambda row: row["selection_mse"])
    number, restart = selected["row"], selected["restart"]
    chosen_index = indices[number]
    fresh_rows = [{**rows[number], "seed": 102}]
    fresh = fit(
        output,
        plan,
        specification,
        fresh_rows,
        device,
        indices=chosen_index[None],
        restarts=32,
    )
    save_fit(output / "fresh", fresh, fresh_rows)
    fresh_restart = int(fresh["query_mse"][0].argmin())
    save_json(
        output / "source_selection.json",
        {
            "selected": selected,
            "settings": rows[number],
            "proposal": provenance[rows[number]["candidate"]],
            "unique_proposals": len(proposals),
            "fresh_restart": fresh_restart,
        },
    )
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {
                name: digest(output / name)
                for name in (
                    "registration.json",
                    "source.npz",
                    "candidate_fits.npz",
                    "fresh.npz",
                    "source_selection.json",
                )
            },
        },
    )
    with np.load(parent / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    rng = np.random.default_rng(plan["seed"] + 2900000)
    x = rng.normal(size=(16384, specification["inputs"])).astype(np.float32)
    y = teacher_predict(x, truth, specification)
    x_tensor, y_tensor = torch.tensor(x, device=device), torch.tensor(y, device=device)
    test = (
        score(
            x_tensor,
            y_tensor,
            torch.tensor(result["weights"][number : number + 1], device=device),
            torch.tensor(result["numeric"][number : number + 1], device=device),
            specification,
        )
        .cpu()
        .numpy()[0]
    )
    test_fresh = (
        score(
            x_tensor,
            y_tensor,
            torch.tensor(fresh["weights"], device=device),
            torch.tensor(fresh["numeric"], device=device),
            specification,
        )
        .cpu()
        .numpy()[0]
    )
    population = sine_population(
        result["weights"][number, ..., 0],
        result["numeric"][number, :, specification["codes"] + 1],
        truth,
        specification,
    )
    fresh_population = sine_population(
        fresh["weights"][0, ..., 0],
        fresh["numeric"][0, :, specification["codes"] + 1],
        truth,
        specification,
    )
    recovery, aligned, columns = align_indices(
        chosen_index, truth["indices"], specification["codes"]
    )
    np.savez_compressed(
        output / "selected_states.npz",
        indices=chosen_index,
        aligned_indices=aligned,
        aligned_columns=columns,
        weights=result["weights"][number, restart],
        numeric=result["numeric"][number, restart],
        fresh_weights=fresh["weights"][0, fresh_restart],
        fresh_numeric=fresh["numeric"][0, fresh_restart],
    )
    summary = {
        "task": specification,
        "method": "canonical soft weights -> signed k-means -> fresh C",
        "recovery": recovery,
        "selected_population_mse": float(population[restart]),
        "fresh_population_mse": float(fresh_population[fresh_restart]),
        "selected_test_mse": float(test[restart]),
        "fresh_test_mse": float(test_fresh[fresh_restart]),
        "fresh_successes": int((fresh_population < 0.1).sum()),
        "fresh_restarts": 32,
        "source_selection": selected,
        "unique_proposals": len(proposals),
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path, nargs="?")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.register:
        register()
    elif args.parent is not None:
        run(args.parent, torch.device("cuda:0"))
    else:
        parser.error("Provide a source run or --register")


if __name__ == "__main__":
    main()

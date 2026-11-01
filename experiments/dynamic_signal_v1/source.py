import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from experiments.dynamic_signal_v1.fit import fit


def align_columns(weights, readout, reference, reference_sign):
    mask = (weights != 0).astype(float)
    orders = []
    for item, signs in zip(mask, np.sign(readout)):
        intersection = item.T @ reference
        union = item.sum(0)[:, None] + reference.sum(0)[None] - intersection
        score = intersection / np.maximum(union, 1)
        score -= 0.25 * (signs[:, None] != reference_sign[None])
        old, new = linear_sum_assignment(-score)
        order = np.empty(item.shape[1], dtype=int)
        order[new] = old
        orders.append(order)
    return np.array(orders)


def extract(plan, raw, output):
    weights, bias, readout, offset = [raw[f"parameter_{i}"].copy() for i in range(4)]
    norm = np.linalg.norm(weights, axis=1).clip(1e-9)
    weights /= norm[:, None]
    bias /= norm
    readout *= norm
    tasks = raw["task_ids"]
    train = tasks < 8
    quality = np.maximum(0.001, np.log(2) - raw["query_loss"])
    for task in range(plan["source_tasks"]):
        ids = tasks == task
        quality[ids] /= quality[ids].sum()
    quality[~train] = 0
    quality /= quality.sum()
    reference_id = np.flatnonzero(train)[raw["query_loss"][train].argmin()]
    reference = (weights[reference_id] != 0).astype(float)
    reference_sign = np.sign(readout[reference_id])
    for _ in range(8):
        orders = align_columns(weights, readout, reference, reference_sign)
        masks = np.stack(
            [(item[:, order] != 0) for item, order in zip(weights, orders)]
        )
        reference = (masks * quality[:, None, None]).sum(0)
    aligned_weights = np.stack([item[:, order] for item, order in zip(weights, orders)])
    aligned_bias = np.take_along_axis(bias, orders, axis=1)
    aligned_readout = np.take_along_axis(readout, orders, axis=1)
    support = np.zeros(reference.size, dtype=int)
    support[np.argsort(-reference.flatten())[:256]] = 1
    profiles = []
    for task in range(8):
        ids = tasks == task
        importance = quality[ids] / quality[ids].sum()
        profiles.append((aligned_weights[ids] * importance[:, None, None]).sum(0))
    profiles = np.stack(profiles, -1).reshape(-1, 8)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-9)
    clustering = KMeans(n_clusters=16, n_init=20, random_state=plan["seed"])
    clustering.fit(profiles[support.astype(bool)])
    labels = clustering.predict(profiles) + 1
    indices = ((aligned_weights != 0).reshape(len(weights), -1) * labels).reshape(
        weights.shape
    )
    consensus = (support * labels).reshape(reference.shape)
    first_seen = list(dict.fromkeys(consensus.flatten().tolist()))
    nonzero = [value for value in first_seen if value]
    nonzero.extend(value for value in range(1, 17) if value not in nonzero)
    mapping = np.zeros(17, dtype=int)
    mapping[nonzero] = np.arange(1, 17)
    indices = mapping[indices]
    consensus = mapping[consensus]
    coefficients = np.zeros((len(weights), 16), dtype=np.float32)
    for index in range(len(weights)):
        for code in range(1, 17):
            selected = indices[index] == code
            if selected.any():
                coefficients[index, code - 1] = aligned_weights[index][selected].mean()
    np.savez_compressed(
        output / "extracted.npz",
        indices=indices,
        consensus=consensus,
        orders=orders,
        reference_id=reference_id,
        frequency=reference,
        profiles=profiles,
        original_query_loss=raw["query_loss"],
        original_query_accuracy=raw["query_accuracy"],
    )
    return indices, [coefficients, aligned_bias, aligned_readout, offset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-id", type=int, required=True)
    parser.add_argument("--calibration-size", type=int, default=1024)
    args = parser.parse_args()
    plan = json.loads(Path("experiments/dynamic_signal_v1/protocol.json").read_text())
    root = Path(plan["run"])
    bank_name = "banks" if args.calibration_size == 1024 else "banks_large_tuned"
    output = root / bank_name / f"noise_{args.noise_id}"
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    data = dict(np.load(root / "source" / f"noise_{args.noise_id}.npz"))
    calibration = json.loads(
        (
            root
            / "calibration_fit"
            / f"noise_{args.noise_id}_n{args.calibration_size}"
            / "selection.json"
        ).read_text()
    )
    config = calibration["adam"]["config"]
    options = copy.deepcopy(plan)
    options["steps"] = 4096
    original = fit(
        options, data, "dense_source", [config], 16384, 4, device, output / "round_4096"
    )
    raw = original
    for keep in [2048, 1024, 512, 256]:
        weights = torch.tensor(raw["parameter_0"], device=device)
        selected = weights.abs().flatten(1).topk(keep, dim=1).indices
        mask = (
            torch.zeros_like(weights.flatten(1))
            .scatter_(1, selected, 1)
            .reshape_as(weights)
        )
        initial = [raw[f"parameter_{i}"] for i in range(4)]
        raw = fit(
            options,
            data,
            "sparse_source",
            [config],
            16384,
            4,
            device,
            output / f"round_{keep}",
            mask=mask,
            initial=initial,
            steps=2048,
        )
    indices, initial = extract(plan, raw, output)
    coded_configs = [
        {
            "optimizer": "adam",
            "lr": rate,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "initial_scale": 1.0,
        }
        for rate in plan["grids"]["codebook_lr"]
    ]
    post = fit(
        options,
        data,
        "source_codebook",
        coded_configs,
        16384,
        4,
        device,
        output / "codebook_refit",
        indices=torch.tensor(
            np.tile(indices, (len(coded_configs), 1, 1)), device=device
        ),
        initial=[np.concatenate([value] * len(coded_configs)) for value in initial],
        steps=4096,
    )
    count = len(indices)
    selected = post["query_loss"].reshape(len(coded_configs), count).argmin(0)
    chosen = selected * count + np.arange(count)
    post = {key: value[chosen] for key, value in post.items()}
    retained = []
    for task in range(plan["source_tasks"]):
        ids = np.flatnonzero((post["task_ids"] == task) & (post["query_loss"] < 0.65))
        retained.extend(ids[np.argsort(post["query_loss"][ids])[:2]])
    retained = np.array(retained, dtype=int)
    task_ids = post["task_ids"][retained]
    role = np.where(task_ids < 8, 0, np.where(task_ids < 10, 1, 2))
    np.savez_compressed(
        output / "bank.npz",
        indices=indices[retained],
        task_ids=task_ids,
        role=role,
        query_loss=post["query_loss"][retained],
        query_accuracy=post["query_accuracy"][retained],
        retained_ids=retained,
    )
    result = {
        "source_tasks": plan["source_tasks"],
        "networks_per_task": 4,
        "ordinary_source_adam": config,
        "codebook_configs": coded_configs,
        "selected_source_codebook_config_ids": selected.tolist(),
        "source_steps": [4096, 2048, 2048, 2048, 2048],
        "post_codebook_steps": 4096,
        "nonzero_budget": 256,
        "codes": 16,
        "original_dense_query_accuracy": float(original["query_accuracy"].mean()),
        "sparse_query_accuracy": float(raw["query_accuracy"].mean()),
        "codebook_query_accuracy": float(post["query_accuracy"].mean()),
        "kept": len(retained),
        "role_counts": [int((role == index).sum()) for index in range(3)],
        "retained_query_accuracy": float(post["query_accuracy"][retained].mean())
        if len(retained)
        else None,
        "extraction": "Source-training weights/masks and readout signs only; no X covariance, no teacher I or permutation. Known budgets are 256 active first-layer weights and 16 scalar codes. Column alignment uses support Jaccard with readout-sign penalty; tying uses source-training weight profiles.",
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BANK", result, flush=True)


if __name__ == "__main__":
    main()

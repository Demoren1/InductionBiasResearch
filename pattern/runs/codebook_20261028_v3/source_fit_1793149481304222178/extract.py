"""Filter solutions, align hidden columns, then infer the shared indexator."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def column_orders(masks, reference):
    orders = []
    for mask in masks:
        old, new = linear_sum_assignment(-(mask.T @ reference))
        order = np.empty(mask.shape[1], dtype=int)
        order[new] = old
        orders.append(order)
    return np.array(orders)


def good_aligned_bank(plan, raw):
    spec = plan["extraction"]
    networks = plan["bank"]["networks_per_task"]
    train_count = plan["bank"]["train_networks"]
    validation_count = plan["bank"]["validation_networks"]
    all_ids = np.arange(len(raw["query_loss"]))
    task_ids = all_ids // networks
    within_task = all_ids % networks
    roles = np.where(
        within_task < train_count,
        0,
        np.where(within_task < train_count + validation_count, 1, 2),
    )
    good = raw["query_loss"] <= spec["maximum_query_bce"]
    if "minimum_query_accuracy" in spec:
        good &= raw["query_accuracy"] >= spec["minimum_query_accuracy"]
    retained = []
    seeds = []
    for task in range(len(plan["source_patterns"])):
        for role in range(3):
            ids = all_ids[good & (task_ids == task) & (roles == role)]
            ids = ids[np.argsort(raw["query_loss"][ids], kind="stable")]
            if "retain_fraction" in spec:
                pool_size = int(((task_ids == task) & (roles == role)).sum())
                quota = max(1, int(np.ceil(pool_size * spec["retain_fraction"])))
                ids = ids[:quota]
            if role == 0:
                ids = ids[: spec["maximum_training_maps_per_task"]]
                if len(ids):
                    seeds.append(ids[0])
            retained.extend(ids)
    retained = np.array(sorted(retained), dtype=int)
    if not len(seeds):
        raise ValueError("No good training solutions; collect a larger or better bank")
    training_ids = retained[roles[retained] == 0]
    quality = np.maximum(0, math.log(2) - raw["query_loss"][training_ids])
    for task in np.unique(task_ids[training_ids]):
        group = task_ids[training_ids] == task
        quality[group] /= quality[group].sum()
    quality /= quality.sum()
    masks = raw["mask"][training_ids]
    best = None
    for seed in seeds[: spec["alignment_starts"]]:
        reference = raw["mask"][seed].copy()
        for _ in range(spec["alignment_iterations"]):
            orders = column_orders(masks, reference)
            aligned = np.stack([m[:, order] for m, order in zip(masks, orders)])
            reference = (aligned * quality[:, None, None]).sum(0)
        agreement = float(np.square(reference).sum())
        if best is None or agreement > best[0]:
            best = agreement, reference.copy(), int(seed)
    agreement, reference, seed = best
    support = np.zeros((11, 8), dtype=np.int64)
    np.put_along_axis(support, np.argsort(-reference, axis=0)[:4], 1, axis=0)
    centers = (support * np.arange(11)[:, None]).sum(0) / 4
    canonical = np.argsort(centers, kind="stable")
    reference, support = reference[:, canonical], support[:, canonical]
    orders = column_orders(raw["mask"][retained], reference)
    bank = {}
    for key, array in raw.items():
        values = array[retained]
        if key in ("parameter_0", "mask"):
            values = np.stack([v[:, order] for v, order in zip(values, orders)])
        elif key == "features":
            values = np.stack([v[:, order, :] for v, order in zip(values, orders)])
        elif key in ("parameter_1", "parameter_2", "column_order"):
            values = np.take_along_axis(values, orders, axis=1)
        bank[key] = values
    bank.update(
        task_ids=task_ids[retained], split_ids=roles[retained], raw_ids=retained
    )
    train_rows = np.flatnonzero(bank["split_ids"] == 0)
    weights = bank["parameter_0"][train_rows]
    readout = bank["parameter_2"][train_rows]
    weights = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-9)
    profiles = []
    for task in range(len(plan["source_patterns"])):
        group = bank["task_ids"][train_rows] == task
        importance = quality[group, None, None] * (readout[group, None, :] > 0)
        profiles.append(
            (weights[group] * importance).sum(0) / np.maximum(importance.sum(0), 1e-9)
        )
    profiles = np.array(profiles).reshape(len(plan["source_patterns"]), 88).T
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-9)
    active = np.flatnonzero(support.flatten())
    selected = profiles[active]
    rng = np.random.default_rng(plan["seed"] + 1)
    best = None
    for _ in range(spec["cluster_starts"]):
        centers = selected[rng.choice(32, 4, replace=False)].copy()
        for _ in range(spec["cluster_iterations"]):
            distance = np.square(selected[:, None] - centers[None]).sum(-1)
            rows, slots = linear_sum_assignment(np.repeat(distance, 8, axis=1))
            labels = np.empty(32, dtype=int)
            labels[rows] = slots // 8
            centers = np.array([selected[labels == k].mean(0) for k in range(4)])
            centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
        error = float(np.square(selected - centers[labels]).sum())
        if best is None or error < best[0]:
            best = error, labels.copy(), centers.copy()
    _, labels, centers = best
    indices = np.zeros(88, dtype=np.int64)
    indices[active] = labels + 1
    all_labels = np.square(profiles[:, None] - centers[None]).sum(-1).argmin(1) + 1
    targets = bank["mask"].reshape(-1, 88).astype(np.int64) * all_labels[None]
    details = {
        "support_frequency": reference,
        "profiles": profiles,
        "centers": centers,
        "alignment_agreement": np.array(agreement),
        "alignment_initial_raw_id": np.array(seed),
        "orders": orders,
    }
    print(
        "GOOD BANK counts by task/role:",
        [
            [
                int(((bank["task_ids"] == task) & (bank["split_ids"] == role)).sum())
                for role in range(3)
            ]
            for task in range(len(plan["source_patterns"]))
        ],
        flush=True,
    )
    return bank, indices.reshape(11, 8), targets.reshape(-1, 11, 8), details

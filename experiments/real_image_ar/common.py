"""Image tasks and fixed-mask numeric fitting for the real-data experiment."""

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST

from experiments.dynamic_signal_v1.fit import fit

BASE = Path("experiments/real_image_ar")


def load_plan():
    plan = json.loads((BASE / "protocol.json").read_text())
    if not 0 < plan["active_features"] < plan["inputs"]:
        raise ValueError("Invalid input-feature budget")
    return plan


def images(plan, name, train=True):
    dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[name](
        plan["image_root"], train=train
    )
    values = F.avg_pool2d(dataset.data[:, None].float() / 255, 2).flatten(1)
    return values.numpy(), dataset.targets.numpy()


def prepare(plan, name):
    root = Path(plan["run"]) / name
    if (root / "prepared.json").exists():
        return
    root.mkdir(parents=True, exist_ok=False)
    source = Path(plan["source_root"]) / f"feature_bank_{name}_v1"
    source_plan = json.loads((source / "protocol.json").read_text())
    info = json.loads((source / "complete.json").read_text())
    values, labels = images(plan, name)
    splits = np.load(source / "source_splits.npz")
    sx, qx = values[splits["support"]], values[splits["query"]]
    sy = np.tile(
        np.repeat([0, 1], source_plan["source_support_per_class"]), (len(sx), 1)
    )
    qy = np.tile(np.repeat([0, 1], source_plan["source_query_per_class"]), (len(qx), 1))
    np.savez_compressed(
        root / "source.npz",
        support_x=sx,
        support_y=sy.astype(np.float32),
        query_x=qx,
        query_y=qy.astype(np.float32),
        test_x=qx,
        test_y=qy.astype(np.float32),
    )
    with np.load(source / "good_bank.npz") as bank:
        kept = {
            key: bank[key] for key in ["mask", "task", "role", "quality", "accuracy"]
        }
    np.savez_compressed(root / "original_bank.npz", **kept)
    # Gradient transfer sees the same union of source support/query images.
    # A deterministic stratified split supplies its own checkpoint selection.
    rng = np.random.default_rng(plan["seed"] + 5)
    train_ids, validation_ids = [], []
    observed = np.unique(
        np.concatenate([splits["support"].ravel(), splits["query"].ravel()])
    )
    for cls in range(6):
        candidates = rng.permutation(observed[labels[observed] == cls])
        train_ids.extend(candidates[:88])
        validation_ids.extend(candidates[88:])
    np.savez_compressed(
        root / "pretrain_data.npz", train_ids=train_ids, validation_ids=validation_ids
    )
    manifest = {
        "source_bank_sha256": hashlib.sha256(
            (source / "good_bank.npz").read_bytes()
        ).hexdigest(),
        "source_pairs": info["pairs"],
        "source_images": len(observed),
        "pretrain_images": len(train_ids),
        "pretrain_validation_images": len(validation_ids),
        "source_classes": list(range(6)),
        "reserved_pairs": source_plan["reserved_pairs"],
    }
    (root / "prepared.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "protocol.json").write_text(json.dumps(plan, indent=2) + "\n")
    make_cases(plan, name, "calibration")


def make_cases(plan, name, role):
    root = Path(plan["run"]) / name
    values, labels = images(plan, name)
    old = Path(plan["source_root"]) / f"feature_bank_{name}_v1"
    bank_splits = np.load(old / "source_splits.npz")
    if role == "calibration":
        pairs = [[0, 5], [2, 3]]
        draws = plan["calibration_draws"]
        support_pool = {c: bank_splits[f"pool_{c}"][100:] for c in range(6)}
        query_pool = {c: bank_splits[f"query_pool_{c}"][128:] for c in range(6)}
        test_values, test_labels = values, labels
    else:
        if not (root / "frozen.json").exists():
            raise RuntimeError("Freeze source choices before creating targets")
        pairs = list(itertools.combinations(range(6, 10), 2))
        draws = plan["target_draws"]
        support_pool, query_pool = {}, {}
        for cls in range(6, 10):
            ids = np.random.default_rng(plan["seed"] + cls).permutation(
                np.flatnonzero(labels == cls)
            )
            support_pool[cls], query_pool[cls] = ids[:2000], ids[2000:4000]
        test_values, test_labels = images(plan, name, train=False)
    cases = [
        {"pair": list(pair), "draw": draw} for pair in pairs for draw in range(draws)
    ]
    for size in plan["support_per_class"]:
        if (root / f"{role}_{size}.npz").exists():
            continue
        support_ids, query_ids, test_ids = [], [], []
        for case_id, case in enumerate(cases):
            rng = np.random.default_rng(plan["seed"] + 5000 + case_id)
            support_ids.append(
                np.concatenate(
                    [rng.permutation(support_pool[c])[:size] for c in case["pair"]]
                )
            )
            query_ids.append(
                np.concatenate(
                    [
                        rng.choice(
                            query_pool[c], plan["query_per_class"], replace=False
                        )
                        for c in case["pair"]
                    ]
                )
            )
            if role == "calibration":
                test_ids.append(query_ids[-1])
            else:
                test_ids.append(
                    np.concatenate(
                        [
                            rng.choice(
                                np.flatnonzero(test_labels == c),
                                plan["test_per_class"],
                                replace=False,
                            )
                            for c in case["pair"]
                        ]
                    )
                )
        support_ids, query_ids, test_ids = map(
            np.asarray, [support_ids, query_ids, test_ids]
        )
        if any(np.intersect1d(s, q).size for s, q in zip(support_ids, query_ids)):
            raise RuntimeError("Support/query image overlap")
        np.savez_compressed(
            root / f"{role}_{size}.npz",
            support_x=values[support_ids],
            support_y=np.tile(np.repeat([0, 1], size), (len(cases), 1)).astype(
                np.float32
            ),
            query_x=values[query_ids],
            query_y=np.tile(
                np.repeat([0, 1], plan["query_per_class"]), (len(cases), 1)
            ).astype(np.float32),
            test_x=test_values[test_ids],
            test_y=np.tile(
                np.repeat([0, 1], len(test_ids[0]) // 2), (len(cases), 1)
            ).astype(np.float32),
        )
        np.savez_compressed(
            root / f"{role}_{size}_ids.npz",
            support=support_ids,
            query=query_ids,
            test=test_ids,
        )
    (root / f"{role}_cases.json").write_text(json.dumps(cases, indent=2) + "\n")


def numeric_configs(plan, method):
    optimizer = "sgd" if method == "sgd" else "adam"
    rates = plan["sgd_lr"] if optimizer == "sgd" else plan["adam_lr"]
    momenta = plan["momentum"] if optimizer == "sgd" else [0.0]
    scales = (
        [1.0]
        if method in ["transfer", "frozen_transfer", "linear"]
        else plan["initial_scale"]
    )
    return [
        {
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": wd,
            "momentum": momentum,
            "initial_scale": scale,
        }
        for lr, wd, momentum, scale in itertools.product(
            rates, plan["weight_decay"], momenta, scales
        )
    ]


def fit_masks(
    plan,
    data,
    masks,
    method,
    configs,
    restarts,
    output,
    device,
    case_ids=None,
    steps=None,
    pretrained=None,
):
    """Make compact per-restart inputs and call the existing numeric solver."""
    cases = len(data["support_y"]) if case_ids is None else len(case_ids)
    tasks = np.repeat(np.arange(cases) if case_ids is None else case_ids, restarts)
    count = len(tasks)
    if masks is None:
        selected = np.tile(np.arange(plan["inputs"]), (count, 1))
    else:
        selected = np.stack([np.flatnonzero(mask) for mask in masks])
        if selected.shape != (count, plan["active_features"]):
            raise ValueError(
                "Each restart must have exactly the declared feature budget"
            )
    prepared = {}
    for split in ["support", "query", "test"]:
        inputs = data[split + "_x"][tasks]
        prepared[split + "_x"] = np.take_along_axis(inputs, selected[:, None], axis=-1)
        prepared[split + "_y"] = data[split + "_y"][tasks]
    initial = None
    kind = "linear" if method in ["linear", "frozen_transfer"] else method
    if method == "frozen_transfer":
        for split in ["support", "query", "test"]:
            prepared[split + "_x"] = np.maximum(
                prepared[split + "_x"] @ pretrained[0] + pretrained[1], 0
            )
    dimensions = prepared["support_x"].shape[-1]
    options = dict(plan, blocks=1, block_size=dimensions)
    options["batch_size"] = min(plan["batch_size"], len(prepared["support_y"][0]))
    if method == "transfer":
        total = len(configs) * count
        torch.manual_seed(plan["seed"] + 400)
        torch.randn(total, dimensions, plan["hidden"], device=device)
        readout = torch.randn(total, plan["hidden"], device=device) / np.sqrt(
            plan["hidden"]
        )
        initial = [
            np.broadcast_to(pretrained[0], (total, *pretrained[0].shape)).copy(),
            np.broadcast_to(pretrained[1], (total, len(pretrained[1]))).copy(),
            readout.cpu().numpy(),
            np.zeros(total, np.float32),
        ]
    arrays = fit(
        options,
        prepared,
        kind,
        configs,
        len(prepared["support_y"][0]),
        1,
        device,
        output,
        initial=initial,
        steps=steps,
    )
    expanded = arrays["task_ids"].copy()
    arrays["task_ids"], arrays["restart_ids"] = (
        expanded // restarts,
        expanded % restarts,
    )
    arrays["selected_inputs"] = selected[expanded].astype(np.int16)
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    metadata = json.loads(output.with_suffix(".json").read_text())
    metadata["rows"] = np.stack(
        [arrays["config_ids"], arrays["task_ids"], arrays["restart_ids"]], -1
    ).tolist()
    metadata["source_case_ids"] = (
        None if case_ids is None else np.asarray(case_ids).tolist()
    )
    metadata["active_model_parameters"] = (
        dimensions + 1
        if kind == "linear"
        else dimensions * plan["hidden"] + 2 * plan["hidden"] + 1
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    output.with_suffix(".done.json").write_text('{"complete": true}\n')
    return arrays

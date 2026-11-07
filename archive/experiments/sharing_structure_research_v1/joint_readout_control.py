"""G28: jointly fit C and a trainable readout with previously frozen I."""

import argparse
import itertools
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_indexator_v1.data import digest, load_archive, save_json
from experiments.darts_indexator_v1.model import balanced_loss, metrics

from .pattern_prefix import timestamp

BASE = Path(__file__).resolve().parent
PATTERN = BASE / "results/20261105/pattern_0110_prefix_g26"
SETS = BASE / "results/20261105/deepsets_prefix_g27"


def selected_index(previous, family=0):
    path = previous / "source" / f"case_{family}_0"
    selection = json.loads((path / "selection.json").read_text())
    number = selection["methods"]["single_fixed"]["candidate"]
    return load_archive(path / "predictors.npz")["indices"][number]


def head_groups(index):
    signatures = np.sort(index, axis=0).T
    return np.unique(signatures, axis=0, return_inverse=True)[1]


def basis(index, codes, device, dtype):
    index = torch.as_tensor(index, device=device, dtype=torch.long)
    return (
        F.one_hot(index.abs(), codes + 1)[..., 1:].to(dtype) * index.sign()[..., None]
    )


def feature(x, index, codes, device, dtype):
    return torch.einsum(
        "nd,dhk->nhk",
        torch.as_tensor(x, device=device, dtype=dtype),
        basis(index, codes, device, dtype),
    )


def teacher_matrix_output(x, weights, matrix, groups, offsets):
    activation = np.maximum(np.einsum("nd,tdh->tnh", x, weights) + 0.15, 0) - 0.15
    readout = matrix[:, groups] / 4
    return np.einsum("tnh,tho->tno", activation, readout) + offsets[:, None]


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G28 frozen-I joint C/W2 numerical control",
        "registered_utc": timestamp(),
        "seed": 26110628017,
        "scope": "Reuse frozen learned I. No new structure search. Learner receives no teacher C/W2. This does not validate extraction with unknown readout.",
        "pattern": {
            "seeds": 3,
            "starts": 32,
            "rates": [0.01, 0.03],
            "steps": 4096,
            "warmup": 64,
            "batch": 128,
            "checkpoint_every": 64,
            "l2": 0.001,
            "clip": 0.5,
            "heads": ["shared", "free"],
            "policy": "Same old 0110 support/query/test. Joint Adam on C, common hidden b, output b and readout. Identical paired C/-C branch budget for both heads. Select each checkpoint/start/rate on query BCE only. Free head has eight independent parameters. Head L2 is mean square, matching scalar-head scale.",
        },
        "sets": {
            "families": 4,
            "targets": 8,
            "starts": 64,
            "steps": 64,
            "checkpoint_every": 4,
            "ridge": 1e-6,
            "line_search": 12,
            "bias": 0.15,
            "codes": 4,
            "hidden": 16,
            "noise": [0.0, 0.2],
            "repeats": 64,
            "support": 4096,
            "validation": 1024,
            "test": 8192,
            "tasks": ["original_sum", "learned_pool_matrix"],
            "heads": ["fixed", "pooled", "free", "random_free"],
            "policy": "Simultaneous damped Gauss-Newton on C and readout, fresh signed starts. For each function, half the free-head starts begin with equal coefficients, half with independent random coefficients. Known b stays fixed. Observed f(0) anchors an unknown output offset. All predictors frozen before test. Pooled head groups inferred only from learned I column signatures. Fully free head has 16 independent coefficients/output; permutation invariance is audited, not assumed.",
            "matrix_task": "Two outputs with a new learned 2x2 matrix after pooling the two local feature channels, expanded across eight points. New Gaussian C and signed matrices. This teacher remains a set function; no unknown-readout structure extraction is performed.",
        },
        "resource_policy": "Only GPUs0-3, packed numerical starts; never6/7 or foreign4/5.",
        "software_tests": False,
    }
    (root / "code_snapshot").mkdir()
    source = Path(__file__)
    shutil.copy2(source, root / "code_snapshot" / source.name)
    plan["code_sha256"] = digest(source)
    plan["parents"] = {
        str(p / "source_frozen.json"): digest(p / "source_frozen.json")
        for p in (PATTERN, SETS)
    }
    save_json(root / "protocol.json", plan)
    pi = selected_index(PATTERN)
    train = load_archive(PATTERN / "data/binary_train.npz")
    rng = np.random.default_rng(plan["seed"])
    random_pi = rng.permutation(pi.flatten()).reshape(pi.shape)
    np.savez_compressed(
        root / "pattern_data.npz", index=pi, random_index=random_pi, **train
    )
    (root / "private").mkdir()
    for family in range(4):
        rng = np.random.default_rng(plan["seed"] + 100003 * family + 101)
        index = selected_index(SETS, family)
        random_index = rng.permutation(index.flatten()).reshape(index.shape)
        old = load_archive(SETS / "target" / f"data_{family}.npz")
        teacher = load_archive(SETS / "private" / f"teacher_{family}.npz")
        old_private = load_archive(SETS / "private" / f"target_{family}.npz")
        teacher_groups = head_groups(teacher["index"])
        if len(np.unique(head_groups(index))) != 2:
            raise ValueError("Expected two groups derived from learned I")
        codes = rng.normal(size=(8, 4))
        matrix = rng.uniform(0.4, 1.5, size=(8, 2, 2))
        matrix *= rng.choice((-1, 1), size=matrix.shape)
        offsets = rng.normal(0, 0.3, size=(8, 2))
        weights = np.sign(teacher["index"])[None] * np.take(
            np.c_[np.zeros(8), codes], np.abs(teacher["index"]), axis=1
        )

        data = {"index": index, "random_index": random_index}
        private = {
            "matrix_codes": codes,
            "matrix_readout": matrix,
            "original_sum_test_x": old_private["test_x"],
            "original_sum_test_y": old_private["test_y"][..., None],
            "permutation_x": old_private["permutation_x"],
        }
        for condition in range(2):
            data[f"original_sum_zero_{condition}"] = old[f"zero_y_{condition}"][
                ..., None
            ]
        for split in ("support", "validation"):
            data[f"original_sum_{split}_x"] = old[f"{split}_x"]
            for condition in range(2):
                data[f"original_sum_{split}_y_{condition}"] = old[
                    f"{split}_y_{condition}"
                ][..., None]
        zero_noise = rng.normal(size=(8, 2))
        for condition, sigma in enumerate(plan["sets"]["noise"]):
            data[f"learned_pool_matrix_zero_{condition}"] = (
                offsets + sigma * zero_noise / math.sqrt(4096 * 64)
            )
        for split, count in [("support", 4096), ("validation", 1024), ("test", 8192)]:
            x = rng.normal(size=(count, 16))
            y = teacher_matrix_output(x, weights, matrix, teacher_groups, offsets)
            if split == "test":
                private[f"learned_pool_matrix_{split}_x"] = x
                private[f"learned_pool_matrix_{split}_y"] = y
            else:
                data[f"learned_pool_matrix_{split}_x"] = x
                noise = rng.normal(size=y.shape)
                for condition, sigma in enumerate(plan["sets"]["noise"]):
                    data[f"learned_pool_matrix_{split}_y_{condition}"] = (
                        y + sigma * noise / 8
                    )
        np.savez_compressed(root / f"sets_data_{family}.npz", **data)
        np.savez_compressed(root / "private" / f"sets_{family}.npz", **private)
    save_json(
        root / "data_frozen.json",
        {
            "frozen_utc": timestamp(),
            "sha256": {p.name: digest(p) for p in root.glob("*_data*.npz")},
            "private_sha256": {
                p.name: digest(p) for p in (root / "private").glob("*.npz")
            },
        },
    )
    policy = json.loads((BASE / "GPU_POLICY.json").read_text())
    policy["active_continuation"] = str(root.relative_to(BASE) / "protocol.json")
    policy["current_execution"] = (
        "G28: frozen-I joint C/W2 control; only packed GPU0-3 numerical fitting. No structure-search generalization claim."
    )
    save_json(BASE / "GPU_POLICY.json", policy)
    print(json.dumps({"prepared": str(root), "utc": timestamp()}), flush=True)


def pattern_predict(feat, parameters, free):
    hidden = torch.einsum("nhk,bk->bnh", feat, parameters[:, :2])
    hidden = (hidden + parameters[:, 2, None, None]).relu()
    head = parameters[:, 3:-1]
    return (hidden * head[:, None]).sum(-1) + parameters[:, -1, None]


def pattern_step(sf, sy, parameters, optimizer, rates, generator, cfg, free):
    ids = torch.randint(
        len(sf), (cfg["batch"],), generator=generator, device=parameters.device
    )
    prediction = pattern_predict(sf[ids], parameters, free)
    loss = balanced_loss(prediction, sy[:, ids])
    penalty = parameters[:, :3].square().sum(1)
    penalty += parameters[:, 3:-1].square().mean(1)
    penalty += parameters[:, -1].square()
    optimizer.zero_grad(set_to_none=True)
    (loss + cfg["l2"] * penalty / 2).sum().backward()
    norm = parameters.grad.norm(dim=1, keepdim=True).clamp_min(1e-12)
    parameters.grad.mul_((cfg["clip"] / norm).clamp_max(1))
    before = parameters.detach().clone()
    optimizer.step()
    with torch.no_grad():
        parameters.copy_(before + rates * (parameters - before))


def pattern_fit(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    cfg = plan["pattern"]
    data = load_archive(root / "pattern_data.npz")
    rows, output = [], {}
    for kind in ("shared", "free", "random_free"):
        free = kind != "shared"
        index = data["random_index"] if kind == "random_free" else data["index"]
        sf = feature(data["support_x"], index, 2, device, torch.float32)
        vf = feature(data["query_x"], index, 2, device, torch.float32)
        sy = torch.tensor(data["support_y"], device=device)
        vy = torch.tensor(data["query_y"], device=device)
        for seed in range(cfg["seeds"]):
            started = time.monotonic()
            rng = np.random.default_rng(plan["seed"] + seed * 1009)
            initial_c = rng.normal(size=(cfg["starts"], 2))
            initial_c *= 0.5 / np.linalg.norm(initial_c, axis=1, keepdims=True)
            initial_c = np.tile(initial_c, (len(cfg["rates"]), 1))
            initial = np.c_[
                initial_c,
                np.zeros(len(initial_c)),
                np.ones((len(initial_c), 8 if free else 1)),
                np.zeros(len(initial_c)),
            ]
            count = len(initial)
            candidates = np.r_[initial, initial.copy()]
            candidates[count:, :2] *= -1
            parameters = torch.tensor(
                candidates, device=device, dtype=torch.float32, requires_grad=True
            )
            rates = torch.tensor(
                np.repeat(cfg["rates"], cfg["starts"]),
                device=device,
                dtype=torch.float32,
            ).repeat(2)[:, None]
            optimizer = torch.optim.Adam([parameters], lr=1.0)
            generator = torch.Generator(device=device).manual_seed(
                plan["seed"] + seed * 71 + 701
            )

            for _ in range(cfg["warmup"]):
                pattern_step(sf, sy, parameters, optimizer, rates, generator, cfg, free)
            with torch.no_grad():
                values = balanced_loss(pattern_predict(vf, parameters, free), vy)
                branch = values[count:] < values[:count]
                chosen = torch.arange(count, device=device) + branch * count
                old_state = optimizer.state[parameters]
                first = old_state["exp_avg"][chosen].clone()
                second = old_state["exp_avg_sq"][chosen].clone()
                parameters = parameters[chosen].detach().clone().requires_grad_()
            optimizer = torch.optim.Adam([parameters], lr=1.0)
            optimizer.state[parameters] = {
                "step": torch.tensor(float(cfg["warmup"])),
                "exp_avg": first,
                "exp_avg_sq": second,
            }
            rates = rates[:count]
            best = torch.full((count,), torch.inf, device=device)
            best_parameters = parameters.detach().clone()
            best_step = torch.zeros(count, dtype=torch.long, device=device)
            history, train, validation, recall = [], [], [], []
            remaining = cfg["steps"] - 2 * cfg["warmup"]
            for iteration in range(remaining + 1):
                if iteration % cfg["checkpoint_every"] == 0 or iteration == remaining:
                    with torch.no_grad():
                        tm = metrics(pattern_predict(sf, parameters, free), sy)
                        vm = metrics(pattern_predict(vf, parameters, free), vy)
                        improved = vm[:, 0] < best
                        best[improved] = vm[improved, 0]
                        best_parameters[improved] = parameters[improved]
                        best_step[improved] = iteration + 2 * cfg["warmup"]
                        history.append(iteration + 2 * cfg["warmup"])
                        train.append(tm[:, 0].cpu().numpy())
                        validation.append(vm[:, 0].cpu().numpy())
                        recall.append(vm[:, 2].cpu().numpy())
                if iteration != remaining:
                    pattern_step(
                        sf, sy, parameters, optimizer, rates, generator, cfg, free
                    )
            key = f"{kind}_{seed}"
            winner = int(best.argmin())
            output.update(
                {
                    key + "_parameters": best_parameters.detach().cpu().numpy(),
                    key + "_steps": np.array(history),
                    key + "_train": np.stack(train),
                    key + "_validation": np.stack(validation),
                    key + "_recall": np.stack(recall),
                    key + "_best_step": best_step.cpu().numpy(),
                }
            )
            rows.append(
                {
                    "key": key,
                    "head": kind,
                    "seed": seed,
                    "winner": winner,
                    "query_bce": float(best[winner]),
                    "seconds": time.monotonic() - started,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    np.savez_compressed(root / "pattern_fits.npz", **output)
    save_json(
        root / "pattern_frozen.json",
        {
            "frozen_utc": timestamp(),
            "rows": rows,
            "sha256": digest(root / "pattern_fits.npz"),
        },
    )


def reg_values(feat, parameters, groups, outputs, bias, fixed, jacobian=False):
    code = parameters[:, :4]
    activation = torch.einsum("nhk,bk->bnh", feat, code) + bias
    hidden = activation.relu() - max(bias, 0)
    if fixed:
        head = parameters.new_ones((len(parameters), 16, outputs)) / 4
    else:
        head = parameters[:, 4:].reshape(len(parameters), -1, outputs)[:, groups] / 4
    prediction = torch.einsum("bnh,bho->bno", hidden, head)
    if not jacobian:
        return prediction
    code_j = torch.einsum(
        "nhk,bnh,bho->bnok", feat, (activation > 0).to(feat.dtype), head
    )
    if fixed:
        return prediction, code_j.flatten(1, 2)
    membership = F.one_hot(groups, (parameters.shape[1] - 4) // outputs).to(feat.dtype)
    grouped = torch.einsum("bnh,hg->bng", hidden, membership) / 4
    identity = torch.eye(outputs, device=feat.device, dtype=feat.dtype)
    head_j = torch.einsum("bng,oq->bnogq", grouped, identity).flatten(-2)
    return prediction, torch.cat([code_j, head_j], -1).flatten(1, 2)


@torch.no_grad()
def regression_fit(feat, y, vfeat, vy, groups, fixed, initial, cfg):
    outputs = y.shape[-1]
    parameters = initial.clone()
    best_parameters = parameters.clone()
    best = parameters.new_full((len(parameters),), torch.inf)
    best_step = torch.zeros(len(parameters), device=parameters.device, dtype=torch.long)
    history, training, validation = [], [], []
    active = torch.ones(len(parameters), device=parameters.device, dtype=torch.bool)
    identity = torch.eye(
        parameters.shape[1], device=parameters.device, dtype=parameters.dtype
    )
    count = y.shape[1] * outputs
    for iteration in range(cfg["steps"] + 1):
        prediction, jac = reg_values(
            feat, parameters, groups, outputs, cfg["bias"], fixed, jacobian=True
        )
        residual = (y - prediction).flatten(1)
        train = residual.square().mean(1)
        if iteration % cfg["checkpoint_every"] == 0 or iteration == cfg["steps"]:
            val = (
                (
                    reg_values(vfeat, parameters, groups, outputs, cfg["bias"], fixed)
                    - vy
                )
                .square()
                .mean((1, 2))
            )
            improved = val < best
            best[improved] = val[improved]
            best_parameters[improved] = parameters[improved]
            best_step[improved] = iteration
            history.append(iteration)
            training.append(train.cpu().numpy())
            validation.append(val.cpu().numpy())
        if iteration == cfg["steps"]:
            break
        gram = jac.transpose(1, 2) @ jac / count
        cross = (jac.transpose(1, 2) @ residual[..., None]) / count
        scale = gram.diagonal(dim1=1, dim2=2).amax(1).clamp_min(1e-12)
        direction = torch.linalg.solve(
            gram + cfg["ridge"] * scale[:, None, None] * identity,
            cross,
        )[..., 0]
        accepted = ~active | (train < 1e-24)
        chosen = parameters.clone()
        for exponent in range(cfg["line_search"]):
            proposal = parameters + 2.0 ** (-exponent) * direction
            value = (
                (reg_values(feat, proposal, groups, outputs, cfg["bias"], fixed) - y)
                .square()
                .mean((1, 2))
            )
            take = (~accepted) & (proposal.abs().amax(1) <= 16) & (value < train)
            chosen[take] = proposal[take]
            accepted |= take
            if accepted.all():
                break
        active &= accepted & (train >= 1e-24)
        parameters = chosen
    return {
        "parameters": best_parameters.cpu().numpy(),
        "validation": best.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "steps": np.array(history),
        "train_history": np.stack(training),
        "validation_history": np.stack(validation),
    }


def sets_fit(root, family, device):
    plan = json.loads((root / "protocol.json").read_text())
    cfg = plan["sets"]
    data = load_archive(root / f"sets_data_{family}.npz")
    rows, arrays = [], {}
    for task in cfg["tasks"]:
        x = data[task + "_support_x"]
        vx = data[task + "_validation_x"]
        for kind in cfg["heads"]:
            if kind == "fixed" and task != "original_sum":
                continue
            started = time.monotonic()
            index = data["random_index"] if kind == "random_free" else data["index"]
            groups = head_groups(index) if kind == "pooled" else np.arange(16)
            groups = torch.tensor(groups, device=device, dtype=torch.long)
            feat = feature(x, index, 4, device, torch.float64)
            vfeat = feature(vx, index, 4, device, torch.float64)
            outputs = data[task + "_support_y_0"].shape[-1]
            n = cfg["starts"]
            rng = np.random.default_rng(plan["seed"] + family * 100003 + 1001)
            signs = np.tile(
                np.array(list(itertools.product((-1, 1), repeat=4))), (4, 1)
            )
            initial_c = signs[None] * np.exp(rng.normal(-0.3, 0.7, (8, n, 4)))
            if kind == "fixed":
                initial = initial_c
            else:
                head = np.ones((8, n, int(groups.max()) + 1, outputs))
                head[:, n // 2 :] = rng.normal(0, 1, head[:, n // 2 :].shape)
                initial = np.concatenate([initial_c, head.reshape(8, n, -1)], -1)
            initial = torch.tensor(
                np.tile(initial.reshape(8 * n, -1), (2, 1)),
                device=device,
                dtype=torch.float64,
            ).clamp(-16, 16)
            ys, vys = [], []
            for condition in range(2):
                anchor = data[f"{task}_zero_{condition}"]
                ys.append(
                    np.repeat(
                        data[f"{task}_support_y_{condition}"] - anchor[:, None],
                        n,
                        axis=0,
                    )
                )
                vys.append(
                    np.repeat(
                        data[f"{task}_validation_y_{condition}"] - anchor[:, None],
                        n,
                        axis=0,
                    )
                )
            y = torch.tensor(np.concatenate(ys), device=device, dtype=torch.float64)
            vy = torch.tensor(np.concatenate(vys), device=device, dtype=torch.float64)
            # Chunk model starts, while preserving the complete registered cohort.
            results = []
            for begin in range(0, len(initial), 512):
                end = min(begin + 512, len(initial))
                results.append(
                    regression_fit(
                        feat,
                        y[begin:end],
                        vfeat,
                        vy[begin:end],
                        groups,
                        kind == "fixed",
                        initial[begin:end],
                        cfg,
                    )
                )
            result = {}
            for key in results[0]:
                if key == "steps":
                    result[key] = results[0][key]
                else:
                    axis = 1 if key.endswith("history") else 0
                    result[key] = np.concatenate([r[key] for r in results], axis=axis)
            validation = result["validation"].reshape(2, 8, n)
            winners = validation.argmin(2)
            key = task + "__" + kind
            result.update(
                winners=winners, index=index, head_groups=groups.cpu().numpy()
            )
            arrays.update({key + "__" + k: v for k, v in result.items()})
            rows.append(
                {
                    "key": key,
                    "task": task,
                    "head": kind,
                    "parameters": initial.shape[-1],
                    "outputs": outputs,
                    "seconds": time.monotonic() - started,
                    "mean_validation": validation.min(2).mean(1).tolist(),
                }
            )
            print(json.dumps({"family": family, **rows[-1]}), flush=True)
            del y, vy, initial
    np.savez_compressed(root / f"sets_fits_{family}.npz", **arrays)
    save_json(
        root / f"sets_frozen_{family}.json",
        {
            "frozen_utc": timestamp(),
            "rows": rows,
            "sha256": digest(root / f"sets_fits_{family}.npz"),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "pattern", "sets"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.root = args.root.resolve()
    torch.set_num_threads(2)
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "pattern":
        pattern_fit(args.root, args.device)
    else:
        sets_fit(args.root, args.family, args.device)


if __name__ == "__main__":
    main()

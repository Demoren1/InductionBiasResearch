"""Source-only calibration and held-class evaluation of learned indexators."""

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from distractor_prior import MaskVAE, compact_assignment
from feature_prior import assignment
from nf_prior import NFVAE
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path("experiments/structure_search_v2")


def make_cases(dataset, out, spec, stage, seeds=4):
    splits = np.load(out / "source_splits.npz")
    if stage == "tune":
        pairs = spec["reserved_pairs"]
        pools = {c: splits[f"pool_{c}"][100:] for c in spec["source_classes"]}
        queries = {c: splits[f"query_pool_{c}"][128:] for c in spec["source_classes"]}
    else:
        pairs = list(itertools.combinations(spec["target_classes"], 2))
        pools, queries = {}, {}
        for c in spec["target_classes"]:
            ids = np.flatnonzero(dataset.targets.numpy() == c)
            rng = np.random.default_rng(spec["seed"] + c)
            ids = rng.permutation(ids)
            pools[c], queries[c] = ids[:500], ids[500:1000]
    support, query, rows = [], [], []
    for pair in pairs:
        for seed in range(seeds):
            rng = np.random.default_rng(281240 + seed)
            support.append(
                np.concatenate([rng.choice(pools[c], 16, replace=False) for c in pair])
            )
            query.append(
                np.concatenate(
                    [rng.choice(queries[c], 64, replace=False) for c in pair]
                )
            )
            rows.append({"pair": list(pair), "seed": seed})
    support, query = np.array(support), np.array(query)
    pixels = F.avg_pool2d(dataset.data[:, None].float() / 255, 2).flatten(1)
    x = pixels[support]
    xv = pixels[query]
    y = torch.tensor([[0.0] * 16 + [1.0] * 16]).expand(len(rows), -1)
    yv = torch.tensor([[0.0] * 64 + [1.0] * 64]).expand(len(rows), -1)
    return x, y, xv, yv, rows, support, query


def fit(x, y, xv, yv, method, options, out, spec, steps, restarts, device):
    cases, inputs, h, slots = (
        len(x),
        x.shape[-1],
        spec["hidden"],
        spec["active_features"],
    )
    base_count, count = cases * restarts, cases * restarts * len(options)
    x = x.to(device).repeat_interleave(restarts, 0).repeat(len(options), 1, 1)
    y = y.to(device).repeat_interleave(restarts, 0).repeat(len(options), 1)
    xv = xv.to(device).repeat_interleave(restarts, 0).repeat(len(options), 1, 1)
    yv = yv.to(device).repeat_interleave(restarts, 0).repeat(len(options), 1)
    dense = method in [
        "dense_adam",
        "dense_sgd",
        "pretrained_adam",
        "frozen_pretrained",
    ]
    k = inputs if dense else slots
    pool = None
    if method in ["component_random", "component_full"]:
        pool = np.load(spec["proposal_pool"])["pool"]
        if method == "component_full":
            k = len(pool)
    torch.manual_seed(281250)
    initial = [
        torch.randn(base_count, k, h, device=device) / np.sqrt(k),
        torch.zeros(base_count, h, device=device),
        torch.randn(base_count, h, device=device) / np.sqrt(h),
        torch.zeros(base_count, device=device),
    ]
    parameters = [
        p.repeat(len(options), *([1] * (p.ndim - 1))).requires_grad_() for p in initial
    ]
    learning_rates = torch.tensor(
        [p[0] for p in options], device=device
    ).repeat_interleave(base_count)
    decay = torch.tensor([p[1] for p in options], device=device).repeat_interleave(
        base_count
    )
    torch.manual_seed(281251)
    z = torch.randn(base_count, 16, device=device).repeat(len(options), 1)
    model = None
    if dense:
        selected = torch.arange(inputs, device=device)[None].expand(count, -1)
    elif method in ["frequency", "functional"]:
        bank = np.load(out / "good_bank.npz")
        training = bank["role"] == 0
        quality = np.maximum(0, np.log(2) - bank["quality"][training])
        # Equal source-task mass, not a vote dominated by the easiest pair.
        for task in np.unique(bank["task"]):
            group = bank["task"][training] == task
            quality[group] /= quality[group].sum()
        if method == "functional":
            values = bank["tokens"][training, :, 2]
        elif "mask" in bank:
            values = bank["mask"][training]
        else:
            values = np.zeros((int(training.sum()), inputs))
            np.put_along_axis(values, bank["indices"][training], 1, axis=1)
        probability = (values * quality[:, None]).sum(0)
        indices = np.sort(np.argsort(-probability, kind="stable")[:slots])
        selected = torch.tensor(indices, device=device)[None].expand(count, -1)
    elif method == "supervised_selector":
        source = np.load(out / "source_observations.npz")
        data = np.concatenate([source["x"], source["query_x"]], 1)
        labels = np.concatenate([source["y"], source["query_y"]], 1)
        covariance = (
            (data - data.mean(1, keepdims=True))
            * (labels - labels.mean(1, keepdims=True))[:, :, None]
        ).mean(1)
        variance = data.var(1).clip(1e-6)
        relevance = (
            (covariance.square() / variance).mean(0)
            if isinstance(covariance, torch.Tensor)
            else (covariance**2 / variance).mean(0)
        )
        indices = np.sort(np.argsort(-relevance)[:slots])
        selected = torch.tensor(indices, device=device)[None].expand(count, -1)
    elif method == "component_full":
        selected = torch.tensor(pool, device=device)[None].expand(count, -1)
    elif method == "component_random":
        pool = torch.tensor(pool, device=device)
        selected = torch.stack(
            [
                pool[torch.randperm(len(pool), device=device)[:slots]].sort().values
                for _ in range(count)
            ]
        )
    elif method == "random":
        selected = torch.stack(
            [
                torch.randperm(inputs, device=device)[:slots].sort().values
                for _ in range(count)
            ]
        )
    else:
        info = json.loads((out / "prior_complete.json").read_text())
        model = (
            MaskVAE(inputs)
            if spec.get("binary_prior")
            else NFVAE(info["tasks"], inputs, slots * inputs)
        ).to(device)
        file = "untrained_prior.pt" if method == "untrained_vae_z" else "prior.pt"
        model.load_state_dict(torch.load(out / file, weights_only=True))
        model.requires_grad_(False).eval()
        with torch.no_grad():
            logits = model.decoder(z)
            if not spec.get("binary_prior"):
                logits = logits.reshape(count, slots, inputs)
            selected = (
                compact_assignment(logits, slots)
                if spec.get("binary_prior")
                else assignment(logits)
            ).argmax(-1)
    if method in ["pretrained_adam", "frozen_pretrained"]:
        state = torch.load(
            out / "pretrained.pt", weights_only=True, map_location=device
        )
        with torch.no_grad():
            parameters[0].copy_(state["weight"][None])
            parameters[1].copy_(state["bias"][None])
        if method == "frozen_pretrained":
            parameters[0].requires_grad_(False)
            parameters[1].requires_grad_(False)
    optimizer_type = torch.optim.SGD if method == "dense_sgd" else torch.optim.Adam
    optimizer = optimizer_type(
        [p for p in parameters if p.requires_grad],
        lr=1.0,
        **({"momentum": 0.9} if method == "dense_sgd" else {}),
    )
    structure_optimizer = None
    if method in ["vae_z", "untrained_vae_z"]:
        z.requires_grad_()
        structure_optimizer = torch.optim.Adam([z], lr=0.03)
    if method == "direct_indices":
        logits = logits.detach().requires_grad_()
        structure_optimizer = torch.optim.Adam([logits], lr=0.03)
    best = [p.detach().clone() for p in parameters]
    best_loss = torch.full((count,), torch.inf, device=device)
    best_indices = selected.clone()
    best_z = z.detach().clone()
    best_step = torch.zeros(count, dtype=torch.long, device=device)
    history = []

    def prediction(data, indices, parameters, soft=None):
        if soft is None:
            chosen = data.gather(2, indices[:, None].expand(-1, data.shape[1], -1))
        else:
            chosen = torch.bmm(data, soft.transpose(1, 2))
        weight, bias, readout, offset = parameters
        hidden = (torch.bmm(chosen, weight) + bias[:, None]).relu()
        return (hidden * readout[:, None]).sum(-1) + offset[:, None]

    for step in range(steps + 1):
        if step % 32 == 0:
            with torch.no_grad():
                if method in ["vae_z", "untrained_vae_z"]:
                    scores = model.decoder(z)
                    selected = (
                        compact_assignment(scores, slots)
                        if spec.get("binary_prior")
                        else assignment(scores.reshape(count, slots, inputs))
                    ).argmax(-1)
                elif method == "direct_indices":
                    selected = (
                        compact_assignment(logits, slots)
                        if spec.get("binary_prior")
                        else assignment(logits)
                    ).argmax(-1)
                validation_logits = prediction(xv, selected, parameters)
                loss = F.binary_cross_entropy_with_logits(
                    validation_logits, yv, reduction="none"
                ).mean(1)
                better = loss < best_loss
                best_loss[better] = loss[better]
                for stored, current in zip(best, parameters):
                    stored[better] = current[better]
                best_indices[better] = selected[better]
                best_z[better] = z[better]
                best_step[better] = step
        if step % 128 == 0:
            history.append({"step": step, "query_bce": best_loss.cpu().tolist()})
        if step == steps:
            break
        soft = None
        structural_step = (
            structure_optimizer is not None
            and step >= spec.get("structure_start", 0)
            and step % spec.get("structure_every", 8) == 0
        )
        if structural_step:
            scores = model.decoder(z) if method != "direct_indices" else logits
            if not spec.get("binary_prior") and method != "direct_indices":
                scores = scores.reshape(count, slots, inputs)
            soft = (
                compact_assignment(scores, slots, straight_through=True)
                if spec.get("binary_prior")
                else assignment(scores, straight_through=True)
            )
            selected = soft.detach().argmax(-1)
        train_logits = prediction(x, selected, parameters, soft)
        loss = (
            F.binary_cross_entropy_with_logits(train_logits, y, reduction="none")
            .mean(1)
            .sum()
        )
        if structural_step and method != "direct_indices":
            loss = loss + 0.0001 * z.square().sum()
        optimizer.zero_grad(set_to_none=True)
        if structure_optimizer is not None:
            structure_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        before = [p.detach().clone() for p in parameters]
        optimizer.step()
        with torch.no_grad():
            for old, current in zip(before, parameters):
                if not current.requires_grad:
                    continue
                shape = [count] + [1] * (current.ndim - 1)
                current.copy_(
                    old
                    + (current - old) * learning_rates.reshape(shape)
                    - old * (learning_rates * decay).reshape(shape)
                )
        if structural_step:
            structure_optimizer.step()
    with torch.no_grad():
        logits = prediction(xv, best_indices, best)
        accuracy = ((logits >= 0) == yv.bool()).float().mean(1)
    return {
        "query_loss": best_loss.cpu().numpy(),
        "query_accuracy": accuracy.cpu().numpy(),
        "indices": best_indices.cpu().numpy(),
        "z": best_z.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        **{f"parameter_{i}": p.cpu().numpy() for i, p in enumerate(best)},
    }, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument("--stage", choices=["tune", "evaluate"], required=True)
    args = parser.parse_args()
    spec = json.loads((ROOT / "feature_protocol.json").read_text())
    out = ROOT / "runs" / ("feature_bank_" + args.dataset + "_v1")
    snapshot = out / ("fit_source_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir()
    for file in ["feature_fit.py", "feature_prior.py", "nf_prior.py"]:
        shutil.copy2(ROOT / file, snapshot / file)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    dataset = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset](
        ROOT / "datasets", train=True
    )
    if args.stage == "tune":
        x, y, xv, yv, cases, support, query = make_cases(dataset, out, spec, "tune")
        np.savez_compressed(out / "tuning_splits.npz", support=support, query=query)
        results = []
        for method in ["dense_adam", "frequency", "dense_sgd"]:
            if method == "dense_sgd":
                options = list(itertools.product([0.01, 0.03, 0.1, 0.3], [0.0, 0.001]))
            else:
                options = list(
                    itertools.product([0.001, 0.003, 0.01, 0.03], [0.0, 0.001, 0.01])
                )
            data, history = fit(
                x, y, xv, yv, method, options, out, spec, 1024, 2, device
            )
            losses = (
                data["query_loss"].reshape(len(options), len(cases), 2).min(-1).mean(-1)
            )
            for option, value in zip(options, losses):
                results.append(
                    {
                        "method": method,
                        "lr": option[0],
                        "wd": option[1],
                        "source_query_bce": float(value),
                    }
                )
            np.savez_compressed(out / ("tuning_" + method + ".npz"), **data)
            (out / ("tuning_" + method + "_history.json")).write_text(
                json.dumps(history)
            )
            print(
                args.dataset,
                method,
                "best",
                min(results, key=lambda r: r["source_query_bce"]),
                flush=True,
            )
            (out / "tuning.json").write_text(json.dumps(results, indent=2))
    else:
        raise NotImplementedError(
            "Held-out evaluation is implemented after source calibration"
        )
    (out / (args.stage + "_complete.json")).write_text(
        json.dumps({"source_only": True, "cases": cases})
    )


if __name__ == "__main__":
    main()

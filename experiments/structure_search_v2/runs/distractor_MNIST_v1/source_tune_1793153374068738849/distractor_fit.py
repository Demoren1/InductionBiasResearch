"""Fair source pretraining, source calibration, then unseen-class evaluation."""

import argparse
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from distractor_data import image_table, layout, observations
from feature_fit import fit
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST

ROOT = Path("experiments/structure_search_v2")


def cases(dataset, out, spec, final=False, seeds=4):
    pools = np.load(out / "source_splits.npz")
    pixels = image_table(dataset)
    permutation = layout(spec["seed"] + 1, spec["blocks"])
    if final:
        pairs = list(itertools.combinations(spec["target_classes"], 2))
        rng = np.random.default_rng(spec["seed"] + 300)
        ids = {
            c: rng.permutation(np.flatnonzero(dataset.targets.numpy() == c))[:1000]
            for c in spec["target_classes"]
        }
    else:
        pairs = spec["reserved_pairs"]
        ids = {c: pools[f"pool_{c}"] for c in spec["source_classes"]}
    xs, qs, ys, qys, records, supports, queries = [], [], [], [], [], [], []
    for pair in pairs:
        for seed in range(seeds):
            rng = np.random.default_rng(281450 + seed)
            support = np.concatenate(
                [rng.choice(ids[c][100:500], 16, replace=False) for c in pair]
            )
            query = np.concatenate(
                [rng.choice(ids[c][628:756], 64, replace=False) for c in pair]
            )
            x, obs_ids = observations(
                pixels, support, support, permutation, 281460 + seed
            )
            q, qids = observations(pixels, query, query, permutation, 281470 + seed)
            xs.append(x)
            qs.append(q)
            ys.append(torch.tensor([0.0] * 16 + [1.0] * 16))
            qys.append(torch.tensor([0.0] * 64 + [1.0] * 64))
            records.append({"pair": list(pair), "seed": seed})
            supports.append(obs_ids)
            queries.append(qids)
    return (
        torch.stack(xs),
        torch.stack(ys),
        torch.stack(qs),
        torch.stack(qys),
        records,
        np.array(supports),
        np.array(queries),
    )


def pretrain(dataset, out, spec, device):
    source = np.load(out / "source_observations.npz")
    pairs = np.array(json.loads((out / "bank_complete.json").read_text())["pairs"])
    inputs = np.concatenate([source["x"], source["query_x"]], 1)
    binary = np.concatenate([source["y"], source["query_y"]], 1)
    labels = np.where(binary > 0, pairs[:, 1, None], pairs[:, 0, None])
    x = torch.tensor(inputs.reshape(-1, inputs.shape[-1]), device=device)
    y = torch.tensor(labels.flatten(), device=device)
    # All source labels used by mask fitness are available to this gradient
    # baseline too. Checkpoint images are separate from both source pools.
    pixels = image_table(dataset)
    pools = np.load(out / "source_splits.npz")
    permutation = layout(spec["seed"] + 1, spec["blocks"])
    xv, yv = [], []
    for index, pair in enumerate([[0, 5], [2, 3], [1, 4]]):
        ids = np.concatenate([pools[f"pool_{c}"][628:756] for c in pair])
        q, _ = observations(pixels, ids, ids, permutation, 281480 + index)
        xv.append(q)
        yv.append(torch.tensor([pair[0]] * 128 + [pair[1]] * 128))
    xv, yv = torch.cat(xv).to(device), torch.cat(yv).to(device)
    options = list(itertools.product([0.001, 0.003, 0.01], [0.001, 0.01, 0.1]))
    m, d, h = len(options), inputs.shape[-1], spec["hidden"]
    torch.manual_seed(281490)
    parameters = [
        torch.randn(m, d, h, device=device) / np.sqrt(d),
        torch.zeros(m, h, device=device),
        torch.randn(m, h, 6, device=device) / np.sqrt(h),
        torch.zeros(m, 6, device=device),
    ]
    parameters = [p.requires_grad_() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1.0)
    rates = torch.tensor([option[0] for option in options], device=device)
    decay = torch.tensor([option[1] for option in options], device=device)
    best = [p.detach().clone() for p in parameters]
    best_loss = torch.full((m,), torch.inf, device=device)
    history = []
    for step in range(4097):
        if step % 128 == 0:
            with torch.no_grad():
                logits = (
                    torch.bmm(
                        (xv @ parameters[0] + parameters[1][:, None]).relu(),
                        parameters[2],
                    )
                    + parameters[3][:, None]
                )
                ce = F.cross_entropy(
                    logits.transpose(1, 2), yv[None].expand(m, -1), reduction="none"
                ).mean(1)
                better = ce < best_loss
                best_loss[better] = ce[better]
                for stored, current in zip(best, parameters):
                    stored[better] = current[better]
                history.append(
                    {"step": step, "source_validation_ce": ce.cpu().tolist()}
                )
                if step % 1024 == 0:
                    print("PRETRAIN", step, float(best_loss.min()), flush=True)
        if step == 4096:
            break
        ids = torch.randint(len(x), (128,), device=device)
        logits = (
            torch.bmm(
                (x[ids] @ parameters[0] + parameters[1][:, None]).relu(), parameters[2]
            )
            + parameters[3][:, None]
        )
        loss = (
            F.cross_entropy(
                logits.transpose(1, 2), y[ids][None].expand(m, -1), reduction="none"
            )
            .mean(1)
            .sum()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        old = [p.detach().clone() for p in parameters]
        optimizer.step()
        with torch.no_grad():
            for before, current in zip(old, parameters):
                shape = [m] + [1] * (current.ndim - 1)
                current.copy_(
                    before
                    + (current - before) * rates.reshape(shape)
                    - before * (rates * decay).reshape(shape)
                )
    selected = int(best_loss.argmin())
    torch.save(
        {
            "weight": best[0][selected],
            "bias": best[1][selected],
            "readout": best[2][selected],
            "offset": best[3][selected],
        },
        out / "pretrained.pt",
    )
    (out / "pretrain_history.json").write_text(json.dumps(history))
    (out / "pretrain_complete.json").write_text(
        json.dumps(
            {
                "same_source_observations": True,
                "source_train_rows": len(x),
                "source_validation_rows": len(xv),
                "lr": options[selected][0],
                "wd": options[selected][1],
                "validation_ce": float(best_loss.min()),
                "target_classes_used": False,
            },
            indent=2,
        )
    )


def evaluate_saved(data, test_pixels, test_labels, records, spec, device):
    restarts = 4
    test_accuracy, test_bce = [], []
    permutation = layout(spec["seed"] + 1, spec["blocks"])
    for case_id, case in enumerate(records):
        pair = case["pair"]
        main = np.concatenate([np.flatnonzero(test_labels.numpy() == c) for c in pair])
        observations_x, _ = observations(
            test_pixels, main, main, permutation, 281500 + case["seed"]
        )
        y = torch.tensor(
            (test_labels[main].numpy() == pair[1]).astype(float), device=device
        )
        ids = slice(case_id * restarts, (case_id + 1) * restarts)
        indices = torch.tensor(data["indices"][ids], device=device)
        w, b, a, c = [
            torch.tensor(data[f"parameter_{i}"][ids], device=device) for i in range(4)
        ]
        losses = torch.zeros(restarts, device=device)
        correct = torch.zeros(restarts, device=device)
        with torch.no_grad():
            for start in range(0, len(main), 256):
                batch = observations_x[start : start + 256].to(device)
                chosen = batch[:, indices].permute(1, 0, 2)
                logits = ((torch.bmm(chosen, w) + b[:, None]).relu() * a[:, None]).sum(
                    -1
                ) + c[:, None]
                labels = y[start : start + 256][None].expand_as(logits)
                losses += F.binary_cross_entropy_with_logits(
                    logits, labels, reduction="none"
                ).sum(1)
                correct += ((logits >= 0) == labels.bool()).sum(1)
        test_accuracy.extend((correct / len(main)).cpu().tolist())
        test_bce.extend((losses / len(main)).cpu().tolist())
    return np.array(test_accuracy), np.array(test_bce)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["MNIST", "FashionMNIST"], required=True)
    parser.add_argument(
        "--stage", choices=["pretrain", "tune", "evaluate"], required=True
    )
    args = parser.parse_args()
    spec = json.loads((ROOT / "distractor_protocol.json").read_text())
    spec.update(
        active_features=spec["slots"],
        binary_prior=True,
        structure_start=512,
        structure_every=16,
    )
    out = ROOT / "runs" / ("distractor_" + args.dataset + "_v1")
    snapshot = out / ("source_" + args.stage + "_" + str(time.time_ns()))
    snapshot.mkdir()
    for file in [
        "distractor_fit.py",
        "distractor_data.py",
        "feature_fit.py",
        "distractor_prior.py",
    ]:
        shutil.copy2(ROOT / file, snapshot / file)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    dataset_type = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[args.dataset]
    dataset = dataset_type(ROOT / "datasets", train=True)
    if args.stage == "pretrain":
        pretrain(dataset, out, spec, device)
    elif args.stage == "tune":
        x, y, xv, yv, records, support, query = cases(dataset, out, spec)
        np.savez_compressed(out / "tuning_splits.npz", support=support, query=query)
        results = []
        for method in [
            "dense_adam",
            "frequency",
            "dense_sgd",
            "pretrained_adam",
            "frozen_pretrained",
        ]:
            rates = (
                [0.01, 0.03, 0.1, 0.3]
                if method == "dense_sgd"
                else [0.001, 0.003, 0.01, 0.03]
            )
            options = list(itertools.product(rates, [0.0, 0.001, 0.01, 0.1, 1.0]))
            data, history = fit(
                x, y, xv, yv, method, options, out, spec, 1024, 2, device
            )
            ce = (
                data["query_loss"]
                .reshape(len(options), len(records), 2)
                .min(-1)
                .mean(-1)
            )
            for option, value in zip(options, ce):
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
                "TUNE",
                method,
                min(
                    (r for r in results if r["method"] == method),
                    key=lambda r: r["source_query_bce"],
                ),
                flush=True,
            )
            (out / "tuning.json").write_text(json.dumps(results, indent=2))
        # Prior and compact numeric calibration are frozen before unseen classes.
        (out / "evaluation_protocol.json").write_text(
            json.dumps(
                {
                    "source_classes": spec["source_classes"],
                    "target_classes": spec["target_classes"],
                    "steps": 1024,
                    "restarts": 4,
                    "split_seeds": 8,
                    "methods": [
                        "frequency",
                        "functional",
                        "supervised_selector",
                        "vae_sample",
                        "vae_z",
                        "untrained_vae_z",
                        "direct_indices",
                        "random",
                        "dense_adam",
                        "dense_sgd",
                        "pretrained_adam",
                        "frozen_pretrained",
                    ],
                    "calibrated_on_source_only": True,
                },
                indent=2,
            )
        )
    else:
        protocol = json.loads((out / "evaluation_protocol.json").read_text())
        tuning = json.loads((out / "tuning.json").read_text())
        x, y, xv, yv, records, support, query = cases(
            dataset, out, spec, final=True, seeds=protocol["split_seeds"]
        )
        np.savez_compressed(out / "target_splits.npz", support=support, query=query)
        (out / "target_cases.json").write_text(json.dumps(records))
        official_test = dataset_type(ROOT / "datasets", train=False)
        test_pixels = image_table(official_test)
        for method in protocol["methods"]:
            if (out / (method + ".npz")).exists():
                continue
            reference = (
                method
                if method
                in ["dense_adam", "dense_sgd", "pretrained_adam", "frozen_pretrained"]
                else "frequency"
            )
            option = min(
                (r for r in tuning if r["method"] == reference),
                key=lambda r: r["source_query_bce"],
            )
            data, history = fit(
                x,
                y,
                xv,
                yv,
                method,
                [(option["lr"], option["wd"])],
                out,
                spec,
                protocol["steps"],
                protocol["restarts"],
                device,
            )
            accuracy, ce = evaluate_saved(
                data, test_pixels, official_test.targets, records, spec, device
            )
            data.update(test_accuracy=accuracy, test_loss=ce)
            np.savez_compressed(out / (method + ".npz"), **data)
            (out / (method + "_history.json")).write_text(json.dumps(history))
            selected = data["query_loss"].reshape(-1, protocol["restarts"]).argmin(1)
            selected_acc = accuracy.reshape(-1, protocol["restarts"])[
                np.arange(len(records)), selected
            ]
            print(
                args.dataset,
                "RESULT",
                method,
                "selected_test_accuracy",
                float(selected_acc.mean()),
                flush=True,
            )
    (out / (args.stage + "_complete.json")).write_text(
        json.dumps({"stage": args.stage, "dataset": args.dataset})
    )


if __name__ == "__main__":
    main()

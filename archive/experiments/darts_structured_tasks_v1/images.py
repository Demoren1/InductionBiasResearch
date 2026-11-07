"""Blind sharing search for a compact DeepSets classifier on image point sets."""

import argparse
import itertools
import json
import math
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import MNIST, FashionMNIST

from .data import BASE, digest, save_json
from .model import BatchedOptimizer, clip_gradient, hard_indices


def points(images, grid):
    intensity = F.avg_pool2d(images[:, None].float() / 255, 28 // grid).flatten(1)
    u, v = torch.meshgrid(
        torch.linspace(-1, 1, grid), torch.linspace(-1, 1, grid), indexing="ij"
    )
    coordinates = torch.stack([u.flatten(), v.flatten()], -1)
    return torch.cat(
        [coordinates[None].expand(len(images), -1, -1), intensity[..., None] * 2 - 1],
        -1,
    )


def permute(x, generator):
    order = torch.rand(x.shape[:2], device=x.device, generator=generator).argsort(-1)
    return x.gather(1, order[..., None].expand(-1, -1, x.shape[-1])).flatten(1)


def oracle_indices(spec):
    count, width = spec["points"], spec["width"]
    first = np.zeros((3 * count, count * width), dtype=np.int16)
    for point in range(count):
        first[point * 3 : (point + 1) * 3, point * width : (point + 1) * width] = (
            np.arange(1, 3 * width + 1).reshape(3, width)
        )
    bias = np.tile(np.arange(1, width + 1), count).astype(np.int16)
    pooling = np.tile(np.eye(width, dtype=np.int16), (count, 1))
    return [first, bias[:, None], pooling]


def prepare(dataset, grid=4, width=8):
    output = BASE / "results/20261104" / ("images_" + dataset)
    output.mkdir(parents=True, exist_ok=False)
    count = grid**2
    spec = {
        "dataset": dataset,
        "grid": grid,
        "points": count,
        "width": width,
        "rho_width": 32,
        "inputs": count * 3,
        "hidden": count * width,
        "codes": [3 * width, width, 1],
        "classes": 10,
        "seed": 261104941,
        "support": 6000,
        "query": 1000,
        "selection": 1000,
        "batch": 128,
        "steps": 4000,
        "fit_steps": 3000,
        "confirmation_steps": 6000,
        "checkpoint_every": 250,
        "numeric_warmup": 128,
        "restarts": 2,
        "fit_restarts": 4,
        "pilot_seeds": [51, 52],
        "confirmation_seeds": [53, 54, 55],
        "minimum_temperature": 0.03,
        "late_entropy": 0.01,
        "zero_probability": 0.5,
        "numeric_l2": 0.0001,
        "gradient_clip": 10.0,
    }
    dataset_class = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[dataset]
    source = dataset_class(
        Path("experiments/structure_search_v2/datasets"), train=True, download=False
    )
    order = np.random.default_rng(spec["seed"]).permutation(len(source))
    arrays = {}
    offset = 0
    for split in ("support", "query", "selection"):
        ids = order[offset : offset + spec[split]]
        offset += spec[split]
        arrays[split + "_points"] = points(source.data[ids], grid).numpy()
        arrays[split + "_y"] = source.targets[ids].numpy()
        arrays[split + "_ids"] = ids
    np.savez_compressed(output / "source.npz", **arrays)
    save_json(output / "specification.json", spec)
    save_json(
        output / "protocol.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "official_test_opened": False,
            "rules": [
                "Full ten-class MNIST/FashionMNIST from the existing official training split; separate 6000/1000/1000 support/query/selection images, with independent random permutations of points during training. Official test is opened only after freezing.",
                "Compact reference: 4x4 image points, phi Linear(3,8)+ReLU, mean pooling, rho Linear(8,32)+ReLU+Linear(32,10). This differs in capacity and resolution from the historical 196-point, width64 pilot.",
                "Student is unrestricted flatten(48)->128 ReLU->8 linear->32 ReLU->10. Learn signed indexators for the first matrix, its biases and the pooling matrix, with 24/8/1 independent base codes. Ground-truth pointwise blocks, bias roles and pooling roles are never supplied to search.",
                "Every dictionary has fixed zero plus trainable C and exactly tied -C. Apply pair-mass-to-zero canonicalization before every argmax. Keep all learned P and categorical I.",
                "The final rho MLP is learned numerically, as is the readout in the Pattern experiment. It is not a generator or encoder of the dataset.",
                "Two numeric replicas jointly share P during search; query chooses numeric refit checkpoints. Mean selection CE chooses hard architecture checkpoints and hyperparameters; top2 configurations confirmed on three new search seeds.",
                "Compare an independently tuned prescribed DeepSets and an independently tuned unrestricted dense student, Adam and momentum SGD, with four starts. Log all numerical and architecture search costs; no equality of FLOPs or source costs is claimed.",
                "No test-based selection, software tests, commits, push or publication.",
            ],
        },
    )
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for name in ("images.py", "model.py", "data.py"):
        shutil.copy2(BASE / name, snapshot / name)
    print("PREPARED", output, flush=True)


def shapes(spec):
    return [
        (spec["inputs"], spec["hidden"]),
        (spec["hidden"], 1),
        (spec["hidden"], spec["width"]),
    ]


def parameter_count(spec):
    width, rho = spec["width"], spec["rho_width"]
    return sum(spec["codes"]) + width * rho + rho + rho * 10 + 10


def materialize(probabilities, numeric, spec, indices=None):
    matrices, offset = [], 0
    for number, codes in enumerate(spec["codes"]):
        values = numeric[..., offset : offset + codes]
        if indices is None:
            complete = torch.cat(
                [torch.zeros_like(values[..., :1]), values, -values], -1
            )
            matrix = torch.einsum("sijk,srk->srij", probabilities[number], complete)
        else:
            index = indices[number][:, None].expand(-1, numeric.shape[1], -1, -1)
            matrix = (
                F.pad(values, (1, 0))
                .gather(2, index.abs().flatten(2))
                .reshape(index.shape)
                * index.sign()
            )
        matrices.append(matrix)
        offset += codes
    return matrices


def prediction(x, matrices, numeric, spec):
    width, rho = spec["width"], spec["rho_width"]
    first, bias, pool = matrices
    hidden = (
        torch.einsum("nd,srdh->srnh", x, first) + bias[..., 0][..., None, :]
    ).relu()
    pooled = torch.einsum("srnh,srhk->srnk", hidden, pool) / math.sqrt(spec["hidden"])
    offset = sum(spec["codes"])
    head = numeric[..., offset : offset + width * rho].reshape(
        *numeric.shape[:2], width, rho
    )
    offset += width * rho
    head_bias = numeric[..., offset : offset + rho]
    offset += rho
    readout = numeric[..., offset : offset + rho * 10].reshape(
        *numeric.shape[:2], rho, 10
    )
    final = numeric[..., offset + rho * 10 :]
    activation = (
        torch.einsum("srnk,srkh->srnh", pooled, head) + head_bias[..., None, :]
    ).relu()
    return torch.einsum("srnh,srho->srno", activation, readout) + final[..., None, :]


def loss_per_model(logits, labels):
    expanded = labels[None, None].expand(logits.shape[:-1])
    return (
        F.cross_entropy(logits.flatten(0, -2), expanded.flatten(), reduction="none")
        .reshape(expanded.shape)
        .mean(-1)
    )


def initialize(rows, restarts, spec, device):
    arrays = []
    for row in rows:
        rng = np.random.default_rng(spec["seed"] + row["seed"] * 100003)
        per_restart = []
        for _ in range(restarts):
            parts = []
            for group, codes in enumerate(spec["codes"]):
                values = np.sort(np.abs(rng.normal(size=codes))) + 0.1
                values *= (0.5 if group == 0 else 0.1 if group == 1 else 1.0) / np.sqrt(
                    np.mean(values**2)
                )
                parts.append(values)
            width, rho = spec["width"], spec["rho_width"]
            parts.extend(
                [
                    rng.normal(size=width * rho) * math.sqrt(2 / width),
                    np.zeros(rho),
                    rng.normal(size=rho * 10) / math.sqrt(rho),
                    np.zeros(10),
                ]
            )
            per_restart.append(np.concatenate(parts))
        arrays.append(per_restart)
    return torch.tensor(np.array(arrays), device=device, dtype=torch.float32)


def read_source(output, device, spec):
    with np.load(output / "source.npz") as archive:
        data = {
            name: torch.as_tensor(archive[name], device=device)
            for name in archive.files
            if not name.endswith("_ids")
        }
    generator = torch.Generator(device=device).manual_seed(spec["seed"] + 120)
    for split in ("query", "selection"):
        data[split + "_x"] = permute(data[split + "_points"], generator)
    return data


@torch.no_grad()
def scores(x, y, matrices, numeric, spec):
    losses = torch.zeros(numeric.shape[:2], device=numeric.device)
    correct = torch.zeros_like(losses)
    for start in range(0, len(x), 256):
        labels = y[start : start + 256]
        logits = prediction(x[start : start + 256], matrices, numeric, spec)
        losses += loss_per_model(logits, labels) * len(labels)
        correct += (logits.argmax(-1) == labels).sum(-1)
    return torch.stack([losses / len(x), correct / len(x)], -1)


def optimize(
    output,
    spec,
    rows,
    device,
    steps,
    restarts,
    phase,
    indices=None,
    inherited=None,
    dense=False,
    architecture=False,
):
    data = read_source(output, device, spec)
    numeric = (
        initialize(rows, restarts, spec, device)
        if inherited is None
        else torch.tensor(inherited, device=device)
    ).requires_grad_()
    numeric_optimizer = BatchedOptimizer(numeric, rows)
    parameters, architecture_optimizers = [], []
    if architecture:
        for shape, codes in zip(shapes(spec), spec["codes"]):
            arrays = [
                np.random.default_rng(
                    spec["seed"] + row["seed"] * 100003 + 17 * codes
                ).normal(0, 1, (*shape, 2 * codes + 1))
                for row in rows
            ]
            alpha = torch.tensor(np.array(arrays), device=device, dtype=torch.float32)
            alpha[..., 0] += math.log(2 * codes)
            parameters.append(alpha.requires_grad_())
            architecture_optimizers.append(
                BatchedOptimizer(alpha, rows, "architecture_lr")
            )
    elif dense:
        for number, shape in enumerate(shapes(spec)):
            arrays = [
                np.random.default_rng(
                    spec["seed"] + row["seed"] * 100003 + 19 * number
                ).normal(size=(restarts, *shape))
                * (
                    math.sqrt(2 / spec["inputs"])
                    if number == 0
                    else 0.0
                    if number == 1
                    else 1.0
                )
                for row in rows
            ]
            parameter = torch.tensor(
                np.array(arrays), device=device, dtype=torch.float32
            ).requires_grad_()
            parameters.append(parameter)
            architecture_optimizers.append(BatchedOptimizer(parameter, rows))
    else:
        indices = [
            torch.as_tensor(value, device=device, dtype=torch.long) for value in indices
        ]
    generator = torch.Generator(device=device).manual_seed(spec["seed"] + 122)
    best = torch.full(numeric.shape[:2], torch.inf, device=device)
    best_numeric = numeric.detach().clone()
    best_matrices = [
        torch.zeros(len(rows), restarts, *shape, device=device)
        for shape in shapes(spec)
    ]
    best_step = torch.zeros_like(best, dtype=torch.long)
    histories, snapshots = [], []
    started = time.perf_counter()
    if architecture:
        with torch.no_grad():
            fixed_probabilities = [parameter.softmax(-1) for parameter in parameters]
        for _ in range(spec["numeric_warmup"]):
            ids = torch.randint(
                len(data["support_y"]),
                (spec["batch"],),
                generator=generator,
                device=device,
            )
            x = permute(data["support_points"][ids], generator)
            matrices = materialize(fixed_probabilities, numeric, spec)
            loss = loss_per_model(
                prediction(x, matrices, numeric, spec), data["support_y"][ids]
            )
            objective = loss.sum() + spec["numeric_l2"] * numeric.square().sum() / 2
            gradient = torch.autograd.grad(objective, numeric)[0]
            numeric_optimizer.step(
                clip_gradient(gradient, spec["gradient_clip"], (-1,))
            )
    for step in range(steps + 1):
        progress = step / max(steps, 1)
        temperature = spec["minimum_temperature"] ** min(
            1, max(0, (progress - 0.3) / 0.7)
        )
        probabilities = (
            [(parameter / temperature).softmax(-1) for parameter in parameters]
            if architecture
            else None
        )
        matrices = (
            parameters if dense else materialize(probabilities, numeric, spec, indices)
        )
        if step % spec["checkpoint_every"] == 0 or step == steps:
            with torch.no_grad():
                query = scores(
                    data["query_x"], data["query_y"], matrices, numeric, spec
                )
                improved = query[..., 0] < best
                best[improved] = query[..., 0][improved]
                best_numeric[improved] = numeric[improved]
                best_step[improved] = step
                for old, matrix in zip(best_matrices, matrices):
                    old[improved] = matrix[improved]
                training_x = permute(
                    data["support_points"][:256],
                    torch.Generator(device=device).manual_seed(spec["seed"] + 124),
                )
                training = scores(
                    training_x, data["support_y"][:256], matrices, numeric, spec
                )
                record = {
                    "step": step,
                    "train": training.cpu().tolist(),
                    "query": query.cpu().tolist(),
                    "seconds": time.perf_counter() - started,
                }
                histories.append(record)
                if architecture and step in (0, steps // 2, steps):
                    snapshots.append(
                        {
                            "step": step,
                            "numeric": numeric.detach().cpu().numpy(),
                            "probabilities": [
                                p.detach().cpu().numpy() for p in probabilities
                            ],
                        }
                    )
                if not torch.isfinite(query).all():
                    raise FloatingPointError("Non-finite image fit")
                if step % 1000 == 0 or step == steps:
                    print(
                        spec["dataset"],
                        phase,
                        "step",
                        step,
                        "queryCE",
                        round(float(query[..., 0].mean()), 4),
                        "accuracy",
                        round(float(query[..., 1].mean()), 4),
                        "seconds",
                        round(time.perf_counter() - started, 1),
                        flush=True,
                    )
        if step == steps:
            break
        ids = torch.randint(
            len(data["support_y"]), (spec["batch"],), generator=generator, device=device
        )
        x = permute(data["support_points"][ids], generator)
        logits = prediction(x, matrices, numeric, spec)
        loss = loss_per_model(logits, data["support_y"][ids])
        objective = loss.sum() + spec["numeric_l2"] * numeric.square().sum() / 2
        if dense:
            objective = (
                objective
                + spec["numeric_l2"] * sum(p.square().sum() for p in parameters) / 2
            )
        gradients = torch.autograd.grad(
            objective,
            [numeric, *parameters] if dense else [numeric],
            retain_graph=architecture,
        )
        if architecture:
            entropy = sum(
                -(p * p.clamp_min(1e-12).log()).sum(-1).flatten(1).mean(-1)
                for p in probabilities
            ) / len(probabilities)
            active = sum(
                (1 - p[..., 0]).flatten(1).mean(-1) for p in probabilities
            ) / len(probabilities)
            sparsity = numeric.new_tensor([row["sparsity"] for row in rows])
            architecture_loss = (
                loss.mean(-1).sum()
                + (
                    sparsity * active
                    + spec["late_entropy"] * max(0, 2 * progress - 1) * entropy
                ).sum()
            )
            architecture_gradients = torch.autograd.grad(architecture_loss, parameters)
        else:
            architecture_gradients = gradients[1:]
        numeric_optimizer.step(
            clip_gradient(gradients[0], spec["gradient_clip"], (-1,))
        )
        for parameter, optimizer, gradient in zip(
            parameters, architecture_optimizers, architecture_gradients
        ):
            dimensions = tuple(range(1 if architecture else 2, parameter.ndim))
            optimizer.step(clip_gradient(gradient, spec["gradient_clip"], dimensions))
            if architecture:
                with torch.no_grad():
                    parameter.sub_(parameter.mean(-1, keepdim=True))
    selection = scores(
        data["selection_x"], data["selection_y"], best_matrices, best_numeric, spec
    )
    result = {
        "numeric": best_numeric.cpu().numpy(),
        "query_ce": best.cpu().numpy(),
        "selection": selection.cpu().numpy(),
        "step": best_step.cpu().numpy(),
        "seconds": np.array(time.perf_counter() - started),
    }
    result.update(
        {f"matrix_{i}": matrix.cpu().numpy() for i, matrix in enumerate(best_matrices)}
    )
    destination = output / phase
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / "fits.npz", **result)
    save_json(destination / "rows.json", rows)
    save_json(destination / "history.json", histories)
    save_json(
        destination / "complete.json",
        {
            "steps": steps,
            "numeric_updates": numeric_optimizer.steps.flatten().cpu().tolist(),
            "architecture_updates": steps if architecture else 0,
            "rows": len(rows),
            "restarts": restarts,
            "batch": spec["batch"],
            "seconds": float(result["seconds"]),
        },
    )
    if snapshots:
        arrays = {
            "steps": np.array([s["step"] for s in snapshots]),
            "numeric": np.array([s["numeric"] for s in snapshots]),
        }
        for group in range(len(spec["codes"])):
            arrays[f"probability_{group}"] = np.stack(
                [s["probabilities"][group] for s in snapshots]
            )
        np.savez_compressed(destination / "snapshots.npz", **arrays)
    return result, snapshots


def search_phase(output, spec, phase, device):
    if phase == "pilot":
        configurations = [
            {
                "configuration": i,
                "numeric_lr": numeric,
                "architecture_lr": alpha,
                "sparsity": sparsity,
            }
            for i, (numeric, alpha, sparsity) in enumerate(
                itertools.product([0.001, 0.003], [0.003, 0.01, 0.03], [0.0, 0.003])
            )
        ]
    else:
        configurations = json.loads((output / "pilot_selection.json").read_text())[
            "top_configurations"
        ]
    rows = [
        {**row, "seed": seed}
        for row in configurations
        for seed in spec[phase + "_seeds"]
    ]
    steps = spec["steps"] if phase == "pilot" else spec["confirmation_steps"]
    _, snapshots = optimize(
        output,
        spec,
        rows,
        device,
        steps,
        spec["restarts"],
        phase + "_search",
        architecture=True,
    )
    fit_rows, indices, inherited = [], [[], [], []], []
    for snapshot, saved in enumerate(snapshots):
        proposals = [
            hard_indices(torch.tensor(p)).numpy() for p in saved["probabilities"]
        ]
        for number, row in enumerate(rows):
            fit_rows.append(
                {
                    **row,
                    "snapshot": snapshot,
                    "root": number,
                    "search_step": saved["step"],
                }
            )
            inherited.append(saved["numeric"][number])
            for group in range(3):
                indices[group].append(proposals[group][number])
    result, _ = optimize(
        output,
        spec,
        fit_rows,
        device,
        spec["fit_steps"],
        spec["restarts"],
        phase + "_hard",
        indices=[np.array(x) for x in indices],
        inherited=np.array(inherited),
    )
    np.savez_compressed(
        output / (phase + "_hard/indices.npz"),
        **{f"indices_{i}": np.array(x) for i, x in enumerate(indices)},
    )
    by_seed = {}
    for number, row in enumerate(fit_rows):
        key = (row["configuration"], row["seed"])
        value = float(result["selection"][number, :, 0].mean())
        if key not in by_seed or value < by_seed[key]["selection_ce"]:
            by_seed[key] = {**row, "row": number, "selection_ce": value}
    ranking = []
    for row in configurations:
        entries = [
            value for key, value in by_seed.items() if key[0] == row["configuration"]
        ]
        ranking.append(
            {
                "settings": row,
                "selection_ce": float(np.mean([x["selection_ce"] for x in entries])),
                "per_seed": entries,
            }
        )
    ranking.sort(key=lambda row: row["selection_ce"])
    save_json(
        output / (phase + "_selection.json"),
        {
            "ranking": ranking,
            "top_configurations": [x["settings"] for x in ranking[:2]],
            "primary": min(ranking[0]["per_seed"], key=lambda row: row["selection_ce"]),
        },
    )
    print(spec["dataset"], phase, "BEST", ranking[0], flush=True)


def controls(output, spec, device):
    rows = []
    for method, rates in (("adam", [0.001, 0.003, 0.01]), ("sgdm", [0.03, 0.1, 0.3])):
        for rate in rates:
            rows.append(
                {
                    "method": method,
                    "optimizer": "adam" if method == "adam" else "sgd",
                    "momentum": 0.0 if method == "adam" else 0.9,
                    "numeric_lr": rate,
                    "seed": 61,
                }
            )
    chosen = {}
    for variant in ("oracle", "dense"):
        indices = (
            [np.repeat(x[None], len(rows), axis=0) for x in oracle_indices(spec)]
            if variant == "oracle"
            else None
        )
        result, _ = optimize(
            output,
            spec,
            rows,
            device,
            spec["steps"],
            spec["fit_restarts"],
            variant,
            indices=indices,
            dense=variant == "dense",
        )
        chosen[variant] = {}
        for method in ("adam", "sgdm"):
            candidates = []
            for number, row in enumerate(rows):
                if row["method"] == method:
                    restart = int(result["query_ce"][number].argmin())
                    candidates.append(
                        {
                            "row": number,
                            "restart": restart,
                            "settings": row,
                            "selection_ce": float(
                                result["selection"][number, restart, 0]
                            ),
                        }
                    )
            chosen[variant][method] = min(
                candidates, key=lambda row: row["selection_ce"]
            )
    save_json(output / "control_selection.json", chosen)


def freeze(output, spec, device):
    selected = json.loads((output / "confirmation_selection.json").read_text())[
        "primary"
    ]
    with np.load(output / "confirmation_hard/indices.npz") as archive:
        indices = [
            archive[f"indices_{i}"][selected["row"] : selected["row"] + 1]
            for i in range(3)
        ]
    rows = [{**selected, "seed": 62}]
    result, _ = optimize(
        output, spec, rows, device, spec["fit_steps"], 8, "fresh", indices=indices
    )
    save_json(
        output / "fresh_selection.json",
        {"restart": int(result["query_ce"][0].argmin()), "settings": rows[0]},
    )
    files = [
        "source.npz",
        "specification.json",
        "protocol.json",
        "confirmation_selection.json",
        "confirmation_hard/fits.npz",
        "confirmation_hard/indices.npz",
        "control_selection.json",
        "oracle/fits.npz",
        "dense/fits.npz",
        "fresh/fits.npz",
        "fresh_selection.json",
    ]
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": {name: digest(output / name) for name in files},
            "official_test_opened": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MNIST", "FashionMNIST"), required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "controls", "pilot", "confirmation", "freeze"),
        required=True,
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "prepare":
        prepare(args.dataset)
        return
    output = BASE / "results/20261104" / ("images_" + args.dataset)
    spec = json.loads((output / "specification.json").read_text())
    device = torch.device("cuda:0")
    if args.stage == "controls":
        controls(output, spec, device)
    elif args.stage in ("pilot", "confirmation"):
        search_phase(output, spec, args.stage, device)
    else:
        freeze(output, spec, device)


if __name__ == "__main__":
    main()

"""Choose fresh numerical fits on development episodes and score fixed test fits."""

import argparse
import json
import math
from itertools import product

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.asrr_benchmark_families_v1.classification import (
    input_tensor,
    read_data,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.msr import (
    project,
    reparameterized_layout,
)
from experiments.real_group_transfer_v1.optimization import (
    Layout,
    losses,
    predict,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .common import CONFIG, OUT, SEED, freeze, register
from .panels import panel_directory

SCREEN_RATES = (0.003, 0.01)
SCREEN_STEPS = (75, 150)
SHORTLIST = 3
CFG = CONFIG["target"]


def layer_weights(parameter, layout, labels, signs, basis):
    if labels is not None:
        return parameter[..., labels] * signs
    if basis is not None:
        codes = parameter.unflatten(-1, (len(basis), basis.shape[2]))
        weights = torch.einsum("brgk,gpk->brgp", codes, basis)
        return (
            weights.reshape(
                *parameter.shape[:2],
                layout.sites,
                layout.channels,
                layout.fan_in,
            )
            .transpose(-1, -2)
            .flatten(2)
        )
    return parameter


def fit_batch(sx, sy, query, layout, structure, rate, steps, seed_blocks):
    labels = signs = basis = None
    size = layout.connections
    scale = math.sqrt(2 / layout.fan_in)
    if "factor" in structure:
        factor = structure["factor"]
        sx = project(sx, factor, layout)
        query = {name: project(x, factor, layout) for name, x in query.items()}
        layout = reparameterized_layout(layout, factor)
        size = layout.connections
        scale = math.sqrt(2 / layout.fan_in)
    elif "basis" in structure:
        basis = structure["basis"]
        size = basis.shape[0] * basis.shape[2]
        scale *= math.sqrt(layout.connections / size)
    if "index" in structure:
        index = structure["index"]
        size = int(index.abs().max())
        labels, signs = index.abs().sub(1).clamp_min(0), index.sign()
    weights, tail = [], []
    for count, seed in seed_blocks:
        weight_generator = torch.Generator(device=sx.device).manual_seed(seed)
        tail_generator = torch.Generator(device=sx.device).manual_seed(
            seed + 1
        )
        weights.append(
            torch.randn(
                count, 1, size, generator=weight_generator, device=sx.device
            )
        )
        tail.append(
            torch.randn(
                count,
                1,
                layout.hidden * (layout.outputs + 1) + layout.outputs,
                generator=tail_generator,
                device=sx.device,
            )
        )
    weights = torch.cat(weights).mul_(structure.get("initial_scale", scale))
    tail = torch.cat(tail)
    tail[..., : layout.hidden] *= 0.1
    tail[..., -layout.outputs :] = 0
    state = torch.cat((weights, tail), dim=-1).requires_grad_()
    optimizer = torch.optim.Adam(
        [state], lr=rate, fused=sx.is_cuda, capturable=sx.is_cuda
    )

    def update():
        optimizer.zero_grad(set_to_none=True)
        weight = layer_weights(state[..., :size], layout, labels, signs, basis)
        risk = losses(predict(sx, weight, state[..., size:], layout), sy)
        risk.mean(-1).sum().backward()
        optimizer.step()

    start, advance = 1, update
    if sx.is_cuda and min(steps) > 3:
        # All episodes in this batch use fixed inputs throughout the numerical fit.
        current = torch.cuda.current_stream(sx.device)
        stream = torch.cuda.Stream(device=sx.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                update()
        current.wait_stream(stream)
        optimizer.zero_grad(set_to_none=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            update()
        start, advance = 4, graph.replay
    snapshots = {}
    for step in range(start, max(steps) + 1):
        advance()
        if step in steps:
            with torch.no_grad():
                weight = layer_weights(
                    state[..., :size], layout, labels, signs, basis
                )
                snapshots[step] = {
                    name: predict(x, weight, state[..., size:], layout)[:, 0]
                    .cpu()
                    .numpy()
                    for name, x in query.items()
                }
    return snapshots


def metric(logits, truth):
    acc = (logits.argmax(-1) == truth).mean(-1)
    selected = np.take_along_axis(logits, truth[..., None], axis=-1)[..., 0]
    ce = (logsumexp(logits, axis=-1) - selected).mean(-1)
    return acc, ce


def model_on_device(path, device):
    if path is None:
        return {}
    data = read_data(path)
    return {
        name: torch.as_tensor(
            values,
            dtype=torch.long if name == "index" else torch.float32,
            device=device,
        )
        for name, values in data.items()
    }


def input_batches(
    dataset, phase, device, *, pack=1, include_clean=False, folder=None
):
    folder = panel_directory(dataset) / phase if folder is None else folder
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    shot = config["shot"]
    start = 0
    paths = sorted(folder.glob("inputs_*.npz"))
    for first in range(0, len(paths), pack):
        shards = paths[first : first + pack]
        panels = [read_data(path) for path in shards]
        parts = [
            (int(path.stem.split("_")[1]), len(panel["sx"]))
            for path, panel in zip(shards, panels, strict=True)
        ]
        data = {
            key: np.concatenate([panel[key] for panel in panels])
            for key in panels[0]
        }
        sx = data["sx"][:, :, :shot].reshape(
            -1, shot * layout.outputs, layout.sites, layout.fan_in
        )
        sy = data["sy"][:, :, :shot].reshape(len(sx), -1)
        queries = {"query": input_tensor(data["qx"], device)}
        if (phase == "test" or include_clean) and "qx_clean" in data:
            queries["clean"] = input_tensor(data["qx_clean"], device)
        yield (
            parts,
            slice(start, start + len(sx)),
            input_tensor(sx, device),
            torch.as_tensor(sy, dtype=torch.long, device=device),
            queries,
        )
        start += len(sx)


def validation(dataset, models, device, rates, steps, replicates):
    layout = Layout(*CONFIG["real"][dataset]["layout"])
    data = read_data(panel_directory(dataset) / "dev" / "answers.npz")
    values = {
        (str(path), rate, step): np.zeros(2)
        for path, rate, step in product(models, rates, steps)
    }
    specs = [(path, model_on_device(path, device)) for path in models]
    for parts, rows, sx, sy, queries in input_batches(
        dataset, "dev", device, pack=6 if device.type == "cuda" else 1
    ):
        labels = data["qy"][rows]
        for (path, structure), rate, rep in product(
            specs, rates, range(replicates)
        ):
            seeds = [
                (count, SEED + i * 1129 + rep * 10111) for i, count in parts
            ]
            predictions = fit_batch(
                sx, sy, queries, layout, structure, rate, steps, seeds
            )
            for step, logits in predictions.items():
                acc, ce = metric(logits["query"], labels)
                values[str(path), rate, step] += [acc.sum(), ce.sum()]
    norm = len(data["qy"]) * replicates
    return [
        {
            "model": name,
            "rate": rate,
            "steps": step,
            "accuracy": float(v[0] / norm),
            "ce": float(v[1] / norm),
        }
        for (name, rate, step), v in values.items()
    ]


def fit_order(row):
    return (
        -row["accuracy"],
        row["ce"],
        row["model"],
        row["steps"],
        row["rate"],
    )


def choose(dataset, method, models, device):
    screen = []
    if len(models) > SHORTLIST:
        screen = validation(
            dataset, models, device, SCREEN_RATES, SCREEN_STEPS, 1
        )
        ranked = list(
            dict.fromkeys(
                row["model"] for row in sorted(screen, key=fit_order)
            )
        )
        models = [path for path in models if str(path) in ranked[:SHORTLIST]]
    full = validation(
        dataset,
        models,
        device,
        CFG["rates"],
        CFG["steps"],
        CFG["seed_replicates"],
    )
    winner = min(full, key=fit_order)
    model_file = next(path for path in models if str(path) == winner["model"])
    out = OUT / dataset
    save_json(
        out / f"{method}_validation.json", {"screen": screen, "full": full}
    )
    winner["model_sha256"] = (
        digest(model_file) if model_file is not None else None
    )
    freeze(out / f"{method}_choice.json", winner)
    print(
        f"{dataset} {method} validation {winner['accuracy']:.4f}, "
        f"lr={winner['rate']} steps={winner['steps']}",
        flush=True,
    )
    return winner


def score_test(dataset, method, models, policy, device):
    folder = OUT / dataset
    output = folder / f"{method}_test.npz"
    if output.exists():
        return
    predictions_file = folder / f"{method}_logits.npz"
    seal_file = folder / f"{method}_seal.json"
    registration = {
        "choice_sha256": digest(folder / f"{method}_choice.json"),
        "models": {
            str(path): digest(path) for path in models if path is not None
        },
    }
    if seal_file.exists():
        freeze(
            seal_file,
            {**registration, "predictions_sha256": digest(predictions_file)},
        )
        logits = read_data(predictions_file)
    else:
        layout = Layout(*CONFIG["real"][dataset]["layout"])
        specs = [model_on_device(path, device) for path in models]
        rate, steps = policy["rate"], policy["steps"]
        predictions = {}
        for parts, _, sx, sy, query in input_batches(
            dataset, "test", device, pack=4 if device.type == "cuda" else 1
        ):
            all_logits = {name: [] for name in query}
            for structure, rep in product(
                specs, range(CFG["seed_replicates"])
            ):
                seeds = [
                    (count, SEED + i * 1129 + rep * 10111 + 8011)
                    for i, count in parts
                ]
                snapshot = fit_batch(
                    sx, sy, query, layout, structure, rate, [steps], seeds
                )[steps]
                for name, values in snapshot.items():
                    all_logits[name].append(values)
            for name, values in all_logits.items():
                predictions.setdefault(name, []).append(np.stack(values))
        logits = {
            name: np.concatenate(values, axis=1)
            for name, values in predictions.items()
        }
        if not all(np.isfinite(v).all() for v in logits.values()):
            raise ValueError("Non-finite target logits")
        atomic_npz(predictions_file, **logits)
        freeze(
            seal_file,
            {**registration, "predictions_sha256": digest(predictions_file)},
        )
    answers = read_data(panel_directory(dataset) / "test" / "answers.npz")
    result = {"units": answers["units"]}
    for name, values in logits.items():
        acc, ce = metric(
            values, np.broadcast_to(answers["qy"], values.shape[:-1])
        )
        result[f"{name}_acc"] = acc.mean(0)
        result[f"{name}_ce"] = ce.mean(0)
    atomic_npz(output, **result)
    print(
        dataset,
        method,
        {
            name: float(x.mean())
            for name, x in result.items()
            if name.endswith("_acc")
        },
        flush=True,
    )


def transfer(dataset, method, models, device):
    out = OUT / dataset
    if (out / f"{method}_test.npz").exists():
        return
    choice = out / f"{method}_choice.json"
    policy = (
        json.loads(choice.read_text())
        if choice.exists()
        else choose(dataset, method, models, device)
    )
    selected = next(path for path in models if str(path) == policy["model"])
    if selected is not None and digest(selected) != policy["model_sha256"]:
        raise ValueError(f"Changed selected model {selected}")
    score_test(dataset, method, [selected], policy, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG["real"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    register()
    torch.set_num_threads(1)
    transfer(args.dataset, "dense", [None], torch.device(args.device))

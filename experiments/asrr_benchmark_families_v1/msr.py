"""MSR first-layer adaptation on the frozen WISDM and Aug-Omniglot panels."""

import argparse
import json
import math
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.real_group_transfer_v1.optimization import losses, predict
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .classification import (
    DATASETS,
    SETTINGS,
    device_setup,
    input_tensor,
    label_tensor,
    model_layout,
    read_data,
    train,
)
from .groups import LOCAL, PLAN, ROOT
from .linear import atomic_npz, tolerance_interval

DESIGNS = {
    "wisdm": ((1, 9), (1, 16), (1, 32), (4, 16), (4, 32), (1, 64)),
    "aug_omniglot": ((1, 32), (4, 32), (1, 64), (4, 64), (4, 128), (1, 784)),
}
OUTER_STEPS = 1600
OUTER_RATE = 0.001
META_BATCH = 12
NATIVE_STEPS = (1, 3, 9)
MODES = ("basis", "initialization")


def project(x, basis, layout):
    sites = min(len(basis), layout.sites)
    blocks = len(basis) // sites
    factors = basis.unflatten(0, (sites, blocks)).expand(layout.sites, -1, -1, -1)
    return torch.einsum("bnsi,skir->bnskr", x, factors).flatten(2, 3)


def reparameterized_layout(layout, basis):
    blocks = len(basis) // min(layout.sites, len(basis))
    return replace(
        layout,
        sites=layout.sites * blocks,
        channels=layout.channels // blocks,
        fan_in=basis.shape[-1],
    )


def forward(x, parameter, output_factor, layout):
    local = parameter[..., : layout.connections].unflatten(
        -1, (layout.sites, layout.fan_in, layout.channels)
    )
    weight = torch.einsum("brsik,skh->brsih", local, output_factor).flatten(2)
    return predict(x, weight, parameter[..., layout.connections :], layout)


def predictions(sx, sy, queries, init, rates, factor, layout, snapshots, *, higher):
    count = [
        layout.connections,
        layout.hidden,
        layout.hidden * layout.outputs,
        layout.outputs,
    ]
    parameter = init.expand(len(sx), 1, -1)
    rate = rates.repeat_interleave(
        rates.new_tensor(count, dtype=torch.long), output_size=sum(count)
    )
    if not higher:
        parameter = parameter.detach().requires_grad_()
    saved = {}
    last = max(snapshots)
    for step in range(1, last + 1):
        error = losses(forward(sx, parameter, factor, layout), sy).mean(-1).sum()
        grad = torch.autograd.grad(error, parameter, create_graph=higher)[0]
        parameter = parameter - rate * grad
        if step in snapshots:
            with torch.set_grad_enabled(higher):
                saved[step] = {
                    name: forward(x, parameter, factor, layout)[:, 0]
                    for name, x in queries.items()
                }
                if not higher:
                    saved[step]["support"] = forward(sx, parameter, factor, layout)[
                        :, 0
                    ]
        if not higher and step < last:
            parameter = parameter.detach().requires_grad_()
    return saved


def source_panels(dataset, device, layout):
    data = read_data(LOCAL / dataset / "source.npz")
    sx, qx = [input_tensor(data[key], device) for key in ("sx", "qx")]
    sy, qy = [label_tensor(data[key], device) for key in ("sy", "qy")]
    order = sy.argsort(stable=True).unflatten(1, (layout.outputs, -1))
    sx = sx[torch.arange(len(sx), device=device)[:, None, None], order]
    clean_count = 5
    if dataset == "aug_omniglot":
        transformed = sx[:, :, clean_count:].flatten(1, 2)
        classes = torch.arange(layout.outputs, device=device)
        labels = classes.repeat_interleave(transformed.shape[1] // layout.outputs)
        qx = torch.cat((transformed, qx), dim=1)
        qy = torch.cat((labels.expand(len(qy), -1), qy), dim=1)
        sx = sx[:, :, :clean_count]
    return sx, qx, qy


def fit_source(dataset, blocks, rank, device):
    cfg, original = PLAN[dataset], model_layout(dataset)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 19001)
    samples = torch.Generator(device=device).manual_seed(cfg["seed"] + 29140)
    sx, qx, qy = source_panels(dataset, device, original)
    classes = torch.arange(original.outputs, device=device)
    if rank == original.fan_in:
        basis = torch.eye(rank, device=device).repeat(blocks, 1, 1)
    else:
        raw = torch.randn(
            blocks, original.fan_in, rank, device=device, generator=generator
        )
        basis = torch.linalg.qr(raw).Q
    layout = reparameterized_layout(original, basis)
    factor = torch.eye(layout.channels, device=device).repeat(layout.sites, 1, 1)
    size = layout.connections + layout.hidden * (1 + layout.outputs) + layout.outputs
    init = torch.randn(size, device=device, generator=generator)
    init[: layout.connections] *= math.sqrt(2 / layout.fan_in)
    init[layout.connections : layout.connections + layout.hidden] *= 0.1
    init[-layout.outputs :] = 0
    rates = torch.full((4,), 0.4, device=device)
    state = [v.requires_grad_() for v in (basis, factor, init, rates)]
    optimizer = torch.optim.Adam(state, lr=OUTER_RATE, fused=True)
    for step in range(OUTER_STEPS):
        shot = cfg["shots"][step % len(cfg["shots"])]
        tasks = torch.randint(len(sx), (META_BATCH,), device=device, generator=samples)
        order = torch.rand(
            (META_BATCH, layout.outputs, sx.shape[2]), device=device, generator=samples
        ).argsort()[..., :shot]
        x = sx[tasks[:, None, None], classes[None, :, None], order].flatten(1, 2)
        y = classes.repeat_interleave(shot).expand(META_BATCH, -1)
        query = {"query": project(qx[tasks], basis, original)}
        logits = predictions(
            project(x, basis, original),
            y,
            query,
            init,
            rates,
            factor,
            layout,
            (1,),
            higher=True,
        )[1]["query"]
        risk = losses(logits[:, None], qy[tasks]).mean()
        optimizer.zero_grad(set_to_none=True)
        risk.backward()
        optimizer.step()
        if (step + 1) % 400 == 0:
            print(
                dataset,
                blocks,
                rank,
                step + 1,
                round(float(risk.detach()), 4),
                flush=True,
            )
    with torch.no_grad():
        # A complete square output factor can be absorbed into fresh coefficients.
        if (torch.linalg.matrix_rank(factor) != layout.channels).any():
            raise ValueError("Noninvertible output factor")
        if not all(torch.isfinite(value).all() for value in state):
            raise ValueError("Nonfinite MSR source model")
        q = torch.linalg.qr(basis).Q
    return dict(
        zip(
            ("basis", "output", "init", "rates", "orthogonal"),
            [value.detach().cpu().numpy() for value in (*state, q)],
            strict=True,
        )
    )


def panel_batch(path, shot, device, layout, *, clean_dev=False):
    data = read_data(path)
    sx = input_tensor(data["sx"][:, :, :shot], device).reshape(
        -1, shot * layout.outputs, layout.sites, layout.fan_in
    )
    sy = label_tensor(data["sy"][:, :, :shot], device).reshape(len(sx), -1)
    queries = {"query": input_tensor(data["qx"], device)}
    if "qx_clean" in data and (clean_dev or path.parent.name == "test"):
        queries["clean"] = input_tensor(data["qx_clean"], device)
    return sx, sy, queries


def evaluations(mode, batch, model, original, steps, rate, seed, *, dev):
    sx, sy, queries = batch
    basis = model["orthogonal" if mode == "basis" else "basis"]
    layout = reparameterized_layout(original, basis)
    px = project(sx, basis, original)
    qx = {key: project(x, basis, original) for key, x in queries.items()}
    if mode == "basis":
        return train(px, sy, qx, None, layout, rate, steps, seed, dev)
    fitted = predictions(
        px,
        sy,
        qx,
        model["init"],
        model["rates"],
        model["output"],
        layout,
        steps,
        higher=False,
    )
    values = {
        k: {name: x.cpu().numpy() for name, x in rows.items()}
        for k, rows in fitted.items()
    }
    if dev:
        labels = sy.cpu().numpy()
        for row in values.values():
            row["support"] = metrics(row["support"], labels)
    else:
        for row in values.values():
            row.pop("support")
    return values


def metrics(logits, labels):
    ce = (
        logsumexp(logits, axis=-1)
        - np.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
    )
    return np.column_stack((ce.mean(1), (logits.argmax(-1) == labels).mean(1)))


def panel_seed(dataset, path, *, target):
    offset = int(path.stem.rsplit("_", 1)[1])
    return PLAN[dataset]["seed"] + 121991 + 1129 * offset + (701 if target else 0)


def train_candidate(job):
    dataset, blocks, rank, gpu = job
    name = f"b{blocks}_r{rank}"
    out, work = ROOT / dataset / "msr", LOCAL / dataset / "dev"
    device, original = device_setup(gpu), model_layout(dataset)
    record = out / f"{name}_dev.json"
    if record.exists():
        return
    model_file = out / f"{name}.npz"
    if not model_file.exists():
        atomic_npz(model_file, **fit_source(dataset, blocks, rank, device))
    model = {key: input_tensor(v, device) for key, v in read_data(model_file).items()}
    labels = read_data(work / "answers.npz")["qy"]
    results = {}
    for shot in PLAN[dataset]["shots"]:
        grid = (
            *(("basis", rate, (0, *SETTINGS["steps"])) for rate in SETTINGS["rates"]),
            ("initialization", None, NATIVE_STEPS),
        )
        scores = {
            (mode, rate, step): [] for mode, rate, steps in grid for step in steps
        }
        for path in sorted(work.glob("inputs_*.npz")):
            batch = panel_batch(path, shot, device, original)
            offset = int(path.stem.rsplit("_", 1)[1])
            query_y = labels[offset : offset + len(batch[0])]
            for mode, rate, steps in grid:
                rows = evaluations(
                    mode,
                    batch,
                    model,
                    original,
                    steps,
                    rate,
                    panel_seed(dataset, path, target=False),
                    dev=True,
                )
                for step, values in rows.items():
                    scores[mode, rate, step].append(
                        np.c_[metrics(values["query"], query_y), values["support"]]
                    )
        results[shot] = [
            {
                "mode": mode,
                "name": name,
                "rate": rate,
                "step": step,
                "codes": rank * original.hidden,
                "dev_metrics": np.concatenate(values).mean(0).tolist(),
            }
            for (mode, rate, step), values in scores.items()
            if step
        ]
    save_json(record, results)
    print(dataset, name, "development scored", flush=True)


def select(dataset):
    out = ROOT / dataset / "msr"
    choices = {}
    for shot in map(str, PLAN[dataset]["shots"]):
        rows = [
            row
            for blocks, rank in DESIGNS[dataset]
            for row in json.loads((out / f"b{blocks}_r{rank}_dev.json").read_text())[
                shot
            ]
        ]
        choices[shot] = {
            mode: min(
                (row for row in rows if row["mode"] == mode),
                key=lambda row: (-row["dev_metrics"][1], row["dev_metrics"][0]),
            )
            for mode in MODES
        }
    save_json(out / "choices.json", choices)
    print(dataset, "frozen choices", choices, flush=True)
    return choices


def test_shard(job):
    dataset, gpu, paths, choices = job
    out, work = ROOT / dataset / "msr", LOCAL / dataset / "msr_predictions"
    device, original = device_setup(gpu), model_layout(dataset)
    names = {row["name"] for rows in choices.values() for row in rows.values()}
    models = {
        name: {
            k: input_tensor(v, device)
            for k, v in read_data(out / f"{name}.npz").items()
        }
        for name in names
    }
    for path in paths:
        shard = work / path.name
        if shard.exists():
            continue
        scores = {}
        for shot, rows in choices.items():
            batch = panel_batch(path, int(shot), device, original)
            for mode, row in rows.items():
                step = row["step"]
                logits = evaluations(
                    mode,
                    batch,
                    models[row["name"]],
                    original,
                    (step,),
                    row["rate"],
                    panel_seed(dataset, path, target=True),
                    dev=False,
                )[step]
                scores.update(
                    {f"{shot}_{kind}_{mode}": a for kind, a in logits.items()}
                )
        atomic_npz(shard, **scores)


def summarize(dataset, shard_hashes):
    root, work = ROOT / dataset, LOCAL / dataset
    truth = read_data(work / "test" / "answers.npz")
    base = read_data(root / "per_episode.npz")
    if not np.array_equal(base["units"], truth["units"]):
        raise ValueError("MSR episodes differ from the original comparison")
    shards = []
    for path, expected in shard_hashes.items():
        if digest(path) != expected:
            raise ValueError(f"Changed MSR prediction shard {path}")
        shards.append(read_data(path))
    labels, units = truth["qy"], truth["units"]
    report, per_episode = {}, {}
    for key in shards[0]:
        shot, kind, mode = key.split("_")
        rows = metrics(np.concatenate([shard[key] for shard in shards]), labels)
        per_episode[key] = rows
        differences = {}
        for control in ("dense", "partial", "ward"):
            delta = 100 * (rows[:, 1] - base[f"s{shot}_{kind}_{control}"][:, 1])
            if dataset == "wisdm":
                delta = np.array(
                    [delta[units == user].mean() for user in np.unique(units)]
                )
            differences[control] = tolerance_interval(delta)
        report.setdefault(shot, {}).setdefault(kind, {})[mode] = {
            "metrics": rows.mean(0).tolist(),
            "delta_pp_95_ci": differences,
        }
    save_json(root / "msr" / "results.json", report)
    atomic_npz(root / "msr" / "per_episode.npz", units=units, **per_episode)
    print(dataset, "MSR results", json.dumps(report), flush=True)


def evaluate_test(dataset, gpus, pool):
    out, work = ROOT / dataset, LOCAL / dataset
    choices = select(dataset)
    paths = sorted((work / "test").glob("inputs_*.npz"))
    hashes = {p.name: digest(p) for p in paths}
    original = json.loads((out / "test_prediction_plan.json").read_text())
    if hashes != original["inputs_sha256"]:
        raise ValueError(f"Changed common test data for {dataset}")
    save_json(
        out / "msr" / "test_prediction_plan.json",
        {
            "models": {
                name: digest(out / "msr" / f"{name}.npz")
                for name in {
                    row["name"] for rows in choices.values() for row in rows.values()
                }
            },
            "choices": choices,
            "inputs_sha256": hashes,
        },
    )
    target = work / "msr_predictions"
    target.mkdir(exist_ok=True)
    pool.map(
        test_shard,
        [(dataset, gpu, paths[i :: len(gpus)], choices) for i, gpu in enumerate(gpus)],
    )
    sealed = {target / path.name: digest(target / path.name) for path in paths}
    save_json(out / "msr" / "sealed.json", {p.name: sha for p, sha in sealed.items()})
    summarize(dataset, sealed)


def run(datasets, gpus):
    for dataset in datasets:
        root = ROOT / dataset / "msr"
        root.mkdir(exist_ok=True)
        registration = {
            "source_sha256": digest(LOCAL / dataset / "source.npz"),
            "code_sha256": digest(Path(__file__)),
            "designs": DESIGNS[dataset],
            "source_steps": OUTER_STEPS,
            "outer_rate": OUTER_RATE,
            "meta_batch": META_BATCH,
            "source": "One higher-order SGD update, alternating support budgets.",
            "adaptation": "First-layer Kronecker factors on the matched classifier. "
            "Basis-only uses orthogonalized input factors and fresh Adam; "
            "initialization also transfers all numerical starts and SGD rates.",
            "data": "Existing fixed panels. Aug-Omniglot source adaptation uses "
            "the clean drawings and outer updates use all the transformed drawings.",
            "selection": "Same development episodes, maximize accuracy then minimize NLL.",
            "native_steps": NATIVE_STEPS,
            "fresh_grid": SETTINGS,
        }
        save_json(root / "registration.json", registration)
    jobs = [(d, b, r) for d in datasets for b, r in DESIGNS[d]]
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        for start in range(0, len(jobs), len(gpus)):
            wave = jobs[start : start + len(gpus)]
            pool.map(train_candidate, [(*row, gpu) for row, gpu in zip(wave, gpus)])
        for dataset in datasets:
            evaluate_test(dataset, gpus, pool)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", choices=DATASETS, nargs="+", default=DATASETS)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    options = parser.parse_args()
    run(options.datasets, options.gpus)

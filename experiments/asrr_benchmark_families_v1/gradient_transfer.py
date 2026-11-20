"""Validate gradient partitions on the frozen WISDM and Aug-Omniglot panels."""

import argparse
import json
import math
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

from experiments.real_group_transfer_v1.optimization import losses, predict
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .classification import (
    DATASETS,
    compact_labels,
    device_setup,
    grid,
    input_tensor,
    key_for,
    model_layout,
    pick,
    read_data,
    train,
)
from .gradient_groups import (
    gradient_kernel,
    gradient_refine,
    gradient_tree,
    group_score,
    orthogonal_groups,
)
from .gradient_synthetic import GRADIENT_CONFIG, GRADIENT_PLAN, OUT
from .groups import LOCAL, PLAN, ROOT, input_bank
from .linear import atomic_npz, tolerance_interval
from .msr import (
    metrics,
    panel_batch,
    panel_seed,
    project,
    reparameterized_layout,
    source_panels,
)

CONFIG = GRADIENT_CONFIG["classification"]
LOCAL_OUT = LOCAL / "gradient_asrr"


def ward_index(dataset):
    old = read_data(ROOT / dataset / "bank.npz")
    if dataset == "aug_omniglot":
        return old["ward_input4_32"]
    cfg = CONFIG[dataset]
    profiles = read_data(LOCAL / dataset / "profiles.npz")["profiles"]
    return compact_labels(
        input_bank(profiles, cfg["blocks"], [cfg["rank"]], partial=False)[cfg["rank"]]
    )


def grouping(index, blocks):
    width = index.shape[1] // blocks
    return np.array(
        [
            np.unique(index[:, h], return_inverse=True)[1]
            for h in range(0, index.shape[1], width)
        ]
    )


def basis_for(groups):
    return np.stack([orthogonal_groups(labels) for labels in groups]).astype(np.float32)


def initial_tail(tasks, layout, device, generator):
    tail = torch.randn(
        tasks,
        1,
        layout.hidden * (layout.outputs + 1) + layout.outputs,
        device=device,
        generator=generator,
    )
    tail[..., : layout.hidden] *= 0.1
    tail[..., -layout.outputs :] = 0
    return tail


def reference(support, sy, original, basis, tail, generator):
    x = project(support, basis, original)
    layout = reparameterized_layout(original, basis)
    coefficients = torch.randn(
        len(x),
        1,
        layout.connections,
        device=x.device,
        generator=generator,
    ) * math.sqrt(2 / layout.fan_in)
    parameter = torch.cat((coefficients, tail), dim=-1).requires_grad_()
    optimizer = torch.optim.Adam(
        [parameter],
        lr=CONFIG["source_adaptation_rate"],
        fused=True,
    )
    for _ in range(CONFIG["source_adaptation_steps"]):
        weight, tail = parameter.split((layout.connections, tail.shape[-1]), dim=-1)
        objective = losses(predict(x, weight, tail, layout), sy).mean(-1).sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    with torch.no_grad():
        weight, tail = parameter.split((layout.connections, tail.shape[-1]), dim=-1)
        sites = min(len(basis), original.sites)
        blocks = len(basis) // sites
        shared = basis.unflatten(0, (sites, blocks)).expand(original.sites, -1, -1, -1)
        local = weight.unflatten(
            -1, (original.sites, blocks, layout.fan_in, layout.channels)
        )
        dense = torch.einsum("sbdr,tnsbrh->tnsdbh", shared, local).flatten(2)
    return dense, tail.detach()


def per_input_gradient(x, y, weight, tail, layout):
    weight = weight.detach().requires_grad_()
    objective = losses(predict(x, weight, tail, layout), y).mean(-1).sum()
    grad = torch.autograd.grad(objective, weight)[0]
    return (
        grad.reshape(len(grad), layout.sites, layout.fan_in, layout.channels)
        .transpose(1, 2)
        .reshape(len(grad), layout.fan_in, layout.hidden)
    )


def source_kernels(job, *, clean_queries=None):
    dataset, shot, gpu, groups = job
    device, layout = device_setup(gpu), model_layout(dataset)
    ordered, qx, qy = source_panels(dataset, device, layout)
    seed = GRADIENT_CONFIG["seed"] + PLAN[dataset]["seed"]
    support_rng = torch.Generator(device=device).manual_seed(seed)
    starts_rng = torch.Generator(device=device).manual_seed(seed + 11047)
    basis = input_tensor(basis_for(groups), device)
    shape = (len(basis), layout.fan_in, layout.fan_in)
    adapted_queries = {"adapted": qx}
    if clean_queries is not None:
        adapted_queries["clean"] = input_tensor(clean_queries, device)
    kernels = {
        name: torch.zeros(shape, device=device) for name in ("start", *adapted_queries)
    }
    classes = torch.arange(layout.outputs, device=device)
    ids = torch.arange(len(ordered), device=device)
    sy = classes.repeat_interleave(shot).expand(len(ordered), -1)
    for _ in range(CONFIG["source_draws"]):
        positions = torch.rand(ordered.shape[:3], device=device, generator=support_rng)
        positions = positions.argsort()[..., :shot]
        sx = ordered[ids[:, None, None], classes[None, :, None], positions].flatten(
            1, 2
        )
        tail = initial_tail(len(sx), layout, device, starts_rng)
        common = torch.randn(
            len(sx),
            1,
            layout.sites,
            1,
            layout.channels,
            device=device,
            generator=starts_rng,
        ) * (math.sqrt(2) / layout.fan_in)
        common = common.expand(-1, -1, -1, layout.fan_in, -1).flatten(2)
        adapted = reference(sx, sy, layout, basis, tail, starts_rng)
        width = layout.hidden // len(basis)
        for (weight, other), queries in (
            ((common, tail), {"start": qx}),
            (adapted, adapted_queries),
        ):
            support_grad = per_input_gradient(sx, sy, weight, other, layout)
            for name, query_x in queries.items():
                query_grad = per_input_gradient(query_x, qy, weight, other, layout)
                paired = zip(
                    support_grad.split(width, dim=-1),
                    query_grad.split(width, dim=-1),
                    strict=True,
                )
                for block, (s, q) in enumerate(paired):
                    support = s.transpose(0, 1).reshape(layout.fan_in, -1)
                    query = q.transpose(0, 1).reshape(layout.fan_in, -1)
                    kernels[name][block] += gradient_kernel(support, query)
    return {
        name: a.cpu().numpy().astype(np.float64) / CONFIG["source_draws"]
        for name, a in kernels.items()
    }


def make_bank(dataset, pool, gpus):
    folder = OUT / dataset
    folder.mkdir(exist_ok=True, parents=True)
    if (folder / "bank.npz").exists():
        return
    count = CONFIG[dataset]["blocks"]
    rank = CONFIG[dataset]["rank"]
    index = ward_index(dataset)
    groups = grouping(index, count)
    arrays = {"ward": index, "normalized": basis_for(groups)}
    if dataset == "wisdm":
        thresholds = read_data(ROOT / dataset / "bank.npz")["partial_input1_0.65"]
        arrays["normalized_threshold"] = basis_for(grouping(thresholds, count))
    shots = PLAN[dataset]["shots"]
    kernels = pool.map(
        source_kernels,
        [(dataset, shot, gpus[i % len(gpus)], groups) for i, shot in enumerate(shots)],
        chunksize=1,
    )
    maps, scores = {"ward": groups}, {}
    for shot, rows in zip(shots, kernels, strict=True):
        for signal, kernel in rows.items():
            for search in ("tree", "refine"):
                name = f"s{shot}_{signal}_{search}"
                initial = (
                    np.stack([gradient_tree(k, rank) for k in kernel])
                    if search == "tree"
                    else groups
                )
                labels = np.stack(
                    [
                        gradient_refine(k, g)
                        for k, g in zip(kernel, initial, strict=True)
                    ]
                )
                maps[name] = labels
                arrays[name] = basis_for(labels)
                scores[name] = [
                    {
                        "score": group_score(k, new),
                        "ward_score": group_score(k, old),
                        "ari_to_ward": adjusted_rand_score(old, new),
                        "sizes": np.bincount(new).tolist(),
                    }
                    for k, old, new in zip(kernel, groups, labels, strict=True)
                ]
    atomic_npz(folder / "groups.npz", **maps)
    save_json(folder / "group_diagnostics.json", scores)
    atomic_npz(folder / "bank.npz", **arrays)
    print(dataset, "fixed", list(arrays), flush=True)


def predict_shard(job, *, root=OUT, local_root=LOCAL_OUT, clean_dev=False):
    dataset, phase, gpu, tasks, specifications = job
    device, original = device_setup(gpu), model_layout(dataset)
    model = read_data(root / dataset / "bank.npz")
    basis = {k: input_tensor(a, device) for k, a in model.items() if k != "ward"}
    folder = local_root / dataset / phase
    for shot, path in tasks:
        destination = folder / f"s{shot}_{path.name}"
        if destination.exists():
            continue
        sx, sy, queries = panel_batch(
            path, int(shot), device, original, clean_dev=clean_dev
        )
        seed = panel_seed(dataset, path, target=(phase == "test"))
        raw = {}
        for name, settings in specifications[shot].items():
            if name == "ward":
                support, readout, index, layout = sx, queries, model[name], original
            else:
                block = basis[name]
                support = project(sx, block, original)
                readout = {k: project(v, block, original) for k, v in queries.items()}
                index = None
                layout = reparameterized_layout(original, block)
            for rate, steps in settings:
                predictions = train(
                    support,
                    sy,
                    readout,
                    index,
                    layout,
                    rate,
                    steps,
                    seed,
                    curves=phase == "dev",
                )
                for step, rows in predictions.items():
                    raw.update(
                        {key_for(name, rate, step, k): v for k, v in rows.items()}
                    )
        atomic_npz(destination, **raw)


def evaluate(
    dataset,
    phase,
    specs,
    gpus,
    pool,
    *,
    root=OUT,
    local_root=LOCAL_OUT,
    clean_dev=False,
):
    original, out = ROOT / dataset, root / dataset
    panel = LOCAL / dataset / phase
    target = local_root / dataset / phase
    target.mkdir(parents=True, exist_ok=True)
    inputs = sorted(panel.glob("inputs_*.npz"))
    digest_map = {path.name: digest(path) for path in inputs}
    expected = json.loads((original / f"{phase}_prediction_plan.json").read_text())
    if digest_map != expected["inputs_sha256"]:
        raise ValueError(f"Changed {dataset}/{phase} inputs")
    path = out / f"{phase}_prediction_plan.json"
    record = {
        "inputs_sha256": digest_map,
        "bank_sha256": digest(out / "bank.npz"),
        "specifications": specs,
    }
    if clean_dev:
        record["clean_dev"] = True
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f"Changed {dataset}/{phase} choices")
    save_json(path, record)
    jobs = [(shot, p) for shot in specs for p in inputs]
    pool.map(
        partial(predict_shard, root=root, local_root=local_root, clean_dev=clean_dev),
        [
            (dataset, phase, gpu, jobs[i :: len(gpus)], specs)
            for i, gpu in enumerate(gpus)
        ],
        chunksize=1,
    )
    hashes = {
        f"s{shot}_{p.name}": digest(target / f"s{shot}_{p.name}") for shot, p in jobs
    }
    save_json(out / f"{phase}_sealed.json", hashes)
    print(dataset, phase, "sealed", len(hashes), flush=True)
    answers = read_data(panel / "answers.npz")
    values = {shot: {} for shot in specs}
    for shot, p in jobs:
        name = f"s{shot}_{p.name}"
        shard = target / name
        if digest(shard) != hashes[name]:
            raise ValueError(f"Changed predictions for {name}")
        offset = int(p.stem.rsplit("_", 1)[1])
        for key, logits in read_data(shard).items():
            rows = (
                logits
                if key.endswith("__support")
                else metrics(logits, answers["qy"][offset : offset + len(logits)])
            )
            values[shot].setdefault(key, []).append(rows)
    return {
        shot: {key: np.concatenate(rows) for key, rows in groups.items()}
        for shot, groups in values.items()
    }, answers["units"]


def select(dataset, gpus, pool):
    baseline = ["ward", "normalized"]
    if dataset == "wisdm":
        baseline.append("normalized_threshold")
    names = {
        str(shot): [f"s{shot}_{name}" for name in CONFIG["search"]]
        for shot in PLAN[dataset]["shots"]
    }
    values, _ = evaluate(
        dataset,
        "dev",
        {shot: grid([*baseline, *items]) for shot, items in names.items()},
        gpus,
        pool,
    )
    choices = {
        shot: {
            **{name: pick(rows, [name]) for name in baseline},
            "gradient": pick(rows, names[shot]),
        }
        for shot, rows in values.items()
    }
    save_json(OUT / dataset / "choices.json", choices)
    save_json(
        OUT / dataset / "dev_curves.json",
        {
            shot: {k: v.mean(0).tolist() for k, v in rows.items()}
            for shot, rows in values.items()
        },
    )
    print(dataset, "frozen validation choices", choices, flush=True)
    return choices


def summarize(dataset, values, units, choices):
    out, prior = OUT / dataset, ROOT / dataset
    msr = read_data(prior / "msr/per_episode.npz")
    old = read_data(prior / "per_episode.npz")
    if not all(np.array_equal(units, source["units"]) for source in (msr, old)):
        raise ValueError(f"Changed target units for {dataset}")
    rank = CONFIG[dataset]["rank"]
    codes = rank * model_layout(dataset).hidden
    saved = json.loads((prior / "msr/choices.json").read_text())
    if any(row["basis"]["codes"] != codes for row in saved.values()):
        raise ValueError("MSR has a different numerical budget")
    results, raw = {}, {}
    for shot, rows in values.items():
        results[shot] = {}
        for kind in ("query", "clean"):
            if not any(key.endswith(f"__{kind}") for key in rows):
                continue
            selected = {
                name: rows[key_for(v["name"], v["rate"], v["step"], kind)]
                for name, v in choices[shot].items()
            }
            selected["msr"] = msr[f"{shot}_{kind}_basis"]
            if dataset == "wisdm":
                selected["threshold"] = old[f"s{shot}_{kind}_partial"]
            delta = {}
            for method, reference in selected.items():
                if method == "gradient":
                    continue
                diff = (selected["gradient"][:, 1] - reference[:, 1]) * 100
                if dataset == "wisdm":
                    diff = np.array([diff[units == p].mean() for p in np.unique(units)])
                delta[method] = tolerance_interval(diff)
            norm = (selected["normalized"][:, 1] - selected["ward"][:, 1]) * 100
            if dataset == "wisdm":
                norm = np.array([norm[units == p].mean() for p in np.unique(units)])
            results[shot][kind] = {
                "metrics": {k: v.mean(0).tolist() for k, v in selected.items()},
                "gradient_delta_pp_95_ci": delta,
                "normalization_delta_pp_95_ci": tolerance_interval(norm),
                "episodes": len(units),
                "codes": codes,
            }
            raw.update(
                {f"s{shot}_{kind}_{name}": data for name, data in selected.items()}
            )
    save_json(out / "results.json", results)
    atomic_npz(out / "per_episode.npz", units=units, **raw)
    print(dataset, "completed", flush=True)


def run(dataset, phase, gpus):
    if not (OUT / "synthetic/results.json").is_file():
        raise ValueError("Run and inspect the synthetic experiment first")
    out = OUT / dataset
    out.mkdir(exist_ok=True, parents=True)
    save_json(
        out / "registration.json",
        {
            "plan_sha256": digest(GRADIENT_PLAN),
            "code_sha256": digest(Path(__file__)),
            "source_sha256": digest(LOCAL / dataset / "source.npz"),
            "ordinary": "Ward by input profiles at the matched rank, same original fit grid.",
            "additional_wisdm_control": "Frozen threshold partition and its normalized variant.",
            "source": "Aug-Omniglot clean support and all transformed source queries; WISDM earlier support and later queries.",
        },
    )
    with mp.get_context("spawn").Pool(min(len(gpus), 8)) as pool:
        if phase in ("dev", "run"):
            make_bank(dataset, pool, gpus)
            choices = select(dataset, gpus, pool)
        else:
            choices = json.loads((out / "choices.json").read_text())
        if phase in ("test", "run"):
            values, units = evaluate(
                dataset,
                "test",
                {
                    shot: {v["name"]: [[v["rate"], [v["step"]]]] for v in rows.values()}
                    for shot, rows in choices.items()
                },
                gpus,
                pool,
            )
            summarize(dataset, values, units, choices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("phase", choices=("dev", "test", "run"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    run(args.dataset, args.phase, args.gpus)

"""Fit source proposals, tune on separate episodes, and score frozen predictions."""

import argparse
import json
import math
import multiprocessing as mp

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.real_group_transfer_v1.optimization import (
    Layout,
    fit_dense,
    losses,
    predict,
)
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .groups import LOCAL, PLAN, ROOT, input_bank, shuffled, signed_bank
from .linear import atomic_npz, tolerance_interval

DATASETS = ("wisdm", "aug_omniglot")
SETTINGS = PLAN["classification"]
FAMILIES = ("partial", "ward")


def model_layout(dataset):
    cfg = PLAN[dataset]
    if dataset == "wisdm":
        sites = cfg["sites"]
        return Layout(
            sites,
            cfg["window"] * 4 // sites,
            cfg["hidden"] // sites,
            len(cfg["classes"]),
        )
    return Layout(1, 784, cfg["hidden"], 5)


def read_data(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)


def label_tensor(x, device):
    return torch.tensor(x, dtype=torch.long, device=device)


def input_tensor(x, device):
    tensor = torch.tensor(x, dtype=torch.float32, device=device)
    return tensor / 255 if x.dtype == np.uint8 else tensor


def device_setup(gpu):
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    return f"cuda:{gpu}"


def source_worker(job):
    dataset, gpu, ids = job
    device = device_setup(gpu)
    cfg = PLAN[dataset]
    folder = LOCAL / dataset
    raw = read_data(folder / "source.npz")
    layout = model_layout(dataset)
    for i in ids:
        output = folder / f"source_fit_{i:03d}.npz"
        if output.exists():
            continue
        take = slice(i, i + 1)
        panel = tuple(
            fn(raw[k][take], device)
            for k, fn in (
                ("sx", input_tensor),
                ("sy", label_tensor),
                ("qx", input_tensor),
                ("qy", label_tensor),
            )
        )
        param, risk, _ = fit_dense(
            panel,
            layout,
            steps=cfg["source_steps"],
            restarts=cfg["source_restarts"],
            device=device,
            seed=cfg["seed"] + 991 * int(i),
        )
        atomic_npz(output, parameters=param[0], ce=risk[0])


def compact_labels(index):
    # Canonicalize code names and orientations without an int16 size limit.
    original = index.ravel()
    keys, inverse = np.unique(np.abs(original), return_inverse=True)
    first = np.full(len(keys), original.size)
    np.minimum.at(first, inverse, np.arange(original.size))
    order = np.argsort(first)
    order = order[keys[order] != 0]
    rename = np.zeros(len(first), dtype=np.int32)
    rename[order] = np.arange(1, len(order) + 1) * np.sign(original[first[order]])
    return (rename[inverse] * np.sign(original)).reshape(index.shape)


def deduplicate(bank, meta):
    seen, result = set(), {}
    for name, raw in bank.items():
        index = compact_labels(raw)
        key = (meta[name]["family"], index.tobytes())
        if key not in seen:
            seen.add(key)
            result[name] = index
    return result, {k: meta[k] for k in result}


def make_bank(dataset, profiles):
    cfg, layout = PLAN[dataset], model_layout(dataset)
    bank, meta = {}, {}
    for method in FAMILIES:
        partial = method == "partial"
        if dataset == "wisdm":
            values = cfg["partial_thresholds" if partial else "ward_budgets"]
            maps = signed_bank(
                profiles,
                np.ones(profiles.shape[1:], dtype=bool),
                values,
                partial=partial,
            )
            for value, index in maps.items():
                name = f"{method}_global_{value:g}"
                bank[name], meta[name] = (
                    index,
                    {"family": method, "input_groups": False},
                )
        for blocks in cfg.get("blocks", [1]):
            grid = cfg["input_thresholds" if partial else "input_groups"]
            for value, index in input_bank(
                profiles, blocks, grid, partial=partial
            ).items():
                name = f"{method}_input{blocks}_{value:g}"
                bank[name], meta[name] = index, {"family": method, "input_groups": True}
    if dataset == "wisdm":
        size = layout.fan_in * layout.channels
        bank["temporal"] = np.tile(
            np.arange(1, size + 1).reshape(layout.fan_in, layout.channels),
            (1, layout.sites),
        )
        meta["temporal"] = {"family": "temporal", "input_groups": False}
    return deduplicate(bank, meta)


def source(dataset, gpus):
    folder, out = LOCAL / dataset, ROOT / dataset
    with np.load(folder / "source.npz") as data:
        count = len(data["units"])
    shards = np.array_split(np.arange(count), len(gpus))
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        pool.map(
            source_worker,
            [(dataset, gpu, ids) for gpu, ids in zip(gpus, shards, strict=True)],
        )
    states = [read_data(folder / f"source_fit_{i:03d}.npz") for i in range(count)]
    params = np.stack([s["parameters"] for s in states])
    ce = np.stack([s["ce"] for s in states])
    order = ce[:, 0].argsort()
    profiles = source_profiles(params[order], model_layout(dataset))
    atomic_npz(folder / "profiles.npz", profiles=profiles, losses=ce, order=order)
    bank, definitions = make_bank(dataset, profiles)
    atomic_npz(out / "bank.npz", **bank)
    save_json(
        out / "bank.json",
        {
            "entries": {
                name: {**definitions[name], "codes": int(abs(idx).max())}
                for name, idx in bank.items()
            },
            "source_profiles_sha256": digest(folder / "profiles.npz"),
            "source_panel_sha256": digest(folder / "source.npz"),
            "bank_sha256": digest(out / "bank.npz"),
        },
    )
    print(
        dataset,
        "source bank",
        {k: int(abs(v).max()) for k, v in bank.items()},
        flush=True,
    )


def train(sx, sy, queries, index, layout, rate, steps, seed, curves):
    size = layout.connections if index is None else int(abs(index).max())
    if size == layout.connections:
        index = None
    if index is not None:
        flat = (
            index.reshape(layout.fan_in, layout.sites, layout.channels)
            .transpose(1, 0, 2)
            .ravel()
        )
        labels = label_tensor(flat, sx.device)
        select = (labels.abs() - 1).clamp_min(0)
        sign = labels.sign().float()
    seed_weight = torch.Generator(device=sx.device).manual_seed(seed)
    seed_tail = torch.Generator(device=sx.device).manual_seed(seed + 1)
    w = torch.randn(len(sx), 1, size, generator=seed_weight, device=sx.device)
    tail = torch.randn(
        len(sx),
        1,
        layout.hidden * (layout.outputs + 1) + layout.outputs,
        generator=seed_tail,
        device=sx.device,
    )
    w *= math.sqrt(2 / layout.fan_in)
    tail[..., : layout.hidden] *= 0.1
    tail[..., -layout.outputs :] = 0
    parameter = torch.cat([w, tail], dim=-1).requires_grad_()
    optimizer = torch.optim.Adam([parameter], lr=rate, fused=True)
    snapshots = {}
    last = max(steps)
    for step in range(last + 1):
        weight = parameter[..., :size]
        tail = parameter[..., size:]
        if index is not None:
            weight = weight[..., select] * sign
        if step in steps:
            with torch.no_grad():
                outputs = {
                    name: predict(x, weight, tail, layout)[:, 0].cpu().numpy()
                    for name, x in queries.items()
                }
                if curves:
                    logits = predict(sx, weight, tail, layout)
                    loss = losses(logits, sy)[:, 0].mean(-1)
                    recall = (logits[:, 0].argmax(-1) == sy).float().mean(-1)
                    outputs["support"] = (
                        torch.stack([loss, recall], dim=-1).cpu().numpy()
                    )
                snapshots[step] = outputs
        if step == last:
            break
        objective = losses(predict(sx, weight, tail, layout), sy).mean(-1).sum()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    return snapshots


def predict_worker(job):
    dataset, gpu, phase, tasks, spec = job
    device = device_setup(gpu)
    cfg = PLAN[dataset]
    layout = model_layout(dataset)
    work, out = LOCAL / dataset, ROOT / dataset
    bank = read_data(out / "bank.npz")
    if phase != "dev":
        bank.update(read_data(out / "shuffle_bank.npz"))
    panel = work / ("dev" if phase == "shuffle_dev" else phase)
    for shot, path in tasks:
        i = int(path.stem.rsplit("_", 1)[1])
        cache = work / f"pred_{phase}" / f"s{shot}_{path.name}"
        if cache.exists():
            continue
        data = read_data(panel / path)
        sx = input_tensor(
            data["sx"][:, :, :shot].reshape(
                -1, layout.outputs * shot, layout.sites, layout.fan_in
            ),
            device,
        )
        sy = label_tensor(data["sy"][:, :, :shot].reshape(len(sx), -1), device)
        queries = {"query": input_tensor(data["qx"], device)}
        if phase == "test" and "qx_clean" in data:
            queries["clean"] = input_tensor(data["qx_clean"], device)
        bundle = {}
        seed = cfg["seed"] + 121991 + 1129 * i + (701 if phase == "test" else 0)
        for name, rates in spec[str(shot)].items():
            for rate, steps in rates:
                snapshots = train(
                    sx,
                    sy,
                    queries,
                    bank.get(name),
                    layout,
                    rate,
                    steps,
                    seed,
                    phase != "test",
                )
                for step, arrays in snapshots.items():
                    prefix = f"{name}__{rate:g}__{step}"
                    bundle.update(
                        {f"{prefix}__{key}": scores for key, scores in arrays.items()}
                    )
        tmp = cache.with_suffix(".partial.npz")
        np.savez(tmp, **bundle)
        tmp.replace(cache)


def evaluate(dataset, phase, specs, gpus):
    cfg, work, out = PLAN[dataset], LOCAL / dataset, ROOT / dataset
    cache = work / f"pred_{phase}"
    cache.mkdir(exist_ok=True)
    panel = work / ("dev" if phase == "shuffle_dev" else phase)
    inputs = sorted(panel.glob("inputs_*.npz"))
    jobs = [(shot, p.relative_to(panel)) for shot in cfg["shots"] for p in inputs]
    plan = {
        "bank_sha256": digest(out / "bank.npz"),
        "shuffle_sha256": digest(out / "shuffle_bank.npz") if phase != "dev" else None,
        "inputs_sha256": {p.name: digest(p) for p in inputs},
        "specs": specs,
    }
    manifest = out / f"{phase}_prediction_plan.json"
    if manifest.exists() and json.loads(manifest.read_text()) != plan:
        raise ValueError(f"Prediction settings changed for {dataset}/{phase}")
    save_json(manifest, plan)
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        pool.map(
            predict_worker,
            [
                (dataset, gpu, phase, jobs[i :: len(gpus)], specs)
                for i, gpu in enumerate(gpus)
            ],
        )
    hashes = {
        f"s{shot}_{p.name}": digest(cache / f"s{shot}_{p.name}") for shot, p in jobs
    }
    save_json(out / f"{phase}_sealed.json", hashes)
    print(dataset, phase, "sealed", len(hashes), flush=True)


def read_metrics(dataset, phase):
    cfg, work, out = PLAN[dataset], LOCAL / dataset, ROOT / dataset
    phases = "dev" if phase == "shuffle_dev" else phase
    with np.load(work / phases / "answers.npz") as answers:
        y, units = answers["qy"], answers["units"]
    hashes = json.loads((out / f"{phase}_sealed.json").read_text())
    metrics = {}
    for shot in cfg["shots"]:
        scores = {}
        for base in sorted(k for k in hashes if k.startswith(f"s{shot}_")):
            path = work / f"pred_{phase}" / base
            if hashes[base] != digest(path):
                raise ValueError(f"Changed prediction archive {path}")
            offset = int(path.stem.rsplit("_", 1)[1])
            with np.load(path) as data:
                for key in data.files:
                    value = data[key]
                    if not key.endswith("__support"):
                        target = y[offset : offset + len(value)]
                        ce = (
                            logsumexp(value, axis=-1)
                            - np.take_along_axis(value, target[..., None], axis=-1)[
                                ..., 0
                            ]
                        )
                        value = np.column_stack(
                            (ce.mean(-1), (value.argmax(-1) == target).mean(-1))
                        )
                    scores.setdefault(key, []).append(value)
        metrics[str(shot)] = {
            key: np.concatenate(values) for key, values in scores.items()
        }
    return metrics, units


def grid(names):
    search = [[lr, [0, *SETTINGS["steps"]]] for lr in SETTINGS["rates"]]
    return {name: search for name in names}


def key_for(name, rate, step, kind="query"):
    return f"{name}__{rate:g}__{step}__{kind}"


def pick(scores, names):
    candidates = []
    for name in names:
        for rate in SETTINGS["rates"]:
            for step in SETTINGS["steps"]:
                ce, acc = scores[key_for(name, rate, step)].mean(0)
                candidates.append((-float(acc), float(ce), step, rate, name))
    acc, ce, step, rate, name = min(candidates)
    return {"name": name, "rate": rate, "step": step, "dev_acc": -acc, "dev_ce": ce}


def select(dataset, gpus):
    cfg, out = PLAN[dataset], ROOT / dataset
    entries = json.loads((out / "bank.json").read_text())["entries"]
    bank = read_data(out / "bank.npz")
    base = {*FAMILIES, "dense"}
    if "temporal" in entries:
        base.add("temporal")
    evaluate(
        dataset,
        "dev",
        {str(shot): grid([*bank, "dense"]) for shot in cfg["shots"]},
        gpus,
    )
    scores, _ = read_metrics(dataset, "dev")
    choices, shuffles, shuffle_specs = {}, {}, {}
    for shot, rows in scores.items():
        choices[shot] = {}
        for family in sorted(base):
            names = (
                [n for n, row in entries.items() if row["family"] == family]
                if family != "dense"
                else ["dense"]
            )
            choices[shot][family] = pick(rows, names)
        rng = np.random.default_rng(cfg["seed"] + int(shot) * 601)
        names = []
        for family in FAMILIES:
            name = choices[shot][family]["name"]
            for i in range(SETTINGS["shuffle_repeats"]):
                key = f"{family}_shuffle_s{shot}_{i}"
                shuffles[key] = shuffled(
                    bank[name], rng, input_groups=entries[name]["input_groups"]
                )
                names.append(key)
        shuffle_specs[shot] = grid(names)
    atomic_npz(out / "shuffle_bank.npz", **shuffles)
    evaluate(dataset, "shuffle_dev", shuffle_specs, gpus)
    control, _ = read_metrics(dataset, "shuffle_dev")
    for shot, rows in control.items():
        for family in FAMILIES:
            names = [n for n in shuffle_specs[shot] if n.startswith(family)]
            aggregate = {
                key_for(family, rate, step): np.mean(
                    [rows[key_for(name, rate, step)] for name in names], axis=0
                )
                for rate in SETTINGS["rates"]
                for step in SETTINGS["steps"]
            }
            choices[shot][f"{family}_shuffle"] = {
                **pick(aggregate, [family]),
                "names": names,
            }
    save_json(out / "choices.json", choices)
    trace = {
        shot: {key: value.mean(0).tolist() for key, value in rows.items()}
        for shot, rows in scores.items()
    }
    save_json(out / "dev_curves.json", trace)
    print(dataset, "frozen development choices", json.dumps(choices), flush=True)


def target(dataset, gpus):
    path = ROOT / dataset / "choices.json"
    choices = json.loads(path.read_text())
    specs = {shot: {} for shot in choices}
    for shot, rows in choices.items():
        for choice in rows.values():
            for name in choice.get("names", [choice["name"]]):
                specs[shot][name] = [[choice["rate"], [choice["step"]]]]
    evaluate(dataset, "test", specs, gpus)
    summarize(dataset, choices)


def summarize(dataset, choices):
    scores, units = read_metrics(dataset, "test")
    out = ROOT / dataset
    report, raw = {}, {}
    for shot, rows in scores.items():
        report[shot] = {}
        for kind in ("query", "clean"):
            if not any(k.endswith(f"__{kind}") for k in rows):
                continue
            selected = {}
            for method, choice in choices[shot].items():
                selected[method] = np.mean(
                    [
                        rows[key_for(name, choice["rate"], choice["step"], kind)]
                        for name in choice.get("names", [choice["name"]])
                    ],
                    axis=0,
                )
                raw[f"s{shot}_{kind}_{method}"] = selected[method]
            pairs = {}
            for a, b in (
                ("partial", "partial_shuffle"),
                ("ward", "ward_shuffle"),
                ("partial", "ward"),
                ("partial", "dense"),
                ("ward", "dense"),
            ):
                diff = (selected[a][:, 1] - selected[b][:, 1]) * 100
                if dataset == "wisdm":
                    diff = np.array([diff[units == p].mean() for p in np.unique(units)])
                pairs[f"{a}_minus_{b}"] = tolerance_interval(diff)
            report[shot][kind] = {
                "metrics": {k: v.mean(0).tolist() for k, v in selected.items()},
                "accuracy_delta_pp_95_ci": pairs,
                "episodes": len(units),
                "independent_units": len(np.unique(units)),
            }
        if "clean" in report[shot]:
            interaction = {}
            for method in FAMILIES:
                for control in ("dense", f"{method}_shuffle"):
                    aug = (
                        raw[f"s{shot}_query_{method}"][:, 1]
                        - raw[f"s{shot}_query_{control}"][:, 1]
                    )
                    clean = (
                        raw[f"s{shot}_clean_{method}"][:, 1]
                        - raw[f"s{shot}_clean_{control}"][:, 1]
                    )
                    interaction[f"{method}_minus_{control}"] = tolerance_interval(
                        (aug - clean) * 100
                    )
            report[shot]["augmentation_interaction_pp_95_ci"] = interaction
    save_json(out / "results.json", report)
    atomic_npz(out / "per_episode.npz", units=units, **raw)
    print(dataset, "results", json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("phase", choices=("source", "select", "test", "run"))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    phases = {"source": source, "select": select, "test": target}
    for method in phases.values() if args.phase == "run" else [phases[args.phase]]:
        method(args.dataset, args.gpus)

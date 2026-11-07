"""G23 observed panels; teacher I/C are confined to generation and later audit."""

import json
import math

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .functional_observability import teacher as module_teacher


def teacher(plan, job, tasks=None):
    family, variant = job["family"], job["variant"]
    rng = np.random.default_rng(plan["seed"] + family * 41)
    if variant == "random":
        index = (rng.uniform(size=(16, 16)) < 0.55) * rng.integers(1, 5, size=(16, 16))
        index *= rng.choice((-1, 1), size=(16, 16))
    else:
        _, index, _, _ = module_teacher(
            family,
            variant,
            {"seed": plan["seed"], "tasks": 1, "codes": plan["codes"]},
            "cpu",
        )
    count = plan["sources"] if tasks is None else tasks
    codes = np.stack(
        [
            np.random.default_rng(plan["seed"] + family * 1009 + task * 10007).normal(
                size=plan["codes"]
            )
            for task in range(count)
        ]
    )
    if variant == "quartet_mix":
        codes /= 2
    return index.astype(np.int16), codes


def predict(x, weights, plan):
    return (torch.einsum("ni,tih->tnh", x, weights) + plan["bias"]).relu().sum(
        -1
    ) / math.sqrt(plan["hidden"])


def weights_from_index(index, codes, device):
    values = np.c_[np.zeros(len(codes)), codes]
    weights = values[:, np.abs(index)] * np.sign(index)
    return torch.tensor(weights, device=device, dtype=torch.float64)


@torch.no_grad()
def prepare(root, plan, job_id, device):
    job = plan["families"][job_id]
    stem = root / "data" / f"family_{job_id:03d}"
    if stem.with_name(stem.name + "_source.npz").exists():
        raise RuntimeError("Source observations already generated")
    index, codes = teacher(plan, job)
    weights = weights_from_index(index, codes, device)
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + job["family"] * 83
    )
    order = np.random.default_rng(plan["seed"] + job["family"] * 47).permutation(
        plan["inputs"]
    )
    t = np.r_[
        0.0, np.geomspace(plan["axis_min"], plan["axis_max"], plan["axis_points"] - 1)
    ]
    vt = (t[:-1] + t[1:]) / 2
    axes = torch.eye(plan["inputs"], device=device, dtype=torch.float64)
    sign = weights.new_tensor([-1, 1])
    axis_x = torch.einsum("s,di,n->sdni", sign, axes, weights.new_tensor(t)).flatten(
        0, 2
    )
    axis_vx = torch.einsum("s,di,n->sdni", sign, axes, weights.new_tensor(vt)).flatten(
        0, 2
    )
    slices, prefix_validation = [], []
    for step in range(1, plan["inputs"] + 1):
        for count, collection in (
            (plan["slice_points"], slices),
            (plan["prefix_validation"], prefix_validation),
        ):
            scales = weights.new_tensor(plan["slice_scales"])[
                torch.arange(count, device=device) % len(plan["slice_scales"])
            ]
            local = (
                torch.randn(
                    (count, step),
                    device=device,
                    dtype=torch.float64,
                    generator=generator,
                )
                * scales[:, None]
                / math.sqrt(step)
            )
            x = weights.new_zeros((count, plan["inputs"]))
            x[:, order[:step]] = local
            collection.append(x)
    panels = {
        "train": torch.cat((axis_x, torch.cat(slices))),
        "axis_validation": axis_vx,
        "prefix_validation": torch.cat(prefix_validation),
        "fit_validation": torch.randn(
            (plan["fit_validation"], plan["inputs"]),
            device=device,
            dtype=torch.float64,
            generator=generator,
        ),
        "meta": torch.randn(
            (plan["meta_points"], plan["inputs"]),
            device=device,
            dtype=torch.float64,
            generator=generator,
        ),
    }
    repeats = {
        "train": torch.cat(
            (
                weights.new_full((len(axis_x),), plan["axis_repeats"]),
                weights.new_full(
                    (plan["inputs"] * plan["slice_points"],), plan["slice_repeats"]
                ),
            )
        ),
        "axis_validation": weights.new_full((len(axis_vx),), plan["axis_repeats"]),
        "prefix_validation": weights.new_full(
            (len(panels["prefix_validation"]),), plan["slice_repeats"]
        ),
        "fit_validation": weights.new_full(
            (plan["fit_validation"],), plan["slice_repeats"]
        ),
        "meta": weights.new_full((plan["meta_points"],), plan["slice_repeats"]),
    }
    arrays = {
        "order": order,
        "axis_t": t,
        "axis_vt": vt,
        "train_repeats": repeats["train"].cpu().numpy(),
    }
    meta = {}
    labels_per_task = 0
    for name, x in panels.items():
        clean = predict(x, weights, plan)
        noise = torch.randn(
            clean.shape, device=device, dtype=torch.float64, generator=generator
        )
        destination = meta if name == "meta" else arrays
        destination[name + "_x"] = x.cpu().numpy()
        for sigma in plan["noise"]:
            destination[f"{name}_y_{sigma}"] = (
                (clean + sigma * noise / repeats[name].sqrt()).cpu().numpy()
            )
        labels_per_task += int(repeats[name].sum())
    np.savez_compressed(stem.with_name(stem.name + "_source.npz"), **arrays)
    np.savez_compressed(stem.with_name(stem.name + "_meta.npz"), **meta)
    save_json(
        stem.with_suffix(".json"),
        {
            "job": job,
            "labels_per_source_task_condition": labels_per_task,
            "source_tasks": plan["sources"],
            "labels_per_family_condition": labels_per_task * plan["sources"],
            "noise_conditions_paired": True,
            "axis_train_points": len(axis_x),
            "axis_validation_points": len(axis_vx),
            "train_points": len(panels["train"]),
            "protocol_sha256": digest(root / "protocol.json"),
        },
    )


def load_source(root, plan, job_id, task, sigma, device):
    path = root / "data" / f"family_{job_id:03d}_source.npz"
    frozen = json.loads((root / "data_frozen.json").read_text())
    if digest(path) != frozen["sha256"][str(path.relative_to(root))]:
        raise RuntimeError("Frozen source observations changed")
    with np.load(path) as data:
        arrays = {
            name: torch.tensor(
                value[task] if "_y_" in name else value,
                device=device,
                dtype=torch.float64,
            )
            for name, value in data.items()
            if name.endswith(("_x", f"_y_{sigma}")) or name == "train_repeats"
        }
        arrays.update({name: data[name] for name in ("order", "axis_t", "axis_vt")})
    return arrays

"""Build single-task WSR and LUT-Q candidates and the multi-task Ward bank."""

import argparse
import json

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import (
    input_tensor,
    read_data,
)
from experiments.asrr_benchmark_families_v1.groups import (
    input_bank,
    signed_bank,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.real_group_transfer_v1.optimization import (
    Layout,
    fit_dense,
    losses,
    predict,
)
from experiments.real_group_transfer_v1.study import (
    DATA as FC100_DATA,
    source_profiles,
)
from experiments.sharing_structure_research_v1.blind_contract import digest
from experiments.srr_approx_v1.synthetic_random_controls import shuffle_groups

from .common import CONFIG, OUT, SEED, WORK, freeze, register
from .hard import fit_indexators
from .panels import panel_directory
from .target import score_test, transfer


def source_tensors(dataset, device):
    path = panel_directory(dataset) / "source.npz"
    data = read_data(path)
    return (
        input_tensor(data["sx"], device),
        torch.as_tensor(data["sy"], device=device, dtype=torch.long),
        input_tensor(data["qx"], device),
        torch.as_tensor(data["qy"], device=device, dtype=torch.long),
    )


def dense_source(dataset, device):
    folder = WORK / dataset
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "dense_sources.npz"
    if path.exists():
        data = read_data(path)
        return data["parameters"], data["ce"]
    layout = Layout(*CONFIG["real"][dataset]["layout"])
    if dataset == "mnist_torus":
        config = CONFIG["mnist"]
        panel = source_tensors(dataset, device)
        parameter, ce, seconds = fit_dense(
            panel,
            layout,
            steps=config["source_steps"],
            restarts=config["source_restarts"],
            device=device,
            seed=SEED + 339,
        )
        print(f"{dataset} Dense source fits {seconds:.1f}s", flush=True)
    elif dataset == "fc100":
        cached = read_data(FC100_DATA.parent / "fc100" / "dense_fits.npz")
        if not np.array_equal(
            cached["records"],
            read_data(folder / "source_images.npz")["records"],
        ):
            raise ValueError(
                "FC100 source drawing indices differ from the saved Dense fits"
            )
        parameter = cached["parameters"]
        with torch.no_grad():
            _, _, qx, qy = source_tensors(dataset, device)
            state = torch.as_tensor(parameter, device=device)
            logits = predict(
                qx,
                state[..., : layout.connections],
                state[..., layout.connections :],
                layout,
            )
            ce = losses(logits, qy).mean(-1).cpu().numpy()
    else:
        cached = [
            read_data(path)
            for path in sorted(
                panel_directory(dataset).glob("source_fit_*.npz")
            )
        ]
        if len(cached) != len(
            read_data(panel_directory(dataset) / "source.npz")["units"]
        ):
            raise ValueError(
                f"Incomplete cached Dense source fits for {dataset}"
            )
        parameter = np.stack([x["parameters"] for x in cached])
        ce = np.stack([x["ce"] for x in cached])
    if np.any(ce[:, 1:] < ce[:, :-1]):
        order = ce.argsort(-1)
        parameter = parameter[np.arange(len(ce))[:, None], order]
        ce = np.take_along_axis(ce, order, axis=1)
    atomic_npz(path, parameters=parameter, ce=ce)
    return parameter, ce


def train_hard(dataset, method, device):
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    settings, codes = CONFIG["hard_source"], config["codes"]
    folder = OUT / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "candidates.json"
    if manifest.exists():
        return [
            folder / row["name"] for row in json.loads(manifest.read_text())
        ]
    parameters, ce = dense_source(dataset, device)
    rows = np.argsort(ce[:, 0])[: settings["tasks"]]
    initial = torch.tensor(parameters[rows, 0], device=device)
    sx, sy, qx, qy = [v[rows] for v in source_tensors(dataset, device)]
    saved = []
    for rate in settings["rates"]:
        generator = torch.Generator(device=device).manual_seed(SEED + 133)

        def loss(weight, tail, *, validation):
            x, y = qx, qy
            if not validation:
                take = torch.randperm(
                    sx.shape[1], device=device, generator=generator
                )[:128]
                x, y = sx[:, take], sy[:, take]
            return losses(
                predict(x, weight[:, None], tail[:, None], layout), y
            )[:, 0].mean(-1)

        candidates, numerics, steps = fit_indexators(
            method,
            initial,
            layout.connections,
            codes,
            loss,
            {**settings, "rate": rate},
        )
        for task, index, score, step in zip(
            rows, candidates, numerics, steps, strict=True
        ):
            path = folder / f"task{task}_rate{rate:g}.npz"
            atomic_npz(path, index=index)
            saved.append(
                {
                    "name": path.name,
                    "codes": codes,
                    "rate": rate,
                    "step": int(step),
                    "numerical_ce": float(score),
                    "sha256": digest(path),
                }
            )
        print(f"{dataset} {method} source rate {rate:g}", flush=True)
    freeze(manifest, saved)
    return [folder / row["name"] for row in saved]


def model_index(index, layout):
    return (
        index.reshape(layout.fan_in, layout.sites, layout.channels)
        .transpose(1, 0, 2)
        .ravel()
    )


def ward(dataset, device):
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    folder = OUT / dataset / "ward_bank"
    folder.mkdir(parents=True, exist_ok=True)
    parameters, ce = dense_source(dataset, device)
    profiles = source_profiles(parameters[np.argsort(ce[:, 0])], layout)
    m = config["codes"]
    groups = m // layout.hidden
    candidates = {
        f"input{blocks}": input_bank(
            profiles, blocks, (groups,), partial=False
        )[groups]
        for blocks in (1, 2, 4, 8)
    }
    if layout.connections <= 10000:
        mask = np.ones(profiles.shape[1:], dtype=bool)
        candidates["global"] = signed_bank(
            profiles, mask, (m,), partial=False
        )[m]
    models = []
    for name, index in candidates.items():
        path = folder / f"{name}.npz"
        if len(np.unique(np.abs(index))) != m:
            raise ValueError(f"Wrong code count for {name}")
        if not path.exists():
            atomic_npz(path, index=model_index(index, layout))
        models.append(path)
    transfer(dataset, "asrr", models, device)
    out = OUT / dataset
    policy = json.loads((out / "asrr_choice.json").read_text())
    name = next(path.stem for path in models if str(path) == policy["model"])
    index = candidates[name]
    paths = []
    rng = np.random.default_rng(SEED + 248)
    for replica in range(CONFIG["target"]["shuffle_replicates"]):
        if name == "global":
            shuffled = shuffle_groups(index, rng)
        else:
            blocks = np.split(index, int(name.removeprefix("input")), axis=1)
            shuffled = np.concatenate(
                [
                    shuffle_groups(block, rng, shared_inputs=True)
                    for block in blocks
                ],
                axis=1,
            )
        if not np.array_equal(
            np.sort(index, axis=0), np.sort(shuffled, axis=0)
        ):
            raise ValueError("A control changed the signed counts of a neuron")
        path = folder / f"shuffle{replica}.npz"
        if not path.exists():
            atomic_npz(path, index=model_index(shuffled, layout))
        paths.append(path)
    freeze(
        out / "shuffled_choice.json",
        {**policy, "parent": name, "replicates": len(paths)},
    )
    score_test(dataset, "shuffled", paths, policy, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG["real"])
    parser.add_argument(
        "methods", nargs="+", choices=("source", "wsr", "lutq", "ward")
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    register()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    for name in args.methods:
        if name == "source":
            dense_source(args.dataset, device)
        elif name == "ward":
            ward(args.dataset, device)
        else:
            transfer(
                args.dataset,
                name,
                train_hard(args.dataset, name, device),
                device,
            )

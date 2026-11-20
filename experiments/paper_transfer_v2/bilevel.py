"""Same-budget continuous MSR-FC and categorical Yeh transfer."""

import argparse
import json
from itertools import product

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.groups import ROOT as FAMILY_OUTPUT
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.msr import (
    predictions,
    project,
    reparameterized_layout,
)
from experiments.real_group_transfer_v1.optimization import (
    Layout,
    initialization,
    losses,
    predict,
)
from experiments.sharing_structure_research_v1.blind_contract import digest

from .common import CONFIG, OUT, SEED, freeze, register
from .linear import budgeted_map
from .source import dense_source, source_tensors
from .target import layer_weights, transfer

CFG = CONFIG["real_bilevel"]


def stratified_sources(dataset, device, layout):
    sx, sy, qx, qy = source_tensors(dataset, device)
    ordered = sy.argsort(stable=True).reshape(len(sx), layout.outputs, -1)
    units = torch.arange(len(sx), device=device)[:, None, None]
    support = sx[units, ordered]
    if dataset == "aug_omniglot":
        support = support[:, :, :5]
    return support, qx, qy


def sample_support(data, shot, generator, classes):
    sx, qx, qy = data
    device = sx.device
    tasks = torch.randperm(len(sx), device=device, generator=generator)[
        : CFG["meta_batch"]
    ]
    shape = (len(tasks), len(classes), sx.shape[2])
    selected = torch.rand(shape, device=device, generator=generator).argsort()[
        ..., :shot
    ]
    x = sx[tasks[:, None, None], classes[None, :, None], selected].flatten(
        1, 2
    )
    y = classes.repeat_interleave(shot).expand(len(tasks), -1)
    q = torch.randperm(qx.shape[1], device=device, generator=generator)[
        : CFG["query_per_task"]
    ]
    return tasks, x, y, qx[tasks][:, q], qy[tasks][:, q]


def train_msr(data, original, num_factors, rank, rate, shot, device):
    generator = torch.Generator(device=device).manual_seed(SEED + 610)
    factors = torch.randn(
        num_factors, original.fan_in, rank, device=device, generator=generator
    )
    factors = torch.linalg.qr(factors).Q
    layout = reparameterized_layout(original, factors)
    output = torch.eye(layout.channels, device=device).repeat(
        layout.sites, 1, 1
    )
    state = initialization(
        layout, layout.connections, 1, 1, device, SEED + 611
    )
    initial = state.detach().flatten()
    inner_rates = torch.full((4,), 0.4, device=device)
    trained = [
        v.requires_grad_() for v in (factors, output, initial, inner_rates)
    ]
    optimizer = torch.optim.Adam(trained, lr=rate, fused=True)
    classes = torch.arange(layout.outputs, device=device)
    for step in range(1, CFG["outer_steps"] + 1):
        _, sx, sy, qx, qy = sample_support(data, shot, generator, classes)
        fitted = predictions(
            project(sx, factors, original),
            sy,
            {"q": project(qx, factors, original)},
            initial,
            inner_rates,
            output,
            layout,
            (1,),
            higher=True,
        )[1]["q"]
        risk = losses(fitted[:, None], qy).mean()
        optimizer.zero_grad(set_to_none=True)
        risk.backward()
        optimizer.step()
        if step in CFG["checkpoints"]:
            with torch.no_grad():
                factor, triangular = torch.linalg.qr(factors)
                if (triangular.diagonal(dim1=-2, dim2=-1).abs() < 1e-6).any():
                    raise ValueError("MSR factor lost rank")
                if (torch.linalg.matrix_rank(output) != layout.channels).any():
                    raise ValueError(
                        "The MSR output factors must be invertible"
                    )
            yield step, factor.cpu().numpy()


def train_yeh(data, layout, exemplar, groups, codes, rate, warm, shot, device):
    generator = torch.Generator(device=device).manual_seed(SEED + 725)
    connections = layout.connections // groups
    rank = codes // groups
    if warm:
        view = exemplar.reshape(layout.sites, layout.fan_in, layout.channels)
        weight = torch.as_tensor(view.transpose(0, 2, 1).copy(), device=device)
        ranks = weight.reshape(groups, connections).argsort(-1).argsort(-1)
    else:
        ranks = torch.rand(
            groups, connections, device=device, generator=generator
        ).argsort(-1)
    assigned = ranks * rank // connections
    logits = (
        torch.randn(
            groups, connections, rank, device=device, generator=generator
        )
        * 0.15
    )
    logits.scatter_add_(
        -1,
        assigned[..., None],
        torch.full_like(assigned[..., None], 5, dtype=logits.dtype),
    )
    logits.requires_grad_()
    numeric = initialization(
        layout, codes, 1, len(data[0]), device, SEED + 726
    )
    numeric = numeric.detach().clone()
    optimizer = torch.optim.Adam([logits], lr=rate, fused=True)
    classes = torch.arange(layout.outputs, device=device)
    for step in range(1, CFG["outer_steps"] + 1):
        tasks, sx, sy, qx, qy = sample_support(data, shot, generator, classes)
        basis = logits.softmax(-1)
        state = numeric[tasks].detach().requires_grad_()
        for _ in range(3):
            weight = layer_weights(
                state[..., :codes], layout, None, None, basis
            )
            fit = losses(predict(sx, weight, state[..., codes:], layout), sy)
            gradient = torch.autograd.grad(
                fit.mean(-1).sum(), state, create_graph=True
            )[0]
            state = state - 0.3 * gradient
        weight = layer_weights(state[..., :codes], layout, None, None, basis)
        risk = losses(
            predict(qx, weight, state[..., codes:], layout), qy
        ).mean()
        entropy = -(basis * logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        (risk + 0.002 * entropy).backward()
        optimizer.step()
        numeric[tasks] = state.detach()
        if step in CFG["checkpoints"]:
            with torch.no_grad():
                probability = logits.softmax(-1)
                orthogonal, triangular = torch.linalg.qr(probability)
                if (triangular.diagonal(dim1=-2, dim2=-1).abs() < 1e-6).any():
                    raise ValueError("The Yeh basis lost rank")
            labels = np.stack(
                [budgeted_map(row) for row in probability.cpu().numpy()]
            )
            labels += np.arange(groups)[:, None] * rank
            index = labels.reshape(
                layout.sites, layout.channels, layout.fan_in
            ).transpose(0, 2, 1)
            yield step, orthogonal.cpu().numpy(), index.ravel()


def build(dataset, method, device):
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    codes, shot = config["codes"], config["shot"]
    folder = OUT / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "candidates.json"
    names = ("msr",) if method == "msr" else ("yeh_map", "yeh_soft")
    if manifest.exists():
        return {
            name: [folder / p for p in values]
            for name, values in json.loads(manifest.read_text()).items()
        }
    data = stratified_sources(dataset, device, layout)
    paths = {name: [] for name in names}
    if method == "msr":
        rank = codes // layout.hidden
        factors = (1, layout.sites if layout.sites > 1 else 4)
        for blocks, rate in product(factors, CFG["msr_rates"]):
            archived = (
                FAMILY_OUTPUT / dataset / "msr" / f"b{blocks}_r{rank}.npz"
            )
            if archived.exists() and rate == CFG["msr_rates"][0]:
                with np.load(archived, allow_pickle=False) as file:
                    basis = file["orthogonal"]
                path = folder / f"archive_b{blocks}_r{rank}.npz"
                atomic_npz(path, factor=basis)
                paths["msr"].append(path)
                continue
            for step, basis in train_msr(
                data, layout, blocks, rank, rate, shot, device
            ):
                path = folder / f"f{blocks}_rate{rate:g}_step{step}.npz"
                atomic_npz(path, factor=basis)
                paths["msr"].append(path)
                print(dataset, "MSR", blocks, rate, step, flush=True)
    else:
        parameters, numerics = dense_source(dataset, device)
        exemplar = parameters[numerics[:, 0].argmin(), 0, : layout.connections]
        groups = layout.sites if layout.sites > 1 else 16
        options = tuple(zip(CFG["yeh_rates"], (True, False), strict=True))
        for rate, warm in options:
            for step, basis, index in train_yeh(
                data, layout, exemplar, groups, codes, rate, warm, shot, device
            ):
                stem = f"{'dense' if warm else 'cold'}_rate{rate:g}_step{step}"
                for name, value in (
                    ("yeh_map", {"index": index}),
                    ("yeh_soft", {"basis": basis}),
                ):
                    path = folder / f"{name}_{stem}.npz"
                    atomic_npz(path, **value)
                    paths[name].append(path)
                print(dataset, "Yeh", rate, step, flush=True)
    freeze(
        manifest,
        {name: [p.name for p in models] for name, models in paths.items()},
    )
    freeze(
        folder / "model_hashes.json",
        {p.name: digest(p) for rows in paths.values() for p in rows},
    )
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=CONFIG["real"])
    parser.add_argument("methods", nargs="+", choices=("msr", "yeh"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    register()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    for method in args.methods:
        for name, bank in build(args.dataset, method, device).items():
            transfer(args.dataset, name, bank, device)

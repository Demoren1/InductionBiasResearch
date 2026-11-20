"""Refit the part of a Ward-initialized basis outside fixed group directions."""

import json
from itertools import product

import numpy as np
import torch

from experiments.asrr_benchmark_families_v1.classification import read_data
from experiments.asrr_benchmark_families_v1.gradient_groups import (
    orthogonal_groups,
)
from experiments.asrr_benchmark_families_v1.gradient_mixture import (
    clean_source_queries,
)
from experiments.asrr_benchmark_families_v1.linear import atomic_npz
from experiments.asrr_benchmark_families_v1.msr import (
    predictions,
    project,
    reparameterized_layout,
)
from experiments.paper_transfer_v2.bilevel import (
    sample_support,
    stratified_sources,
)
from torch.nn import functional as F
from experiments.paper_transfer_v2.common import CONFIG
from experiments.paper_transfer_v2.source import dense_source
from experiments.real_group_transfer_v1.optimization import (
    Layout,
    initialization,
    losses,
)
from experiments.real_group_transfer_v1.study import source_profiles
from experiments.sharing_structure_research_v1.blind_contract import save_json

from .bank import block_profiles
from .bases import factor_model, ward_profiles
from .common import OUT, SEED

CHECKPOINTS = (200, 400, 800)
BLOCKS = {
    "wisdm": (1, 4),
    "fc100": (1, 16),
    "aug_omniglot": (4, 8),
    "mnist_torus": (4, 16),
}


def train(dataset, original, initial_basis, anchored, device):
    shot = CONFIG["real"][dataset]["shot"]
    support = stratified_sources(dataset, device, original)
    clean = None
    if dataset == "aug_omniglot":
        clean = (
            torch.tensor(
                clean_source_queries()[:, -support[1].shape[1] :],
                device=device,
                dtype=torch.float32,
            )
            / 255
        )
    rng = torch.Generator(device=device).manual_seed(SEED + 1627)
    initial = torch.tensor(initial_basis, device=device, dtype=torch.float32)
    fixed = initial[..., :anchored]
    free = initial[..., anchored:].clone().requires_grad_()
    layout = reparameterized_layout(original, initial)
    output = torch.eye(layout.channels, device=device).repeat(
        layout.sites, 1, 1
    )
    state = initialization(layout, layout.connections, 1, 1, device, SEED + 11)
    parameters = state.detach().flatten().requires_grad_()
    rates = torch.full((4,), -0.4, device=device, requires_grad=True)
    optimizer = torch.optim.Adam(
        (free, parameters, rates), lr=0.002, fused=True
    )
    classes = torch.arange(original.outputs, device=device)
    for step in range(1, max(CHECKPOINTS) + 1):
        residual = free - fixed @ (fixed.mT @ free)
        variable = residual / residual.norm(dim=-2, keepdim=True).clamp_min(
            1e-6
        )
        basis = torch.cat((fixed, variable), dim=-1)
        ids, sx, sy, qx, qy = sample_support(support, shot, rng, classes)
        q = {"query": project(qx, basis, original)}
        if clean is not None:
            q["clean"] = project(clean[ids], basis, original)
        logits = predictions(
            project(sx, basis, original),
            sy,
            q,
            parameters,
            rates.sigmoid(),
            output,
            layout,
            (3,),
            higher=True,
        )[3]
        risk = losses(logits["query"][:, None], qy).mean()
        if clean is not None:
            clean_risk = losses(
                logits["clean"][:, None], support[2][ids]
            ).mean()
            risk = 0.5 * (risk + clean_risk)
        optimizer.zero_grad(set_to_none=True)
        risk.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            with torch.no_grad():
                residual = free - fixed @ (fixed.mT @ free)
                variable, triangular = torch.linalg.qr(residual)
                if (triangular.diagonal(dim1=-2, dim2=-1).abs() < 1e-6).any():
                    raise ValueError("The learned complement lost rank")
                candidate = torch.cat((fixed, variable), dim=-1)
            print(
                dataset,
                "bilevel",
                len(initial),
                anchored,
                step,
                float(risk.detach()),
                flush=True,
            )
            yield step, candidate.cpu().numpy()


def train_ridge(dataset, original, initial_basis, anchored, device):
    shot = CONFIG["real"][dataset]["shot"]
    source = stratified_sources(dataset, device, original)
    rng = torch.Generator(device=device).manual_seed(SEED + 493)
    initial = torch.tensor(initial_basis, device=device, dtype=torch.float32)
    fixed = initial[..., :anchored]
    variable = initial[..., anchored:].clone().requires_grad_()
    log_penalty = torch.tensor(-2.0, device=device, requires_grad=True)
    optimizer = torch.optim.Adam((variable, log_penalty), lr=0.003, fused=True)
    classes = torch.arange(original.outputs, device=device)
    for step in range(1, max(CHECKPOINTS) + 1):
        residual = variable - fixed @ (fixed.mT @ variable)
        normalized = residual / residual.norm(dim=-2, keepdim=True).clamp_min(
            1e-6
        )
        basis = torch.cat((fixed, normalized), dim=-1)
        _, sx, sy, qx, qy = sample_support(source, shot, rng, classes)
        xs = project(sx, basis, original).flatten(2)
        xq = project(qx, basis, original).flatten(2)
        gram = xs @ xs.mT
        energy = gram.diagonal(dim1=-2, dim2=-1).mean(-1)
        penalty = log_penalty.clamp(-8, 3).exp() * energy.clamp_min(1e-6)
        identity = torch.eye(xs.shape[1], device=device)
        labels = F.one_hot(sy, original.outputs).float() - 1 / original.outputs
        dual = torch.linalg.solve(
            gram + penalty[:, None, None] * identity, labels
        )
        scores = 5 * ((xq @ xs.mT) @ dual)
        risk = F.cross_entropy(scores.flatten(0, 1), qy.flatten())
        optimizer.zero_grad(set_to_none=True)
        risk.backward()
        optimizer.step()
        if step in CHECKPOINTS:
            with torch.no_grad():
                residual = variable - fixed @ (fixed.mT @ variable)
                orthogonal, triangular = torch.linalg.qr(residual)
                if (triangular.diagonal(dim1=-2, dim2=-1).abs() < 1e-6).any():
                    raise ValueError("The ridge-trained complement lost rank")
                candidate = torch.cat((fixed, orthogonal), dim=-1)
            print(
                dataset,
                "ridge-refit",
                len(initial),
                anchored,
                step,
                float(risk.detach()),
                flush=True,
            )
            yield step, candidate.cpu().numpy()


def build_meta(dataset, device, *, stage):
    folder = OUT / dataset / stage
    folder.mkdir(parents=True, exist_ok=True)
    manifest = folder / "bank.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    config = CONFIG["real"][dataset]
    layout = Layout(*config["layout"])
    rank = config["codes"] // layout.hidden
    weights, validation = dense_source(dataset, device)
    profiles = source_profiles(weights[np.argsort(validation[:, 0])], layout)
    kernel = read_data(OUT / dataset / "kernels.npz")["adapted"]
    result = {}
    anchor_count = rank // 3 if stage == "ridge" else rank // 2
    learner = train_ridge if stage == "ridge" else train
    for count, anchored in product(BLOCKS[dataset], (anchor_count, 0)):
        values = kernel.reshape(count, -1, layout.fan_in, layout.fan_in).sum(1)
        bases = []
        for profile, signal in zip(
            block_profiles(profiles, count), values, strict=True
        ):
            basis = orthogonal_groups(ward_profiles(profile, rank))
            scores = (basis * (signal @ basis)).sum(0)
            bases.append(basis[:, np.argsort(-scores)])
        initial = factor_model(bases, layout)["factor"]
        tag = "ridge" if stage == "ridge" else "bilevel"
        family = f"anchored_{tag}" if anchored else f"ward_warm_{tag}"
        for step, basis in learner(dataset, layout, initial, anchored, device):
            name = f"{family}_{count}_{step}"
            path = folder / f"{name}.npz"
            atomic_npz(path, factor=basis)
            result[name] = {"family": family, "path": str(path)}
    save_json(manifest, result)
    return result

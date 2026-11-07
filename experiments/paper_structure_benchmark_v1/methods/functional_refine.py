"""Source-pooled hard assignment refinement using functional gradient/GGN scores."""

import argparse
import json
import math
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)
from ..protocol.real_boundary import StudyBoundary
from ..fitting.adaptation import values as basis_values


def forward(x, index, parameter, codes, hidden, outputs):
    dictionary = torch.cat(
        (parameter[..., :1] * 0, parameter[..., :codes], -parameter[..., :codes]), -1
    )
    labels = torch.where(index < 0, codes - index, index).flatten(1)
    weight = dictionary.gather(2, labels[:, None].expand(-1, parameter.shape[1], -1))
    weight = weight.reshape(len(index), parameter.shape[1], x.shape[-1], hidden)
    pre = (
        torch.einsum("tnd,rtdh->rtnh", x, weight)
        + parameter[:, :, None, codes : codes + hidden]
    )
    head = parameter[..., codes + hidden : -outputs].reshape(
        len(index), parameter.shape[1], hidden, outputs
    )
    value = (
        torch.einsum("rtnh,rtho->rtno", pre.relu(), head) / math.sqrt(hidden)
        + parameter[:, :, None, -outputs:]
    )
    return value, pre, head, weight, dictionary


def risk(value, target, scale):
    return ((value - target[None]).square().mean((-1, -2)) * scale[None].square()).mean(
        -1
    )


def label_key(index):
    lookup, names, next_label = {0: 0}, {}, 1
    for value in index.ravel():
        absolute = abs(int(value))
        if absolute not in lookup:
            lookup[absolute] = next_label
            names[absolute] = 1 if value > 0 else -1
            next_label += 1
    normalized = np.asarray(
        [
            0
            if value == 0
            else lookup[abs(int(value))]
            * names[abs(int(value))]
            * (1 if value > 0 else -1)
            for value in index.ravel()
        ],
        dtype=np.int16,
    )
    return normalized.tobytes()


@torch.no_grad()
def discrete_step(index, parameter, x, y, scale, codes, hidden, outputs):
    value, pre, head, weight, dictionary = forward(
        x, index, parameter, codes, hidden, outputs
    )
    current = risk(value, y, scale)
    gate = (pre > 0).to(x.dtype)
    count = x.shape[1]
    signal = (
        2
        * torch.einsum("rtno,rtho->rtnh", value - y[None], head)
        * gate
        / (count * outputs * math.sqrt(hidden))
    )
    gradient = (
        torch.einsum("tnd,rtnh->rtdh", x, signal) * scale[None, :, None, None].square()
    )
    diagonal = (
        2
        * torch.einsum("tnd,rtnh,rth->rtdh", x.square(), gate, head.square().sum(-1))
        / (count * outputs * hidden)
    )
    diagonal *= scale[None, :, None, None].square()
    cost = torch.einsum("rtdh,rtk->rdhk", gradient - diagonal * weight, dictionary)
    cost += 0.5 * torch.einsum("rtdh,rtk->rdhk", diagonal, dictionary.square())
    cost /= len(x)
    categorical = torch.where(index < 0, codes - index, index)
    old_cost = cost.gather(-1, categorical[..., None])[..., 0]
    best_cost, best_label = cost.min(-1)
    replacement = torch.where(best_label > codes, codes - best_label, best_label)
    improvement = (old_cost - best_cost).flatten(1)
    valid = ((replacement != index) & ((old_cost - best_cost) > 1e-10)).flatten(1)
    improvement = torch.where(valid, improvement, -torch.inf)
    total = index[0].numel()
    top = improvement.topk(min(64, total), dim=1).indices
    replacement = replacement.flatten(1)
    flat = index.flatten(1)
    chosen, accepted = (
        flat.clone(),
        torch.zeros(len(index), dtype=torch.bool, device=x.device),
    )
    eligible = valid.sum(1)
    rows = torch.arange(len(index), device=x.device)[:, None]
    tolerance = 1e-7 * current.clamp_min(1)
    for maximum in (64, 32, 16, 8, 4, 2, 1):
        move = (
            torch.arange(top.shape[1], device=x.device)[None]
            < eligible.clamp_max(maximum)[:, None]
        )
        proposal = flat.clone()
        proposal[rows, top] = torch.where(move, replacement[rows, top], flat[rows, top])
        loss = risk(
            forward(x, proposal.reshape_as(index), parameter, codes, hidden, outputs)[
                0
            ],
            y,
            scale,
        )
        take = (~accepted) & (loss < current - tolerance)
        chosen[take] = proposal[take]
        accepted |= take
        if accepted.all():
            break
    # Exact single-edge alternatives can rescue a misleading diagonal quadratic ranking.
    if not accepted.all():
        positions = top[:, :16]
        dimensions, neurons = positions // hidden, positions % hidden
        selected_label = replacement[rows, positions]
        selected_label = torch.where(
            selected_label < 0, codes - selected_label, selected_label
        )
        new_weight = dictionary.gather(
            2, selected_label[:, None].expand(-1, len(x), -1)
        )
        old_weight = weight.flatten(2).gather(
            2, positions[:, None].expand(-1, len(x), -1)
        )
        delta = (new_weight - old_weight).permute(0, 2, 1)[..., None]
        inputs = x[:, :, dimensions].permute(2, 3, 0, 1)
        before = pre.permute(0, 3, 1, 2)[rows, neurons]
        coefficient = head.permute(0, 2, 1, 3)[rows, neurons]
        activation_change = (
            (before + delta * inputs).relu() - before.relu()
        ) / math.sqrt(hidden)
        output_change = activation_change[..., None] * coefficient[:, :, :, None]
        changed_loss = risk(value, y, scale)[:, None] + (
            (
                (
                    2 * (value - y[None])[:, None] * output_change
                    + output_change.square()
                ).mean((-1, -2))
            )
            * scale[None, None].square()
        ).mean(-1)
        allowed = valid.gather(1, positions)
        changed_loss = torch.where(allowed, changed_loss, torch.inf)
        best_loss, alternative = changed_loss.min(1)
        take = (~accepted) & (best_loss < current - tolerance)
        row = torch.arange(len(index), device=x.device)
        position = positions[row, alternative]
        chosen[row[take], position[take]] = replacement[row[take], position[take]]
    return chosen.reshape_as(index), (chosen != flat).sum(1)


def run(root, codes, device, steps):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "source" / "current_functional_proposals"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    if protocol["loss"] != "mse":
        raise ValueError("This registered ablation instantiates squared-loss GGN")
    h, o = protocol["hidden"], protocol["outputs"]
    base_folder = root / "fits" / f"current_k{codes}"
    baseline = np.load(base_folder / "selected.npz")
    base_index, initial = baseline["index"], baseline["source_parameters"].copy()
    amplitude = float(
        np.linalg.norm(baseline["basis"], axis=-1)[base_index != 0].mean()
    )
    initial[:, :codes] *= amplitude
    reference_name = json.loads((base_folder / "selected.json").read_text())[
        "candidate"
    ]
    origins = json.loads(
        (root / "source" / "current_proposals" / "candidates.json").read_text()
    )["candidates"]
    reference = next(row["anchor"] for row in origins if row["name"] == reference_name)
    support, selection = [
        dict(np.load(root / "public" / f"{name}.npz"))
        for name in ("support", "numerical")
    ]
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (support, "x"),
            (support, "y"),
            (selection, "x"),
            (selection, "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    center, scale = y.mean(1), y.std((1, 2), correction=0).clamp_min(1e-8)
    y, vy = (
        (y - center[:, None]) / scale[:, None, None],
        (vy - center[:, None]) / scale[:, None, None],
    )
    initial[:, codes + h : -o] /= scale.cpu().numpy()[:, None]
    initial[:, -o:] = (initial[:, -o:] - center.cpu().numpy()) / scale.cpu().numpy()[
        :, None
    ]
    rates = [0.0003, 0.001, 0.003, 0.01]
    parameter = torch.tensor(
        np.repeat(initial[None], len(rates), 0),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    index = torch.tensor(
        np.repeat(base_index[None], len(rates), 0), device=device, dtype=torch.long
    )
    optimizer = ScenarioAdam(parameter, rates)
    generator = torch.Generator(device=device).manual_seed(775003)
    with torch.no_grad():
        reference_value = basis_values(
            vx,
            torch.tensor(baseline["basis"], device=device),
            torch.tensor(baseline["source_parameters"], device=device)[None],
            h,
            o,
        )
        reference_risk = float(
            (reference_value - (vy * scale[:, None, None] + center[:, None])[None])
            .square()
            .mean()
        )
        decoded_risk = float(
            risk(forward(vx, index[:1], parameter[:1], codes, h, o)[0], vy, scale)[0]
        )
    if abs(reference_risk - decoded_risk) > 5e-5 * max(1, reference_risk):
        raise RuntimeError(
            "The learned starting predictor must be preserved by the basis conversion"
        )
    save_json(
        folder / "registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "base_model_sha256": digest(base_folder / "selected.npz"),
            "trigger": "Paired development MNIST error is11.1% above Dense, worse29/32tasks. No target query responses opened.",
            "codes": codes,
            "initial_codebook_scale": amplitude,
            "initial_reference_source_H0_loss": reference_risk,
            "initial_decoded_source_H0_loss": decoded_risk,
            "numeric_rates": rates,
            "outer_steps": steps,
            "numeric_steps_per_outer": 20,
            "structure_score": "Sum over source tasks of g*delta+.5*diag(GGN)*delta^2; all labels including zero permitted",
            "acceptance": "True source H0 functional loss decreases at fixed numbers; try at most64/32/16/8/4/2/1 entries then16 exact single-edge alternatives",
            "selection": "H0-diverse trajectory proposals plus unchanged baseline; identical common source numerical refitting and source M selection follow in separate jobs",
            "ground_truth": "No true parameters, masks, groups or coordinate roles; only learned source models and observations",
        },
    )
    shutil.copy2(__file__, folder / "executed_code.py")
    torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    checkpoints, indices, losses, training, movement = [], [], [], [], []
    status = "failed"
    try:
        with counter:
            for step in range(steps + 1):
                if step % 10 == 0 or step == steps:
                    with torch.no_grad():
                        train = risk(
                            forward(x, index, parameter, codes, h, o)[0], y, scale
                        )
                        score = risk(
                            forward(vx, index, parameter, codes, h, o)[0], vy, scale
                        )
                        checkpoints.append(step)
                        indices.append(index.cpu().numpy().astype(np.int16))
                        losses.append(score.cpu().numpy())
                        training.append(train.cpu().numpy())
                        print(
                            json.dumps(
                                {
                                    "functional_outer": step,
                                    "source_H0_min": float(score.min()),
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                if step == steps:
                    break
                for _ in range(20):
                    positions = torch.randint(
                        x.shape[1], (512,), generator=generator, device=device
                    )
                    value = forward(x[:, positions], index, parameter, codes, h, o)[0]
                    objective = (
                        (value - y[None, :, positions]).square().mean((-1, -2)).sum()
                    )
                    optimizer.step(
                        parameter, torch.autograd.grad(objective, parameter)[0]
                    )
                index, changes = discrete_step(
                    index, parameter.detach(), vx, vy, scale, codes, h, o
                )
                movement.append(changes.cpu().numpy())
        history = np.stack(losses)
        requested = [(0, 0)] + [
            (int(history[:, row].argmin()), row) for row in range(len(rates))
        ]
        for milestone in (100, 200, steps):
            checkpoint = int(np.argmin(np.abs(np.asarray(checkpoints) - milestone)))
            requested.append((checkpoint, int(history[checkpoint].argmin())))
        proposals, metadata, seen = {}, [], set()
        for checkpoint, row in requested:
            proposal = indices[checkpoint][row]
            key = label_key(proposal)
            if key in seen:
                continue
            seen.add(key)
            name = f"candidate_{len(metadata):03d}"
            proposals[name] = proposal
            metadata.append(
                {
                    "name": name,
                    "codes": codes,
                    "anchor": int(reference),
                    "outer_step": int(checkpoints[checkpoint]),
                    "numeric_rate": rates[row],
                    "source_H0_loss": float(history[checkpoint, row]),
                    "baseline": checkpoint == 0,
                }
            )
        np.savez_compressed(folder / "candidates.npz", **proposals)
        save_json(folder / "candidates.json", {"candidates": metadata})
        np.savez_compressed(
            folder / "curves.npz",
            steps=np.asarray(checkpoints),
            indices=np.stack(indices),
            train_history=np.stack(training),
            validation_history=history,
            changed_positions=np.stack(movement),
        )
        status = "complete"
    finally:
        torch.cuda.synchronize(device)
        save_json(
            folder / "metrics.json",
            {
                "status": status,
                "wall_seconds": time.perf_counter() - started,
                "torch_operator_flops": counter.get_total_flops(),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--codes", type=int, default=128)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.codes, torch.device(args.device), args.steps)


if __name__ == "__main__":
    main()

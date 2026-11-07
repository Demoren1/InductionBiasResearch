"""Batch independent G23 fixed-I codebook fits without changing their objective."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_data import load_source
from .shared_structure_numeric import fresh_codes


def features(x, indices, codes):
    index = torch.as_tensor(indices, device=x.device).long()
    basis = F.one_hot(index.abs(), codes + 1)[..., 1:].to(x.dtype)
    basis *= index.sign()[..., None]
    return torch.matmul(x, basis.flatten(2)).reshape(
        len(indices), len(x), index.shape[-1], codes
    )


@torch.no_grad()
def packed_losses(feature, y, code, tasks, plan):
    total = code.new_zeros(code.shape[:-1])
    for begin in range(0, y.shape[-1], plan["numeric_chunk"]):
        end = min(y.shape[-1], begin + plan["numeric_chunk"])
        activation = torch.bmm(
            code, feature[:, begin:end].flatten(1, 2).transpose(-1, -2)
        ).reshape(*code.shape[:-1], end - begin, plan["hidden"])
        residual = (activation + plan["bias"]).relu().sum(-1) / math.sqrt(
            plan["hidden"]
        ) - y[tasks, begin:end]
        total += residual.square().sum(-1)
    return total / y.shape[-1]


@torch.no_grad()
def packed_statistics(feature, y, code, tasks, plan):
    total = code.new_zeros(code.shape[:-1])
    gram = code.new_zeros((*code.shape[:-1], plan["codes"], plan["codes"]))
    score = torch.zeros_like(code)
    for begin in range(0, y.shape[-1], plan["numeric_chunk"]):
        end = min(y.shape[-1], begin + plan["numeric_chunk"])
        local = feature[:, begin:end]
        activation = torch.bmm(code, local.flatten(1, 2).transpose(-1, -2)).reshape(
            *code.shape[:-1], end - begin, plan["hidden"]
        )
        activation += plan["bias"]
        jacobian = (
            torch.einsum("bpnh,bnhk->bpnk", (activation > 0).to(code.dtype), local)
            / math.sqrt(plan["hidden"])
        ).contiguous()
        residual = y[tasks, begin:end] - activation.relu().sum(-1) / math.sqrt(
            plan["hidden"]
        )
        gram += torch.matmul(jacobian.transpose(-1, -2), jacobian)
        score += (jacobian * residual[..., None]).sum(-2)
        total += residual.square().sum(-1)
    return total / y.shape[-1], gram / y.shape[-1], score / y.shape[-1]


def pack_models(code, active):
    counts = active.sum(-1)
    width = int(counts.max())
    slots = active.to(torch.int8).argsort(dim=-1, descending=True, stable=True)[
        :, :width
    ]
    valid = torch.arange(width, device=code.device)[None] < counts[:, None]
    selected = code.gather(1, slots[..., None].expand(-1, -1, code.shape[-1]))
    return selected, slots, valid


@torch.no_grad()
def fit_codes(indices, x, y, vx, vy, initial, plan, *, max_steps=None):
    """Fit B indexators x T tasks x S signed starts; select each independently."""
    started = time.monotonic()
    feature, validation_feature = (
        features(x, indices, plan["codes"]),
        features(vx, indices, plan["codes"]),
    )
    code = initial[None].expand(len(indices), *initial.shape).clone()
    code.clamp_(-plan["axis_code_max"], plan["axis_code_max"])
    max_steps = plan["numeric_code_steps"] if max_steps is None else max_steps
    best_score = code.new_full(code.shape[:-1], torch.inf)
    best_code = code.clone()
    best_step = torch.zeros(code.shape[:-1], device=code.device, dtype=torch.long)
    alive = torch.ones_like(best_step, dtype=torch.bool)
    flat_code = code.flatten(1, 2)
    train = code.new_zeros(code.shape[:-1])
    validation = torch.zeros_like(train)
    identity = torch.eye(plan["codes"], device=code.device, dtype=code.dtype)
    training_history, validation_history, alive_history = [], [], []
    for step in range(max_steps + 1):
        if alive.any():
            packed, slots, valid = pack_models(flat_code, alive.flatten(1, 2))
            tasks = slots // initial.shape[1]
            if step < max_steps:
                train_packed, gram, score = packed_statistics(
                    feature, y, packed, tasks, plan
                )
            else:
                train_packed = packed_losses(feature, y, packed, tasks, plan)
            validation_packed = packed_losses(
                validation_feature, vy, packed, tasks, plan
            )
            for value, packed_value in (
                (train, train_packed),
                (validation, validation_packed),
            ):
                flat = value.flatten(1, 2)
                flat.scatter_(
                    1, slots, torch.where(valid, packed_value, flat.gather(1, slots))
                )
        better = validation < best_score
        best_score[better], best_code[better], best_step[better] = (
            validation[better],
            code[better],
            step,
        )
        training_history.append(train.clone())
        validation_history.append(validation.clone())
        alive_history.append(alive.sum(-1))
        if step == max_steps or not alive.any():
            break
        scale = torch.linalg.eigvalsh(gram)[..., -1].clamp_min(1e-12)
        packed_direction = torch.linalg.solve(
            gram + plan["numeric_ridge"] * scale[..., None, None] * identity,
            score[..., None],
        )[..., 0]
        direction = torch.zeros_like(flat_code)
        direction.scatter_(
            1,
            slots[..., None].expand(-1, -1, plan["codes"]),
            torch.where(valid[..., None], packed_direction, 0),
        )
        accepted, proposed = ~alive, code.clone()
        accepted_flat, proposed_flat = accepted.flatten(1, 2), proposed.flatten(1, 2)
        for exponent in range(plan["numeric_line_search"]):
            candidate = flat_code + 2.0**-exponent * direction
            pending, positions, available = pack_models(candidate, ~accepted_flat)
            value = packed_losses(
                feature, y, pending, positions // initial.shape[1], plan
            )
            take = available & (pending.abs().amax(-1) <= plan["axis_code_max"])
            take &= value < train.flatten(1, 2).gather(1, positions)
            index = positions[..., None].expand(-1, -1, plan["codes"])
            proposed_flat.scatter_(
                1,
                index,
                torch.where(take[..., None], pending, proposed_flat.gather(1, index)),
            )
            accepted_flat.scatter_(
                1, positions, accepted_flat.gather(1, positions) | take
            )
            if accepted.all():
                break
        alive &= accepted
        code = proposed
        flat_code = code.flatten(1, 2)
    if not torch.isfinite(best_code).all() or not torch.isfinite(best_score).all():
        raise FloatingPointError("Nonfinite batched signed-C fit")
    best_score_cpu, best_step_cpu = best_score.cpu(), best_step.cpu()
    training_cpu = torch.stack(training_history).cpu()
    validation_cpu = torch.stack(validation_history).cpu()
    alive_cpu = torch.stack(alive_history).cpu()
    winners = best_score.argmin(-1)
    selected = best_code.gather(
        2, winners[..., None, None].expand(*winners.shape, 1, plan["codes"])
    ).squeeze(2)
    winners_cpu = winners.cpu()
    elapsed = time.monotonic() - started
    histories = []
    for batch in range(len(indices)):
        tasks = []
        for task in range(y.shape[0]):
            zero = torch.nonzero(alive_cpu[:, batch, task] == 0).flatten()
            last = int(zero[0]) if len(zero) else len(training_cpu) - 1
            winner = int(winners_cpu[batch, task])
            tasks.append(
                {
                    "selected_start": winner,
                    "selected_step": int(best_step_cpu[batch, task, winner]),
                    "validation_mse": float(best_score_cpu[batch, task, winner]),
                    "history": [
                        {
                            "step": step,
                            "train_mse": training_cpu[step, batch, task].tolist(),
                            "validation_mse": validation_cpu[
                                step, batch, task
                            ].tolist(),
                            "active_starts": int(alive_cpu[step, batch, task]),
                        }
                        for step in range(last + 1)
                    ],
                    "best_validation_per_start": best_score_cpu[batch, task].tolist(),
                    "best_step_per_start": best_step_cpu[batch, task].tolist(),
                    "seconds": elapsed / (len(indices) * y.shape[0]),
                    "timing_kind": "amortized over independent indexators/tasks in one batch",
                }
            )
        histories.append(tasks)
    return selected, histories, elapsed


def fit_job(root, plan, job_id, device, batch_size):
    common = root / "common"
    phase_hash = digest(common / "protocol.json")
    path = common / "banks" / f"job_{job_id:03d}"
    bank = json.loads(path.with_suffix(".json").read_text())
    destination = common / "fits" / f"job_{job_id:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Common fit job already completed")
    destination.mkdir(exist_ok=True)
    with np.load(path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    family_id, sigma = bank["family_id"], bank["noise"]
    family = plan["families"][family_id]["family"]
    observed = [
        load_source(root, plan, family_id, task, sigma, device)
        for task in range(plan["sources"])
    ]
    x, vx = observed[0]["train_x"], observed[0]["fit_validation_x"]
    y = torch.stack([data[f"train_y_{sigma}"] for data in observed])
    vy = torch.stack([data[f"fit_validation_y_{sigma}"] for data in observed])
    initial = torch.stack(
        [fresh_codes(plan, family, task, device) for task in range(plan["sources"])]
    )
    missing = []
    for number, candidate in enumerate(bank["candidates"]):
        output = destination / candidate["signature"]
        if output.with_suffix(".json").exists():
            previous = json.loads(output.with_suffix(".json").read_text())
            if previous["phase_protocol_sha256"] != phase_hash or previous[
                "arrays_sha256"
            ] != digest(output.with_suffix(".npz")):
                raise RuntimeError(
                    "Existing numeric fit changed or belongs to another phase"
                )
        else:
            missing.append(number)
    for begin in range(0, len(missing), batch_size):
        numbers = missing[begin : begin + batch_size]
        codes, histories, seconds = fit_codes(
            indices[numbers], x, y, vx, vy, initial, plan
        )
        for local, number in enumerate(numbers):
            signature = bank["candidates"][number]["signature"]
            output = destination / signature
            np.savez_compressed(
                output.with_suffix(".npz"),
                index=indices[number],
                codes=codes[local].cpu().numpy(),
            )
            save_json(
                output.with_suffix(".json"),
                {
                    "job_id": job_id,
                    "candidate": number,
                    "signature": signature,
                    "numeric_fits": histories[local],
                    "meta_labels_opened": False,
                    "phase_protocol_sha256": phase_hash,
                    "arrays_sha256": digest(output.with_suffix(".npz")),
                    "seconds": seconds / len(numbers),
                    "batch": {
                        "candidates": numbers,
                        "elapsed_seconds": seconds,
                        "candidate_seconds_are_amortized": True,
                        "execution": "same independent GN and per-start backtracking/checkpoint, vectorized I/tasks/starts",
                    },
                },
            )
        print(
            json.dumps(
                {
                    "common_job": job_id,
                    "candidates": numbers,
                    "of": len(indices),
                    "batch_seconds": seconds,
                    "source_fit_validation": [
                        [r["validation_mse"] for r in task] for task in histories
                    ],
                }
            ),
            flush=True,
        )
    artefacts = [
        str((destination / candidate["signature"]).relative_to(common))
        for candidate in bank["candidates"]
    ]
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": job_id,
            "bank_sha256": digest(path.with_suffix(".json")),
            "fit_artefacts": artefacts,
            "phase_protocol_sha256": phase_hash,
            "meta_labels_opened": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}:
        raise RuntimeError("Explicit GPU0--3 required; GPUs6/7 are reserved")
    torch.set_num_threads(1)
    plan = json.loads((args.root / "protocol.json").read_text())
    phase = json.loads((args.root / "common/protocol.json").read_text())
    if phase["batched_numeric_execution"]["code_sha256"] != digest(Path(__file__)):
        raise RuntimeError("Batched numeric code differs from registered execution")
    for job_id in range(
        args.shard, len(plan["families"]) * len(plan["noise"]), args.shards
    ):
        fit_job(args.root, plan, job_id, args.device, args.batch_size)


if __name__ == "__main__":
    main()

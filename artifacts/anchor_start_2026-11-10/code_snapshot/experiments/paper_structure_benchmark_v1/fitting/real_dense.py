"""Pack real-data source tasks/restarts without repeating the input tensor."""

import argparse
import itertools
import json
import math
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from ..protocol.real_boundary import StudyBoundary
from .adaptation import losses
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.sharing_structure_research_v1.shared_structure_bilevel_optimization import (
    ScenarioAdam,
)


def predict(x, parameter, dimensions, hidden, outputs):
    boundary = dimensions * hidden
    weight = parameter[..., :boundary].reshape(*parameter.shape[:2], dimensions, hidden)
    bias = parameter[..., boundary : boundary + hidden]
    head = parameter[..., boundary + hidden : -outputs].reshape(
        *parameter.shape[:2], hidden, outputs
    )
    activation = (torch.einsum("tnd,rtdh->rtnh", x, weight) + bias[:, :, None]).relu()
    return (
        torch.einsum("rtnh,rtho->rtno", activation, head) / math.sqrt(hidden)
        + parameter[:, :, None, -outputs:]
    )


def cross_entropy(value, target):
    return -(value.log_softmax(-1) * target[None]).sum(-1).mean(-1)


def run(root, device, steps):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    folder = root / "source" / "dense"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o = [protocol[key] for key in ("inputs", "hidden", "outputs")]
    loss_kind = protocol["loss"]
    rows = [
        {"lr": lr, "weight_decay": decay, "restart": restart}
        for lr, decay, restart in itertools.product(
            (0.001, 0.003, 0.01), (0.0, 1e-4), range(2)
        )
    ]
    save_json(
        folder / "registered.json",
        {
            "settings": rows,
            "steps": steps,
            "checkpoint_every": 25,
            "source_sha256": {
                name: digest(root / "public" / f"{name}.npz")
                for name in ("support", "numerical")
            },
            "code_sha256": digest(Path(__file__)),
            "input_repetition": False,
            "used_panels": ["support", "numerical"],
        },
    )
    shutil.copy2(__file__, folder / "executed_code.py")
    support = dict(np.load(root / "public" / "support.npz"))
    numerical = dict(np.load(root / "public" / "numerical.npz"))
    x, y, vx, vy = [
        torch.tensor(panel[key], device=device, dtype=torch.float32)
        for panel, key in (
            (support, "x"),
            (support, "y"),
            (numerical, "x"),
            (numerical, "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    tasks, count = len(x), len(rows)
    generator = torch.Generator(device=device).manual_seed(753001)
    length = d * h + h + h * o + o
    parameter = torch.randn((count, tasks, length), generator=generator, device=device)
    parameter[..., : d * h] *= math.sqrt(2 / d)
    parameter[..., d * h : d * h + h] *= 0.1
    parameter[..., -o:] = 0
    parameter.requires_grad_()
    optimizer = ScenarioAdam(parameter, [row["lr"] for row in rows])
    decay = parameter.new_tensor([row["weight_decay"] for row in rows])[:, None, None]
    best = parameter.new_full((count, tasks), torch.inf)
    selected = parameter.detach().clone()
    selected_step = torch.zeros((count, tasks), device=device, dtype=torch.long)
    history = {"steps": [], "train": [], "validation": [], "recall": []}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    counter = FlopCounterMode(display=False)
    status = "failed"
    try:
        with counter:
            for step in range(steps + 1):
                if step % 25 == 0 or step == steps:
                    with torch.no_grad():
                        value = predict(x, parameter, d, h, o)
                        validation_value = predict(vx, parameter, d, h, o)
                        train = losses(value, y, loss_kind)
                        score = losses(validation_value, vy, loss_kind)
                        recall = (
                            (
                                (validation_value.argmax(-1) == vy.argmax(-1)[None])
                                .float()
                                .mean(-1)
                            )
                            if loss_kind == "cross_entropy"
                            else score.sqrt()
                        )
                        improved = score < best
                        best[improved], selected[improved], selected_step[improved] = (
                            score[improved],
                            parameter[improved],
                            step,
                        )
                        history["steps"].append(step)
                        history["train"].append(train.cpu().numpy())
                        history["validation"].append(score.cpu().numpy())
                        history["recall"].append(recall.cpu().numpy())
                    if step % 250 == 0:
                        print(
                            json.dumps(
                                {
                                    "dense_step": step,
                                    "median_numerical_accuracy_or_rmse": float(
                                        recall.median()
                                    ),
                                    "seconds": time.perf_counter() - started,
                                }
                            ),
                            flush=True,
                        )
                if step == steps:
                    break
                positions = torch.randint(
                    x.shape[1],
                    (min(512, x.shape[1]),),
                    device=device,
                    generator=generator,
                )
                loss = losses(
                    predict(x[:, positions], parameter, d, h, o),
                    y[:, positions],
                    loss_kind,
                ).sum()
                gradient = (
                    torch.autograd.grad(loss, parameter)[0] + decay * parameter.detach()
                )
                optimizer.step(parameter, gradient)
        torch.cuda.synchronize(device)
        rank = best.argsort(0)[:4].T
        task = torch.arange(tasks, device=device)[:, None]
        ensemble = selected[rank, task]
        chosen = selected[best.argmin(0), torch.arange(tasks, device=device)]
        np.savez_compressed(
            folder / "models.npz",
            parameters=chosen.cpu().numpy(),
            ensemble_parameters=ensemble.cpu().numpy(),
            numerical_cross_entropy=best.cpu().numpy(),
            numerical_loss=best.cpu().numpy(),
            selected_step=selected_step.cpu().numpy(),
            steps=np.array(history["steps"]),
            train_history=np.stack(history["train"]),
            validation_history=np.stack(history["validation"]),
            validation_recall=np.stack(history["recall"]),
        )
        status = "complete"
    finally:
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
                "models_sha256": digest(folder / "models.npz")
                if status == "complete"
                else None,
                "target_access": False,
                "structure_selection_access": False,
            },
        )
        boundary.save(folder / "access_receipt.json", status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()
    run(args.root.resolve(), torch.device(args.device), args.steps)


if __name__ == "__main__":
    main()

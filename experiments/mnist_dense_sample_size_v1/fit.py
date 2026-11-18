"""Fit the registered Dense sample-size conditions and retain late states."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    cross_entropy,
    predict,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1 import (
    shared_structure_bilevel_optimization as bilevel,
)
from experiments.sharing_structure_research_v1.blind_contract import save_json


def run(root, device):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root)
    boundary.install()
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o = (protocol[k] for k in ("inputs", "hidden", "outputs"))
    rows = protocol["settings"]
    with np.load(root / "public/H.npz") as panel:
        vx = torch.tensor(panel["x"], device=device, dtype=torch.float32) / 255
        vy = torch.tensor(panel["y"], device=device, dtype=torch.float32)
    for count in protocol["training_examples_per_class"]:
        folder = root / "fits" / f"shots_{count}"
        folder.mkdir(exist_ok=False)
        with np.load(root / "public" / f"support_{count}.npz") as panel:
            x = torch.tensor(panel["x"], device=device, dtype=torch.float32) / 255
            y = torch.tensor(panel["y"], device=device, dtype=torch.float32)
        generator = torch.Generator(device=device).manual_seed(
            protocol["initialization_seed"]
        )
        length = d * h + h + h * o + o
        parameter = torch.randn(
            (len(rows), len(x), length), generator=generator, device=device
        )
        parameter[..., : d * h] *= math.sqrt(2 / d)
        parameter[..., d * h : d * h + h] *= 0.1
        parameter[..., -o:] = 0
        parameter.requires_grad_()
        optimizer = bilevel.ScenarioAdam(parameter, [row["lr"] for row in rows])
        decay = parameter.new_tensor([r["weight_decay"] for r in rows])[:, None, None]
        best = parameter.new_full(parameter.shape[:2], torch.inf)
        selected = parameter.detach().clone()
        selected_step = torch.zeros_like(best, dtype=torch.long)
        history = {
            key: [] for key in ("steps", "train", "H", "train_accuracy", "H_accuracy")
        }
        snapshots = []
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for step in range(protocol["steps"] + 1):
            if step in protocol["snapshots"]:
                snapshots.append(parameter.detach().cpu().numpy().copy())
            if step % protocol["checkpoint_every"] == 0:
                with torch.no_grad():
                    train_logits = predict(x, parameter, d, h, o)
                    H_logits = predict(vx, parameter, d, h, o)
                    train = cross_entropy(train_logits, y)
                    score = cross_entropy(H_logits, vy)
                    improved = score < best
                    best[improved] = score[improved]
                    selected[improved] = parameter[improved]
                    selected_step[improved] = step
                    history["steps"].append(step)
                    history["train"].append(train.cpu().numpy())
                    history["H"].append(score.cpu().numpy())
                    for key, logits, labels in (
                        ("train_accuracy", train_logits, y),
                        ("H_accuracy", H_logits, vy),
                    ):
                        history[key].append(
                            (logits.argmax(-1) == labels.argmax(-1)[None])
                            .float()
                            .mean(-1)
                            .cpu()
                            .numpy()
                        )
            if step % 500 == 0:
                print(
                    json.dumps(
                        {
                            "shots": count,
                            "step": step,
                            "median_H_best_ce": float(best.min(0).values.median()),
                            "seconds": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
            if step == protocol["steps"]:
                break
            positions = torch.randint(
                x.shape[1],
                (min(512, x.shape[1]),),
                generator=generator,
                device=device,
            )
            objective = cross_entropy(
                predict(x[:, positions], parameter, d, h, o), y[:, positions]
            ).sum()
            gradient = torch.autograd.grad(objective, parameter)[0]
            optimizer.step(parameter, gradient + decay * parameter.detach())
        torch.cuda.synchronize(device)
        np.savez_compressed(
            folder / "models.npz",
            selected_parameters=selected.detach().cpu().numpy(),
            final_parameters=parameter.detach().cpu().numpy(),
            selected_step=selected_step.cpu().numpy(),
            best_H_ce=best.cpu().numpy(),
            snapshot_steps=np.asarray(protocol["snapshots"]),
            snapshots=np.stack(snapshots),
            **{key: np.stack(value) for key, value in history.items()},
        )
        save_json(
            folder / "metrics.json",
            {
                "status": "complete",
                "shots": count,
                "seconds": time.perf_counter() - started,
                "target_access": False,
            },
        )
    boundary.save(root / "fit_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()

"""Prepare shared Dense observations with Adam and full-gradient polishing."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    cross_entropy,
    predict,
)
from experiments.paper_structure_benchmark_v1.fitting.real_dense import (
    run as fit_dense,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def run(root, output, device, *, initial_hidden=None):
    boundary = StudyBoundary(root)
    boundary.install()
    protocol = json.loads((root / "protocol.json").read_text())
    d, h, o = [protocol[k] for k in ("inputs", "hidden", "outputs")]
    folder = root / "source/dense"
    if (folder / "adam_models.npz").exists():
        metrics = json.loads((folder / "metrics.json").read_text())
        if (
            metrics["status"] != "complete"
            or digest(folder / "adam_models.npz") != metrics["models_sha256"]
        ):
            raise RuntimeError("Cannot resume from an incomplete Adam checkpoint")
        save_json(
            output / "source_layout_fix.json",
            {
                "reason": "L-BFGS requires contiguous parameter gradients",
                "adam_retrained": False,
                "adam_models_sha256": digest(folder / "adam_models.npz"),
                "source_code_sha256": digest(Path(__file__)),
            },
        )
    else:
        fit_dense(root, device, 2500, initial_hidden=initial_hidden)
        shutil.copy2(folder / "models.npz", folder / "adam_models.npz")
    with np.load(folder / "adam_models.npz") as archive:
        original = dict(archive)
    settings = json.loads((folder / "registered.json").read_text())["settings"]
    ranking = original["numerical_loss"].argsort(0)[:4]
    states = np.ascontiguousarray(original["ensemble_parameters"].transpose(1, 0, 2))
    parameter = torch.tensor(
        states, device=device, dtype=torch.float64, requires_grad=True
    )
    x, y, vx, vy = [
        torch.tensor(
            np.load(root / "public" / f"{panel}.npz")[key],
            device=device,
            dtype=torch.float64,
        )
        for panel, key in (
            ("support", "x"),
            ("support", "y"),
            ("numerical", "x"),
            ("numerical", "y"),
        )
    ]
    x, vx = x / 255, vx / 255
    decay = parameter.new_tensor(
        [
            [settings[int(ranking[r, task])]["weight_decay"] for task in range(len(x))]
            for r in range(4)
        ]
    )[..., None]
    optimizer = torch.optim.LBFGS(
        [parameter],
        max_iter=20,
        max_eval=40,
        history_size=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    best = torch.tensor(np.sort(original["numerical_loss"], axis=0)[:4], device=device)
    chosen = parameter.detach().clone()
    history = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(predict(x, parameter, d, h, o), y).sum()
        loss = loss + (decay * parameter.square()).sum() / 2
        loss.backward()
        return loss

    for block in range(11):
        if block:
            optimizer.step(closure)
        objective = float(closure().detach())
        with torch.no_grad():
            train = cross_entropy(predict(x, parameter, d, h, o), y)
            validation = cross_entropy(predict(vx, parameter, d, h, o), vy)
            improved = validation < best
            best[improved] = validation[improved].to(best.dtype)
            chosen[improved] = parameter[improved]
        history.append(
            {
                "iterations": block * 20,
                "objective": objective,
                "train_ce": train.cpu().numpy().tolist(),
                "H_ce": validation.cpu().numpy().tolist(),
                "gradient_max": float(parameter.grad.abs().max()),
            }
        )
        print(json.dumps({"polish_block": block, "objective": objective}), flush=True)
    rank = best.argsort(0)
    tasks = torch.arange(len(x), device=device)
    selected = chosen[rank[0], tasks]
    ensemble = chosen[rank.T, tasks[:, None]]
    with torch.no_grad():
        train_logits = predict(x, selected[None], d, h, o)
        H_logits = predict(vx, selected[None], d, h, o)
        quality = {
            "train_ce": cross_entropy(train_logits, y)[0].cpu().numpy().tolist(),
            "H_ce": cross_entropy(H_logits, vy)[0].cpu().numpy().tolist(),
            "train_accuracy": (train_logits[0].argmax(-1) == y.argmax(-1))
            .double()
            .mean(-1)
            .cpu()
            .numpy()
            .tolist(),
            "H_accuracy": (H_logits[0].argmax(-1) == vy.argmax(-1))
            .double()
            .mean(-1)
            .cpu()
            .numpy()
            .tolist(),
            "polish_gradient_tolerance_reached": history[-1]["gradient_max"] <= 1e-7,
            "preparation": (
                "Twelve Adam fits and four H-retained "
                "full-gradient continuations per task"
            ),
        }
    np.savez_compressed(
        folder / "models.npz",
        parameters=selected.float().cpu().numpy(),
        ensemble_parameters=ensemble.float().cpu().numpy(),
        numerical_loss=best.cpu().numpy(),
        numerical_cross_entropy=best.cpu().numpy(),
    )
    np.savez_compressed(
        folder / "polish_states.npz", final=parameter.detach().cpu().numpy()
    )
    save_json(
        folder / "polish.json",
        {
            "history": history,
            "quality": quality,
            "models_sha256": digest(folder / "models.npz"),
        },
    )
    save_json(output / "dense_quality.json", quality)
    np.savez_compressed(
        output / "dense_curves.npz",
        **{
            key: original[key]
            for key in (
                "steps",
                "train_history",
                "validation_history",
                "numerical_loss",
                "selected_step",
            )
        },
    )
    pair = output / "pair"
    (pair / "code_snapshot").mkdir(parents=True)
    save_json(
        pair / "registration.json",
        {
            "previous_dense_sha256": digest(folder / "models.npz"),
            "meaning": (
                "Compatibility key: same Dense preparation as this run's G34; "
                "no historical comparator"
            ),
        },
    )
    boundary.save(folder / "polish_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), torch.device(args.device))


if __name__ == "__main__":
    main()

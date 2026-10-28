"""Source-only controls for the scrambled-filter collection pilot."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scrambled_pattern import predict


def dense_predict(x, parameters):
    weight, bias, readout, offset = parameters
    return (((x @ weight) + bias[:, None]).relu() * readout[:, None]).sum(-1) + offset[
        :, None
    ]


def main():
    out = Path("experiments/structure_search_v2/runs/scrambled_pattern_source_pilot_v1")
    shutil.copy2(__file__, out / "controls_source.py")
    spec = json.loads((out / "protocol.json").read_text())
    torch.set_num_threads(4)
    device = torch.device("cuda")
    data = np.load(out / "source_data.npz")
    x = torch.tensor(data["x"], device=device)
    y = torch.tensor(data["y"], device=device)
    tasks = len(y)
    variance = y[:, spec["support"] :].var(1, unbiased=False).clamp_min(1e-8)
    records = []
    # Dense source calibration: independent numerical weights and three rates.
    rates = [0.001, 0.003, 0.01]
    replicas = 4
    count = tasks * replicas * len(rates)
    labels = y.repeat_interleave(replicas * len(rates), 0)
    scales = variance.repeat_interleave(replicas * len(rates))
    torch.manual_seed(spec["seed"] + 2)
    parameters = [
        torch.randn(count, 64, 32, device=device) * 0.1,
        torch.zeros(count, 32, device=device),
        torch.randn(count, 32, device=device) * 0.1,
        torch.zeros(count, device=device),
    ]
    parameters = [p.requires_grad_() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1.0)
    lr = torch.tensor(rates, device=device).repeat_interleave(replicas).repeat(tasks)
    best = torch.full((count,), torch.inf, device=device)
    for step in range(6145):
        if step % 256 == 0:
            with torch.no_grad():
                error = (
                    dense_predict(x[spec["support"] :], parameters)
                    - labels[:, spec["support"] :]
                ).square().mean(1) / scales
                best = torch.minimum(best, error)
                record = {
                    "method": "dense_adam",
                    "step": step,
                    "best_mean_nmse": float(
                        best.reshape(tasks, -1).min(1).values.mean()
                    ),
                    "all_nmse": error.cpu().tolist(),
                }
                records.append(record)
                print(record["method"], step, record["best_mean_nmse"], flush=True)
        if step == 6144:
            break
        ids = torch.randint(spec["support"], (128,), device=device)
        loss = (
            (dense_predict(x[ids], parameters) - labels[:, ids]).square().mean(1)
            / scales
        ).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Scale the actual Adam update per model, retaining independent moments.
        old = [p.detach().clone() for p in parameters]
        optimizer.step()
        with torch.no_grad():
            for before, after in zip(old, parameters):
                shape = [count] + [1] * (after.ndim - 1)
                after.copy_(before + (after - before) * lr.reshape(shape))
    summary = {
        "dense_source_best_nmse": best.reshape(tasks, -1).min(1).values.cpu().tolist()
    }
    # Correct permutation is used only for this explicitly named oracle control.
    for method in ["oracle_codebook", "frozen_random_codebooks"]:
        torch.manual_seed(spec["seed"] + 1)
        if method == "oracle_codebook":
            teacher = np.load(out / "hidden_teacher.npz")
            permutations = torch.tensor(teacher["permutation"], device=device)[None]
            population, restarts = 1, 16
        else:
            population, restarts = spec["population"], spec["restarts"]
            permutations = torch.stack(
                [torch.randperm(64, device=device) for _ in range(population)]
            )
        parameters = [
            torch.randn(population, tasks, restarts, 4, 8, device=device) * 0.5,
            torch.full((population, tasks, restarts, 4), -1.0, device=device),
            torch.ones(population, tasks, restarts, 4, device=device),
            torch.zeros(population, tasks, restarts, device=device),
        ]
        parameters = [p.requires_grad_() for p in parameters]
        optimizer = torch.optim.Adam(parameters, lr=spec["learning_rate"])
        best = torch.full((population, tasks, restarts), torch.inf, device=device)
        for step in range(6145):
            if step % 256 == 0:
                with torch.no_grad():
                    error = torch.zeros_like(best)
                    for ids in torch.arange(
                        spec["support"], len(x), device=device
                    ).split(256):
                        residual = (
                            predict(x[ids], permutations, parameters)
                            - y[None, :, None, ids]
                        )
                        error += (
                            residual.square().sum(-1)
                            / spec["query"]
                            / variance[None, :, None]
                        )
                    best = torch.minimum(best, error)
                    value = float(best.min(-1).values.mean(-1).min())
                    records.append(
                        {"method": method, "step": step, "best_mean_nmse": value}
                    )
                    print(method, step, value, flush=True)
            if step == 6144:
                break
            ids = torch.randint(spec["support"], (128,), device=device)
            residual = predict(x[ids], permutations, parameters) - y[None, :, None, ids]
            loss = (residual.square().mean(-1) / variance[None, :, None]).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        summary[method + "_best_mean_nmse"] = float(best.min(-1).values.mean(-1).min())
    (out / "controls_history.json").write_text(json.dumps(records, indent=2))
    (out / "controls_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

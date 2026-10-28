"""Fit dense source networks from observations without teacher layouts."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def predict(inputs, weights):
    return torch.einsum("bd,mdh->mbh", inputs, weights).relu().sum(-1) / 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v1")
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--starts", type=int, default=4)
    args = parser.parse_args()
    if (args.out / "source_complete.json").exists():
        parser.error("Source run already exists; use a fresh output directory")
    torch.set_num_threads(4)
    device = torch.device("cuda")
    data = np.load(args.out / "source_data.npz")
    spec = json.loads((args.out / "solver_protocol.json").read_text())
    x = torch.tensor(data["x"], device=device)
    y = torch.tensor(data["y"], device=device).repeat_interleave(args.starts, 0)
    tasks, support = len(data["y"]), spec["source_support"]
    count = tasks * args.starts
    torch.manual_seed(283001)
    weights = (torch.randn(count, 64, 36, device=device) * 0.1).requires_grad_()
    penalties = torch.tensor(
        [0.0 if start < args.starts // 2 else 0.0001 for start in range(args.starts)],
        device=device,
    ).repeat(tasks)
    optimizer = torch.optim.Adam([weights], lr=0.01)
    scale = y[:, :support].var(1, unbiased=False).clamp_min(1e-8)
    best, best_error = (
        weights.detach().clone(),
        torch.full((count,), torch.inf, device=device),
    )
    history = []
    for step in range(args.steps + 1):
        if step % 256 == 0:
            with torch.no_grad():
                error = torch.zeros(count, device=device)
                for ids in torch.arange(support, len(x), device=device).split(256):
                    error += (predict(x[ids], weights) - y[:, ids]).square().sum(1)
                error /= (len(x) - support) * scale
                improve = error < best_error
                best_error[improve] = error[improve]
                best[improve] = weights[improve]
                history.append({"step": step, "query_nmse": best_error.cpu().tolist()})
            if step % 1024 == 0:
                print(
                    "SOURCE",
                    step,
                    float(best_error.mean()),
                    float(best_error.reshape(tasks, args.starts).min(1).values.mean()),
                    flush=True,
                )
        if step == args.steps:
            break
        ids = torch.randint(support, (256,), device=device)
        loss = (predict(x[ids], weights) - y[:, ids]).square().mean(1) / scale
        loss = (loss + penalties * weights.abs().sum((1, 2))).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    np.savez_compressed(
        args.out / "source_models.npz",
        weights=best.cpu().numpy(),
        quality=best_error.cpu().numpy(),
        penalties=penalties.cpu().numpy(),
        task=np.arange(count) // args.starts,
    )
    (args.out / "source_history.json").write_text(json.dumps(history))
    (args.out / "source_complete.json").write_text(
        json.dumps(
            {
                "steps": args.steps,
                "starts": args.starts,
                "tasks": tasks,
                "read_files": ["source_data.npz", "solver_protocol.json"],
                "regularization": "two unpenalized and two L1-sum .0001 starts by default",
                "best_source_query_mean": float(
                    best_error.reshape(tasks, args.starts).min(1).values.mean()
                ),
            },
            indent=2,
        )
    )
    shutil.copy2(__file__, args.out / "source_snapshot.py")


if __name__ == "__main__":
    main()

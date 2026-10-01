"""Build vector importance for trained outer/Kronecker weight factors.

Magnitude importance is the absolute masked factor value, normalized within
each factor to remove the a*c, b/c scale ambiguity. Functional importance is
the validation-BCE increase when a factor coordinate is set to zero with all
other trained weights frozen. The tensor product of magnitude factors exactly
recovers the normalized absolute first-layer weight map.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import make_dataset  # noqa: E402
from evaluation.factor_importance import support  # noqa: E402
from models.mlp import BatchedMaskedMLP  # noqa: E402

FACTOR_METHODS = (
    "outer_input4", "outer_hidden4", "outer_dense",
    "kron_left4", "kron_right4", "kron_dense",
)


def normalize_factor(values: torch.Tensor) -> torch.Tensor:
    values = values.abs().flatten(1)
    return values / values.amax(dim=1, keepdim=True).clamp_min(1e-12)


def factor_values(state: dict, method: str) -> torch.Tensor:
    repeats = state["repeats"]
    index = state["methods"].index(method)
    local = slice((index % 3) * repeats, (index % 3 + 1) * repeats)
    if method.startswith("outer"):
        a = state["outer_a"][local] * state["outer_a_mask"][local]
        b = state["outer_b"][local] * state["outer_b_mask"][local]
    else:
        a = state["kron_a"][local] * state["kron_a_mask"][local]
        b = state["kron_b"][local] * state["kron_b_mask"][local]
    return torch.cat((normalize_factor(a), normalize_factor(b)), dim=1)


def evaluate_ablation(state: dict, method: str, data: dict,
                      device: torch.device, batch_size: int) -> dict:
    repeats = state["repeats"]
    index = state["methods"].index(method)
    sl = slice(index * repeats, (index + 1) * repeats)
    first_layer = state["first_layer"][sl].to(device)
    affected = support("outer" if method.startswith("outer") else "kronecker").to(device)
    ablated = first_layer[None] * (1 - affected[:, None])
    removed = ((first_layer[None] != 0).float() * affected[:, None]).sum((2, 3))
    model = BatchedMaskedMLP(16 * repeats, 8, 8).to(device)
    with torch.no_grad():
        model.w1.copy_(ablated.reshape(-1, 8, 8))
        model.b1.copy_(state["b1"][sl].to(device).repeat(16, 1))
        model.w2.copy_(state["w2"][sl].to(device).repeat(16, 1).unsqueeze(-1))
        model.b2.copy_(state["b2"][sl].to(device).repeat(16).unsqueeze(-1))
        x, y = data["x"].to(device), data["y"].to(device)
        losses = model.val_loss(x, y, batch_size).reshape(16, repeats)
    delta = losses - state["best_val_bce"][sl].to(device)[None]
    return {
        "factor_magnitude": factor_values(state, method),
        "delta_bce": delta.T.cpu(),
        "delta_bce_per_removed_edge": (delta / removed.clamp_min(1)).T.cpu(),
        "removed_edges": removed.T.cpu(),
        "mean_delta_bce": float(delta.mean()),
        "fraction_positive": float((delta > 0).float().mean()),
        "mean_removed_edges": float(removed.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    run = json.loads((args.run_dir / "summary.json").read_text())
    summary = {"protocol": {
        "source": str(args.run_dir / "summary.json"),
        "importance": "normalized absolute masked factors; frozen-weight BCE ablation",
        "length": 16,
    }, "patterns": {}}
    tensors = {}
    for pattern in run["patterns"]:
        state = torch.load(args.run_dir / f"pattern_{pattern}_states.pt",
                           weights_only=True, map_location="cpu")
        data = make_dataset(pattern, run["protocol"]["eval_samples"],
                            seed=200000 + int(pattern, 2), pos_fraction=0.5)
        tensors[pattern] = {}
        summary["patterns"][pattern] = {}
        for method in FACTOR_METHODS:
            result = evaluate_ablation(state, method, data, device, args.batch_size)
            tensors[pattern][method] = {
                key: value for key, value in result.items() if torch.is_tensor(value)
            }
            summary["patterns"][pattern][method] = {
                key: value for key, value in result.items() if not torch.is_tensor(value)
            }
        print(f"[weight-importance] {pattern} complete", flush=True)
    torch.save(tensors, args.run_dir / "factor_importance.pt")
    (args.run_dir / "factor_importance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("[weight-importance] saved", flush=True)


if __name__ == "__main__":
    main()

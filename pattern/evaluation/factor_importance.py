"""Measure factor-coordinate importance by validation-loss ablation.

Each trained vector or Kronecker factor coordinate controls a set of matrix
entries. With downstream weights frozen, this removes those active entries
and records the BCE change. The output is a 16-dimensional importance vector
per model, ready for a later vector-VAE dataset if factorized masks work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generate import make_dataset  # noqa: E402
from models.mlp import BatchedMaskedMLP  # noqa: E402


def support(method: str) -> torch.Tensor:
    affected = torch.zeros(16, 8, 8)
    if method == "outer":
        for input_index in range(8):
            affected[input_index, input_index, :] = 1
        for hidden_index in range(8):
            affected[8 + hidden_index, :, hidden_index] = 1
    elif method == "kronecker":
        # A is 2x4 and B is 4x2; M=A kron B.
        for a_row in range(2):
            for a_col in range(4):
                affected[a_row * 4 + a_col,
                         a_row * 4:(a_row + 1) * 4,
                         a_col * 2:(a_col + 1) * 2] = 1
        for b_row in range(4):
            for b_col in range(2):
                affected[8 + b_row * 2 + b_col,
                         b_row::4, b_col::2] = 1
    else:
        raise ValueError(method)
    assert torch.all(affected.sum((1, 2)) == 8)
    return affected


def evaluate_bce(model: BatchedMaskedMLP, data: dict, batch_size: int) -> torch.Tensor:
    x = data["x"].to(model.mask.device)
    y = data["y"].to(model.mask.device)
    return model.val_loss(x, y, batch_size)


def ablation_for_method(state: dict, method: str, validation: dict,
                        device: torch.device, batch_size: int) -> dict:
    repeats = state["repeats"]
    method_index = state["methods"].index(method)
    sl = slice(method_index * repeats, (method_index + 1) * repeats)
    masks = state["mask"][sl].to(device)
    w1 = state["w1"][sl].to(device)
    b1 = state["b1"][sl].to(device)
    w2 = state["w2"][sl].to(device)
    b2 = state["b2"][sl].to(device)
    affected = support(method).to(device)
    ablated = masks[None] * (1 - affected[:, None])
    count_removed = (masks[None] * affected[:, None]).sum((2, 3))
    model = BatchedMaskedMLP(16 * repeats, 8, 8).to(device)
    with torch.no_grad():
        model.load_masks(ablated.reshape(-1, 8, 8))
        model.w1.copy_(w1.repeat(16, 1, 1))
        model.b1.copy_(b1.repeat(16, 1))
        model.w2.copy_(w2.repeat(16, 1).unsqueeze(-1))
        model.b2.copy_(b2.repeat(16).unsqueeze(-1))
        ablated_bce = evaluate_bce(model, validation, batch_size).reshape(16, repeats)
    base_bce = state["best_val_bce"][sl].to(device)
    delta = ablated_bce - base_bce[None]
    if method == "outer":
        factors = torch.cat((
            torch.sigmoid(state["outer_input_logits"]),
            torch.sigmoid(state["outer_hidden_logits"]),
        ), dim=1).to(device)
    else:
        factors = torch.cat((
            torch.sigmoid(state["kron_left_logits"]).flatten(1),
            torch.sigmoid(state["kron_right_logits"]).flatten(1),
        ), dim=1).to(device)
    # The factor values determine ranking but have a scale ambiguity; the
    # frozen-weight ablation delta measures task sensitivity directly.
    normalized_delta = delta / count_removed.clamp_min(1)
    return {
        "factor_values": factors.cpu(),
        "ablation_delta_bce": delta.T.cpu(),
        "ablation_delta_per_removed_edge": normalized_delta.T.cpu(),
        "removed_edges": count_removed.T.cpu(),
        "mean_delta_bce": float(delta.mean()),
        "mean_positive_delta_bce": float(delta.clamp_min(0).mean()),
        "fraction_positive": float((delta > 0).float().mean()),
        "mean_removed_edges": float(count_removed.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    summary_path = args.run_dir / "summary.json"
    run_summary = json.loads(summary_path.read_text())
    patterns = list(run_summary["patterns"])
    out = {"patterns": {}, "protocol": {
        "source": str(summary_path),
        "validation_seed": "200000 + pattern integer",
        "ablation": "remove factor-controlled active connections, no retraining",
        "factor_vector_length": 16,
    }}
    tensors = {}
    for pattern in patterns:
        state = torch.load(args.run_dir / f"pattern_{pattern}_states.pt",
                           weights_only=True, map_location="cpu")
        validation = make_dataset(pattern, args.eval_samples,
                                  seed=200000 + int(pattern, 2), pos_fraction=0.5)
        out["patterns"][pattern] = {}
        tensors[pattern] = {}
        for method in ("outer", "kronecker"):
            result = ablation_for_method(state, method, validation, device,
                                         args.batch_size)
            tensors[pattern][method] = {
                key: value for key, value in result.items() if torch.is_tensor(value)
            }
            out["patterns"][pattern][method] = {
                key: value for key, value in result.items() if not torch.is_tensor(value)
            }
        print(f"[factor-importance] {pattern} complete", flush=True)
    torch.save(tensors, args.run_dir / "factor_importance.pt")
    (args.run_dir / "factor_importance_summary.json").write_text(
        json.dumps(out, indent=2) + "\n")
    print("[factor-importance] saved", flush=True)


if __name__ == "__main__":
    main()

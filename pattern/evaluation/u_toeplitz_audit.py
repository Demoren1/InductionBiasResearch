"""Audit Toeplitz structure in saved pattern U pilots, after hidden alignment.

The support score compares 32 active edges with the ideal Toeplitz mask.
For continuous U, support is a top-32 proxy from mean |W| over the 16 tasks;
for binary U it is the actual nonzero assignment/gate. The subspace score is
the average squared projection of the four ideal Toeplitz tap bases into the
learned U column space. It is invariant to code/basis changes, conditional on
the hidden-column permutation selected by support overlap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pattern.evaluation.u_reparameterization_pilot import (
    SharedUModel, analytic_assignment,
)


def full_u(model: SharedUModel) -> torch.Tensor:
    if model.method == "direct_continuous":
        return model.u.detach()
    if model.method == "kronecker_continuous":
        return torch.einsum("ia,jb->ijab", model.u1, model.u2).reshape(model.n_edges, -1).detach()
    if model.method == "kronecker_binary":
        u1 = torch.nn.functional.one_hot(model.u1_logits.argmax(1), model.seq_len).float()
        u2 = torch.nn.functional.one_hot(model.u2_logits.argmax(1), 5).float()
        gate = model.hard_active(model.gate_logits)
        return (torch.einsum("ia,jb->ijab", u1, u2).reshape(model.n_edges, -1)
                * gate[:, None]).detach()
    return model.assignment_matrix().detach()[:, 1:]


def binary_codes(model: SharedUModel) -> np.ndarray | None:
    if model.method == "kronecker_binary":
        input_code = model.u1_logits.argmax(1).detach().numpy()
        hidden_code = model.u2_logits.argmax(1).detach().numpy()
        gate = model.hard_active(model.gate_logits).bool()
        code = 1 + 5 * input_code[:, None] + hidden_code[None, :]
        return np.where(gate.reshape(model.seq_len, 8).numpy(), code, 0)
    if "binary" in model.method or model.method.endswith(("_assignment", "_tied")):
        return model.assignment_matrix().argmax(1).reshape(model.seq_len, 8).detach().numpy()
    return None


def pairwise_group_f1(found: np.ndarray, gold: np.ndarray) -> dict:
    shared = (found > 0) & (gold > 0)
    f = found[shared]
    g = gold[shared]
    rows, cols = np.triu_indices(f.size, k=1)
    f_same = f[rows] == f[cols]
    g_same = g[rows] == g[cols]
    tp = int((f_same & g_same).sum())
    pred = int(f_same.sum())
    truth = int(g_same.sum())
    return {
        "common_active_edges": int(shared.sum()),
        "same_group_pair_precision": tp / pred if pred else 0.0,
        "same_group_pair_recall": tp / truth if truth else 0.0,
        "same_group_pair_f1": 2 * tp / (pred + truth) if pred + truth else 0.0,
        "same_group_pairs_predicted": pred,
        "same_group_pairs_ideal": truth,
    }


def audit(path: Path) -> dict:
    summary = json.loads((path / "summary.json").read_text())
    method = summary["method"]
    if method == "learned_binary_first_order_bilevel":
        method = "learned_binary"
    seq_len = summary.get("seq_len", 8)
    model = SharedUModel(method, summary["seed"], seq_len,
                         summary.get("column_quota", False),
                         summary.get("free_cardinality", False))
    state = torch.load(path / "best.pt", map_location="cpu", weights_only=True)
    mismatch = model.load_state_dict(state["state_dict"], strict=False)
    if mismatch.unexpected_keys or set(mismatch.missing_keys) - {"gate_bias"}:
        raise ValueError(f"checkpoint mismatch in {path}: {mismatch}")
    model.eval()
    codes = binary_codes(model)
    weights = model.first_layer().detach().numpy()
    if codes is None:
        mean_importance = np.abs(weights).mean(0)
        support = np.zeros(model.n_edges, dtype=bool)
        support[np.argpartition(mean_importance.ravel(), -32)[-32:]] = True
        support = support.reshape(seq_len, 8)
        support_source = "top32_mean_abs_effective_W"
    else:
        support = codes > 0
        support_source = "binary_U_nonzero_assignment"
    gold_codes = analytic_assignment(seq_len).argmax(1).reshape(seq_len, 8).numpy()
    gold_support = gold_codes > 0
    overlap = np.array([[(support[:, j] & gold_support[:, k]).sum()
                         for k in range(8)] for j in range(8)])
    old_columns, new_columns = linear_sum_assignment(-overlap)
    permutation = np.empty(8, dtype=int)
    permutation[new_columns] = old_columns
    aligned = support[:, permutation]
    intersection = int((aligned & gold_support).sum())
    result = {
        "method": method,
        "seed": summary["seed"],
        "column_quota": model.column_quota,
        "free_cardinality": model.free_cardinality,
        "support_source": support_source,
        "active_edges": int(support.sum()),
        "active_edges_by_column": support.sum(0).astype(int).tolist(),
        "hidden_permutation_new_to_old": permutation.tolist(),
        "toeplitz_support_intersection": intersection,
        "toeplitz_support_iou": intersection / int((aligned | gold_support).sum()),
        "toeplitz_support_precision": intersection / int(support.sum()) if support.any() else 0.0,
        "toeplitz_support_recall": intersection / 32,
        "toeplitz_support_hamming_matches": int((aligned == gold_support).sum()),
        "mean_test_acc_sanity_only": summary["mean_test_acc"],
    }
    # Orthogonal projection of each learned W onto the four-diagonal
    # rectangular Toeplitz family. The normalized variant removes arbitrary
    # positive hidden-unit scales before assessing the repeated tap values.
    aligned_weights = weights[:, :, permutation]
    offsets = np.arange(seq_len)[:, None] - np.arange(8)[None, :]
    def toeplitz_energy(weight_matrix: np.ndarray) -> float:
        projected = np.zeros_like(weight_matrix)
        for tap in range(4):
            positions = offsets == tap
            projected[:, positions] = weight_matrix[:, positions].mean(axis=1)[:, None]
        energy = np.square(projected).sum(axis=(1, 2))
        total = np.square(weight_matrix).sum(axis=(1, 2))
        return float(np.mean(energy / np.maximum(total, 1e-12)))
    result["weight_toeplitz_explained_energy"] = toeplitz_energy(aligned_weights)
    column_norms = np.linalg.norm(aligned_weights, axis=1, keepdims=True)
    result["weight_toeplitz_explained_energy_column_normalized"] = toeplitz_energy(
        aligned_weights / np.maximum(column_norms, 1e-12))
    if codes is not None:
        result.update(pairwise_group_f1(codes[:, permutation], gold_codes))
        result["distinct_active_codes"] = int(np.unique(codes[codes > 0]).size)
    u = full_u(model).reshape(seq_len, 8, -1)[:, permutation].reshape(model.n_edges, -1)
    tap_basis = analytic_assignment(seq_len)[:, 1:]
    tap_basis = tap_basis / tap_basis.norm(dim=0, keepdim=True)
    basis = torch.linalg.svd(u, full_matrices=False).U
    rank = int(torch.linalg.matrix_rank(u))
    result["u_effective_rank"] = rank
    result["toeplitz_tap_subspace_projection"] = float(
        (basis[:, :rank].T @ tap_basis).square().sum() / 4)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for directory in sorted(args.root.iterdir()):
        if not (directory / "best.pt").exists():
            continue
        result = audit(directory)
        (directory / "toeplitz_audit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(directory.name, "IoU", round(result["toeplitz_support_iou"], 3),
              "tap", round(result["toeplitz_tap_subspace_projection"], 3),
              "groupF1", round(result.get("same_group_pair_f1", -1), 3))


if __name__ == "__main__":
    main()

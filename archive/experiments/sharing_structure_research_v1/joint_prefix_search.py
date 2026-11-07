"""Common integer prefixes with independently fitted signed task codebooks."""

import math
import time

import numpy as np
import torch
from torch.nn import functional as F

from .count_exchanges import search_counts
from .count_projection import integer_counts, integer_moves
from .count_qp_active import function_qp
from .shared_structure_numeric import code_prediction, features
from .shared_structure_numeric_batch import fit_codes


@torch.no_grad()
def census(data, initial, tasks, refit, solver, plan, device):
    """Search every connection/group from empty prefixes; bootstrap I is absent."""
    started = time.monotonic()
    order = data["order"]
    x = torch.tensor(data["search_train_x"], device=device)
    vx = torch.tensor(data["prefix_validation_x"], device=device)
    y = torch.tensor(data["search_train_y"][tasks], device=device)
    vy = torch.tensor(data["prefix_validation_y"][tasks], device=device)
    code = torch.tensor(initial[tasks], device=device)
    prefix = np.zeros((1, 0), dtype=np.int16)
    totals = np.array([plan["hidden"]], dtype=np.int16)
    choices = np.r_[
        0, np.arange(1, plan["codes"] + 1), -np.arange(1, plan["codes"] + 1)
    ]
    labels = torch.tensor(choices[1:], device=device)
    trace, indices, codebooks = [], [], []
    for step in range(1, plan["inputs"] + 1):
        begin = (step - 1) * plan["slice_points"]
        end = step * plan["slice_points"]
        block = x[begin:end]
        target_y = y[:, begin:end]
        parent = torch.tensor(prefix, device=device).long()
        basis = F.one_hot(parent.abs(), plan["codes"] + 1)[..., 1:].to(x.dtype)
        basis *= parent.sign()[..., None]
        weights = torch.einsum("prk,tk->tpr", basis, code)
        base = (
            torch.einsum("nr,tpr->tnp", block[:, order[: step - 1]], weights)
            + plan["bias"]
        )
        code_values = code[:, labels.abs() - 1] * labels.sign()
        atoms = (
            base[..., None]
            + block[:, order[step - 1]][None, :, None, None]
            * code_values[:, None, None, :]
        ).relu()
        matrix = (atoms - base.relu()[..., None]).flatten(2) / math.sqrt(plan["hidden"])
        target = target_y - torch.einsum(
            "tnp,p->tn", base.relu() / math.sqrt(plan["hidden"]), y.new_tensor(totals)
        )
        grams = (matrix.transpose(1, 2) @ matrix / plan["slice_points"]).cpu().numpy()
        gram = grams.mean(0)
        cross = torch.einsum("tnq,tn->q", matrix, target).cpu().numpy() / target.numel()
        eigenvalues, vectors = np.linalg.eigh(gram)
        threshold = max(float(eigenvalues[-1]), 1e-30) * plan["rank_relative"]
        rank = int((eigenvalues > threshold).sum())
        record = {
            "prefix_length": step,
            "codebook_before": code.cpu().tolist(),
            "parameters": len(cross),
            "rank": rank,
            "condition": float(eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > 0
            else None,
            "minimum_gram_eigenvalue": float(eigenvalues[0]),
            "single_task_minimum_gram_eigenvalues": np.linalg.eigvalsh(grams)[
                :, 0
            ].tolist(),
            "accepted": False,
        }
        if rank != len(cross):
            trace.append({**record, "reason": "rank_deficient_dictionary"})
            break
        estimate = vectors @ ((vectors.T @ cross) / eigenvalues)
        other = estimate.reshape(len(prefix), len(choices) - 1)
        rounded = integer_counts(np.c_[totals - other.sum(1), other], totals)
        constrained, qp = function_qp(
            gram, cross, rounded[:, 1:].flatten(), totals, plan
        )
        record["qp"] = qp
        if not qp["converged"]:
            trace.append({**record, "reason": "qp_not_converged"})
            break
        other = constrained.reshape(len(prefix), len(choices) - 1)
        rounded = integer_counts(np.c_[totals - other.sum(1), other], totals)
        if solver == "unit":
            rounded, integer = integer_moves(rounded, gram, cross, plan)
        else:
            rounded, integer = search_counts(
                rounded, gram, cross, plan, kind="parent_pair", device=device
            )
        record["integer"] = integer
        pre_refit_mse = float(
            (
                torch.einsum(
                    "tnq,q->tn", matrix, y.new_tensor(rounded[:, 1:].flatten())
                )
                - target
            )
            .square()
            .mean()
        )
        extended = np.c_[
            np.repeat(prefix, len(choices), axis=0), np.tile(choices, len(prefix))
        ]
        keep = rounded.flatten() > 0
        prefix, totals = extended[keep], rounded.flatten()[keep]
        index = np.zeros((plan["inputs"], plan["hidden"]), dtype=np.int16)
        index[order[:step]] = np.repeat(prefix, totals, axis=0).T
        direct = code_prediction(features(block, index, plan["codes"]), code, plan)
        direct_mse = float((direct - target_y).square().mean())
        metric_error = abs(direct_mse - pre_refit_mse)
        if metric_error > 1e-8 * (1 + abs(direct_mse)):
            raise ArithmeticError(
                "Joint count metric disagrees with actual hard forward"
            )
        record["pre_refit_training_mse"] = pre_refit_mse
        record["metric_forward_absolute_difference"] = metric_error
        if refit:
            fitted, histories, _ = fit_codes(
                index[None],
                x[:end],
                y[:, :end],
                vx[: step * plan["prefix_validation"]],
                vy[:, : step * plan["prefix_validation"]],
                code[:, None],
                plan,
                max_steps=plan["code_refit_steps"],
            )
            code = fitted[0]
            record["refit"] = histories[0]
        validation_begin = (step - 1) * plan["prefix_validation"]
        validation_end = step * plan["prefix_validation"]
        prediction = code_prediction(features(block, index, plan["codes"]), code, plan)
        validation = code_prediction(
            features(vx[validation_begin:validation_end], index, plan["codes"]),
            code,
            plan,
        )
        record.update(
            {
                "accepted": True,
                "reason": "joint_integer_prefix",
                "distinct_prefixes": len(prefix),
                "codebook_after": code.cpu().tolist(),
                "training_mse_per_task": (prediction - target_y)
                .square()
                .mean(1)
                .cpu()
                .tolist(),
                "validation_mse_per_task": (
                    validation - vy[:, validation_begin:validation_end]
                )
                .square()
                .mean(1)
                .cpu()
                .tolist(),
            }
        )
        trace.append(record)
        indices.append(index)
        codebooks.append(code.cpu().numpy())
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {
        "complete": len(indices) == plan["inputs"],
        "indices": np.stack(indices)
        if indices
        else np.zeros((0, plan["inputs"], plan["hidden"]), dtype=np.int16),
        "codebooks": np.stack(codebooks)
        if codebooks
        else np.zeros((0, len(tasks), plan["codes"])),
        "trace": trace,
        "seconds": time.monotonic() - started,
    }

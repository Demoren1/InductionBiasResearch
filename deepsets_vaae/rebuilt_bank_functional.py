"""Functional contexts and label-free hidden alignment for rebuilt DeepSets banks.

All saved teachers are retained for audit, while pooled model contexts and
score baselines use only the train split. Source quality is metadata only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.nn import functional as F


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _metadata_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metadata_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _extract_state(bank: dict[str, Any]) -> tuple[dict[str, Tensor], str]:
    state = bank.get("state_dict", bank)
    if not isinstance(state, dict):
        raise ValueError("bank must contain a tensor state_dict")
    normalized = dict(state)
    for target, alternatives in {
        "weight": ("weight", "weights"),
        "readout": ("readout", "a"),
        "per_image_offset": ("per_image_offset", "offset"),
    }.items():
        if target not in normalized:
            for key in alternatives:
                if key in normalized:
                    normalized[target] = normalized[key]
                    break
    for key in ("weight", "masks", "bias", "readout"):
        if key not in normalized or not torch.is_tensor(normalized[key]):
            # Older bank schemas keep masks/weights beside the model state.
            if key in bank and torch.is_tensor(bank[key]):
                normalized[key] = bank[key]
            else:
                raise ValueError(f"bank lacks tensor state field {key!r}")
    if "per_image_offset" not in normalized:
        normalized["per_image_offset"] = torch.zeros(
            normalized["weight"].shape[0], dtype=normalized["weight"].dtype
        )
    return normalized, "state_dict" if "state_dict" in bank else "top_level"


def _moments_for_bank(
    path: Path,
    probe_x: Tensor,
    device: torch.device,
    *,
    teacher_chunk: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    bank = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bank, dict):
        raise ValueError(f"bank {path} must load to a dictionary")
    state, state_location = _extract_state(bank)
    weight = state["weight"].detach().float().cpu()
    masks = state["masks"].detach().float().cpu()
    bias = state["bias"].detach().float().cpu()
    readout = state["readout"].detach().float().cpu()
    offset = state["per_image_offset"].detach().float().cpu().reshape(-1)
    if weight.ndim != 3 or weight.shape[1] != 784:
        raise ValueError(f"{path}: weight must have shape [M, 784, H]")
    teachers, features, hidden = weight.shape
    if masks.shape != weight.shape or bias.shape != (teachers, hidden) or readout.shape != (teachers, hidden):
        raise ValueError(f"{path}: weight/mask/bias/readout shapes are inconsistent")
    if offset.shape != (teachers,):
        raise ValueError(f"{path}: per_image_offset must have shape [M]")
    probes = len(probe_x)
    psi = torch.empty(teachers, probes, hidden, dtype=torch.float32)
    q_signed = torch.empty(teachers, features, hidden, dtype=torch.float32)
    q_abs = torch.empty_like(q_signed)
    q_rms = torch.empty_like(q_signed)

    for start in range(0, teachers, teacher_chunk):
        stop = min(teachers, start + teacher_chunk)
        x = probe_x.to(device=device, dtype=torch.float32)
        w = weight[start:stop].to(device)
        mask = masks[start:stop].to(device)
        effective = w * mask
        b = bias[start:stop].to(device)
        a = readout[start:stop].to(device)
        activation = torch.tanh(torch.einsum("pf,mfh->mph", x, effective) + b[:, None, :])
        psi_chunk = activation * a[:, None, :]
        gain = (1.0 - activation.square()) * a[:, None, :]
        signed = effective * torch.einsum("pf,mph->mfh", x, gain) / probes
        absolute = effective.abs() * torch.einsum(
            "pf,mph->mfh", x.abs(), gain.abs()
        ) / probes
        rms = effective.abs() * torch.einsum(
            "pf,mph->mfh", x.square(), gain.square()
        ).div(probes).clamp_min(0).sqrt()
        psi[start:stop] = psi_chunk.cpu()
        q_signed[start:stop] = signed.cpu()
        q_abs[start:stop] = absolute.cpu()
        q_rms[start:stop] = rms.cpu()

    metadata: dict[str, Any] = {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "state_location": state_location,
        "teacher_count": teachers,
        "feature_count": features,
        "hidden_count": hidden,
    }
    for key in ("source_task", "candidate_recipe", "quality", "density", "edges_per_mask", "seed"):
        if key in bank:
            metadata[key] = _metadata_value(bank[key])
    if isinstance(bank.get("metadata"), dict):
        for key in ("source_task", "candidate_recipe", "quality", "density", "seed"):
            if key in bank["metadata"]:
                metadata[key] = _metadata_value(bank["metadata"][key])
    raw_state = {
        "weight": weight,
        "masks": masks,
        "bias": bias,
        "readout": readout,
        "per_image_offset": offset,
    }
    moments = {
        "psi": psi,
        "q_signed_mean": q_signed,
        "q_abs_mean": q_abs,
        "q_rms": q_rms,
        "teacher_scores": psi.sum(dim=-1) + offset[:, None],
        "raw_state": raw_state,
    }
    del bank, state, weight, masks, bias, readout, offset
    return moments, metadata


def _signature(psi: Tensor, q_abs: Tensor) -> Tensor:
    """Create one scale-normalized ``[P+F,H]`` signature per teacher."""
    psi_columns = psi.abs()
    q_columns = q_abs.abs()
    psi_columns = F.normalize(psi_columns, p=2, dim=1, eps=1e-12)
    q_columns = F.normalize(q_columns, p=2, dim=1, eps=1e-12)
    return F.normalize(torch.cat((psi_columns, q_columns), dim=1), p=2, dim=1, eps=1e-12)


def _match_columns(consensus: Tensor, signature: Tensor) -> Tensor:
    """Return source hidden columns in canonical row order by Hungarian cosine."""
    similarity = consensus.T @ signature
    rows, columns = linear_sum_assignment(-similarity.detach().cpu().numpy())
    order = torch.empty(consensus.shape[1], dtype=torch.long)
    order[torch.as_tensor(rows, dtype=torch.long)] = torch.as_tensor(columns, dtype=torch.long)
    return order


def _gather_hidden(values: Tensor, orders: Tensor) -> Tensor:
    """Apply a ``[teacher,H]`` destination-to-source order to batched arrays."""
    if values.shape[0] != orders.shape[0] or values.shape[-1] != orders.shape[-1]:
        raise ValueError("hidden-order dimensions do not match values")
    index = orders.reshape(orders.shape[0], *([1] * (values.ndim - 2)), orders.shape[1])
    index = index.expand_as(values)
    return values.gather(-1, index)


def _fit_consensus(
    signatures: list[Tensor],
    train_rows: list[Tensor],
    total_teachers: list[int],
    *,
    rounds: int,
) -> tuple[Tensor, list[Tensor]]:
    references = [(task, int(row)) for task, rows in enumerate(train_rows) for row in rows]
    first_task, first_row = references[0]
    consensus = signatures[first_task][first_row].clone()
    orders = [torch.full((count, consensus.shape[1]), -1, dtype=torch.long)
              for count in total_teachers]
    for _ in range(rounds):
        total = torch.zeros_like(consensus)
        for task, rows in enumerate(train_rows):
            for row_value in rows:
                row = int(row_value)
                signature = signatures[task][row]
                order = _match_columns(consensus, signature)
                orders[task][row] = order
                total += signature[:, order]
        consensus = F.normalize(total / len(references), p=2, dim=0, eps=1e-12)
    return consensus, orders


def _save_functional_figures(
    output_dir: Path,
    psi: Tensor,
    q_signed: Tensor,
    q_abs: Tensor,
    representative_mask: Tensor,
    representative_task: int,
    representative_train_row: int,
    representative_heldout_row: int,
) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    psi_fig, psi_axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True,
                                    constrained_layout=True)
    psi_limit = max(1e-12, max(
        float(psi[representative_task, row].abs().max())
        for row in (representative_train_row, representative_heldout_row)
    ))
    for ax, row, role in (
        (psi_axes[0], representative_train_row, "train teacher"),
        (psi_axes[1], representative_heldout_row, "heldout teacher"),
    ):
        image = ax.imshow(psi[representative_task, row].T, aspect="auto",
                          interpolation="nearest", cmap="coolwarm",
                          vmin=-psi_limit, vmax=psi_limit)
        ax.set_title(f"{role}, teacher row {row}")
        ax.set_xlabel("Probe image")
        ax.set_ylabel("Hidden neuron")
    psi_fig.suptitle("Aligned hidden contribution ψ, two teachers")
    psi_fig.colorbar(image, ax=psi_axes.tolist(), shrink=.82)
    psi_path = figure_dir / "psi_two_teachers.png"
    psi_fig.savefig(psi_path, dpi=150)
    plt.close(psi_fig)
    representative_row = representative_train_row
    signed_map = q_signed[representative_task, representative_row]
    signed_limit = max(1e-12, float(signed_map.abs().max()))
    items = [
        ("q_signed_mean.png", signed_map, "Aligned mean signed q",
         "Feature position", "Hidden neuron", "coolwarm", -signed_limit, signed_limit),
        ("q_abs_mean.png", q_abs[representative_task, representative_row],
         "Aligned mean |q|", "Feature position", "Hidden neuron", "viridis", None, None),
        ("mask_support.png", representative_mask,
         "Aligned effective support mask", "Feature position", "Hidden neuron", "Greys", None, None),
    ]
    paths = [str(psi_path.resolve())]
    for name, values, title, x_label, y_label, cmap, vmin, vmax in items:
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        image = ax.imshow(values.detach().cpu().numpy().T, aspect="auto",
                          interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        fig.colorbar(image, ax=ax, shrink=.82)
        fig.tight_layout()
        path = figure_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path.resolve()))
    return paths


@torch.no_grad()
def extract_full_functional(
    banks_paths: Sequence[str | Path],
    probe_x: Tensor,
    probe_ids: Tensor | Sequence[int],
    out: str | Path,
    seed: int,
    device: str | torch.device,
    *,
    train_fraction: float = 0.8,
    alignment_rounds: int = 3,
    teacher_chunk: int = 32,
    make_figures: bool = True,
) -> dict[str, Any]:
    """Extract aligned bank functions, q moments, and quality-free contexts.

    Each bank is split once with a fixed local RNG. Alignment uses only the
    train subset, across all supplied source tasks; held-out teachers are
    matched to the final train consensus. Teacher quality and candidate recipe
    metadata are kept separately for audit and never enter signatures/features.

    Returns the same CPU dictionary written to ``functional_context.pt``.
    """
    paths = [Path(path).resolve() for path in banks_paths]
    if len(paths) != 4:
        raise ValueError("expected exactly four source-task bank paths")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if alignment_rounds < 1 or teacher_chunk < 1:
        raise ValueError("alignment_rounds and teacher_chunk must be positive")
    probe_x = torch.as_tensor(probe_x, dtype=torch.float32)
    if probe_x.ndim != 2 or probe_x.shape[1] != 784 or probe_x.shape[0] < 1:
        raise ValueError("probe_x must have shape [P,784]")
    probe_ids = torch.as_tensor(probe_ids).detach().cpu().reshape(-1)
    if probe_ids.numel() != probe_x.shape[0]:
        raise ValueError("probe_ids must identify every row of probe_x")
    device = torch.device(device)
    output_path = Path(out).resolve()
    if output_path.suffix.lower() == ".pt":
        context_path = output_path
        output_dir = output_path.parent
    else:
        output_dir = output_path
        context_path = output_dir / "functional_context.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_moments: list[dict[str, Tensor]] = []
    bank_metadata: list[dict[str, Any]] = []
    train_rows: list[Tensor] = []
    heldout_rows: list[Tensor] = []
    signatures: list[Tensor] = []
    teacher_counts: list[int] = []
    for task, path in enumerate(paths):
        moments, metadata = _moments_for_bank(
            path, probe_x, device, teacher_chunk=teacher_chunk
        )
        count = moments["psi"].shape[0]
        if count < 2:
            raise ValueError("each bank must contain at least two teachers")
        train_count = int(round(train_fraction * count))
        train_count = min(max(1, train_count), count - 1)
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + 7919 * (task + 1))
        permutation = torch.randperm(count, generator=generator)
        train_rows.append(permutation[:train_count])
        heldout_rows.append(permutation[train_count:])
        signatures.append(_signature(moments["psi"], moments["q_abs_mean"]))
        teacher_counts.append(count)
        all_moments.append(moments)
        bank_metadata.append(metadata)

    consensus, orders = _fit_consensus(
        signatures, train_rows, teacher_counts, rounds=alignment_rounds
    )
    for task, held_rows in enumerate(heldout_rows):
        for row_value in held_rows:
            row = int(row_value)
            orders[task][row] = _match_columns(consensus, signatures[task][row])
    order_tensor = torch.stack(orders)
    del signatures

    psi = torch.stack([_gather_hidden(item["psi"], order_tensor[task])
                       for task, item in enumerate(all_moments)])
    q_signed = torch.stack([_gather_hidden(item["q_signed_mean"], order_tensor[task])
                            for task, item in enumerate(all_moments)])
    q_abs = torch.stack([_gather_hidden(item["q_abs_mean"], order_tensor[task])
                         for task, item in enumerate(all_moments)])
    q_rms = torch.stack([_gather_hidden(item["q_rms"], order_tensor[task])
                         for task, item in enumerate(all_moments)])
    aligned_masks = [
        _gather_hidden(item["raw_state"]["masks"], order_tensor[task])
        for task, item in enumerate(all_moments)
    ]
    # Keep full-bank occupancy for audit, but train-only occupancy is the
    # context feature so held-out teachers cannot leak into model inputs.
    occupancy = torch.stack([mask.mean(dim=0) for mask in aligned_masks])
    train_mask_pool = torch.cat([
        mask[rows] for mask, rows in zip(aligned_masks, train_rows)
    ], dim=0)
    occupancy_train = train_mask_pool.mean(dim=0)
    teacher_probe_outputs = torch.stack([item["teacher_scores"] for item in all_moments])

    # Functional amplitude is normalized per teacher before global moments,
    # preventing high-readout teachers from dominating node context.
    psi_scale = psi.square().mean(dim=(-1, -2), keepdim=True).sqrt().clamp_min(1e-8)
    psi_norm = psi / psi_scale
    pooled_psi = torch.cat([
        psi_norm[task, rows] for task, rows in enumerate(train_rows)
    ], dim=0)
    node_mean = pooled_psi.mean(dim=0).T
    node_std = pooled_psi.std(dim=0, unbiased=False).T
    node_context = torch.cat((node_mean, node_std), dim=1)

    # q arrays remain raw and complete. Context features use one RMS scale per
    # teacher, which makes moment summaries comparable across readout scales.
    q_scale = q_rms.flatten(2).amax(dim=-1).clamp_min(1e-8)[..., None, None]
    q_signed_norm = q_signed / q_scale
    q_abs_norm = q_abs / q_scale
    q_rms_norm = q_rms / q_scale
    q_abs_pool = torch.cat([
        q_abs_norm[task, rows] for task, rows in enumerate(train_rows)
    ], dim=0)
    q_signed_pool = torch.cat([
        q_signed_norm[task, rows] for task, rows in enumerate(train_rows)
    ], dim=0)
    q_rms_pool = torch.cat([
        q_rms_norm[task, rows] for task, rows in enumerate(train_rows)
    ], dim=0)
    edge_context = torch.stack((
        q_abs_pool.mean(dim=0),
        q_abs_pool.std(dim=0, unbiased=False),
        q_signed_pool.mean(dim=0),
        q_rms_pool.mean(dim=0),
        q_rms_pool.std(dim=0, unbiased=False),
        occupancy_train,
    ), dim=-1)

    teacher_stride_indices = [rows[::26] for rows in train_rows]
    teacher_scores_train_stride26 = torch.cat([
        q_abs_norm[task, indices]
        for task, indices in enumerate(teacher_stride_indices)
    ], dim=0)
    teacher_score_task_rows = torch.tensor([
        (task, int(row))
        for task, indices in enumerate(teacher_stride_indices)
        for row in indices
    ], dtype=torch.long)
    source_mean_scores = torch.stack([
        q_abs_norm[task, rows].mean(dim=0)
        for task, rows in enumerate(train_rows)
    ])
    mean_score = q_abs_pool.mean(dim=0)
    source_mean_probe_outputs = torch.stack([
        teacher_probe_outputs[task, rows].mean(dim=0)
        for task, rows in enumerate(train_rows)
    ])
    teacher_probe_outputs_train_stride26 = torch.stack([
        teacher_probe_outputs[task, indices]
        for task, indices in enumerate(teacher_stride_indices)
    ])
    mean_probe_output = source_mean_probe_outputs.mean(dim=0)

    # Keep representative raw and aligned full states for a train and held-out
    # teacher per source task. Other rows are recoverable by the hashes and
    # saved permutation tensor without duplicating each large source bank.
    representative_states: list[dict[str, Any]] = []
    representative_task = 0
    representative_row = int(train_rows[0][0])
    representative_heldout_row = int(heldout_rows[0][0])
    representative_mask = torch.empty_like(all_moments[0]["raw_state"]["masks"][representative_row])
    for task, item in enumerate(all_moments):
        raw = item["raw_state"]
        selected_train = int(train_rows[task][0])
        selected_heldout = int(heldout_rows[task][0])
        selected = []
        for role, row in (("train", selected_train), ("heldout", selected_heldout)):
            order = order_tensor[task, row]
            raw_state = {
                "weight": raw["weight"][row].clone(),
                "masks": raw["masks"][row].clone(),
                "bias": raw["bias"][row].clone(),
                "readout": raw["readout"][row].clone(),
                "per_image_offset": raw["per_image_offset"][row].clone(),
            }
            aligned_state = {
                "weight": raw_state["weight"].gather(1, order[None, :].expand(784, -1)),
                "masks": raw_state["masks"].gather(1, order[None, :].expand(784, -1)),
                "bias": raw_state["bias"].gather(0, order),
                "readout": raw_state["readout"].gather(0, order),
                "per_image_offset": raw_state["per_image_offset"].clone(),
            }
            selected.append({
                "role": role,
                "teacher_row": row,
                "alignment_order": order.clone(),
                "raw_unaligned_state": raw_state,
                "aligned_state": aligned_state,
            })
            if task == representative_task and role == "train":
                representative_mask = aligned_state["masks"]
        representative_states.append({"source_task_index": task, "teachers": selected})

    context: dict[str, Any] = {
        "node_context": node_context.float().cpu(),
        "edge_context": edge_context.float().cpu(),
        "node": node_context.float().cpu(),
        "edge": edge_context.float().cpu(),
        "psi": psi.float().cpu(),
        "psi_norm": psi_norm.float().cpu(),
        "q_signed_mean": q_signed.float().cpu(),
        "q_abs_mean": q_abs.float().cpu(),
        "q_rms": q_rms.float().cpu(),
        "mask_occupancy": occupancy.float().cpu(),
        "mask_occupancy_train": occupancy_train.float().cpu(),
        "source_mean_scores": source_mean_scores.float().cpu(),
        "teacher_scores": teacher_scores_train_stride26.float().cpu(),
        "teacher_score_task_rows": teacher_score_task_rows,
        "teacher_score_rows_per_task": torch.stack(teacher_stride_indices),
        "mean_score": mean_score.float().cpu(),
        "source_mean_probe_outputs": source_mean_probe_outputs.float().cpu(),
        "teacher_probe_outputs": teacher_probe_outputs_train_stride26.float().cpu(),
        "mean_probe_output": mean_probe_output.float().cpu(),
        "probe_x": probe_x.detach().cpu().clone(),
        "probe_ids": probe_ids,
        "train_rows": torch.stack(train_rows),
        "heldout_rows": torch.stack(heldout_rows),
        "alignment_orders": order_tensor,
        "final_alignment_consensus": consensus.float().cpu(),
        "representative_states": representative_states,
        "bank_references": bank_metadata,
        "feature_definitions": {
            "psi": "tanh(probe_x @ (weight*masks) + bias) * readout; aligned hidden axis",
            "q_signed_mean": "mean_p[x[p,i] * effective_weight[m,i,h] * readout[m,h] * (1-tanh(pre[p,m,h])^2)]",
            "q_abs_mean": "mean_p absolute value of the same per-probe q contribution",
            "q_rms": "sqrt(mean_p q_contribution^2)",
            "node_context": "concatenate per-probe mean and population std of per-teacher RMS-normalized psi over train-split teachers from all four banks only; shape [H,2P]",
            "node": "legacy alias of node_context; train-split teachers only",
            "edge_context": "[abs-q mean, abs-q std, signed-q mean, q-RMS mean, q-RMS std, train-only mask occupancy] after per-teacher max-q-RMS normalization over train-split teachers only; shape [F,H,6]",
            "edge": "legacy alias of edge_context; train-split teachers only",
            "mask_occupancy": "aligned mask occupancy over all teachers, retained for audit only [T,F,H]",
            "mask_occupancy_train": "aligned mask occupancy over train-split teachers only, used in edge_context [F,H]",
            "source_mean_scores": "per-task mean per-teacher RMS-normalized abs-q map over train-split teachers only [T,F,H]",
            "teacher_scores": "per-teacher RMS-normalized abs-q maps for train-split rows at stride 26 [sum_t stride_count,F,H]",
            "mean_score": "mean normalized abs-q map across train-split teachers from all source tasks [F,H]",
            "source_mean_probe_outputs": "mean network output over train-split teachers for each fixed probe image; not label or quality score",
            "teacher_probe_outputs": "network outputs for train-split rows at stride 26 on fixed probe images",
            "mean_probe_output": "mean of source_mean_probe_outputs over source tasks for each probe image",
            "quality_nonuse": "Bank quality/validation metadata is retained only in bank_references and is excluded from alignment and every context feature.",
            "heldout_nonuse": "Held-out teachers are excluded from all pooled contexts and score baselines; full per-teacher raw/aligned arrays and audit occupancy remain available.",
        },
        "split": {
            "train_fraction": train_fraction,
            "heldout_fraction": 1.0 - train_fraction,
            "split_seed": int(seed),
            "split_rule": "per task, randperm(M) with local seed + 7919*(task_index+1); first rounded fraction trains consensus, rest held out",
            "alignment_rounds": alignment_rounds,
            "signature": "L2-normalized columns of concat(L2-normalized abs-psi probe profile, L2-normalized abs-q feature profile)",
        },
    }
    figure_paths: list[str] = []
    if make_figures:
        figure_paths = _save_functional_figures(
            output_dir, psi, q_signed, q_abs, representative_mask,
            representative_task, representative_row, representative_heldout_row,
        )
    manifest_path = context_path.with_name("functional_context_manifest.json")
    context["manifest_path"] = str(manifest_path)
    context["figure_paths"] = figure_paths
    context_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(context, context_path)
    manifest = {
        "context_path": str(context_path),
        "context_sha256": _sha256(context_path),
        "seed": int(seed),
        "device_for_moment_computation": str(device),
        "source_tasks": len(paths),
        "teacher_counts": teacher_counts,
        "train_counts": [len(rows) for rows in train_rows],
        "heldout_counts": [len(rows) for rows in heldout_rows],
        "probe_count": int(probe_x.shape[0]),
        "probe_ids": probe_ids.tolist(),
        "bank_references": bank_metadata,
        "figures": figure_paths,
        "quality_and_recipe_usage": "metadata only; no quality, validation score, density, or candidate_recipe enters alignment or context features",
        "hidden_state_alignment": "same saved permutation is applied to psi, q moments, weight, mask, bias, and readout; offset is invariant",
        "alignment_training_rows_only": True,
        "pooled_context_and_score_baselines_training_rows_only": True,
        "heldout_teachers_used_as_context_inputs": False,
        "all_bank_teachers_retained": True,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
    return context

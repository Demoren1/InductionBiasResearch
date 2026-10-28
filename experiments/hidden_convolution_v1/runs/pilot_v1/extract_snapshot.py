"""Infer sparse sharing from fitted weights; rank candidates on source data."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent


def quantize(weights, codes, separate_zero):
    flat = weights.reshape(-1)
    if separate_zero:
        logs = np.log(np.abs(flat) + 1e-8)[:, None]
        split = KMeans(2, n_init=5, random_state=283010).fit(logs)
        active = split.labels_ == split.cluster_centers_.argmax()
        if active.sum() < codes:
            active[:] = True
        fit = KMeans(codes, n_init=5, random_state=283011).fit(flat[active, None])
        centers = np.r_[0, fit.cluster_centers_.flatten()]
        indices = np.zeros(len(flat), dtype=int)
        indices[active] = 1 + fit.labels_
    else:
        fit = KMeans(codes + 1, n_init=5, random_state=283012).fit(flat[:, None])
        zero = int(np.abs(fit.cluster_centers_).argmin())
        order = np.r_[zero, np.delete(np.arange(codes + 1), zero)]
        inverse = np.argsort(order)
        centers = fit.cluster_centers_.flatten()[order]
        centers[0] = 0
        indices = inverse[fit.labels_]
    return indices.reshape(weights.shape), centers.astype("float32")


def align_columns(indices, reference):
    active, target = indices != 0, reference != 0
    cost = np.not_equal(active[:, :, None], target[:, None, :]).sum(0)
    old, new = linear_sum_assignment(cost)
    order = np.empty(indices.shape[1], dtype=int)
    order[new] = old
    return indices[:, order], order


def code_groups(indices, codes):
    presence = np.stack([(indices == k).any(0) for k in range(1, codes + 1)])
    # Groups are inferred from similar column occurrence patterns; no number
    # of groups or group size is supplied by the benchmark.
    distance = np.nan_to_num(pdist(presence.astype(float), metric="jaccard"))
    labels = fcluster(linkage(distance, method="complete"), 0.2, criterion="distance")
    return labels, presence


def align_codes(indices, reference, codes):
    groups, present = code_groups(indices, codes)
    ref_groups, ref_present = code_groups(reference, codes)
    ids, ref_ids = np.unique(groups), np.unique(ref_groups)
    support = np.stack([present[groups == group].any(0) for group in ids])
    ref_support = np.stack(
        [ref_present[ref_groups == group].any(0) for group in ref_ids]
    )
    intersections = (support[:, None, :] & ref_support[None, :, :]).sum(-1)
    old, new = linear_sum_assignment(-intersections)
    group_pairs = {ids[a]: ref_ids[b] for a, b in zip(old, new)}
    votes = np.zeros((codes + 1, codes + 1), dtype=int)
    np.add.at(votes, (indices.flatten(), reference.flatten()), 1)
    cost = -votes[1:, 1:].astype(float)
    for current, target in group_pairs.items():
        # Prefer coherent group mappings, with a finite cost so imperfect
        # groups do not make a global code-label assignment impossible.
        cost[np.ix_(groups == current, ref_groups != target)] += indices.size
    old, new = linear_sum_assignment(cost)
    mapping = np.zeros(codes + 1, dtype=int)
    mapping[old + 1] = new + 1
    return mapping[indices], mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v1")
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    if (args.out / "extraction_complete.json").exists():
        parser.error("Extraction already completed; use a fresh output directory")
    torch.set_num_threads(4)
    spec = json.loads((args.out / "solver_protocol.json").read_text())
    source = np.load(args.out / "source_models.npz")
    candidates, initial = [], []
    for weights in source["weights"]:
        for separated in [False, True]:
            indices, centers = quantize(weights, spec["codes"], separated)
            candidates.append(indices)
            initial.append(centers)
    candidates = np.array(candidates)
    device = torch.device("cuda")
    indices = torch.tensor(candidates, device=device)
    values = torch.tensor(np.array(initial), device=device).requires_grad_()
    task = np.repeat(source["task"], 2)
    observations = np.load(args.out / "source_data.npz")
    x = torch.tensor(observations["x"], device=device)
    y = torch.tensor(observations["y"][task], device=device)
    support = spec["source_support"]
    scale = y[:, :support].var(1, unbiased=False).clamp_min(1e-8)
    optimizer = torch.optim.Adam([values], lr=0.01)
    best, quality = (
        values.detach().clone(),
        torch.full((len(y),), torch.inf, device=device),
    )
    torch.manual_seed(283013)

    def predict(batch):
        numbers = values.clone()
        numbers[:, 0] = 0
        weights = numbers.gather(1, indices.flatten(1)).reshape(-1, 64, 36)
        return torch.einsum("bd,mdh->mbh", batch, weights).relu().sum(-1) / 6

    history = []
    for step in range(args.steps + 1):
        if step % 128 == 0:
            with torch.no_grad():
                error = torch.zeros(len(y), device=device)
                for ids in torch.arange(support, len(x), device=device).split(128):
                    error += (predict(x[ids]) - y[:, ids]).square().sum(1)
                error /= (len(x) - support) * scale
                improve = error < quality
                quality[improve] = error[improve]
                best[improve] = values[improve]
                best[:, 0] = 0
                history.append({"step": step, "query_nmse": quality.cpu().tolist()})
            if step % 512 == 0:
                print("QUANTIZED", step, float(quality.mean()), flush=True)
        if step == args.steps:
            break
        ids = torch.randint(support, (128,), device=device)
        loss = ((predict(x[ids]) - y[:, ids]).square().mean(1) / scale).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    quality = quality.cpu().numpy()
    choose = np.arange(len(source["weights"])) * 2 + quality.reshape(-1, 2).argmin(1)
    raw = candidates[choose]
    quality = quality[choose]
    best = best.cpu().numpy()[choose]
    selected = []
    for task_id in np.unique(source["task"]):
        ids = np.flatnonzero(source["task"] == task_id)
        selected.extend(ids[np.argsort(quality[ids])[:2]])
    selected = np.array(selected)
    reference = int(selected[quality[selected].argmin()])
    aligned, orders, mappings = [], [], []
    for matrix in raw:
        columns, order = align_columns(matrix, raw[reference])
        matrix, mapping = align_codes(columns, raw[reference], spec["codes"])
        aligned.append(matrix)
        orders.append(order)
        mappings.append(mapping)
    np.savez_compressed(
        args.out / "structure_bank.npz",
        raw_indices=raw,
        indices=np.array(aligned),
        quality=quality,
        selected=selected,
        train=selected[source["task"][selected] % 4 != 0],
        validation=selected[source["task"][selected] % 4 == 0],
        column_orders=np.array(orders),
        code_maps=np.array(mappings),
        reference=np.array(reference),
        task=source["task"],
        numeric_values=best,
        chosen_quantizer=choose % 2,
    )
    (args.out / "extraction_history.json").write_text(json.dumps(history))
    summary = {
        "raw_source_models": len(raw),
        "good_bank": len(selected),
        "train": int((source["task"][selected] % 4 != 0).sum()),
        "validation": int((source["task"][selected] % 4 == 0).sum()),
        "best_quantized_query_mean": float(quality[selected[::2]].mean()),
        "quantization_inputs": ["fitted weights", "budget of codes", "fixed zero code"],
        "candidate_ranking": "refit numeric codebooks using source observations and source query",
        "alignment_inputs": ["inferred sparse indexators", "source-query quality"],
        "teacher_read": False,
        "windows_or_filter_count_supplied": False,
        "validation_split": "entire source functions with task id divisible by 4",
        "code_group_distance": "complete-linkage Jaccard, cut .2",
    }
    (args.out / "extraction_complete.json").write_text(json.dumps(summary, indent=2))
    shutil.copy2(__file__, args.out / "extract_snapshot.py")
    print(summary, flush=True)


if __name__ == "__main__":
    main()

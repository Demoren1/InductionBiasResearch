"""GPU pilot for learned partitions with a source-selected number of groups."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch


def basis(indices):
    labels = torch.unique(indices[indices > 0])
    if not len(labels):
        return torch.zeros(len(indices), 0, device=indices.device)
    assignments = (indices[:, None] == labels[None]).to(torch.float64)
    return assignments / assignments.sum(0).sqrt()


def coefficient_draw(count, reference, residual):
    dimensions, groups = reference.shape
    coefficients = (
        torch.randn(count, groups, device=reference.device) / groups**0.5
    )
    weights = coefficients @ reference.T
    if residual:
        extra = torch.randn(count, dimensions, device=reference.device)
        extra -= (extra @ reference) @ reference.T
        extra *= residual / extra.norm(dim=1, keepdim=True)
        weights += extra
    return weights


def observations(weights, count, noise):
    inputs = torch.randn(
        len(weights), count, weights.shape[1], device=weights.device
    )
    signal = torch.einsum("tnp,tp->tn", inputs, weights)
    labels = signal + noise * torch.randn_like(signal)
    return inputs, labels, signal


def least_squares(inputs, labels, subspace=None):
    design = inputs if subspace is None else inputs @ subspace
    if design.shape[-1] == 0:
        return torch.zeros(len(inputs), inputs.shape[-1], device=inputs.device)
    solution = (torch.linalg.pinv(design, rtol=1e-10) @ labels[..., None])[
        ..., 0
    ]
    return solution if subspace is None else solution @ subspace.T


def ridge(inputs, labels, strength):
    samples = inputs.shape[1]
    gram = inputs @ inputs.transpose(1, 2)
    gram += strength * samples * torch.eye(samples, device=inputs.device)
    dual = torch.linalg.solve(gram, labels[..., None])
    return (inputs.transpose(1, 2) @ dual)[..., 0]


def ward_hierarchy(source_weights):
    dimensions = source_weights.shape[1]
    means = source_weights.T.clone()
    counts = torch.ones(dimensions, device=means.device)
    active = torch.ones(dimensions, dtype=torch.bool, device=means.device)
    indices = torch.arange(1, dimensions + 1, device=means.device)
    hierarchy = {dimensions: indices.clone()}
    for groups in range(dimensions, 0, -1):
        distances = torch.cdist(means, means).square()
        distances *= (
            counts[:, None] * counts[None] / (counts[:, None] + counts[None])
        )
        distances[~(active[:, None] & active[None])] = torch.inf
        distances.fill_diagonal_(torch.inf)
        zero_cost = counts * means.square().sum(1)
        zero_cost[~active] = torch.inf
        pair_cost, pair_flat = distances.flatten().min(0)
        remove_cost, remove_id = zero_cost.min(0)
        if remove_cost <= pair_cost:
            removed = int(remove_id)
            indices[indices == removed + 1] = 0
        else:
            kept, removed = divmod(int(pair_flat), dimensions)
            total = counts[kept] + counts[removed]
            means[kept] = (
                counts[kept] * means[kept] + counts[removed] * means[removed]
            ) / total
            counts[kept] = total
            indices[indices == removed + 1] = kept + 1
        active[removed] = False
        hierarchy[groups - 1] = indices.clone()
    return hierarchy


def query_mse(weights, query):
    inputs, labels, _ = query
    return (
        (torch.einsum("tnp,tp->tn", inputs, weights) - labels).square().mean(1)
    )


def positive_recall(prediction, target):
    positive = target > 0
    return ((prediction > 0) & positive).sum(1) / positive.sum(1).clamp_min(1)


def gradient_curves(inputs, labels, query, subspace, steps, interval):
    design = inputs if subspace is None else inputs @ subspace
    query_design = query[0] if subspace is None else query[0] @ subspace
    count, samples, dimensions = design.shape
    values = torch.zeros(count, dimensions, device=inputs.device)
    if dimensions:
        largest = torch.linalg.svdvals(design)[:, 0].square() / samples
        rate = 0.9 / largest.clamp_min(1e-12)
    records = []
    for step in range(steps + 1):
        prediction = torch.einsum("tnk,tk->tn", design, values)
        if step % interval == 0 or step == steps:
            heldout = torch.einsum("tnk,tk->tn", query_design, values)
            records.append(
                {
                    "step": step,
                    "train_mse": float((prediction - labels).square().mean()),
                    "query_mse": float((heldout - query[1]).square().mean()),
                    "train_positive_recall": float(
                        positive_recall(prediction, labels).mean()
                    ),
                    "query_positive_recall": float(
                        positive_recall(heldout, query[1]).mean()
                    ),
                }
            )
        if step < steps and dimensions:
            gradient = (
                torch.einsum("tnk,tn->tk", design, prediction - labels)
                / samples
            )
            values -= rate[:, None] * gradient
    weights = values if subspace is None else values @ subspace.T
    return weights, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        default="experiments/adaptive_indexators_v1/protocol.json",
    )
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    plan = json.loads(protocol_path.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; no CPU fallback")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(protocol_path.read_text())
    (output / "environment.json").write_text(
        json.dumps(
            {
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(device),
                "protocol_sha256": hashlib.sha256(
                    protocol_path.read_bytes()
                ).hexdigest(),
                "source_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "started_unix": time.time(),
            },
            indent=2,
        )
        + "\n"
    )
    all_rows, selections, curves, timings = [], [], [], []
    dimensions = plan["dimensions"]
    for truth_groups in plan["true_groups"]:
        for residual in plan["residual_norms"]:
            for bank in range(plan["banks"]):
                started = time.perf_counter()
                seed = plan["seed"] + 10000 * truth_groups + 100 * bank
                seed += int(residual * 1000)
                torch.manual_seed(seed)
                true_indices = torch.cat(
                    [
                        torch.zeros(
                            plan["inactive"], dtype=torch.long, device=device
                        ),
                        torch.arange(
                            1, truth_groups + 1, device=device
                        ).repeat_interleave(
                            (dimensions - plan["inactive"]) // truth_groups
                        ),
                    ]
                )
                true_indices = true_indices[
                    torch.randperm(dimensions, device=device)
                ]
                reference = basis(true_indices)
                source_truth = coefficient_draw(
                    plan["source_tasks"], reference, residual
                )
                source_x, source_y, _ = observations(
                    source_truth, plan["source_examples"], plan["noise_std"]
                )
                source_weights = least_squares(source_x, source_y)
                torch.cuda.synchronize()
                source_seconds = time.perf_counter() - started
                hierarchy_started = time.perf_counter()
                hierarchy = ward_hierarchy(source_weights)
                pca, _, _ = torch.linalg.svd(
                    source_weights.T, full_matrices=False
                )
                torch.cuda.synchronize()
                hierarchy_seconds = time.perf_counter() - hierarchy_started
                calibration_started = time.perf_counter()
                calibration_truth = coefficient_draw(
                    plan["calibration_tasks"], reference, residual
                )
                calibration_x, calibration_y, _ = observations(
                    calibration_truth,
                    max(plan["support_sizes"]),
                    plan["noise_std"],
                )
                calibration_query = observations(
                    calibration_truth,
                    plan["query_examples"],
                    plan["noise_std"],
                )
                frozen = {}
                for size in plan["support_sizes"]:
                    x, y = calibration_x[:, :size], calibration_y[:, :size]
                    path_losses = []
                    for groups in range(dimensions + 1):
                        fitted = least_squares(x, y, basis(hierarchy[groups]))
                        path_losses.append(
                            float(query_mse(fitted, calibration_query).mean())
                        )
                    pca_losses = []
                    for rank in range(pca.shape[1] + 1):
                        fitted = least_squares(x, y, pca[:, :rank])
                        pca_losses.append(
                            float(query_mse(fitted, calibration_query).mean())
                        )
                    ridge_losses = [
                        float(
                            query_mse(
                                ridge(x, y, value), calibration_query
                            ).mean()
                        )
                        for value in plan["ridge_grid"]
                    ]
                    selected_k = min(
                        range(len(path_losses)), key=path_losses.__getitem__
                    )
                    selected_rank = min(
                        range(len(pca_losses)), key=pca_losses.__getitem__
                    )
                    ridge_id = min(
                        range(len(ridge_losses)), key=ridge_losses.__getitem__
                    )
                    frozen[size] = {
                        "groups": selected_k,
                        "rank": selected_rank,
                        "ridge": plan["ridge_grid"][ridge_id],
                        "partition_query_path": path_losses,
                        "pca_query_path": pca_losses,
                        "ridge_query_path": ridge_losses,
                    }
                    selections.append(
                        {
                            "truth_groups": truth_groups,
                            "residual": residual,
                            "bank": bank,
                            "size": size,
                            **frozen[size],
                        }
                    )
                torch.cuda.synchronize()
                calibration_seconds = time.perf_counter() - calibration_started
                tag = f"k{truth_groups}_r{residual:.2f}_b{bank}"
                (output / f"{tag}_frozen.json").write_text(
                    json.dumps(frozen, indent=2) + "\n"
                )
                target_started = time.perf_counter()
                torch.manual_seed(seed + 5000000)
                target_truth = coefficient_draw(
                    plan["target_tasks"], reference, residual
                )
                target_x, target_y, _ = observations(
                    target_truth, max(plan["support_sizes"]), plan["noise_std"]
                )
                target_query = observations(
                    target_truth, plan["query_examples"], plan["noise_std"]
                )
                target_test = observations(
                    target_truth, plan["test_examples"], plan["noise_std"]
                )
                saved_weights = {}
                for size in plan["support_sizes"]:
                    x, y = target_x[:, :size], target_y[:, :size]
                    selected = frozen[size]
                    indices = hierarchy[selected["groups"]]
                    structured = basis(indices)
                    random_basis = basis(
                        indices[torch.randperm(dimensions, device=device)]
                    )
                    models = {
                        "dense_min_norm": least_squares(x, y),
                        "dense_ridge": ridge(x, y, selected["ridge"]),
                        "adaptive_partition": least_squares(x, y, structured),
                        "base_partition": least_squares(x, y, reference),
                        "source_pca": least_squares(
                            x, y, pca[:, : selected["rank"]]
                        ),
                        "random_partition": least_squares(x, y, random_basis),
                    }
                    for method, space in [
                        ("dense_gd", None),
                        ("adaptive_gd", structured),
                    ]:
                        models[method], history = gradient_curves(
                            x,
                            y,
                            target_query,
                            space,
                            plan["steps"],
                            plan["log_every"],
                        )
                        curves.extend(
                            {
                                "truth_groups": truth_groups,
                                "residual": residual,
                                "bank": bank,
                                "size": size,
                                "method": method,
                                **row,
                            }
                            for row in history
                        )
                    for method, weights in models.items():
                        prediction = torch.einsum(
                            "tnp,tp->tn", target_test[0], weights
                        )
                        population_mse = (
                            (weights - target_truth).square().sum(1)
                        )
                        train_mse = (
                            (torch.einsum("tnp,tp->tn", x, weights) - y)
                            .square()
                            .mean(1)
                        )
                        test_mse = (
                            (prediction - target_test[1]).square().mean(1)
                        )
                        recall = positive_recall(prediction, target_test[2])
                        for task in range(plan["target_tasks"]):
                            all_rows.append(
                                {
                                    "truth_groups": truth_groups,
                                    "residual": residual,
                                    "bank": bank,
                                    "size": size,
                                    "task": task,
                                    "method": method,
                                    "selected_groups": selected["groups"],
                                    "selected_rank": selected["rank"],
                                    "population_mse": float(
                                        population_mse[task]
                                    ),
                                    "train_mse": float(train_mse[task]),
                                    "test_mse": float(test_mse[task]),
                                    "positive_recall": float(recall[task]),
                                    "success": int(
                                        population_mse[task]
                                        <= plan["success_mse"]
                                    ),
                                }
                            )
                        saved_weights[f"n{size}_{method}"] = (
                            weights.cpu().numpy()
                        )
                    saved_weights[f"n{size}_indices"] = indices.cpu().numpy()
                torch.cuda.synchronize()
                timings.append(
                    {
                        "bank": tag,
                        "source_seconds": source_seconds,
                        "hierarchy_seconds": hierarchy_seconds,
                        "calibration_seconds": calibration_seconds,
                        "target_seconds": time.perf_counter() - target_started,
                    }
                )
                np.savez_compressed(
                    output / f"{tag}.npz",
                    true_indices=true_indices.cpu().numpy(),
                    source_weights=source_weights.cpu().numpy(),
                    target_truth=target_truth.cpu().numpy(),
                    **saved_weights,
                )
                print(
                    "DONE",
                    tag,
                    {size: frozen[size]["groups"] for size in frozen},
                    flush=True,
                )
                with (output / "metrics.csv").open("w") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=list(all_rows[0])
                    )
                    writer.writeheader()
                    writer.writerows(all_rows)
                for name, values in [
                    ("selection", selections),
                    ("curves", curves),
                    ("timings", timings),
                ]:
                    (output / f"{name}.json").write_text(
                        json.dumps(values, indent=2) + "\n"
                    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "banks": len(timings),
                "metric_rows": len(all_rows),
                "finished_unix": time.time(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

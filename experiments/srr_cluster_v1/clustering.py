"""Matched profile geometries and clustering algorithms."""

import sys
import time
from importlib import import_module, machinery, util
from pathlib import Path

import numpy as np
import torch
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.spatial.distance import pdist

from experiments.srr_approx_v1.core import canonical, orient

METHODS = ("ward", "single", "complete", "lloyd", "signed", "emeans")
VENDOR = Path("artifacts/local/srr_clustering_2026-11-15/vendor")


def lifted(profiles):
    dimension = profiles.shape[1]
    left, right = np.triu_indices(dimension)
    norms = np.linalg.norm(profiles, axis=1).clip(1e-30)
    result = profiles[:, left] * profiles[:, right] / norms[:, None]
    result[:, left != right] *= np.sqrt(2)
    return result


def author_emeans():
    """Load the pinned PyTorch backend without importing the NumPy backend."""
    name = "ibr_pinned_emeans"
    if name not in sys.modules:
        spec = machinery.ModuleSpec(name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(VENDOR.resolve())]
        sys.modules[name] = util.module_from_spec(spec)
    return import_module(f"{name}.e_means_quantized").QuantizedEmeans


def repair(labels, residual, count):
    """Split high-residual non-singleton members until all labels are used."""
    labels = labels.copy()
    sizes = np.bincount(labels, minlength=count)
    missing = np.flatnonzero(sizes == 0)
    order = np.argsort(-residual, kind="stable")
    cursor = 0
    for group in missing:
        while sizes[labels[order[cursor]]] <= 1:
            cursor += 1
        position = order[cursor]
        sizes[labels[position]] -= 1
        labels[position] = group
        sizes[group] = 1
        cursor += 1
    return labels, len(missing)


def nearest(data, centers, signed=False):
    distances, labels, signs = [], [], []
    center_norm = centers.square().sum(1)
    for batch in data.split(256):
        product = batch @ centers.T
        costs = batch.square().sum(1)[:, None] + center_norm[None]
        costs = costs - 2 * (product.abs() if signed else product)
        value, label = costs.min(1)
        sign = torch.where(product.gather(1, label[:, None]).flatten() >= 0, 1.0, -1.0)
        distances.append(value.clamp_min(0))
        labels.append(label)
        signs.append(sign if signed else torch.ones_like(sign))
    return torch.cat(distances), torch.cat(labels), torch.cat(signs)


def source_objective(data, labels, signs, count):
    counts = np.bincount(labels, minlength=count).clip(1)
    centers = np.zeros((count, data.shape[1]))
    np.add.at(centers, labels, data * signs[:, None])
    centers /= counts[:, None]
    residual = np.sum((data * signs[:, None] - centers[labels]) ** 2, axis=1)
    return float(residual.mean()), residual


def centroid_fit(features, count, method, seed, device):
    """Source-only selection of fixed initializations and e-means recipes."""
    data = torch.as_tensor(features, dtype=torch.float32, device=device)
    candidates, diagnostics = [], []
    for restart in range(4):
        run_seed = seed + restart
        initial_ids = np.random.default_rng(run_seed).choice(
            len(data), count, replace=False
        )
        initial = data[torch.as_tensor(initial_ids, device=device)].clone()
        recipes = (("auto", 5), ("quality", 5), ("quality", 25))
        if method != "emeans":
            recipes = (("full", 25),)
        for law, steps in recipes:
            started = time.perf_counter()
            if method == "emeans":
                model = author_emeans()(
                    data.shape[1],
                    count,
                    nepochs=steps,
                    batch_law=law,
                    seed=run_seed,
                    assignment_backend="native",
                    quantizer=None,
                    assignment_chunk_size=256,
                )
                _, labels = model.train(data, init_centroids=initial)
                labels = labels.cpu().numpy()
                signs = np.ones(len(data))
                history = model.obj.cpu().tolist()
                recipe = model.recipe
                _, residual = source_objective(features, labels, signs, count)
                labels, repairs = repair(labels, residual, count)
            else:
                centers, history, repairs = initial, [], 0
                previous = previous_sign = None
                for _ in range(steps):
                    cost, label, sign = nearest(data, centers, method == "signed")
                    history.append(float(cost.mean()))
                    fixed, changed = repair(
                        label.cpu().numpy(), cost.cpu().numpy(), count
                    )
                    repairs += changed
                    label = torch.as_tensor(fixed, device=device)
                    centers = torch.zeros_like(centers)
                    centers.index_add_(0, label, data * sign[:, None])
                    centers /= torch.bincount(label, minlength=count)[:, None]
                    if (
                        previous is not None
                        and torch.equal(previous, label)
                        and torch.equal(previous_sign, sign)
                    ):
                        break
                    previous = label.clone()
                    previous_sign = sign.clone()
                labels, signs = label.cpu().numpy(), sign.cpu().numpy()
                recipe = {"steps": len(history), "initialization": "random_data"}
            objective, _ = source_objective(features, labels, signs, count)
            diagnostics.append(
                {
                    "seed": run_seed,
                    "recipe": recipe,
                    "history": history,
                    "source_sse_per_position": objective,
                    "empty_group_repairs": repairs,
                    "seconds": time.perf_counter() - started,
                }
            )
            candidates.append(
                (objective, labels.copy(), signs.copy(), len(diagnostics) - 1)
            )
    _, labels, signs, chosen = min(candidates, key=lambda item: item[0])
    return labels, signs, {"chosen": chosen, "runs": diagnostics}


def build_bank(weights, method, budgets, scopes, *, device="cpu", seed=3511159150):
    """Build matched scope/budget candidates, keeping local allocation fixed."""
    _, dimensions, hidden = weights.shape
    bank, records = {}, []
    for scope in scopes:
        if scope == "global":
            parts = [weights.reshape(len(weights), -1).T]
        elif scope == "local":
            parts = [weights[:, :, neuron].T for neuron in range(hidden)]
        elif scope == "input":
            parts = [weights.transpose(1, 0, 2).reshape(dimensions, -1)]
        else:
            raise ValueError(scope)
        signed_geometry = scope != "input"
        actual_method = "lloyd" if scope == "input" and method == "signed" else method
        transformed = [
            lifted(part) if signed_geometry and method != "signed" else part
            for part in parts
        ]
        hierarchical = method in ("ward", "single", "complete", "average")
        trees = []
        if hierarchical:
            trees = [linkage(pdist(part), method=method) for part in transformed]
        for budget in budgets:
            if budget == dimensions * hidden:
                continue
            clusters = budget // hidden if scope in ("local", "input") else budget
            if clusters < 1 or clusters > len(parts[0]):
                continue
            started = time.perf_counter()
            indices, details = [], []
            for number, (profiles, features) in enumerate(zip(parts, transformed)):
                if clusters == 1:
                    labels, detail = np.zeros(len(profiles), dtype=int), {}
                elif hierarchical:
                    labels = cut_tree(trees[number], n_clusters=[clusters])[:, 0]
                    detail = {}
                else:
                    labels, fitted_signs, detail = centroid_fit(
                        features,
                        clusters,
                        actual_method,
                        seed + number * 1009,
                        device,
                    )
                if method == "signed" and signed_geometry and clusters > 1:
                    value = (labels + 1) * fitted_signs.astype(np.int16)
                else:
                    value = orient(profiles, labels) if signed_geometry else labels + 1
                indices.append(value)
                details.append(detail)
            if scope == "global":
                index = indices[0].reshape(dimensions, hidden)
            elif scope == "input":
                index = (indices[0][:, None] - 1) * hidden + np.arange(1, hidden + 1)
            else:
                index = np.stack(
                    [
                        np.sign(value) * (np.abs(value) + neuron * clusters)
                        for neuron, value in enumerate(indices)
                    ],
                    axis=1,
                )
            name = f"{method}_{scope}_m{budget}"
            bank[name] = canonical(index.astype(np.int16))
            records.append(
                {
                    "name": name,
                    "method": method,
                    "scope": scope,
                    "codes": int(np.abs(index).max()),
                    "seconds": time.perf_counter() - started,
                    "clustering": details,
                }
            )
    return bank, records

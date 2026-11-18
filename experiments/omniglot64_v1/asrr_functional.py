"""Refine aSRR with input covariance and vary sharing across neuron blocks."""

import argparse
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical
from experiments.srr_cluster_v1.clustering import centroid_fit

from .clustering import CLUSTER_SEED
from .source import (
    BANKS,
    CLASSES,
    CODES,
    HIDDEN,
    INPUTS,
    OUTPUT,
    ROOT,
    WEIGHTS,
    read_json,
    save_groups,
)
from .transfer import PLAN

INPUT_GROUPS = CODES // HIDDEN
SWEEPS = 4
METHODS = (
    "asrr_fit_activations",
    "asrr_fit_predictive",
    "asrr_two_partitions",
    "asrr_four_partitions",
    "asrr_two_functional",
    "asrr_signed_inputs",
)


def fit_partition(profiles, metric, initial):
    """Alternate code refitting and exact one-pixel assignment changes."""
    labels = initial.copy()
    weighted = metric @ profiles
    teacher_energy = float(np.sum(profiles * weighted))
    one_hot = np.eye(INPUT_GROUPS)
    sizes = np.bincount(labels, minlength=INPUT_GROUPS)
    diagonals = np.diag(metric)
    order = np.argsort(-diagonals, kind="stable")

    membership = one_hot[labels]
    centers = np.linalg.solve(
        membership.T @ metric @ membership, membership.T @ weighted
    )
    losses = [teacher_energy - float(np.sum(centers * (membership.T @ weighted)))]
    moves = []
    for _ in range(SWEEPS):
        center_gram = centers @ centers.T
        center_norm = np.diag(center_gram)
        difference_norm = center_norm[None, :] + center_norm[:, None] - 2 * center_gram
        signal = (weighted - (metric @ membership) @ centers) @ centers.T
        changed = 0
        for pixel in order:
            old = labels[pixel]
            if sizes[old] == 1:
                continue
            change = diagonals[pixel] * difference_norm[old]
            change -= 2 * (signal[pixel] - signal[pixel, old])
            new = int(np.argmin(change))
            if change[new] >= -1e-12:
                continue
            labels[pixel] = new
            sizes[old] -= 1
            sizes[new] += 1
            signal -= metric[:, pixel, None] * (center_gram[new] - center_gram[old])
            changed += 1
        membership = one_hot[labels]
        rhs = membership.T @ weighted
        centers = np.linalg.solve(membership.T @ metric @ membership, rhs)
        losses.append(teacher_energy - float(np.sum(centers * rhs)))
        moves.append(changed)
        if not changed:
            break
    return labels, {"loss": losses, "moved_inputs": moves}


def neuron_importance(parameters, images, aligned):
    """Approximate the logit sensitivity of a normalized first-layer column."""
    raw = parameters[:, :WEIGHTS].reshape(-1, INPUTS, HIDDEN)
    bias = parameters[:, WEIGHTS : WEIGHTS + HIDDEN]
    head = parameters[:, WEIGHTS + HIDDEN : -CLASSES].reshape(-1, HIDDEN, CLASSES)
    head -= head.mean(axis=-1, keepdims=True)
    scales = np.linalg.norm(raw, axis=1)
    gate_rate = ((images @ raw + bias[:, None]) > 0).mean(axis=1)
    relevance = scales * np.sqrt(gate_rate) * np.linalg.norm(head, axis=-1)
    normal = raw / np.maximum(scales[:, None], 1e-12)
    importance = np.empty_like(relevance)
    for task, (source, target) in enumerate(zip(normal, aligned, strict=True)):
        _, order = linear_sum_assignment(-target.T @ source)
        importance[task] = relevance[task, order]
    rms = np.sqrt(np.square(importance).mean(axis=1, keepdims=True))
    return np.clip(importance / np.maximum(rms, 1e-12), 0.25, 3.0)


def indices(partitions, signs=None):
    labels = np.repeat(np.stack(partitions, axis=1), HIDDEN // len(partitions), axis=1)
    index = labels * HIDDEN + np.arange(1, HIDDEN + 1)
    if signs is not None:
        index = index * signs[:, None].astype(np.int16)
    return canonical(index)


def cluster_neuron_blocks(profiles, block_count):
    partitions = []
    for number, block in enumerate(np.split(profiles, block_count, axis=2)):
        points = block.transpose(1, 0, 2).reshape(INPUTS, -1)
        labels, _, _ = centroid_fit(
            points, INPUT_GROUPS, "lloyd", CLUSTER_SEED + number * 1009, "cpu"
        )
        partitions.append(labels)
    return indices(partitions)


def construct():
    destination = OUTPUT / "asrr_functional_registration.json"
    if destination.exists():
        raise FileExistsError(destination)
    boundary = StudyBoundary(ROOT)
    boundary.install()
    torch.set_num_threads(2)
    started = time.perf_counter()
    dense = ROOT / "source/dense/models.npz"
    registration = read_json(OUTPUT / "registration.json")
    dense_hash = digest(dense)
    if dense_hash != read_json(dense.with_name("metrics.json"))["models_sha256"]:
        raise ValueError("Changed Dense source fits")
    (aligned, average, _, ranking) = source_profiles(dense, INPUTS, HIDDEN)
    images = []
    for phase in ("support", "numerical"):
        path = ROOT / "public" / f"{phase}.npz"
        if digest(path) != registration["source_panel_sha256"][phase]:
            raise ValueError(f"Changed source inputs: {phase}")
        with np.load(path) as panel:
            images.append(panel["x"][ranking].astype(float) / 255)
    images = np.concatenate(images, axis=1)
    with np.load(dense) as data:
        parameters = data["parameters"][ranking].astype(float)
    importance = neuron_importance(parameters, images, aligned)
    x = images.reshape(-1, INPUTS)
    centered = x - x.mean(axis=0)
    metric = centered.T @ centered / len(centered)
    ridge = float(np.trace(metric) / INPUTS)
    metric.flat[:: INPUTS + 1] += ridge

    baselines = {}
    for method in ("asrr", "asrr_lloyd"):
        metadata = read_json(OUTPUT / f"{method}_source.json")["candidates"][0]
        path = BANKS / f"{metadata['name']}.npz"
        if digest(path) != metadata["model_sha256"]:
            raise ValueError(f"Changed starting partition: {method}")
        with np.load(path) as model:
            baselines[method] = np.unique(model["index"][:, 0], return_inverse=True)[1]
    common_seconds = time.perf_counter() - started
    ward, lloyd = (baselines[method] for method in ("asrr", "asrr_lloyd"))
    weighted = average * importance[:, None, :]
    features = average.transpose(1, 0, 2).reshape(INPUTS, -1)
    predictive = weighted.transpose(1, 0, 2).reshape(INPUTS, -1)

    started = time.perf_counter() - common_seconds
    labels, trace = fit_partition(features, metric, lloyd)
    name = "asrr_fit_activations"
    save_groups(
        name,
        [(name, indices([labels]), {"starting": "asrr_lloyd", **trace})],
        started,
        None,
    )
    started = time.perf_counter() - common_seconds
    labels, trace = fit_partition(predictive, metric, lloyd)
    name = "asrr_fit_predictive"
    save_groups(
        name,
        [(name, indices([labels]), {"starting": "asrr_lloyd", **trace})],
        started,
        None,
    )

    started = time.perf_counter()
    name = "asrr_two_partitions"
    save_groups(
        name, [(name, indices([ward, lloyd]), {"partitions": 2})], started, None
    )

    started = time.perf_counter() - common_seconds
    name = "asrr_four_partitions"
    index = cluster_neuron_blocks(average, 4)
    save_groups(name, [(name, index, {"partitions": 4})], started, None)

    started = time.perf_counter() - common_seconds
    parts, traces = [], []
    for block, initial in zip(
        np.split(weighted, 2, axis=2), (ward, lloyd), strict=True
    ):
        points = block.transpose(1, 0, 2).reshape(INPUTS, -1)
        labels, trace = fit_partition(points, metric, initial)
        parts.append(labels)
        traces.append(trace)
    name = "asrr_two_functional"
    save_groups(
        name,
        [(name, indices(parts), {"partitions": 2, "trace": traces})],
        started,
        None,
    )

    started = time.perf_counter() - common_seconds
    labels, signs, _ = centroid_fit(
        features, INPUT_GROUPS, "signed", CLUSTER_SEED, "cpu"
    )
    name = "asrr_signed_inputs"
    save_groups(name, [(name, indices([labels], signs), {})], started, None)

    save_json(
        destination,
        {
            "methods": METHODS,
            "codes": CODES,
            "source_sha256": dense_hash,
            "target_plan_sha256": digest(PLAN),
            "code_selection": "Maximum development accuracy, cross-entropy tie-breaker",
            "covariance_ridge": ridge,
            "source_images_for_covariance": len(centered),
            "sweeps": SWEEPS,
            "timing": "Includes shared input preparation for new constructions; two cached starting partitions are charged separately",
        },
    )
    boundary.save(OUTPUT / "asrr_functional_source_access.json", "complete")


def construct_eight_partitions():
    name = "asrr_eight_partitions"
    destination = OUTPUT / f"{name}_source.json"
    model = BANKS / f"{name}.npz"
    if destination.exists() or model.exists():
        raise FileExistsError(f"Already constructed: {name}")
    boundary = StudyBoundary(ROOT)
    boundary.install()
    torch.set_num_threads(2)
    started = time.perf_counter()
    dense = ROOT / "source/dense/models.npz"
    source_hash = digest(dense)
    reference = read_json(OUTPUT / "asrr_functional_registration.json")
    if source_hash != reference["source_sha256"]:
        raise ValueError("Changed Dense archive since the four-partition experiment")
    _, average, _, _ = source_profiles(dense, INPUTS, HIDDEN)
    index = cluster_neuron_blocks(average, 8)
    save_groups(
        name,
        [(name, index, {"partitions": 8, "source_sha256": source_hash})],
        started,
        None,
    )
    boundary.save(OUTPUT / f"{name}_source_access.json", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eight-partitions", action="store_true")
    if parser.parse_args().eight_partitions:
        construct_eight_partitions()
    else:
        construct()

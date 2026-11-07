"""Official image evaluation after source-only architecture selection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torchvision.datasets import MNIST, FashionMNIST

from experiments.darts_indexator_v1.model import canonical_signed_probabilities

from .data import BASE, digest, save_json
from .images import materialize, oracle_indices, permute, points, prediction, scores


def rename_groups(indices, reference, codes):
    positive = np.zeros((codes, codes), dtype=int)
    negative = np.zeros_like(positive)
    for source in range(1, codes + 1):
        for target in range(1, codes + 1):
            positive[source - 1, target - 1] = (
                (indices == source) & (reference == target)
            ).sum()
            negative[source - 1, target - 1] = (
                (indices == -source) & (reference == target)
            ).sum()
    left, right = linear_sum_assignment(-np.maximum(positive, negative))
    labels = np.zeros(codes + 1, dtype=int)
    for source, target in zip(left, right):
        labels[source + 1] = (target + 1) * (
            1 if positive[source, target] >= negative[source, target] else -1
        )
    return labels[np.abs(indices)] * np.sign(indices)


def reference_alignment(indices, spec):
    reference = oracle_indices(spec)
    mask, expected = indices[0] != 0, reference[0] != 0
    pool, expected_pool = indices[2] != 0, reference[2] != 0
    cost = (mask[:, :, None] != expected[:, None, :]).sum(0)
    cost += 3 * (pool[:, None, :] != expected_pool[None, :, :]).sum(-1)
    source, target = linear_sum_assignment(cost)
    order = source[np.argsort(target)]
    reordered = [indices[0][:, order], indices[1][order], indices[2][order]]
    aligned = [
        rename_groups(value, truth, codes)
        for value, truth, codes in zip(reordered, reference, spec["codes"])
    ]
    diagnostics = {}
    for name, value, truth in zip(("input", "bias", "pool"), aligned, reference):
        active, expected = value != 0, truth != 0
        true_positive = int((active & expected).sum())
        precision = true_positive / max(int(active.sum()), 1)
        recall = true_positive / max(int(expected.sum()), 1)
        diagnostics[name] = {
            "categorical_disagreements": int((value != truth).sum()),
            "positions": int(value.size),
            "active_entries": int(active.sum()),
            "reference_active_entries": int(expected.sum()),
            "mask_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        }
    return diagnostics, aligned, order


def evaluate(output, device):
    spec = json.loads((output / "specification.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen image artifact changed: {name}")
    manifest = output / "heldout_manifest.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    dataset_class = {"MNIST": MNIST, "FashionMNIST": FashionMNIST}[spec["dataset"]]
    dataset = dataset_class(
        Path("experiments/structure_search_v2/datasets"), train=False, download=False
    )
    test_points = points(dataset.data, spec["grid"]).to(device)
    y = dataset.targets.to(device)
    generator = torch.Generator(device=device).manual_seed(spec["seed"] + 9000)
    x = permute(test_points, generator)
    save_json(
        manifest,
        {
            "created_unix": time.time(),
            "frozen_utc": frozen["frozen_utc"],
            "samples": len(y),
            "permutation_seed": spec["seed"] + 9000,
            "official_test_opened_after_freeze": True,
        },
    )
    selection = json.loads((output / "confirmation_selection.json").read_text())[
        "primary"
    ]
    controls = json.loads((output / "control_selection.json").read_text())
    with np.load(output / "confirmation_hard/fits.npz") as archive:
        fits = dict(archive)
    row = selection["row"]
    candidates = {"learned_inherited": (fits, row, int(fits["query_ce"][row].argmin()))}
    for variant, methods in controls.items():
        with np.load(output / variant / "fits.npz") as archive:
            result = dict(archive)
        for method, choice in methods.items():
            candidates[variant + "_" + method] = (
                result,
                choice["row"],
                choice["restart"],
            )
    with np.load(output / "fresh/fits.npz") as archive:
        result = dict(archive)
    chosen = json.loads((output / "fresh_selection.json").read_text())
    candidates["learned_fresh"] = result, 0, chosen["restart"]
    metrics, states = {}, {}
    predictions = {"labels": y.cpu().numpy()}
    for name, (result, number, restart) in candidates.items():
        numeric = torch.as_tensor(result["numeric"][number : number + 1], device=device)
        matrices = [
            torch.as_tensor(result[f"matrix_{i}"][number : number + 1], device=device)
            for i in range(3)
        ]
        tested = scores(x, y, matrices, numeric, spec).cpu().numpy()[0]
        item = {
            "test_ce": float(tested[restart, 0]),
            "test_accuracy": float(tested[restart, 1]),
            "all_restart_accuracy": tested[:, 1].tolist(),
            "query_selected_restart": restart,
            "selection_ce": float(result["selection"][number, restart, 0]),
            "selected_step": int(result["step"][number, restart]),
        }
        one_numeric = numeric[:, restart : restart + 1]
        one_matrices = [matrix[:, restart : restart + 1] for matrix in matrices]
        logits = []
        with torch.no_grad():
            for batch in x.split(256):
                logits.append(prediction(batch, one_matrices, one_numeric, spec)[0, 0])
            predicted = torch.cat(logits).argmax(-1)
            predictions[name] = predicted.cpu().numpy()
            probabilities = []
            for permutation in range(8):
                rng = torch.Generator(device=device).manual_seed(
                    spec["seed"] + 9100 + permutation
                )
                inputs = permute(test_points[:128], rng)
                probabilities.append(
                    prediction(inputs, one_matrices, one_numeric, spec).softmax(-1)[
                        0, 0
                    ]
                )
            probabilities = torch.stack(probabilities)
            item["permutation_probability_rms"] = float(
                (probabilities - probabilities.mean(0, keepdim=True))
                .square()
                .mean()
                .sqrt()
            )
            item["permutation_prediction_agreement"] = float(
                (probabilities.argmax(-1) == probabilities[0].argmax(-1)).float().mean()
            )
        metrics[name] = item
        states[name + "_numeric"] = one_numeric.cpu().numpy()[0, 0]
        for group, matrix in enumerate(one_matrices):
            states[f"{name}_matrix_{group}"] = matrix.cpu().numpy()[0, 0]
    with np.load(output / "confirmation_hard/indices.npz") as archive:
        indices = [archive[f"indices_{i}"][row] for i in range(3)]
    diagnostics, aligned, order = reference_alignment(indices, spec)
    states["hidden_column_order"] = order
    for group, (actual, matching, reference) in enumerate(
        zip(indices, aligned, oracle_indices(spec))
    ):
        states[f"indices_{group}"] = actual
        states[f"aligned_indices_{group}"] = matching
        states[f"reference_indices_{group}"] = reference
    with np.load(output / "confirmation_search/snapshots.npz") as archive:
        raw = [
            torch.tensor(
                archive[f"probability_{i}"][
                    selection["snapshot"], selection["root"] : selection["root"] + 1
                ],
                device=device,
            )
            for i in range(3)
        ]
        numeric = torch.tensor(
            archive["numeric"][
                selection["snapshot"], selection["root"] : selection["root"] + 1
            ],
            device=device,
        )
    canonical = [canonical_signed_probabilities(p) for p in raw]
    original_matrices = materialize(raw, numeric, spec)
    canonical_matrices = materialize(canonical, numeric, spec)
    projection = {
        "maximum_matrix_differences": [
            float((a - b).abs().max())
            for a, b in zip(original_matrices, canonical_matrices)
        ]
    }
    raw_indices = []
    for group, (p, codes) in enumerate(zip(raw, spec["codes"])):
        chosen = p.argmax(-1)
        index = torch.where(chosen > codes, codes - chosen, chosen).long()
        raw_indices.append(index)
        states[f"raw_probability_{group}"] = p.cpu().numpy()[0]
        states[f"canonical_probability_{group}"] = canonical[group].cpu().numpy()[0]
    for name, matrices in (
        ("soft", original_matrices),
        ("raw_argmax", materialize(None, numeric, spec, raw_indices)),
        (
            "projected_argmax",
            materialize(
                None,
                numeric,
                spec,
                [torch.tensor(index[None], device=device).long() for index in indices],
            ),
        ),
    ):
        projection[name + "_metrics"] = (
            scores(x, y, matrices, numeric, spec).cpu().tolist()[0]
        )
    np.savez_compressed(output / "selected_states.npz", **states)
    np.savez_compressed(output / "test_predictions.npz", **predictions)
    save_json(
        output / "selected_indexators.json",
        {
            "codes": spec["codes"],
            "input": indices[0].tolist(),
            "bias": indices[1].ravel().tolist(),
            "pool": indices[2].tolist(),
            "reference_alignment": diagnostics,
            "reference_is_not_ground_truth_for_real_labels": True,
        },
    )
    save_json(
        output / "summary.json",
        {
            "task": spec,
            "methods": metrics,
            "reference_agreement": diagnostics,
            "projection": projection,
            "selection": selection,
            "test_used_for_selection": False,
            "parameter_counts": {
                "numerical_codes": sum(spec["codes"]),
                "rho": spec["width"] * spec["rho_width"]
                + spec["rho_width"]
                + spec["rho_width"] * 10
                + 10,
                "assignment_logits": sum(
                    int(np.prod(shape)) * (2 * codes + 1)
                    for shape, codes in zip(
                        [indices[0].shape, indices[1].shape, indices[2].shape],
                        spec["codes"],
                    )
                ),
            },
        },
    )
    print(
        spec["dataset"],
        {name: round(item["test_accuracy"], 4) for name, item in metrics.items()},
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("MNIST", "FashionMNIST"), required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(
        BASE / "results/20261104" / ("images_" + args.dataset),
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()

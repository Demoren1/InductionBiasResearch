"""Evaluate frozen choices and compare actual assignments after selection."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment, minimize_scalar

from experiments.darts_indexator_v1.model import canonical_signed_probabilities

from .data import digest, save_json
from .model import fixed_weights, predict, score, soft_weights


def align_indices(indices, truth, codes):
    """Exhaust global group/sign names and optimally match hidden columns."""
    best = None
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            labels = np.r_[0, np.array(permutation) * np.array(signs)]
            renamed = labels[np.abs(indices)] * np.sign(indices)
            costs = (renamed[:, :, None] != truth[:, None, :]).sum(0)
            source, target = linear_sum_assignment(costs)
            error = int(costs[source, target].sum())
            if best is None or error < best[0]:
                columns = source[np.argsort(target)]
                best = error, renamed[:, columns], columns
    error, aligned, columns = best
    active, expected = aligned != 0, truth != 0
    true_positive = int((active & expected).sum())
    precision = true_positive / max(int(active.sum()), 1)
    recall = true_positive / max(int(expected.sum()), 1)
    return (
        {
            "categorical_errors": error,
            "categorical_positions": int(indices.size),
            "active_edges": int(active.sum()),
            "true_active_edges": int(expected.sum()),
            "mask_precision": precision,
            "mask_recall": recall,
            "mask_f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "groups_used": len(np.unique(np.abs(indices[indices != 0]))),
        },
        aligned,
        columns,
    )


def sine_population(weights, amplitude, truth, specification):
    direction = np.asarray(truth["direction"], dtype=np.float64)
    low = direction * specification["low_frequency"]
    high = direction * specification["frequency"]

    def kernel(left, right):
        return 0.5 * (
            np.exp(-((left - right) ** 2).sum(-1) / 2)
            - np.exp(-((left + right) ** 2).sum(-1) / 2)
        )

    energy = (
        specification["low_amplitude"] ** 2 * kernel(low, low)
        + kernel(high, high)
        + 2 * specification["low_amplitude"] * kernel(low, high)
    )
    correlation = specification["low_amplitude"] * kernel(weights, low) + kernel(
        weights, high
    )
    return (
        amplitude**2 * kernel(weights, weights) - 2 * amplitude * correlation + energy
    )


def sine_geometry(truth, specification):
    direction = np.asarray(truth["direction"], dtype=np.float64)

    def optimum_amplitude(frequency):
        weights = frequency * direction
        low = specification["low_frequency"] * direction
        high = specification["frequency"] * direction

        def kernel(a, b):
            return 0.5 * (
                np.exp(-np.square(a - b).sum() / 2)
                - np.exp(-np.square(a + b).sum() / 2)
            )

        return (
            specification["low_amplitude"] * kernel(weights, low)
            + kernel(weights, high)
        ) / max(kernel(weights, weights), 1e-15)

    points = []
    v = torch.tensor(direction, dtype=torch.float64)
    low = v * specification["low_frequency"]
    high = v * specification["frequency"]
    for name, interval in (
        ("low", (0.05, 1.5)),
        ("high", (specification["frequency"] - 1, specification["frequency"] + 1)),
    ):
        optimum = minimize_scalar(
            lambda f: float(
                sine_population(
                    f * direction, optimum_amplitude(f), truth, specification
                )
            ),
            bounds=interval,
            method="bounded",
            options={"xatol": 1e-13},
        )
        frequency = float(optimum.x)
        amplitude = float(optimum_amplitude(frequency))
        parameter = torch.tensor(
            np.r_[frequency * direction, amplitude],
            dtype=torch.float64,
            requires_grad=True,
        )

        def objective(p):
            def kernel(a, b):
                return 0.5 * (
                    (-(a - b).square().sum() / 2).exp()
                    - (-(a + b).square().sum() / 2).exp()
                )

            weights, a = p[:-1], p[-1]
            energy = (
                specification["low_amplitude"] ** 2 * kernel(low, low)
                + kernel(high, high)
                + 2 * specification["low_amplitude"] * kernel(low, high)
            )
            return (
                a.square() * kernel(weights, weights)
                - 2
                * a
                * (
                    specification["low_amplitude"] * kernel(weights, low)
                    + kernel(weights, high)
                )
                + energy
            )

        gradient = torch.autograd.grad(objective(parameter), parameter)[0]
        hessian = torch.autograd.functional.hessian(objective, parameter)
        points.append(
            {
                "basin": name,
                "frequency": frequency,
                "amplitude": amplitude,
                "population_mse": float(optimum.fun),
                "gradient_norm": float(gradient.norm()),
                "minimum_hessian_eigenvalue": float(
                    torch.linalg.eigvalsh(hessian).min()
                ),
            }
        )
    return points


def evaluate(output, device):
    spec = json.loads((output / "specification.json").read_text())
    frozen = json.loads((output / "frozen.json").read_text())
    for name, expected in frozen["sha256"].items():
        if digest(output / name) != expected:
            raise RuntimeError(f"Frozen artifact changed: {name}")
    with np.load(output / "heldout.npz") as archive:
        heldout = {
            name: torch.as_tensor(archive[name], device=device)
            for name in archive.files
        }
    with np.load(output / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    architecture = json.loads((output / "architecture_selection.json").read_text())
    source_row = architecture["primary"]["row"]
    with np.load(output / "confirmation/hard_fits.npz") as archive:
        architecture_fits = dict(archive)
    query_restart = int(architecture_fits["query_mse"][source_row].argmin())
    candidates = {
        "learned_inherited": (architecture_fits, source_row, query_restart, spec)
    }
    control_choices = json.loads((output / "controls/selection.json").read_text())
    for variant, methods in control_choices.items():
        with np.load(output / "controls" / (variant + ".npz")) as archive:
            result = dict(archive)
        for method, chosen in methods.items():
            local = {**spec, "head": "full" if variant == "dense_full" else "shared"}
            candidates[variant + "_" + method] = (
                result,
                chosen["row"],
                chosen["restart"],
                local,
            )
    with np.load(output / "fresh_evaluation.npz") as archive:
        fresh = dict(archive)
    choice = json.loads((output / "fresh_selection.json").read_text())
    candidates["learned_fresh"] = fresh, 0, choice["query_selected_restart"], spec
    metrics, states = {}, {}
    for name, (result, row, restart, local) in candidates.items():
        weights = torch.tensor(result["weights"][row : row + 1], device=device)
        numeric = torch.tensor(result["numeric"][row : row + 1], device=device)
        test = (
            score(heldout["test_x"], heldout["test_y"], weights, numeric, local)
            .cpu()
            .numpy()[0]
        )
        threshold = 0.1 if spec["kind"] == "sine" else 0.01 * spec["target_variance"]
        item = {
            "query_mse": float(result["query_mse"][row, restart]),
            "selection_mse": float(result["selection_mse"][row, restart]),
            "test_mse": float(test[restart]),
            "test_nmse": float(test[restart] / spec["target_variance"]),
            "all_restart_mean_test_mse": float(test.mean()),
            "successes": int((test < threshold).sum()),
            "restarts": len(test),
            "query_selected_restart": restart,
            "selected_step": int(result["step"][row, restart]),
        }
        if spec["kind"] == "sine":
            risk = sine_population(
                result["weights"][row, ..., 0],
                result["numeric"][row, :, spec["codes"] + 1],
                truth,
                spec,
            )
            item.update(
                {
                    "population_mse": float(risk[restart]),
                    "population_successes": int((risk < 0.1).sum()),
                }
            )
        else:
            inputs = heldout["permutation_x"].reshape(-1, spec["inputs"])
            with torch.no_grad():
                values = predict(
                    inputs,
                    weights[:, restart : restart + 1],
                    numeric[:, restart : restart + 1],
                    local,
                ).reshape(64, 16)
                item["permutation_rms"] = float(
                    (
                        (values - values.mean(-1, keepdim=True)).square().mean()
                        / spec["target_variance"]
                    ).sqrt()
                )
        metrics[name] = item
        states[name + "_weights"] = result["weights"][row, restart]
        states[name + "_numeric"] = result["numeric"][row, restart]
    indices = architecture_fits["indices"][source_row]
    recovery, aligned, columns = align_indices(indices, truth["indices"], spec["codes"])
    states.update(
        {
            "indices": indices,
            "aligned_indices": aligned,
            "aligned_columns": columns,
            "true_indices": truth["indices"],
        }
    )
    primary = architecture["primary"]
    with np.load(output / "confirmation/snapshots.npz") as archive:
        probabilities = torch.tensor(
            archive["probabilities"][
                primary["snapshot"], primary["root"] : primary["root"] + 1
            ],
            device=device,
        )
        source_numeric = torch.tensor(
            archive["numeric"][
                primary["snapshot"], primary["root"] : primary["root"] + 1
            ],
            device=device,
        )
    canonical = canonical_signed_probabilities(probabilities)
    raw_classes = probabilities.argmax(-1)
    raw_indices = torch.where(
        raw_classes > spec["codes"], spec["codes"] - raw_classes, raw_classes
    ).long()
    source_weights = soft_weights(probabilities, source_numeric, spec["codes"])
    projected_weights = soft_weights(canonical, source_numeric, spec["codes"])
    projection = {
        "maximum_soft_weight_difference": float(
            (source_weights - projected_weights).abs().max()
        ),
        "maximum_probability_sum_error": float((canonical.sum(-1) - 1).abs().max()),
    }
    for name, weights in (
        ("soft", source_weights),
        ("raw_argmax", fixed_weights(raw_indices, source_numeric, spec["codes"])),
        (
            "projected_argmax",
            fixed_weights(
                torch.tensor(indices[None], device=device).long(),
                source_numeric,
                spec["codes"],
            ),
        ),
    ):
        projection[name + "_test_mse"] = (
            score(heldout["test_x"], heldout["test_y"], weights, source_numeric, spec)
            .cpu()
            .tolist()[0]
        )
    states.update(
        {
            "raw_probabilities": probabilities.cpu().numpy()[0],
            "canonical_probabilities": canonical.cpu().numpy()[0],
            "source_numeric": source_numeric.cpu().numpy()[0],
            "raw_indices": raw_indices.cpu().numpy()[0],
        }
    )
    np.savez_compressed(output / "selected_states.npz", **states)
    summary = {
        "task": spec,
        "methods": metrics,
        "recovery": recovery,
        "projection": projection,
        "selection": architecture,
        "test_used_for_selection": False,
    }
    if spec["kind"] == "sine":
        summary["population_stationary_points"] = sine_geometry(truth, spec)
    save_json(output / "summary.json", summary)
    print(json.dumps({"methods": metrics, "recovery": recovery}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    evaluate(args.output, torch.device(args.device))


if __name__ == "__main__":
    main()

"""Post-freeze audit of the conditional log-contrast recovery assumptions."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.paper_structure_benchmark_v1.reporting.evaluate import write_csv
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def audit(base):
    output, root = base / "report", base / "confirmation"
    freeze = json.loads((output / "frozen_results.json").read_text())
    for name, value in freeze["files_sha256"].items():
        if digest(base / name) != value:
            raise RuntimeError(f"Frozen file changed: {name}")
    generation = json.loads((root / "private/generation.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    metadata = {row["name"]: row for row in generation["cases"]}
    rows = []
    for sizes in protocol["cases"]:
        case, codes, hidden = sizes["name"], sizes["codes"], sizes["hidden"]
        with np.load(root / "private" / f"{case}.npz") as private:
            index, codebooks = private["index"], private["codes"]
        positions = np.nonzero(index)
        groups = np.abs(index[positions]).astype(int) - 1
        neurons = positions[1]
        membership = np.eye(codes)[groups]

        def contrasts(matrix):
            result = matrix.copy()
            for neuron in range(hidden):
                active = neurons == neuron
                if active.any():
                    result[active] -= result[active].mean(axis=0)
            return result

        operator = contrasts(membership)
        vectors, singular, _ = np.linalg.svd(operator, full_matrices=False)
        rank = int((singular > singular[0] * 1e-10).sum())
        true_space = vectors[:, :rank]
        ideal = operator @ np.log(np.maximum(np.abs(codebooks), 1e-300)).T
        ideal_singular = np.linalg.svd(ideal, compute_uv=False)
        n_max = int(np.bincount(groups, minlength=codes).max())
        projector_bound = float(1 / (8 * np.sqrt(n_max)))
        noise_bound = float(ideal_singular[rank - 1] / (16 * np.sqrt(n_max)))
        used = [set(np.abs(index[:, h])) - {0} for h in range(hidden)]
        pair_components = []
        for code in range(1, codes + 1):
            remaining = {h for h in range(hidden) if code in used[h]}
            components = 0
            while remaining:
                components += 1
                stack = [remaining.pop()]
                while stack:
                    current = stack.pop()
                    neighbors = {
                        h for h in remaining if len(used[current] & used[h]) >= 2
                    }
                    stack.extend(neighbors)
                    remaining -= neighbors
            pair_components.append(components)
        signs = np.sign(codebooks.T).astype(int)
        profiles = {
            min(tuple(profile.tolist()), tuple((-profile).tolist()))
            for profile in signs
        }
        for condition in protocol["conditions"]:
            observed = np.empty((len(groups), len(codebooks)))
            cosine, clamped = [], 0
            for task, code in enumerate(codebooks):
                truth = np.sign(index) * np.r_[0.0, code][np.abs(index)]
                truth /= np.maximum(np.linalg.norm(truth, axis=0), 1e-300)
                with np.load(
                    root / "source" / f"{case}_c{condition}_t{task}" / "learned.npz"
                ) as learned:
                    weight = learned["dense_parameters"][
                        : sizes["inputs"] * hidden
                    ].reshape(sizes["inputs"], hidden)
                direction = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-300)
                similarity = np.abs(truth.T @ direction)
                left, right = linear_sum_assignment(1 - similarity)
                order = right[np.argsort(left)]
                cosine.extend(similarity[left, right].tolist())
                values = np.abs(direction[:, order][positions])
                clamped += int((values < 1e-12).sum())
                observed[:, task] = np.log(np.maximum(values, 1e-12))
            observed = contrasts(observed)
            estimated = np.linalg.svd(observed, full_matrices=False)[0][:, :rank]
            principal_cosines = np.linalg.svd(
                true_space.T @ estimated, compute_uv=False
            )
            projector_error = float(
                np.sqrt(max(0.0, 1 - min(float(principal_cosines[-1]), 1.0) ** 2))
            )
            contrast_error = float(np.linalg.norm(observed - ideal, ord=2))
            rows.append(
                {
                    "variant": metadata[case]["variant"],
                    "case": case,
                    "condition": condition,
                    "source_noise": generation["noise"][condition],
                    "active_connections": len(groups),
                    "max_group_size": n_max,
                    "code_neuron_components": codes - rank,
                    "contrast_rank": rank,
                    "pair_graph_components": "/".join(map(str, pair_components)),
                    "all_pair_graphs_connected": all(n == 1 for n in pair_components),
                    "distinct_sign_profiles_modulo_flip": len(profiles),
                    "oracle_aligned_projector_error": projector_error,
                    "sufficient_projector_bound": float(projector_bound),
                    "projector_bound_satisfied": projector_error < projector_bound,
                    "log_contrast_spectral_error": contrast_error,
                    "sufficient_noise_bound": noise_bound,
                    "noise_bound_satisfied": contrast_error < noise_bound,
                    "error_to_noise_bound": contrast_error / noise_bound,
                    "min_plane_alignment_cosine": float(min(cosine)),
                    "mean_plane_alignment_cosine": float(np.mean(cosine)),
                    "log_floor_count": clamped,
                    "support_is_oracle": True,
                    "hidden_alignment_is_oracle": True,
                }
            )
    write_csv(output / "theory_conditions.csv", rows)
    save_json(
        output / "theory_diagnostic.json",
        {
            "role": "Post-freeze audit; true support and teacher hidden alignment.",
            "log_floor": 1e-12,
            "rank": "True rank used only by the diagnostic, never by the learner.",
            "guarantee": "Conditional unsigned grouping, not end-to-end G34 recovery.",
            "rows": rows,
        },
    )
    print(json.dumps({"theory_diagnostic_rows": len(rows)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    audit(args.base.resolve())


if __name__ == "__main__":
    main()

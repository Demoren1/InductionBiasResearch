"""Post-freeze optimizer diagnostic using a data-based C initialization."""

import json
from datetime import UTC, datetime

import numpy as np
import torch

from .data import BASE, digest, save_json
from .evaluate import sine_population
from .train import fit, save_fit


def initialization(data, indices, rows, restarts, codes, seed):
    moment = (data["support_x"] * data["support_y"][:, None]).mean(0)
    initial = []
    for index, row in zip(indices, rows):
        labels = np.abs(index.ravel())
        signed = np.sign(index.ravel())
        sums = np.bincount(labels, weights=signed * moment, minlength=codes + 1)[1:]
        sizes = np.bincount(labels, minlength=codes + 1)[1:]
        base = sums / np.maximum(sizes, 1)
        rng = np.random.default_rng(seed + row["seed"] * 100003)
        replicas = []
        for _ in range(restarts):
            values = base + 0.02 * np.linalg.norm(base) * rng.normal(
                size=codes
            ) / np.sqrt(codes)
            expanded = np.r_[0.0, values][labels] * signed
            values *= row["radius"] / max(np.linalg.norm(expanded), 1e-12)
            replicas.append(np.r_[values, 0.0, 1.0, 0.0])
        initial.append(replicas)
    return np.array(initial, dtype=np.float32)


def run(parent, device):
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    output = parent / "moment_codebook_control"
    output.mkdir(exist_ok=False)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "purpose": "Post-freeze diagnostic, not primary pre-test selection. Separate C optimizer failures from the measured capacity of already fixed I. Initialize C by signed group averages of mean(support_x*support_y), followed by the same physical norm grid and2% jitter; no true coefficients or directions enter this initializer. Compare with the already reported dense moment initialization.",
            "source_only_selection": True,
            "test_not_used_for_settings": True,
        },
    )
    with np.load(parent / "source.npz") as archive:
        source = dict(archive)
    with np.load(parent / "selected_states.npz") as archive:
        direct = archive["indices"]
    with np.load(
        parent.parent / (parent.name + "_quantized/selected_states.npz")
    ) as archive:
        quantized = archive["indices"]
    # Oracle assignments are supplied only to a separately named control.
    with np.load(parent / "teacher_audit_only.npz") as archive:
        oracle = archive["indices"]
    methods = {"direct": direct, "quantized": quantized, "oracle": oracle}
    rows, indices = [], []
    for method, index in methods.items():
        for radius in (0.5, 2.0, 6.0, 8.0):
            for rate in (0.003, 0.01, 0.03):
                rows.append(
                    {
                        "method": method,
                        "radius": radius,
                        "numeric_lr": rate,
                        "seed": 111,
                    }
                )
                indices.append(index)
    initial = initialization(source, indices, rows, 4, spec["codes"], plan["seed"])
    result = fit(
        parent,
        plan,
        spec,
        rows,
        device,
        indices=np.array(indices),
        initial=initial,
        restarts=4,
    )
    save_fit(output / "calibration", result, rows)
    chosen = {}
    for method in methods:
        options = []
        for number, row in enumerate(rows):
            if row["method"] != method:
                continue
            restart = int(result["query_mse"][number].argmin())
            options.append(
                {
                    "row": number,
                    "restart": restart,
                    "selection_mse": float(result["selection_mse"][number, restart]),
                    "settings": row,
                }
            )
        chosen[method] = min(options, key=lambda row: row["selection_mse"])
    fresh_rows = [{**chosen[method]["settings"], "seed": 112} for method in methods]
    final_indices = np.array(list(methods.values()))
    initial = initialization(
        source, final_indices, fresh_rows, 32, spec["codes"], plan["seed"]
    )
    result = fit(
        parent,
        plan,
        spec,
        fresh_rows,
        device,
        indices=final_indices,
        initial=initial,
        restarts=32,
    )
    result["indices"] = final_indices
    save_fit(output / "fresh", result, fresh_rows)
    selected_restarts = result["query_mse"].argmin(-1)
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source_selection": chosen,
            "query_selected_restarts": selected_restarts.tolist(),
            "sha256": {
                name: digest(output / name)
                for name in ("registration.json", "calibration.npz", "fresh.npz")
            },
        },
    )
    with np.load(parent / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    summary = {}
    for number, row in enumerate(fresh_rows):
        risks = sine_population(
            result["weights"][number, ..., 0],
            result["numeric"][number, :, spec["codes"] + 1],
            truth,
            spec,
        )
        summary[row["method"]] = {
            "selected_population_mse": float(risks[selected_restarts[number]]),
            "successes": int((risks < 0.1).sum()),
            "restarts": len(risks),
            "all_population_mse": risks.tolist(),
            "settings": row,
        }
    save_json(output / "summary.json", summary)
    print(spec["name"], summary, flush=True)


def main():
    torch.set_num_threads(2)
    for frequency in (4, 6, 8):
        run(BASE / f"results/20261104/sine_{frequency}", torch.device("cuda:0"))


if __name__ == "__main__":
    main()

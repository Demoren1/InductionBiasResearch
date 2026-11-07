"""Summarize development data without opening target samples or structural truth."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = (
    "Dense Adam",
    "Dense + GN/revival",
    "Current",
    "LUT-Q + refit",
    "Yeh MAP + refit",
    "Yeh soft",
    "MSR-FC",
)


def dense_prediction(x, p, d, h, o):
    weight = p[: d * h].reshape(d, h)
    bias = p[d * h : d * h + h]
    head = p[d * h + h : -o].reshape(h, o)
    return np.maximum(x @ weight + bias, 0) @ head / math.sqrt(h) + p[-o:]


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def run(root):
    protocol = load_json(root / "protocol.json")
    output = root / "source_report"
    output.mkdir(exist_ok=True)
    rows = []
    for case in protocol["cases"]:
        name = case["name"]
        dataset = "Pattern 0110" if case["inputs"] == 11 else "DeepSets"
        d, h, o = [case[key] for key in ("inputs", "hidden", "outputs")]
        for condition in protocol["conditions"]:
            tag = f"{name}_c{condition}"
            meta = dict(np.load(root / "public" / tag / "meta.npz"))
            tasks = len(meta) // 2
            dense_folder = root / "dense" / tag
            dense = np.load(dense_folder / "fitted.npz")["parameters"]
            prepared = [
                np.load(root / "source" / f"{tag}_t{t}" / "learned.npz")[
                    "dense_parameters"
                ]
                for t in range(tasks)
            ]
            common_cost = load_json(root / "metrics" / f"{tag}_source.json")
            dense_cost = load_json(
                root.parent
                / "dense_cost_measurement"
                / "metrics"
                / f"case_{'00' if d == 11 else '03'}_c0_replay.json"
            )
            dense_cost["wall_seconds"] = load_json(dense_folder / "frozen.json").get(
                "seconds"
            )
            all_costs = [
                load_json(root / "metrics" / f"{tag}_{phase}.json")
                for phase in ("source", "seeds", "proposals", "fit", "select")
            ]
            for method, parameters in (
                ("Dense Adam", dense),
                ("Dense + GN/revival", prepared),
            ):
                risk = np.mean(
                    [
                        np.mean(
                            (
                                dense_prediction(meta[f"x_{t}"], parameters[t], d, h, o)
                                - meta[f"y_{t}"]
                            )
                            ** 2
                        )
                        for t in range(tasks)
                    ]
                )
                seconds = load_json(dense_folder / "frozen.json").get("seconds")
                if method != "Dense Adam":
                    seconds = common_cost.get("wall_seconds")
                rows.append(
                    {
                        "dataset": dataset,
                        "case": name,
                        "condition": condition,
                        "method": method,
                        "source_meta_mse": float(risk),
                        "offline_seconds": seconds,
                        "torch_operator_flops": (
                            dense_cost if method == "Dense Adam" else common_cost
                        ).get("torch_operator_flops"),
                        "gpu_peak_mib": (
                            dense_cost if method == "Dense Adam" else common_cost
                        ).get("cuda_peak_allocated_bytes", 0)
                        / 2**20,
                    }
                )
            chosen = load_json(root / "fits" / tag / "joint_selected.json")
            if chosen:
                rows.append(
                    {
                        "dataset": dataset,
                        "case": name,
                        "condition": condition,
                        "method": "Current",
                        "source_meta_mse": chosen["methods"]["joint_functional"][
                            "meta_mse"
                        ],
                        "offline_seconds": sum(p["wall_seconds"] for p in all_costs),
                        "torch_operator_flops": sum(
                            p["torch_operator_flops"] for p in all_costs
                        ),
                        "gpu_peak_mib": max(
                            p["cuda_peak_allocated_bytes"] for p in all_costs
                        )
                        / 2**20,
                    }
                )
            for method, directory, suffix, numerical_metrics in (
                ("LUT-Q + refit", "lutq", "", "refit_metrics.json"),
                ("Yeh MAP + refit", "yeh", "", "refit_metrics.json"),
                ("Yeh soft", "yeh", "_continuous", "continuous_refit_metrics.json"),
                ("MSR-FC", "msr_maml", "_continuous", "continuous_refit_metrics.json"),
            ):
                folder = root / "competitors" / tag / directory
                selected = load_json(folder / f"selected{suffix}.json")
                if not selected:
                    continue
                costs = [
                    dense_cost if suffix else common_cost,
                    load_json(folder / "metrics.json"),
                    load_json(folder / numerical_metrics),
                ]
                rows.append(
                    {
                        "dataset": dataset,
                        "case": name,
                        "condition": condition,
                        "method": method,
                        "source_meta_mse": selected["source_meta_mse"],
                        "offline_seconds": sum(p["wall_seconds"] for p in costs),
                        "torch_operator_flops": sum(
                            p["torch_operator_flops"] for p in costs
                        ),
                        "gpu_peak_mib": max(
                            p["cuda_peak_allocated_bytes"] for p in costs
                        )
                        / 2**20,
                    }
                )
    fields = (
        "dataset",
        "case",
        "condition",
        "method",
        "source_meta_mse",
        "offline_seconds",
        "torch_operator_flops",
        "gpu_peak_mib",
    )
    with (output / "source_selection.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Development source comparison",
        "",
        "These are source structure-selection losses, not independent transfer results. "
        "Target data and true indexators remain unopened. All numbers include "
        "required dense preprocessing when it is used for initialization. Continuous "
        "Yeh/MSR need DenseAdam only, while hard common refits need the full descriptor preparation. "
        "Dense operation/memory counts come from a one-time unchanged measurement replay per shape; "
        "its numerical outputs never enter selection. "
        "Torch FLOPs exclude NumPy/SciPy and unsupported operators.",
        "",
        "| Dataset | Method | Cases | Source MSE | Offline seconds | Torch TFLOPs | GPU peak MiB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in ("Pattern 0110", "DeepSets"):
        for method in METHODS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            if not selected:
                continue
            values = {}
            for column in (
                "source_meta_mse",
                "offline_seconds",
                "torch_operator_flops",
                "gpu_peak_mib",
            ):
                column_values = [
                    row[column] for row in selected if row[column] is not None
                ]
                values[column] = (
                    float(np.mean(column_values)) if column_values else None
                )
            flops = values["torch_operator_flops"]
            memory = values["gpu_peak_mib"]
            lines.append(
                f"| {dataset} | {method} | {len(selected)} | "
                f"{values['source_meta_mse']:.2e} | {values['offline_seconds']:.2f} | "
                f"{'-' if flops is None else f'{flops / 1e12:.2f}'} | "
                f"{'-' if memory is None else f'{memory:.2f}'} |"
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    for ax, dataset in zip(axes, ("Pattern 0110", "DeepSets")):
        available = [
            method
            for method in METHODS
            if any(
                row["dataset"] == dataset and row["method"] == method for row in rows
            )
        ]
        for position, method in enumerate(available):
            values = [
                max(row["source_meta_mse"], 1e-16)
                for row in rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            ax.scatter([position] * len(values), values, s=45, zorder=3)
        ax.set_xticks(range(len(available)), available, rotation=35, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("Source selection MSE (floor $10^{-16}$)")
        ax.set_title(dataset)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Development results; no target or structural audit")
    fig.savefig(output / "source_selection.png", dpi=180)
    fig.savefig(output / "source_selection.pdf")
    plt.close(fig)
    save = {
        "source_selection_only": True,
        "targets_opened": False,
        "true_parameters_opened": False,
        "rows": len(rows),
    }
    (output / "provenance.json").write_text(json.dumps(save, indent=2) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()

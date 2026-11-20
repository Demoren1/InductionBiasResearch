"""Structural recovery on binary Pattern and the real-valued 0110 benchmark."""

import argparse
import io
import json
import runpy
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_structure_benchmark_v1.protocol.prepare import prepare
from experiments.paper_structure_benchmark_v1.reporting.evaluate import align_index
from experiments.paper_structure_benchmark_v1.reporting.source_report import (
    dense_prediction,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from experiments.srr_approx_v1.core import canonical, signed_forest


REVISION = "d0a66916151b5ca5d3b8b21cfab37013c6873d16"
BENCHMARK = Path("artifacts/benchmark_2026-11-07/synthetic_confirmation")
OUTPUT = Path("artifacts/asrr_pattern_2026-11-19")
SEEDS = (281067, 281068, 281069)
STARTS, RETAINED = 8, 4


def historical_file(name):
    return subprocess.check_output(["git", "show", f"{REVISION}:pattern/{name}"])


def binary_fits(work, plan):
    # Run just the unpruned round of the original Pattern training procedure.
    trainer = work / "bank.py"
    trainer.write_bytes(historical_file("codebook/bank.py"))
    collect = runpy.run_path(str(trainer))["collect"]
    split_file = historical_file("runs/codebook_20261028_v4/splits.npz")
    with np.load(io.BytesIO(split_file)) as data:
        splits = {k: torch.from_numpy(data[k]) for k in ("support", "query", "probe")}
    bits = (torch.arange(2048)[:, None] >> torch.arange(10, -1, -1)) & 1
    x = (2 * bits - 1).float()
    windows = ((x + 1) / 2).unfold(1, 4, 1)
    patterns = plan["source_patterns"]
    labels = torch.stack([
        (windows == torch.tensor(list(map(int, pattern)))).all(-1).any(-1)
        for pattern in patterns
    ]).float()
    settings = dict(plan["bank"], active_edges=[88], networks_per_task=STARTS)
    for seed in SEEDS:
        run = dict(plan, seed=seed, bank=settings)
        fitted, _ = collect(run, x, labels, splits, work)
        # Undo the collector's coordinate-based ordering before aSRR sees a fit.
        weights = np.take_along_axis(
            fitted["parameter_0"],
            fitted["column_order"].argsort(1)[:, None, :],
            axis=2,
        ).reshape(len(patterns), STARTS, 11, 8)
        losses = fitted["query_loss"].reshape(len(patterns), STARTS)
        retained = losses.argsort(axis=1, kind="stable")[:, :RETAINED]
        tasks = np.arange(len(patterns))[:, None]
        accuracy = fitted["query_accuracy"].reshape(len(patterns), STARTS)
        yield f"binary_{seed}", weights[tasks, retained], losses[tasks, retained], {
            "retained_mean_accuracy": float(accuracy[tasks, retained].mean()),
            "best_per_task_mean_accuracy": float(
                accuracy[tasks, retained[:, :1]].mean()
            ),
            "retained_mean_balanced_bce": float(losses[tasks, retained].mean()),
        }


def regression_fits(benchmark_data):
    # The public compact archive kept source task zero in each of three families.
    for condition in (0, 1):
        weights, losses, sources = [], [], []
        for case in range(3):
            label = f"case_{case:02d}_c{condition}"
            path = BENCHMARK / "dense" / label / "fitted.npz"
            with np.load(path) as data:
                parameters = data["parameters"][0]
            weights.append(parameters[:88].reshape(11, 8))
            with np.load(benchmark_data / "public" / label / "selection.npz") as data:
                prediction = dense_prediction(data["x_0"], parameters, 11, 8, 1)
                losses.append(float(np.mean((prediction - data["y_0"]) ** 2)))
            sources.append({"file": str(path), "sha256": digest(path)})
        info = {"sources": sources, "validation_mse": losses}
        yield (
            f"regression_c{condition}", np.array(weights)[:, None],
            np.array(losses)[:, None], info,
        )


def aligned_profiles(work, weights, losses):
    tasks, starts = losses.shape
    path = work / "source.npz"
    np.savez(
        path,
        parameters=weights[np.arange(tasks), losses.argmin(1)].reshape(tasks, -1),
        ensemble_parameters=weights.reshape(tasks, starts, -1),
        numerical_cross_entropy=losses.T,
    )
    _, profiles, _, ranking = source_profiles(path, 11, 8)
    return profiles, int(ranking[0])


def extract(profiles, codes):
    cuts, _ = signed_forest(profiles, (codes, codes + 1))
    # Diagnostic only: allow one Ward cluster to stand for fixed zero.
    extra = cuts[codes + 1].copy()
    labels = np.abs(extra)
    energy = [
        np.square(profiles[:, labels == j]).mean() for j in range(1, codes + 2)
    ]
    extra[labels == 1 + np.argmin(energy)] = 0
    return {"asrr": cuts[codes], "zero_cluster": canonical(extra)}


def score(predictions, truth, codes):
    rows = {}
    for method, index in predictions.items():
        aligned, row = align_index(index, truth, codes)
        active = truth != 0
        row["signed_matches_on_true_active"] = int(((aligned == truth) & active).sum())
        row["mask_overlap_max"] = round(row["mask_recall"] * int(active.sum()))
        rows[method] = row
    return rows


def single_0110(work, retained, patterns):
    task = patterns.index("0110")
    for seed in SEEDS:
        weights = retained[f"binary_{seed}_weights"][task:task + 1]
        losses = retained[f"binary_{seed}_losses"][task:task + 1]
        profiles, _ = aligned_profiles(work, weights, losses)
        yield f"single_0110_{seed}", extract(profiles, 2)


def draw_indices(output, indices, binary_truth, regression_truth):
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cases = (
        ("12 двоичных шаблонов", f"binary_{SEEDS[0]}", binary_truth, 4),
        ("0110, двоичные метки", f"single_0110_{SEEDS[0]}", regression_truth, 2),
        ("0110, регрессия без шума", "regression_c0", regression_truth, 2),
    )
    colors = ["#f0f1f2", "#2d6f95", "#ec9951", "#74946a", "#9272a4"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(8, 8.4))
    for row, (label, case, truth, codes) in enumerate(cases):
        column_order = np.argsort((truth != 0).argmax(0))
        for col, method in enumerate(("truth", "asrr", "zero_cluster")):
            ax = axes[row, col]
            if method == "truth":
                aligned = truth
            else:
                aligned, _ = align_index(indices[f"{case}_{method}"], truth, codes)
            matrix = aligned[:, column_order]
            ax.imshow(np.abs(matrix), cmap=cmap, norm=norm, interpolation="nearest")
            for i, h in zip(*np.nonzero(matrix < 0)):
                ax.text(h, i, "−", ha="center", va="center", color="white")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(("Эталон", "aSRR", "Разрез с нулевой группой")[col])
            if col == 0:
                ax.set_ylabel(label)
            if method != "truth":
                ax.set_xlabel(f"Совпало {np.count_nonzero(aligned == truth)} из 88")
    legend = [Patch(color=c, label=str(i)) for i, c in enumerate(colors)]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
               title="0 — отключённая связь, цвет — код, минус — обратный знак")
    fig.tight_layout(rect=(0, 0.065, 1, 1))
    fig.savefig(output / "indexators.png", dpi=150)
    plt.close(fig)


def main(output, benchmark_data):
    torch.set_num_threads(4)
    plan = json.loads(historical_file("codebook/protocol_v4.json"))
    patterns = plan["source_patterns"]
    output.mkdir(parents=True, exist_ok=True)
    metadata, indices, retained = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="asrr-pattern-") as scratch:
        work = Path(scratch)
        if benchmark_data is None:
            benchmark_data = work / "benchmark"
            prepare(benchmark_data, "confirmation", 3, 16)
        for sources, codes in (
            (binary_fits(work, plan), 4), (regression_fits(benchmark_data), 2),
        ):
            for name, weights, losses, info in sources:
                profiles, anchor = aligned_profiles(work, weights, losses)
                metadata[name] = dict(
                    info, tasks=len(losses), codes=codes, anchor=anchor
                )
                retained[f"{name}_weights"] = weights
                retained[f"{name}_losses"] = losses
                for method, index in extract(profiles, codes).items():
                    indices[f"{name}_{method}"] = index
                print(f"Extracted {name}", flush=True)

        for name, candidates in single_0110(work, retained, patterns):
            metadata[name] = {"tasks": 1, "codes": 2, "anchor": 0}
            for method, index in candidates.items():
                indices[f"{name}_{method}"] = index

        frozen = output / "fitted_indices.npz"
        np.savez_compressed(frozen, **indices)
        frozen_hash = digest(frozen)
        np.savez_compressed(output / "retained_sources.npz", **retained)
        # Structural truth enters only after all fitted-source outputs are frozen.
        reference = Path("archive/reports/final/assets/2026-11-03")
        with np.load(reference / "pattern_heatmap_data.npz") as data:
            binary_truth = data["analytic_indices"]
        with np.load(BENCHMARK / "private/case_00.npz") as data:
            regression_truth = data["index"]
        results = {}
        for name, info in metadata.items():
            truth = binary_truth if name.startswith("binary") else regression_truth
            candidates = {
                key: indices[f"{name}_{key}"] for key in ("asrr", "zero_cluster")
            }
            results[name] = dict(info, recovery=score(candidates, truth, info["codes"]))

        # Exact teacher weights isolate cross-task alignment from Ward clustering.
        task_codes = np.array([[0, *(2 * int(bit) - 1 for bit in p)] for p in patterns])
        weights = task_codes[:, np.abs(binary_truth)] * np.sign(binary_truth)
        normalized = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        controls = {
            "known_neuron_order": score(extract(normalized, 4), binary_truth, 4)
        }
        for anchor, pattern in enumerate(patterns):
            # All exact solutions tie. Give each possible reference first priority.
            priority = np.ones((len(patterns), 1))
            priority[anchor] = 0
            profiles, _ = aligned_profiles(work, weights[:, None], priority)
            controls[pattern] = score(extract(profiles, 4), binary_truth, 4)

        manifest_path = BENCHMARK.parent / "published_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        regression_controls = {}
        for case in range(3):
            name = f"case_{case:02d}"
            private = Path("private") / f"{name}.npz"
            expected = manifest["records"][f"synthetic_confirmation/{private}"]
            if digest(benchmark_data / private) != expected["source_sha256"]:
                raise ValueError(f"Regenerated source teacher differs for {name}")
            with np.load(benchmark_data / private) as data:
                truth, codes = data["index"], data["codes"]
            lookup = np.pad(codes, ((0, 0), (1, 0)))
            exact_weights = np.sign(truth) * lookup[:, np.abs(truth)]
            profiles, _ = aligned_profiles(
                work, exact_weights[:, None], np.zeros((len(codes), 1))
            )
            regression_controls[name] = score(extract(profiles, 2), truth, 2)

    save_json(output / "results.json", {
        "fitted_indices_sha256": frozen_hash,
        "binary_protocol_git_revision": REVISION,
        "source_patterns": patterns,
        "binary_dense_starts": STARTS,
        "binary_retained_starts": RETAINED,
        "binary_adam_steps": plan["bank"]["steps_per_round"],
        "results": results,
        "binary_exact_source_controls": controls,
        "regression_exact_source_controls": regression_controls,
    })
    draw_indices(output, indices, binary_truth, regression_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--benchmark-data", type=Path)
    args = parser.parse_args()
    main(args.output, args.benchmark_data)

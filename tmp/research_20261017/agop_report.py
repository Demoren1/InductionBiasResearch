"""Export AGOP tables and figures into the ignored local cache."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from agop import BRANCHES, OUT, ROOT, BranchNetwork, load, read, replace
from matplotlib import pyplot as plt

from multimask.common import matches, references, topk
from pattern.io import digest, save_json

HOME = Path(__file__).resolve().parent
ASSETS = HOME / ".cache/report_assets"
LABELS = {
    "functional": "Original NF(W,G,F)",
    "branch_agop": "NF(W,G) + MLP(AGOP)",
    "branch_functional": "NF(W,G) + MLP(F)",
    "branch_zero": "NF(W,G) + MLP(0)",
    "agop": "NF(W,G,AW)",
    "diagonal": "NF(W,G,diagonal)",
    "zero": "NF(W,G,0)",
}
COLORS = ("#65778f", "#218c75", "#306eb8", "#b07535")
SEEDS = (4213, 4214, 4215)


def run_path(representation, seed):
    home = HOME if representation == "functional" else OUT / representation
    return home / "runs" / f"original_{seed}"


def csv_file(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ASSETS / f"agop_{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def figure(fig, name):
    for extension in ("png", "svg"):
        fig.savefig(
            ASSETS / f"agop_{name}.{extension}", dpi=175, bbox_inches="tight"
        )
    plt.close(fig)


def baseline_transfer():
    path = ROOT / "multimask/runs/nf_vae_v3/evaluations/z_gumbel_20261013"
    values = read(path / "metrics.json")["families"]
    result = {}
    for family, value in values.items():
        initial = load(path / family / "generated_masks/initial_z.pt")
        result[family] = {
            "methods": {
                "optimized": {
                    "iou": value["selected_iou"],
                    **value["methods"]["optimized_z"],
                },
                "best_prior": {
                    "iou": float(initial["iou"][value["best_random_index"]]),
                    **value["methods"]["best_random_z"],
                },
            }
        }
    return result


@torch.no_grad()
def interventions():
    if (OUT / "interventions.json").exists():
        return read(OUT / "interventions.json")
    config, cards, _ = load(OUT / "multimask_cards.pt")
    full = load(OUT / "full_agop.pt")
    output = []
    for representation in BRANCHES:
        for seed in SEEDS:
            model = BranchNetwork("original", config).eval()
            model.load_state_dict(
                load(run_path(representation, seed) / "best.pt")["model"]
            )
            for index, (task, card) in enumerate(cards.items()):
                inputs = replace(
                    {**card, "full_agop": full[task]}, representation
                )["features"]
                inputs = inputs[card["split"] == 2]
                blank = inputs.clone()
                blank[:, 68 * 8 * 2 :] = 0
                for name, x in (("own", inputs), ("blank", blank)):
                    scores = model(index, x)[0]
                    output.append(
                        {
                            "representation": representation,
                            "seed": seed,
                            "task": task,
                            "input": name,
                            "iou": float(
                                matches(topk(scores), task)[0].mean()
                            ),
                        }
                    )
    return output


def full_transfer(rows, search_mode):
    """Compare paired model seeds with fixed tasks and search randomness."""
    selected = [
        r
        for r in rows
        if r["representation"] in BRANCHES and r["search_mode"] == search_mode
    ]
    optimized = [r for r in selected if r["method"] == "optimized"]
    if len(optimized) != len(BRANCHES) * len(SEEDS) * 2:
        raise ValueError("Full transfer must finish for all three seeds")
    aggregate = []
    for kind in BRANCHES:
        for family in "AB":
            for method in ("best_prior", "optimized"):
                group = [
                    r
                    for r in selected
                    if r["representation"] == kind
                    and r["family"] == family
                    and r["method"] == method
                ]
                record = {
                    "representation": kind,
                    "family": family,
                    "method": method,
                    "seeds": len(group),
                }
                for key in ("iou", "bce", "excess_bce"):
                    values = [r[key] for r in group]
                    record[f"{key}_mean"] = float(np.mean(values))
                    record[f"{key}_sd"] = float(np.std(values, ddof=1))
                aggregate.append(record)
    csv_file(f"{search_mode}_aggregate", aggregate)
    overall = []
    for kind in BRANCHES:
        record = {
            "representation": kind,
            "search_mode": search_mode,
            "seeds": 3,
        }
        for metric in ("iou", "bce", "excess_bce"):
            values = [
                np.mean(
                    [
                        r[metric]
                        for r in optimized
                        if r["representation"] == kind and r["seed"] == seed
                    ]
                )
                for seed in SEEDS
            ]
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_sd"] = float(np.std(values, ddof=1))
        overall.append(record)
    csv_file(f"{search_mode}_overall", overall)
    paired = []
    for control in BRANCHES[1:]:
        for seed in SEEDS:
            record = {"control": control, "seed": seed}
            for metric in ("iou", "bce"):
                means = [
                    np.mean(
                        [
                            r[metric]
                            for r in optimized
                            if r["representation"] == kind
                            and r["seed"] == seed
                        ]
                    )
                    for kind in (BRANCHES[0], control)
                ]
                record[f"agop_minus_control_{metric}"] = float(
                    means[0] - means[1]
                )
            paired.append(record)
    csv_file(f"{search_mode}_paired", paired)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for row, family in enumerate("AB"):
        for column, metric in enumerate(("iou", "bce")):
            axis = axes[row, column]
            for index, (kind, color) in enumerate(zip(BRANCHES, COLORS[1:])):
                values = [
                    next(
                        r[metric]
                        for r in optimized
                        if r["representation"] == kind
                        and r["family"] == family
                        and r["seed"] == seed
                    )
                    for seed in SEEDS
                ]
                axis.plot(
                    range(3),
                    values,
                    marker=("o", "s", "^")[index],
                    fillstyle="none",
                    color=color,
                    label=LABELS[kind],
                )
            axis.set(
                title=f"Held-out {family}: optimized {metric}",
                xlabel="Generator initialization seed",
            )
            axis.set_xticks(range(3), SEEDS)
            if metric == "iou":
                minimum = min(
                    r["iou"] for r in optimized if r["family"] == family
                )
                axis.set_ylim(max(0, minimum - 0.05), 1.02)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(search_mode)
    figure(fig, f"{search_mode}_transfer_seeds")
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    history = []
    for column, kind in enumerate(BRANCHES):
        for row, family in enumerate("AB"):
            axis = axes[row, column]
            for seed in SEEDS:
                path = (
                    run_path(kind, seed) / f"transfer_{search_mode}" / family
                )
                values = read(path / "search/history.json")
                history.extend(
                    {
                        "representation": kind,
                        "seed": seed,
                        "family": family,
                        **value,
                    }
                    for value in values
                )
                axis.plot(
                    [r["step"] for r in values],
                    [r["source_query_bce"] for r in values],
                    label=str(seed),
                )
            axis.set(
                title=f"{family}: {LABELS[kind]}",
                xlabel="z step",
                ylabel="Mean source query BCE, 8 restarts",
            )
    axes[0, 0].legend(title="Generator seed", fontsize=8)
    fig.suptitle(search_mode)
    figure(fig, f"{search_mode}_search")
    csv_file(f"{search_mode}_search", history)
    runtime = []
    for kind in BRANCHES:
        for seed in SEEDS:
            folder = run_path(kind, seed) / f"transfer_{search_mode}"
            completion = read(folder / "COMPLETE.json")
            seconds = completion.get("elapsed_seconds")
            timing = "monotonic"
            if seconds is None:
                timing = "filesystem_birthtime"
                seconds = (
                    folder / "COMPLETE.json"
                ).stat().st_mtime - folder.stat().st_birthtime
            counts = [
                read(folder / family / "search/history.json")[-1][
                    "evaluated_masks"
                ]
                for family in "AB"
            ]
            runtime.append(
                {
                    "representation": kind,
                    "seed": seed,
                    "search_mode": search_mode,
                    "elapsed_seconds": seconds,
                    "timing_source": timing,
                    "hard_fitness_cache_entries": sum(counts),
                }
            )
    csv_file(f"{search_mode}_runtime", runtime)
    return {
        "aggregate": aggregate,
        "paired": paired,
        "generator_seeds": SEEDS,
        "families": ["A", "B"],
        "runtime": runtime,
        "overall": overall,
    }


def compare_methods(summary):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for row, family in enumerate("AB"):
        for column, metric in enumerate(("iou", "bce")):
            axis = axes[row, column]
            for kind, color in zip(BRANCHES, COLORS[1:]):
                values = [
                    next(
                        r
                        for r in summary[mode]["aggregate"]
                        if r["representation"] == kind
                        and r["family"] == family
                        and r["method"] == "optimized"
                    )
                    for mode in ("gumbel", "evolution")
                ]
                axis.errorbar(
                    range(2),
                    [r[f"{metric}_mean"] for r in values],
                    yerr=[r[f"{metric}_sd"] for r in values],
                    marker="o",
                    capsize=4,
                    color=color,
                    label=LABELS[kind],
                )
            axis.set(
                title=f"Held-out {family}: {metric}, mean ± SD",
                xticks=range(2),
                xticklabels=["Gumbel", "Evolution"],
            )
            if metric == "iou":
                axis.set_ylim(0.7, 1.08)
    axes[0, 0].legend(fontsize=8)
    figure(fig, "methods")


def mask_panels(seed, search_mode):
    """Align probabilities and selected edges to the same reference columns."""
    fig, axes = plt.subplots(2, 4, figsize=(10, 8), layout="constrained")
    fig.suptitle(f"{search_mode}, generator seed {seed}: aligned scores/masks")
    for row, family in enumerate("AB"):
        reference = references(f"{family}_0000")[0]
        axes[row, 0].imshow(reference, aspect="auto", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{family}: reference")
        for column, representation in enumerate(BRANCHES, 1):
            path = (
                run_path(representation, seed)
                / f"transfer_{search_mode}"
                / family
            )
            masks = load(path / "sealed_masks.pt")["masks"][:1]
            probabilities = load(path / "results.pt")["probabilities"]
            _, variant, order = matches(masks, f"{family}_0000")
            mask = masks[0, :, order[0]]
            scores = probabilities[:, order[0]]
            image = axes[row, column].imshow(
                scores, aspect="auto", vmin=0, vmax=1
            )
            analytical = references(f"{family}_0000")[variant[0]]
            ys, xs = torch.where(analytical == 1)
            axes[row, column].scatter(
                xs,
                ys,
                marker="s",
                s=15,
                facecolors="none",
                edgecolors="#ff6b6b",
                linewidths=0.6,
            )
            ys, xs = torch.where(mask == 1)
            axes[row, column].scatter(xs, ys, marker="s", s=4, color="white")
            axes[row, column].set_title(
                f"{LABELS[representation]}\nvariant {int(variant[0])}"
            )
        for axis in axes[row]:
            axis.set(xlabel="Aligned neuron", ylabel="Input coordinate")
    fig.colorbar(
        image, ax=axes[:, 1:], label="Decoder sigmoid score", shrink=0.7
    )
    name = "masks" if seed == 4213 else f"masks_{seed}"
    figure(fig, name if search_mode == "gumbel" else f"evolution_{name}")


def main():
    torch.set_num_threads(1)
    ASSETS.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "path"})
    training, history, transfer, pattern, prior = [], [], [], [], []
    for representation in LABELS:
        for seed in SEEDS:
            path = run_path(representation, seed)
            if not (path / "COMPLETE.json").exists():
                continue
            values = read(path / "diagnostics.json")
            prior.extend(
                {"representation": representation, "seed": seed, **row}
                for row in values["prior"]
            )
            training.append(
                {
                    "representation": representation,
                    "seed": seed,
                    "parameters": read(path / "protocol.json")["parameters"],
                    "best_epoch": values["best_epoch"],
                    **values["bank_test"],
                }
            )
            history.extend(
                {
                    "representation": representation,
                    "seed": seed,
                    **{
                        k: v for k, v in row.items() if not isinstance(v, dict)
                    },
                }
                for row in read(path / "history.json")
            )
            if representation not in ("functional", *BRANCHES):
                continue
            optimized = (
                seed == 4213
                or (path / "transfer_gumbel/COMPLETE.json").exists()
            )
            stage = "transfer_gumbel" if optimized else "transfer_prior"
            report = (
                baseline_transfer()
                if representation == "functional" and seed == 4213
                else read(path / stage / "metrics.json")
            )
            reports = [("gumbel" if optimized else "prior", report)]
            evolutionary = path / "transfer_evolution"
            if (
                representation in BRANCHES
                and (evolutionary / "COMPLETE.json").exists()
            ):
                reports.append(
                    ("evolution", read(evolutionary / "metrics.json"))
                )
            for search_mode, report in reports:
                for family, row in report.items():
                    transfer.extend(
                        {
                            "representation": representation,
                            "seed": seed,
                            "family": family,
                            "method": method,
                            "search_mode": search_mode,
                            "selected_search_step": row.get(
                                "selected_search_step"
                            ),
                            **metrics,
                        }
                        for method, metrics in row["methods"].items()
                        if method in ("optimized", "best_prior", "zero")
                    )
    for path in sorted((OUT / "pattern").glob("*/diagnostics.json")):
        representation, seed = path.parent.name.rsplit("_", 1)
        value = read(path)
        pattern.append(
            {
                "representation": representation,
                "seed": int(seed),
                **{k: v for k, v in value.items() if k != "bank_test"},
                **value["bank_test"],
            }
        )
    for name, rows in (
        ("training", training),
        ("history", history),
        ("transfer", transfer),
        ("pattern", pattern),
        ("prior_structure", prior),
    ):
        csv_file(name, rows)
    intervention = interventions()
    csv_file("interventions", intervention)
    summaries = {}
    main_kinds = ("functional", *BRANCHES)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
    for index, representation in enumerate(main_kinds):
        values = [
            row["iou"]
            for row in training
            if row["representation"] == representation
        ]
        summaries[representation] = {
            "bank_iou_mean": float(np.mean(values)),
            "bank_iou_sd": float(np.std(values, ddof=1)),
        }
        axes[0].bar(
            index,
            np.mean(values),
            yerr=np.std(values, ddof=1),
            capsize=3,
            color=COLORS[index],
        )
        for column, family in enumerate("AB", 1):
            rows = [
                r
                for r in transfer
                if r["representation"] == representation
                and r["seed"] == 4213
                and r["family"] == family
                and r["search_mode"] == "gumbel"
            ]
            for offset, method in ((-0.17, "best_prior"), (0.17, "optimized")):
                value = next(r for r in rows if r["method"] == method)
                axes[column].bar(
                    index + offset,
                    value["iou"],
                    width=0.32,
                    color=COLORS[index],
                    alpha=0.45 if offset < 0 else 1,
                )
    for axis, title in zip(
        axes,
        (
            "Source teacher test: mean ± SD, 3 seeds",
            "Held-out A: prior / optimized, seed 4213",
            "Held-out B: prior / optimized, seed 4213",
        ),
    ):
        axis.set(ylim=(0, 1.04), ylabel="Nearest-reference IoU", title=title)
        axis.set_xticks(
            range(4), [LABELS[k] for k in main_kinds], rotation=22, ha="right"
        )
    figure(fig, "comparison")
    full_summary = {
        mode: full_transfer(transfer, mode) for mode in ("gumbel", "evolution")
    }
    compare_methods(full_summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for representation, color in zip(main_kinds, COLORS):
        for seed in SEEDS:
            rows = [
                r
                for r in history
                if r["representation"] == representation and r["seed"] == seed
            ]
            for axis, key in zip(
                axes, ("validation_loss", "validation_iou", "validation_kl")
            ):
                axis.plot(
                    [r["epoch"] for r in rows],
                    [r[key] for r in rows],
                    color=color,
                    alpha=0.7,
                    label=LABELS[representation] if seed == 4213 else None,
                )
    for axis, title in zip(
        axes,
        (
            "Validation surrogate loss",
            "Validation reference IoU",
            "Unweighted KL",
        ),
    ):
        axis.set(xlabel="Epoch", title=title)
    axes[0].legend(fontsize=7)
    figure(fig, "learning")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for index, representation in enumerate(main_kinds):
        rows = [r for r in pattern if r["representation"] == representation]
        for axis, key in zip(axes, ("iou", "prior_exact")):
            values = [r[key] for r in rows]
            axis.bar(
                index,
                np.mean(values),
                yerr=np.std(values, ddof=1),
                capsize=3,
                color=COLORS[index],
            )
    for axis, title in zip(
        axes,
        (
            "Pattern: posterior reference IoU",
            "Pattern: exact random-prior fraction",
        ),
    ):
        axis.set(title=title, ylim=(0, 1.04))
        axis.set_xticks(
            range(4), [LABELS[k] for k in main_kinds], rotation=22, ha="right"
        )
    figure(fig, "pattern")
    for mode in ("gumbel", "evolution"):
        for seed in SEEDS:
            mask_panels(seed, mode)
    sources = {
        str(path.relative_to(ROOT)): digest(path)
        for representation in main_kinds
        for seed in SEEDS
        for path in (run_path(representation, seed) / "best.pt",)
    }
    evidence = [
        HOME / "agop.py",
        HOME / "agop_report.py",
        HOME / "experiment.py",
        HOME / "run.sh",
        ROOT / "multimask/config.json",
        OUT / "protocol.json",
        OUT / "pattern_protocol.json",
        OUT / "multimask_cards.pt",
        OUT / "pattern_cards.pt",
        OUT / "full_agop.pt",
        OUT / "full_transfer_20261018.json",
        OUT / "evolution_transfer_20261018.json",
    ]
    for name in ("protocol.json", "pattern_protocol.json"):
        evidence.append(Path(read(OUT / name)["bank"]) / "manifest.json")
    evidence.extend((OUT / "pattern").glob("*/best.pt"))
    evidence.extend((OUT / "pattern").glob("*/diagnostics.json"))
    evidence.extend(OUT.glob("branch_*/runs/*/transfer_*/metrics.json"))
    evidence.extend(OUT.glob("branch_*/runs/*/transfer_*/*/sealed_masks.pt"))
    evidence.extend(OUT.glob("branch_*/runs/*/transfer_*/*/results.pt"))
    evidence.extend(
        OUT.glob("branch_*/runs/*/transfer_*/*/search/history.json")
    )
    sources.update(
        {str(path.relative_to(ROOT)): digest(path) for path in evidence}
    )
    save_json(
        ASSETS / "agop_summary.json",
        {
            "training": summaries,
            "all_training": training,
            "transfer": transfer,
            "full_transfer": full_summary,
            "pattern": pattern,
            "prior_structure": prior,
            "interventions": intervention,
            "protocol": read(OUT / "protocol.json"),
            "sources": sources,
        },
    )
    print(
        json.dumps(
            {
                "training": summaries,
                "pattern_runs": len(pattern),
                "transfer_rows": len(transfer),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

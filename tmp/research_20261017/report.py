"""Rebuild cached figures/tables; Markdown reports are reviewed snapshots."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

from multimask.common import references

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pattern.io import digest

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
OUT = HOME / "reports"
ASSETS = HOME / ".cache/report_assets"
KINDS = ("original", "reverse", "shared", "cvae", "cvae_elbo")
LABELS = ("12E / 1D", "1E / 12D", "1E / 1D", "cVAE surrogate", "cVAE ELBO")
COLORS = ("#566a86", "#218c75", "#b07535", "#306eb8", "#8546a1")
SEEDS = (4213, 4214, 4215)


def read(path):
    return json.loads(path.read_text())


def csv_file(name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (ASSETS / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def figure(fig, name):
    fig.savefig(ASSETS / f"{name}.png", dpi=175, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def fmt(x, precision=4):
    return f"{x:.{precision}f}"


def main():
    ASSETS.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "path",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    stats, history, transfer = {}, [], {}
    for kind in KINDS:
        for seed in SEEDS:
            run = HOME / "runs" / f"{kind}_{seed}"
            stats[kind, seed] = read(run / "diagnostics.json")
            for row in read(run / "history.json"):
                history.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        **{
                            k: v
                            for k, v in row.items()
                            if not isinstance(v, dict)
                        },
                    }
                )
            if kind == "original" and seed == 4213:
                old = (
                    ROOT
                    / "multimask/runs/nf_vae_v3/evaluations/z_gumbel_20261013"
                )
                report = read(old / "metrics.json")["families"]
                transfer[kind, seed] = {}
                for family, values in report.items():
                    initial = torch.load(
                        old / family / "generated_masks/initial_z.pt",
                        weights_only=True,
                        map_location="cpu",
                    )
                    transfer[kind, seed][family] = {
                        "head": 0,
                        "selected_search_step": values["selected_search_step"],
                        "methods": {
                            "optimized": {
                                "iou": values["selected_iou"],
                                **values["methods"]["optimized_z"],
                            },
                            "best_prior": {
                                "iou": float(
                                    initial["iou"][values["best_random_index"]]
                                ),
                                **values["methods"]["best_random_z"],
                            },
                        },
                    }
            else:
                stage = "transfer_gumbel" if seed == 4213 else "transfer_prior"
                assert (run / stage / "COMPLETE.json").exists(), run / stage
                transfer[kind, seed] = read(run / stage / "metrics.json")
    architecture = []
    for kind, label in zip(KINDS, LABELS):
        protocol = read(HOME / "runs" / f"{kind}_4213/protocol.json")
        values = [stats[kind, s]["bank_test"]["iou"] for s in SEEDS]
        architecture.append(
            {
                "kind": kind,
                "label": label,
                "parameters": protocol["parameters"],
                "test_iou_mean": float(np.mean(values)),
                "test_iou_sd": float(np.std(values, ddof=1)),
                "kl_mean": float(
                    np.mean([stats[kind, s]["bank_test"]["kl"] for s in SEEDS])
                ),
            }
        )
    latent = read(HOME / "latent_probe.json")
    csv_file("architecture", architecture)
    csv_file("training", history)
    csv_file("latent_interventions", latent)
    transfer_rows = [
        {
            "kind": kind,
            "seed": seed,
            "family": family,
            "method": method,
            **result,
        }
        for (kind, seed), report in transfer.items()
        for family, values in report.items()
        for method, result in values["methods"].items()
    ]
    csv_file("transfer", transfer_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    bars = axes[0].bar(
        range(5),
        [r["test_iou_mean"] for r in architecture],
        yerr=[r["test_iou_sd"] for r in architecture],
        capsize=3,
        color=COLORS,
    )
    axes[0].bar_label(bars, fmt="%.3f", padding=4)
    axes[0].set(ylim=(0, 1.16), title="Bank test: mean and SD / 3 model seeds")
    axes[0].set_xticks(range(5), LABELS, rotation=18)
    for offset, mode, label, color in (
        (-0.25, "posterior", "Own posterior", "#306eb8"),
        (0, "wrong_family_posterior", "Other-family posterior", "#b07535"),
        (0.25, "zero", "Zero latent", "#218c75"),
    ):
        means = [
            np.mean(
                [
                    r["iou"]
                    for r in latent
                    if r["kind"] == kind and r["mode"] == mode
                ]
            )
            for kind in KINDS
        ]
        axes[1].bar(
            np.arange(5) + offset, means, width=0.23, label=label, color=color
        )
    axes[1].set_xticks(range(5), LABELS, rotation=18)
    axes[1].set(ylim=(0, 1.15), title="Source-validation: intervention in z")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_ylabel("Nearest-reference IoU")
        ax.grid(axis="y", alpha=0.18)
    figure(fig, "sharing")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for row, family in enumerate("AB"):
        for offset, method, label, color in (
            (-0.18, "best_prior", "Source-selected prior", "#aeb6bf"),
            (0.18, "optimized", "Gumbel optimized", "#306eb8"),
        ):
            data = [
                transfer[k, 4213][family]["methods"][method] for k in KINDS
            ]
            for col, metric, title in (
                (0, "iou", "Selected IoU"),
                (1, "excess_bce", "Held-out excess BCE"),
            ):
                values = [max(0, d[metric]) for d in data]
                bars = axes[row, col].bar(
                    np.arange(5) + offset,
                    values,
                    width=0.34,
                    color=color,
                    label=label,
                )
                axes[row, col].bar_label(
                    bars, fmt="%.3f", fontsize=7, padding=3
                )
                axes[row, col].set_xticks(range(5), LABELS, rotation=16)
                axes[row, col].set_title(f"{family}: {title}")
                axes[row, col].grid(axis="y", alpha=0.18)
        axes[row, 0].set_ylim(0, 1.14)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed seed 4213; no held-out selection of hyperparameters")
    figure(fig, "transfer")

    for name, selected in (
        ("sharing_training", KINDS[:3]),
        ("cvae_training", ("shared", "cvae", "cvae_elbo")),
    ):
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
        for kind in selected:
            index = KINDS.index(kind)
            for seed in SEEDS:
                rows = [
                    r
                    for r in history
                    if r["kind"] == kind and r["seed"] == seed
                ]
                for ax, metric in zip(
                    axes,
                    ("validation_loss", "validation_iou", "validation_kl"),
                ):
                    ax.plot(
                        [r["epoch"] for r in rows],
                        [r[metric] for r in rows],
                        color=COLORS[index],
                        alpha=0.55,
                        label=LABELS[index] if seed == 4213 else None,
                    )
        for ax, label in zip(
            axes,
            ("Validation objective*", "Validation IoU", "KL (unweighted)"),
        ):
            ax.set(xlabel="Epoch", ylabel=label)
            ax.grid(alpha=0.18)
        axes[1].legend(fontsize=8)
        fig.suptitle(
            "*ELBO and surrogate losses have different scales"
            if "cvae" in name
            else "Sharing: each line is one model seed"
        )
        figure(fig, name)

    pattern = read(HOME / "pattern/metrics.json")
    csv_file("pattern_controls", pattern["records"])
    names = (
        "trained_zero",
        "trained_prior",
        "uniform_random_32",
        "permuted_reference_columns",
        "permuted_reference_rows",
        "uncanonicalized_frequency",
        "canonical_frequency",
    )
    rows = [
        next(
            r
            for r in pattern["records"]
            if r["kind"] == name
            and (name != "trained_prior" or r["scale"] == 1)
        )
        for name in names
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout="constrained")
    x = np.arange(len(rows))
    axes[0].bar(
        x - 0.18,
        [r["mean_raw_iou"] for r in rows],
        0.34,
        label="Raw IoU",
        color="#306eb8",
    )
    axes[0].bar(
        x + 0.18,
        [r["mean_aligned_iou"] for r in rows],
        0.34,
        label="Column-aligned IoU",
        color="#218c75",
    )
    axes[0].set_xticks(
        x,
        (
            "D(0)",
            "D(N(0,I))",
            "Random 32",
            "Permute columns",
            "Permute rows",
            "Raw frequency",
            "Canonical frequency",
        ),
        rotation=30,
        ha="right",
    )
    axes[0].set(
        ylabel="Mean IoU",
        ylim=(0, 1.12),
        title="Alignment does not invent input support",
    )
    axes[0].legend(fontsize=8)
    prior = [r for r in pattern["records"] if r["kind"] == "trained_prior"]
    axes[1].plot(
        [r["scale"] for r in prior],
        [r["aligned_exact_fraction"] for r in prior],
        "o-",
        color="#306eb8",
    )
    axes[1].set(
        xscale="log",
        xlabel="Prior standard deviation",
        ylabel="Exact-mask fraction",
        ylim=(0, 1.05),
        title="Literal zero differs from random initial z",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
    figure(fig, "pattern_controls")
    panels = torch.load(
        HOME / "pattern/panels.pt", weights_only=True, map_location="cpu"
    )
    fig, axes = plt.subplots(
        1, len(panels["names"]), figsize=(12, 3.2), layout="constrained"
    )
    for ax, name, mask in zip(axes, panels["names"], panels["masks"]):
        ax.imshow(mask, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
        ax.set_title(name.replace("_", "\n"), fontsize=8)
        ax.set_xticks((0, 7))
        ax.set_yticks((0, 10))
    figure(fig, "pattern_masks")

    fig, axes = plt.subplots(2, 4, figsize=(10, 6), layout="constrained")
    for row, family in enumerate("AB"):
        base = HOME / "runs/cvae_elbo_4213/transfer_gumbel" / family
        sealed = torch.load(
            base / "sealed_masks.pt", weights_only=True, map_location="cpu"
        )
        result = torch.load(
            base / "results.pt", weights_only=True, map_location="cpu"
        )
        masks, orders = sealed["masks"], result["orders"]
        panels = (
            (
                "Nearest reference",
                references(f"{family}_0000")[result["variants"][0]],
            ),
            ("Zero latent", masks[2][:, orders[2]]),
            ("Optimized mask", masks[0][:, orders[0]]),
            ("Decoder sigmoid", result["probabilities"][:, orders[0]]),
        )
        for ax, (name, values) in zip(axes[row], panels):
            image = ax.imshow(
                values,
                vmin=0,
                vmax=1,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
            ax.set(title=name, xlabel="Hidden column")
            ax.set_xticks((0, 3, 7))
            ax.set_yticks((0, 16, 33, 34, 50, 67))
        axes[row, 0].set_ylabel(f"Family {family} / input")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    fig.suptitle("cVAE ELBO: aligned masks and decoder output")
    figure(fig, "cvae_masks")

    zero_route = []
    for seed in SEEDS:
        route = HOME / "runs" / f"reverse_{seed}" / "transfer_prior_prior_head"
        assert (route / "COMPLETE.json").exists(), route
        route = read(route / "metrics.json")
        zero_route.append(
            (
                seed,
                " / ".join(
                    fmt(route[f]["methods"]["best_prior"]["iou"]) for f in "AB"
                ),
                " / ".join(
                    fmt(route[f]["methods"]["best_prior"]["bce"], 5)
                    for f in "AB"
                ),
            )
        )
    recovered = read(HOME / "runs/shared_4214_no_stop/diagnostics.json")
    summary = {
        "architecture": architecture,
        "pattern": pattern,
        "transfer": transfer_rows,
        "source_head_routing": zero_route,
        "shared_no_stop": recovered,
        "note": (
            "Three training seeds; one full gradient-search seed. "
            "No bank rebuild; original seed 4213 transfer is cached."
        ),
    }
    (ASSETS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    sources = []
    for path in sorted((HOME / "runs").glob("*/best.pt")):
        sources.append(
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        )
    for path in (
        ROOT / "multimask/runs/nf_vae_v3/checkpoints/best.pt",
        ROOT / "pattern/runs/nf_vae_v2/checkpoints/best.pt",
        ROOT / "data/multimask/banks/imp32_128_mlp_v3/manifest.json",
        ROOT
        / "data/pattern/banks/imp32_fullsupport_1000_20261009/manifest.json",
    ):
        sources.append(
            {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        )
    (ASSETS / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print("Figures and tables complete:", OUT)


if __name__ == "__main__":
    main()

"""Frozen Omniglot comparison for compensated aSRR block exchanges."""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.asrr_anchor_init_v1.omniglot_transfer import score_episodes
from experiments.asrr_anchor_init_v1.transfer import public_panel
from experiments.paper_asrr_v1.extract import source_profiles
from experiments.paper_compact_v1.source import OMNIGLOT as COMPACT_ROOT
from experiments.paper_compact_v1.source import OUTPUT as COMPACT_OUTPUT
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .asrr_functional import indices
from .merge_cost import QuadraticBank, joint_basis, optimize, relu_bound
from .merge_cost_synthetic import OUTPUT, summary
from .source import (
    BANKS,
    CLASSES,
    CODES,
    HIDDEN,
    INPUTS,
    KEY,
    ROOT,
    WEIGHTS,
    read_json,
)
from .source import OUTPUT as PRIOR
from .transfer import PANELS, SETTING, STUDY, job, run_fits

PREFIX = "asrr_merge_"
BASELINE = "four_partitions"
OPTIONS = {
    "identity_exchange": ("identity", True, False),
    "covariance_equal": ("covariance", False, False),
    "covariance_exchange": ("covariance", True, False),
    "covariance_rematch": ("covariance", True, True),
}
TARGET_JOBS = OUTPUT / "target_jobs.json"
CHOICES = OUTPUT / "choices_frozen.json"
PANELS_PLAN = OUTPUT / "target_panels.json"
SOURCE_DENSE = ROOT / "source/dense/models.npz"
PANEL_SEEDS = tuple(range(7611193750, 7611193758))
EPISODES = 128
MOVES = 32
RIDGE_FRACTION = 0.05


@dataclass
class Source:
    images: np.ndarray
    origin: np.ndarray
    bias: np.ndarray
    head: np.ndarray
    output_bias: np.ndarray
    order: np.ndarray
    banks: dict
    ridge: float


def source_data():
    expected = read_json(PRIOR / "registration.json")
    actual = digest(SOURCE_DENSE)
    if (
        actual
        != read_json(SOURCE_DENSE.with_name("metrics.json"))["models_sha256"]
    ):
        raise ValueError("Changed source Dense fits")
    images = []
    for panel in ("support", "numerical"):
        path = ROOT / "public" / f"{panel}.npz"
        if digest(path) != expected["source_panel_sha256"][panel]:
            raise ValueError(f"Changed source data: {panel}")
        with np.load(path, allow_pickle=False) as archive:
            images.append(archive["x"].astype(np.float64) / 255)
    direction, _, _, order = source_profiles(SOURCE_DENSE, INPUTS, HIDDEN)
    with np.load(SOURCE_DENSE, allow_pickle=False) as archive:
        params = archive["parameters"][order].astype(np.float64)
    images = np.concatenate(images, axis=1)[order]
    origin = images.mean(axis=1, keepdims=True)
    centered = images - origin
    covariance = centered.transpose(0, 2, 1) @ centered / images.shape[1]
    pooled = covariance.mean(axis=0)
    ridge = float(np.trace(pooled) / INPUTS * RIDGE_FRACTION)
    metric = 0.5 * (covariance + pooled) + ridge * np.eye(INPUTS)
    raw = params[:, :WEIGHTS].reshape(-1, INPUTS, HIDDEN)
    scale = np.maximum(np.linalg.norm(raw, axis=1), 1e-12)
    align = np.stack(
        [
            linear_sum_assignment(-u.T @ (w / s))[1]
            for u, w, s in zip(direction, raw, scale, strict=True)
        ]
    )
    bias = np.take_along_axis(
        params[:, WEIGHTS : WEIGHTS + HIDDEN] / scale, align, 1
    )
    head = params[:, WEIGHTS + HIDDEN : -CLASSES].reshape(-1, HIDDEN, CLASSES)
    head = (
        (head - head.mean(axis=2, keepdims=True))
        * scale[:, :, None]
        / np.sqrt(HIDDEN)
    )
    head = np.take_along_axis(head, align[:, :, None], axis=1)
    importance = np.linalg.norm(head, axis=2)
    identity = np.broadcast_to(np.eye(INPUTS), metric.shape)
    banks = {
        name: QuadraticBank(direction, m, importance)
        for name, m in (("identity", identity), ("covariance", metric))
    }
    return Source(
        images, origin, bias, head, params[:, -CLASSES:], order, banks, ridge
    )


def read_partitions(record, blocks):
    path = BANKS / f"{record['name']}.npz"
    if digest(path) != record["model_sha256"]:
        raise ValueError(f"Changed indexator: {path}")
    with np.load(path, allow_pickle=False) as data:
        index = data["index"]
    return [
        np.unique(block[:, 0], return_inverse=True)[1]
        for block in np.split(index, blocks, axis=1)
    ]


def construct():
    if not (OUTPUT / "synthetic.json").exists():
        raise ValueError("Finish synthetic experiments first")
    if CHOICES.exists():
        raise FileExistsError(CHOICES)
    boundary = StudyBoundary(ROOT)
    boundary.install()
    seeds = read_json(
        PRIOR.with_name("omniglot64_asrr_eight_partitions_2026-11-18")
        / CHOICES.name
    )
    policies = seeds["policies"]
    policies[BASELINE]["blocks"] = 4
    policies["eight_partitions"]["blocks"] = 8
    initial = read_partitions(policies[BASELINE], 4)
    source = source_data()
    traces = {}
    for name, (metric, exchange, rematch) in OPTIONS.items():
        started = time.perf_counter()
        partitions, trace = optimize(
            source.banks[metric],
            initial,
            exchange=exchange,
            rematch=rematch,
            moves=MOVES,
        )
        index = indices(partitions)
        if int(index.max()) != CODES or np.any(index <= 0):
            raise ValueError(f"Code budget changed: {name}")
        candidate = f"{PREFIX}{name}"
        path = BANKS / f"{candidate}.npz"
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, index=index)
        policies[name] = {
            "name": candidate,
            "model_sha256": digest(path),
            "blocks": 4,
        }
        traces[name] = {
            "seconds": time.perf_counter() - started,
            "trace": trace,
            "max_exact_gain_error": max(
                (
                    abs(row["predicted_gain"] - row["refit_gain"])
                    for row in trace[1:]
                ),
                default=0.0,
            ),
        }
        print(
            f"{name}: {trace[0]['q']:.4f} -> {trace[-1]['q']:.4f}; {trace[-1]['groups']}",
            flush=True,
        )
    save_json(
        CHOICES, {"policies": policies, "numeric_rule": seeds["numeric_rule"]}
    )
    save_json(OUTPUT / "source_search.json", traces)
    save_json(
        OUTPUT / "registration.json",
        {
            "dataset": "Omniglot",
            "inputs": INPUTS,
            "hidden": HIDDEN,
            "codes": CODES,
            "choices_sha256": digest(CHOICES),
            "source_sha256": digest(SOURCE_DENSE),
            "source_tasks": read_json(PRIOR / "registration.json")[
                "source_tasks"
            ],
            "source_images_per_task": source.images.shape[1],
            "metric": "half task covariance + half pooled within-task covariance + ridge",
            "ridge_fraction": RIDGE_FRACTION,
            "ridge": source.ridge,
            "source_importance": "norm of the centered head after absorbing input weight scale",
            "options": OPTIONS,
            "maximum_moves": MOVES,
            "panel_seeds": PANEL_SEEDS,
            "episodes_per_panel": EPISODES,
            "fit_seed_base": 7611193950,
            "primary": "covariance_rematch minus four_partitions",
            "synthetic_results_sha256": digest(OUTPUT / "synthetic.json"),
            "implementation_sha256": digest(__file__),
        },
    )
    boundary.save(OUTPUT / "source_access.json", "complete")


def prepare():
    if TARGET_JOBS.exists():
        raise FileExistsError(TARGET_JOBS)
    config = read_json(OUTPUT / "registration.json")
    if digest(CHOICES) != config["choices_sha256"]:
        raise ValueError("Changed frozen indexators")
    choices = read_json(CHOICES)
    panels, jobs = {}, []
    for number, seed in enumerate(config["panel_seeds"]):
        phase = f"test_merge_cost_{number:02d}"
        panels[phase] = public_panel(
            SETTING,
            phase,
            config["episodes_per_panel"],
            seed,
            directory=PANELS,
            target=True,
        )
        jobs.extend(
            {
                **job(
                    row,
                    phase,
                    panels[phase]["sha256"],
                    seed=config["fit_seed_base"] + number * EPISODES,
                ),
                **({"input_blocks": row["blocks"]} if "blocks" in row else {}),
            }
            for row in choices["policies"].values()
        )
    # Give each GPU its expensive WSR fit before the smaller folded jobs.
    jobs.sort(key=lambda item: "input_blocks" in item)
    save_json(
        PANELS_PLAN, {"panels": panels, "choices_sha256": digest(CHOICES)}
    )
    save_json(TARGET_JOBS, jobs)
    print(
        f"Frozen {len(jobs)} fits on {len(panels) * EPISODES} new episodes",
        flush=True,
    )


def pair_diagnostic(images, *, tail_fraction=None):
    x = images.reshape(-1, images.shape[-1])
    x = x - x.mean(axis=0)
    eligible = x.std(axis=0) > 0.02
    x = x[:, eligible] / x[:, eligible].std(axis=0)
    gram = x.T @ x / len(x)
    a, b = np.triu_indices(gram.shape[0], k=1)
    gamma = gram[a, b]
    if tail_fraction is None:
        strata = (
            ("uncorrelated", abs(gamma) <= 0.1),
            ("correlated", gamma >= 0.9),
        )
    else:
        lo, hi = np.quantile(gamma, [tail_fraction, 1 - tail_fraction])
        strata = (
            ("lowest_correlation", gamma <= lo),
            ("highest_correlation", gamma >= hi),
        )
    rows = {}
    for label, mask in strata:
        pairs = np.flatnonzero(mask)
        selected = np.random.default_rng(7611193900).choice(
            pairs, size=min(len(pairs), 512), replace=False
        )
        left, right = x[:, a[selected]], x[:, b[selected]]
        sum_input, target = left + right, 2 * left - right
        c = (sum_input * target).sum(axis=0) / np.square(sum_input).sum(axis=0)
        loss = np.square(target - c * sum_input).mean(axis=0)
        prediction = (1 - gamma[selected]) * 4.5
        rows[label] = {
            "sampled_pairs": len(selected),
            "mean_correlation": float(gamma[selected].mean()),
            "mean_mse": float(loss.mean()),
            "max_absolute_formula_error": float(
                np.max(abs(loss - prediction))
            ),
        }
    return rows


def source_diagnostics():
    out = OUTPUT / "source_diagnostics.json"
    if out.exists():
        raise FileExistsError(out)
    source = source_data()
    compact = read_json(COMPACT_OUTPUT / "registration.json")
    if (
        compact["omniglot_source_tasks"]
        != read_json(OUTPUT / "registration.json")["source_tasks"]
    ):
        raise ValueError(
            "Held-out source drawings have different source tasks"
        )
    heldout_path = COMPACT_ROOT / "public/structure.npz"
    if digest(heldout_path) != compact["omniglot_source_sha256"]["structure"]:
        raise ValueError("Changed held-out source drawings")
    with np.load(heldout_path, allow_pickle=False) as archive:
        heldout = archive["x"][source.order].astype(np.float64) / 255
    bank = source.banks["covariance"]
    rows = {}
    for name, row in read_json(CHOICES)["policies"].items():
        if "blocks" not in row:
            continue
        partition = read_partitions(row, row["blocks"])
        fits = [bank.fit(labels) for labels in partition]
        results = {}
        for matching, rematch in (("fixed", False), ("matched", True)):
            placement = bank.assignment(fits, rematch)
            approximation = bank.directions_from_fits(fits, placement)
            kl, bound = relu_bound(
                heldout,
                bank.directions,
                approximation,
                source.bias,
                source.head,
                source.output_bias,
                origin=source.origin,
            )
            results[matching] = {
                "training_q": float(bank.losses(fits, placement).sum()),
                "mean_source_holdout_kl": float(kl.mean()),
                "mean_source_holdout_bound": float(bound.mean()),
                "median_kl_over_bound": float(np.median(kl / bound)),
                "violations": int(np.count_nonzero(kl > bound + 1e-10)),
                "per_task_kl": kl.tolist(),
                "per_task_bound": bound.tolist(),
            }
        results["rank"] = joint_basis(partition).shape[1]
        rows[name] = results
    save_json(
        out,
        {
            "tasks": len(heldout),
            "holdout_drawings_per_task": heldout.shape[1],
            "holdout_sha256": digest(heldout_path),
            "two_standardized_pixels": pair_diagnostic(source.images),
            "models": rows,
        },
    )
    print(
        {
            m: round(r["matched"]["mean_source_holdout_kl"], 5)
            for m, r in rows.items()
        },
        flush=True,
    )


def projection_statistics(panel, basis):
    all_x = np.concatenate(
        [panel[k + "_x"] for k in ("train", "valid", "query")], axis=1
    )
    all_y = np.concatenate(
        [panel[k + "_y"] for k in ("train", "valid", "query")], axis=1
    )
    z = all_x @ basis.astype(all_x.dtype)
    counts = all_y.sum(axis=1)
    support = panel["train_y"].sum(axis=1)
    mean = (all_y.transpose(0, 2, 1) @ z) / counts[..., None]
    support_mean = (
        panel["train_y"].transpose(0, 2, 1) @ z[:, : panel["train_x"].shape[1]]
    ) / support[..., None]
    within = (
        (all_y.transpose(0, 2, 1) @ np.square(z).sum(axis=-1, keepdims=True))[
            ..., 0
        ]
        - counts * np.square(mean).sum(axis=-1)
    ) / (counts - 1)
    a, b = np.triu_indices(CLASSES, k=1)
    all_diff = mean[:, a] - mean[:, b]
    sample_diff = support_mean[:, a] - support_mean[:, b]
    empirical_noise = np.square(sample_diff - all_diff).sum(axis=-1)
    trace_noise = within[:, a] + within[:, b]
    finite_noise = (1 / support - 1 / counts) * within
    return {
        "signal": np.square(all_diff).sum(axis=-1).mean(axis=1),
        "noise_trace": trace_noise.mean(axis=1),
        "finite_noise": (finite_noise[:, a] + finite_noise[:, b]).mean(axis=1),
        "observed_noise": empirical_noise.mean(axis=1),
    }


def rank_diagnostics():
    path = OUTPUT / "rank_diagnostics.json"
    if path.exists():
        raise FileExistsError(path)
    policies = read_json(CHOICES)["policies"]
    partitions = {
        name: read_partitions(policies[name], policies[name]["blocks"])
        for name in (BASELINE, "eight_partitions")
    }
    partitions["union_four_and_eight"] = (
        partitions[BASELINE] + partitions["eight_partitions"]
    )
    bases = {name: joint_basis(p) for name, p in partitions.items()}
    data = {"development": {"path": str(PANELS / "development.npz")}}
    splits = {"development": data, "test": read_json(PANELS_PLAN)["panels"]}
    reports = {}
    for split, cohorts in splits.items():
        statistics = {name: [] for name in bases}
        for record in cohorts.values():
            with np.load(record["path"], allow_pickle=False) as archive:
                panel = dict(archive)
            if "answer_path" in record:
                if digest(record["answer_path"]) != record["answer_sha256"]:
                    raise ValueError("Changed held-out answers")
                with np.load(
                    record["answer_path"], allow_pickle=False
                ) as archive:
                    panel["query_y"] = archive["y"]
            for name, basis in bases.items():
                statistics[name].append(projection_statistics(panel, basis))
        base = statistics[BASELINE]
        comparisons = {}
        for name, values in statistics.items():
            if name == BASELINE:
                continue
            delta = {
                k: np.concatenate(
                    [a[k] - b[k] for a, b in zip(values, base, strict=True)]
                )
                for k in base[0]
            }
            comparisons[name] = {
                "signal_difference": summary(delta["signal"]),
                "noise_trace_difference": summary(delta["noise_trace"]),
                "iid_mse_gain_n10": summary(
                    delta["signal"] - delta["noise_trace"] / 10
                ),
                "finite_pool_mse_gain_n10": summary(
                    delta["signal"] - delta["finite_noise"]
                ),
                "observed_mse_gain_n10": summary(
                    delta["signal"] - delta["observed_noise"]
                ),
                "episodes": len(delta["signal"]),
            }
        reports[split] = comparisons
    save_json(
        path, {"ranks": {n: b.shape[1] for n, b in bases.items()}, **reports}
    )


def score():
    path = OUTPUT / "target_results.json"
    if path.exists():
        raise FileExistsError(path)
    rule = read_json(CHOICES)["numeric_rule"]
    results = score_episodes(OUTPUT, STUDY, {"key": KEY, "rule": rule})
    pooled = {
        name: np.concatenate([r[name] for r in results.values()])
        for name in next(iter(results.values()))
    }
    paired = {}
    for a, b in (
        *((key, BASELINE) for key in pooled if key != BASELINE),
        ("covariance_exchange", "identity_exchange"),
        ("covariance_exchange", "covariance_equal"),
        ("covariance_rematch", "covariance_exchange"),
        ("covariance_rematch", "wsr"),
    ):
        change = pooled[a] - pooled[b]
        paired[f"{a}_minus_{b}"] = {
            "accuracy": summary(change[:, 0]),
            "ce": summary(change[:, 1]),
        }
    rows = {
        key: {"accuracy": summary(value[:, 0]), "ce": summary(value[:, 1])}
        for key, value in pooled.items()
    }
    save_json(
        path,
        {"episodes": len(pooled[BASELINE]), "methods": rows, "paired": paired},
    )
    np.savez_compressed(OUTPUT / "target_episode_scores.npz", **pooled)
    print(
        {
            key: round(100 * row["accuracy"]["mean"], 3)
            for key, row in rows.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "construct",
            "prepare",
            "source-diagnostics",
            "run",
            "score",
            "rank",
        ),
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()
    if args.action == "run":
        run_fits("target", args.gpus, root=ROOT, output=OUTPUT)
    elif args.action == "rank":
        if not (OUTPUT / "predictions_frozen.json").exists():
            raise ValueError(
                "Freeze all predictions before reading query answers"
            )
        rank_diagnostics()
    else:
        {
            "construct": construct,
            "prepare": prepare,
            "source-diagnostics": source_diagnostics,
            "score": score,
        }[args.action]()

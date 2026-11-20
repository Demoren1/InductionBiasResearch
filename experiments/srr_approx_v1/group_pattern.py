"""Recover shared hidden supports from Dense proposals and source responses."""

import argparse
import io
import json
from math import comb
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, xlogy

from experiments.sharing_structure_research_v1.blind_contract import save_json
from experiments.srr_approx_v1.core import canonical, signed_forest
from experiments.srr_approx_v1.group_transfer import OUT
from experiments.srr_approx_v1.pattern import SEEDS, historical_file, score


ARCHIVE = Path("artifacts/asrr_pattern_2026-11-19/retained_sources.npz")
REFERENCE = Path("archive/reports/final/assets/2026-11-03/pattern_heatmap_data.npz")


def source_data():
    """The benchmark generator is isolated from the estimator."""
    plan = json.loads(historical_file("codebook/protocol_v4.json"))
    x = (np.arange(2048)[:, None] >> np.arange(10, -1, -1)) & 1
    windows = np.lib.stride_tricks.sliding_window_view(x, 4, axis=1)
    y = np.array(
        [
            (windows == np.array(list(pattern), dtype=int)).all(-1).any(-1)
            for pattern in plan["source_patterns"]
        ]
    )
    splits = historical_file("runs/codebook_20261028_v4/splits.npz")
    with np.load(io.BytesIO(splits)) as data:
        return x, y.astype(float), data["support"], data["validation"]


def states(x, subsets):
    return np.stack(
        [x[:, subset] @ (2 ** np.arange(len(subset))) for subset in subsets]
    )


def codebook_from_atoms(subsets, levels, dimensions, budget):
    """Read the signed direction of each learned atom, then apply signed Ward."""
    hidden, tasks = levels.shape
    mask = np.zeros((dimensions, hidden), dtype=bool)
    profiles = np.zeros((tasks, dimensions, hidden))
    for h, subset in enumerate(subsets):
        mask[subset, h] = True
        signs = 2 * ((levels[h, :, None] >> np.arange(len(subset))) & 1) - 1
        profiles[:, subset, h] = signs
    group = signed_forest(profiles[:, mask, None], (budget,))[0][budget]
    result = np.zeros(mask.shape, dtype=np.int16)
    result[mask] = group.ravel()
    return canonical(result)


def refit_head(atoms, y):
    # Each task has its own intercept and readout. Weak ridge makes fits finite.
    design = np.concatenate([np.ones((*atoms.shape[:2], 1)), atoms], axis=2)

    def objective(raw):
        weights = raw.reshape(len(y), -1)
        z = np.einsum("tnk,tk->tn", design, weights)
        value = (np.logaddexp(0, z) - y * z).sum() + 0.005 * (weights**2).sum()
        grad = np.einsum("tn,tnk->tk", expit(z) - y, design) + 0.01 * weights
        return value, grad.ravel()

    fitted = minimize(
        objective,
        np.zeros(len(y) * design.shape[-1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 150},
    )
    return fitted.x.reshape(len(y), -1)


def predict_atoms(x, subsets, levels, head):
    codes = states(x, subsets)
    atoms = (codes[:, None, :] == levels[:, :, None]).transpose(1, 2, 0)
    return head[:, :1] + np.einsum("tnh,th->tn", atoms, head[:, 1:])


def bernoulli_kl(p, q):
    q = np.clip(q, 1e-10, 1.0 - 1e-10)
    return xlogy(p, p / q) + xlogy(1 - p, (1 - p) / (1 - q))


def recover(weights, x, y, budget):
    """Pursue recurrent coordinate sets with task-specific binary ridge units."""
    tasks, _, dimensions, hidden = weights.shape
    ranking = np.argsort(-abs(weights.transpose(0, 1, 3, 2)), axis=-1)
    ranking = ranking.reshape(-1, dimensions)
    # Nominally at least eight observations per binary cell. Width is inferred.
    max_width = min(dimensions, int(np.log2(len(x) / 8)))
    proposals = []
    for width in range(1, max_width + 1):
        subsets = np.unique(np.sort(ranking[:, :width], axis=1), axis=0)
        codes = states(x, subsets)
        width_states = 2**width
        offset = (np.arange(len(subsets)) * width_states)[:, None]
        flat = (codes + offset).ravel()
        size = len(subsets) * width_states
        counts = np.bincount(flat, minlength=size)
        positives = np.stack(
            [
                np.bincount(flat, weights=np.tile(target, len(subsets)), minlength=size)
                for target in y
            ]
        )
        proposals.append((subsets, codes, flat, counts, positives))
    selected, task_levels, trace = [], [], []
    predicted = np.broadcast_to(y.mean(axis=1, keepdims=True), y.shape)
    for _ in range(hidden):
        best = None
        for subsets, codes, flat, counts, positives in proposals:
            expected = np.stack(
                [
                    np.bincount(
                        flat, weights=np.tile(p, len(subsets)), minlength=len(counts)
                    )
                    for p in predicted
                ]
            )
            # Bernoulli deviance of the observed conditional response relative
            # to the current predictor. Shrink rare states towards that predictor.
            null = (expected + 0.5) / (counts + 1)
            observed = (positives + null) / (counts + 1)
            gains = (counts * bernoulli_kl(observed, null)).reshape(
                tasks, len(subsets), -1
            )
            level = gains.argmax(axis=2)
            strength = gains.max(axis=2).sum(axis=0)
            for j, subset in enumerate(subsets):
                if tuple(subset) in selected:
                    strength[j] = -np.inf
            j = strength.argmax()
            if best is None or strength[j] > best[0]:
                best = (strength[j], tuple(subsets[j]), level[:, j].copy())
        merit, subset, level = best
        selected.append(subset)
        task_levels.append(level)
        trace.append(float(merit))
        features = (
            states(x, selected)[:, None, :] == np.array(task_levels)[:, :, None]
        ).transpose(1, 2, 0)
        head = refit_head(features, y)
        predicted = expit(predict_atoms(x, selected, np.array(task_levels), head))
    subsets = [np.array(s) for s in selected]
    levels = np.array(task_levels)
    return (
        codebook_from_atoms(subsets, levels, dimensions, budget),
        subsets,
        levels,
        head,
        trace,
    )


def run():
    x, y, train, validation = source_data()
    with np.load(ARCHIVE) as data:
        fits = {seed: data[f"binary_{seed}_weights"] for seed in SEEDS}
    outputs, metadata = {}, {}
    for seed, weights in fits.items():
        index, subsets, levels, head, trace = recover(weights, x[train], y[:, train], 4)
        logits = predict_atoms(x[validation], subsets, levels, head)
        outputs[str(seed)] = index
        metadata[str(seed)] = {
            "subsets": [s.tolist() for s in subsets],
            "levels": levels.tolist(),
            "gain": trace,
            "support_examples": len(train),
            "validation_examples": len(validation),
            "validation_accuracy": float(((logits > 0) == y[:, validation]).mean()),
        }
    OUT.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(OUT / "pattern_indices.npz", **outputs)
    # The planted structure is first accessed after all three estimators return.
    with np.load(REFERENCE) as oracle:
        assessment = score(outputs, oracle["analytic_indices"], 4)
    save_json(
        OUT / "pattern_results.json",
        {key: {**row, **metadata[key]} for key, row in assessment.items()},
    )
    print(
        {
            key: (row["signed_entry_errors"], metadata[key]["validation_accuracy"])
            for key, row in assessment.items()
        },
        flush=True,
    )


def ablate():
    """Change candidate evidence or the sign estimator with other inputs fixed."""
    x, y, train, _ = source_data()
    structure = json.loads((OUT / "pattern_results.json").read_text())
    results = {}
    for seed in SEEDS:
        with np.load(ARCHIVE) as data:
            w = data[f"binary_{seed}_weights"]
        tasks, _, dims, hidden = w.shape
        ranks = np.argsort(-abs(w.transpose(0, 1, 3, 2)), axis=-1).reshape(-1, dims)
        widths = []
        for k in range(1, min(dims, int(np.log2(len(train) / 8))) + 1):
            sets, counts = np.unique(
                np.sort(ranks[:, :k], axis=1), axis=0, return_counts=True
            )
            best = np.argsort(-counts, kind="stable")[:hidden]
            p = counts[best].sum() / len(ranks)
            chance = min(1.0, hidden / comb(dims, k))
            value = float(bernoulli_kl(p, chance)) if p > chance else 0.0
            widths.append((value, sets[best]))
        frequent = max(widths, key=lambda a: a[0])[1]
        levels = []
        for subset in frequent:
            z = states(x[train], [subset])[0]
            n = np.bincount(z, minlength=2 ** len(subset))
            base = y[:, train].mean(axis=1, keepdims=True)
            hits = np.stack(
                [
                    np.bincount(z, weights=target, minlength=len(n))
                    for target in y[:, train]
                ]
            )
            levels.append((n * bernoulli_kl((hits + base) / (n + 1), base)).argmax(1))
        results[f"{seed}_frequency_support"] = codebook_from_atoms(
            frequent, np.array(levels), dims, 4
        )
        learned = structure[str(seed)]["subsets"]
        norm = w / np.maximum(np.linalg.norm(w, axis=2, keepdims=True), 1e-12)
        mask = np.zeros((dims, hidden), dtype=bool)
        profiles = np.zeros((tasks, dims, hidden))
        for h, subset in enumerate(learned):
            local = norm[:, :, subset, :]
            best = np.sum(local**2, axis=2).argmax(axis=-1)
            chosen = np.take_along_axis(local, best[:, :, None, None], axis=3)[..., 0]
            profiles[:, subset, h] = chosen.mean(axis=1)
            mask[subset, h] = True
        groups = signed_forest(profiles[:, mask, None], (4,))[0][4]
        index = np.zeros((dims, hidden), dtype=np.int16)
        index[mask] = groups.ravel()
        results[f"{seed}_learned_support_dense_signatures"] = canonical(index)
        rng = np.random.default_rng(seed + 50000)
        randomized = np.take_along_axis(w, rng.random(w.shape).argsort(axis=2), axis=2)
        results[f"{seed}_random_proposals"] = recover(
            randomized, x[train], y[:, train], 4
        )[0]
    with np.load(REFERENCE) as data:
        scores = score(results, data["analytic_indices"], 4)
    save_json(OUT / "pattern_ablations.json", scores)
    print({k: v["signed_entry_errors"] for k, v in scores.items()}, flush=True)


def figure():
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from experiments.paper_structure_benchmark_v1.reporting.evaluate import align_index

    with np.load(REFERENCE) as data:
        truth = data["analytic_indices"]
    with np.load(OUT / "pattern_anchor_indices.npz") as data:
        warm = align_index(data["391101_warm_native_zero_cluster"], truth, 4)[0]
        found = align_index(data["391101_warm_pursuit"], truth, 4)[0]
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.8), layout="constrained")
    colors = ListedColormap(["#f4f4f4", "#b3d8f0", "#a2ddbf", "#f5d58b", "#ccb9e3"])
    for ax, matrix, title in zip(
        axes,
        (truth, warm, found),
        ("Эталон", "Общий старт и Ward", "Проверка групп по ответам"),
    ):
        ax.imshow(abs(matrix), cmap=colors, norm=BoundaryNorm(np.arange(6) - 0.5, 5))
        for (i, j), label in np.ndenumerate(matrix):
            if label:
                ax.text(j, i, str(label), ha="center", va="center", fontsize=8)
        ax.set(
            title=title,
            xlabel="Скрытый нейрон",
            xticks=np.arange(8),
            xticklabels=np.arange(1, 9),
            yticks=np.arange(11),
            yticklabels=np.arange(1, 12),
        )
        ax.set_xticks(np.arange(9) - 0.5, minor=True)
        ax.set_yticks(np.arange(12) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1)
        ax.tick_params(which="both", length=0)
    axes[0].set_ylabel("Вход")
    fig.savefig(OUT / "pattern_recovered.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("fit", "ablate", "figure"), default="fit", nargs="?"
    )
    phase = parser.parse_args().phase
    {"fit": run, "ablate": ablate, "figure": figure}[phase]()

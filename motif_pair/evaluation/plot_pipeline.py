"""Make lightweight, headless diagnostic plots from motif-pair artifacts.

The plotting pipeline is deliberately read-only with respect to an experiment:
it never trains a model or regenerates data.  Every stage is independent, so
an early/interrupted run still gets a useful ``plots/manifest.json`` explaining
which later plots were unavailable and why.

Example::

    python evaluation/plot_pipeline.py \
      --run-root outputs/ood/pair_disjoint_aligned_seed_42 --pdf
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable

# Some batch containers expose a read-only home directory.  Keep Matplotlib's
# tiny font/cache files in a safe temporary location rather than emitting a
# warning for every plotting invocation.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/motif_pair_matplotlib")
import matplotlib

# Must be selected before importing pyplot: runners are often SSH/headless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


SUMMARY_RE = re.compile(
    r"val_loss \(BCE\) min=(?P<minimum>[-+0-9.eE]+) "
    r"p50=(?P<median>[-+0-9.eE]+) max=(?P<maximum>[-+0-9.eE]+)"
)


class ArtifactMissing(FileNotFoundError):
    """An optional experiment stage was not produced yet."""


def _task_gap(task: str) -> int:
    return config.parse_task(task).gap


def _read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _stage_missing(message: str) -> FileNotFoundError:
    return ArtifactMissing(message)


def _save(fig: plt.Figure, stem: str, plots_dir: Path, pdf: bool) -> list[str]:
    outputs: list[str] = []
    png = plots_dir / f"{stem}.png"
    fig.savefig(png, dpi=160, bbox_inches="tight")
    outputs.append(png.name)
    if pdf:
        path = plots_dir / f"{stem}.pdf"
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path.name)
    plt.close(fig)
    return outputs


def _add_stage(manifest: dict[str, Any], name: str, action: Callable[[], list[str]]) -> None:
    """Run a plot stage without letting an absent later artifact abort the run."""
    try:
        outputs = action()
    except ArtifactMissing as error:
        manifest["stages"][name] = {"status": "missing", "reason": str(error), "outputs": []}
    except FileNotFoundError as error:
        manifest["stages"][name] = {"status": "skipped", "reason": str(error), "outputs": []}
    except Exception as error:  # Keep a diagnostic manifest even for malformed artifacts.
        manifest["stages"][name] = {
            "status": "failed", "reason": f"{type(error).__name__}: {error}", "outputs": []
        }
    else:
        manifest["stages"][name] = {"status": "generated", "outputs": outputs}


def _ordered_tasks(split: dict[str, Any]) -> list[str]:
    train = split.get("train_tasks", [])
    test = split.get("test_tasks", [])
    if not isinstance(train, list) or not isinstance(test, list):
        raise ValueError("split needs train_tasks and test_tasks lists")
    return [str(task) for task in train + test]


def _choose_by_gap(tasks: Iterable[str], limit: int) -> list[str]:
    """Round-robin tasks by gap, avoiding a plot dominated by one condition."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for task in tasks:
        grouped[_task_gap(task)].append(task)
    selected: list[str] = []
    while len(selected) < limit and any(grouped.values()):
        for gap in config.GAPS:
            if grouped[gap] and len(selected) < limit:
                selected.append(grouped[gap].pop(0))
    return selected


def _plot_split(split_path: Path, plots_dir: Path, pdf: bool) -> list[str]:
    if not split_path.is_file():
        raise _stage_missing(f"missing split manifest: {split_path}")
    split = _read_json(split_path)
    train = [str(x) for x in split.get("train_tasks", [])]
    test = [str(x) for x in split.get("test_tasks", [])]
    if not train or not test:
        raise ValueError("split contains no train or test tasks")
    train_counts = [sum(_task_gap(task) == gap for task in train) for gap in config.GAPS]
    test_counts = [sum(_task_gap(task) == gap for task in test) for gap in config.GAPS]
    pair = lambda task: config.task_components(task)[:2]
    train_pairs = {pair(task) for task in train}
    test_pairs = {pair(task) for task in test}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    x = np.arange(len(config.GAPS))
    axes[0].bar(x - .19, train_counts, .38, label="meta-train")
    axes[0].bar(x + .19, test_counts, .38, label="held-out")
    axes[0].set(xticks=x, xticklabels=config.GAPS, xlabel="gap", ylabel="tasks", title="Task split by gap")
    axes[0].legend(frameon=False)
    axes[1].axis("off")
    pair_disjoint = not (train_pairs & test_pairs)
    lines = [
        f"train / test tasks: {len(train)} / {len(test)}",
        f"unique (A, B) pairs: {len(train_pairs)} / {len(test_pairs)}",
        f"pair-disjoint: {pair_disjoint}",
        f"manifest pair_disjoint: {split.get('pair_disjoint', False)}",
        f"split seed: {split.get('split_seed', 'unknown')}",
    ]
    axes[1].text(.04, .92, "\n".join(lines), va="top", fontsize=11, family="monospace")
    return _save(fig, "01_split", plots_dir, pdf)


def _default_data_dir(run_root: Path) -> Path:
    local = run_root / "data"
    return local if local.is_dir() else config.DATA_DIR


def _plot_data(split_path: Path, data_dir: Path, plots_dir: Path, pdf: bool, max_tasks: int) -> list[str]:
    if not split_path.is_file():
        raise _stage_missing(f"missing split manifest: {split_path}")
    split = _read_json(split_path)
    tasks = _choose_by_gap(_ordered_tasks(split), max_tasks)
    rows: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    for task in tasks:
        path = data_dir / f"val_{task}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict) or "y" not in payload or "ones_count" not in payload:
            continue
        y = payload["y"].detach().cpu().numpy()
        counts = payload["ones_count"].detach().cpu().numpy()
        rows.append((task, float(np.mean(y)), counts[y == 0], counts[y == 1]))
    if not rows:
        raise _stage_missing(f"no readable validation sets in {data_dir}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    labels = [f"g{_task_gap(task)}" for task, *_ in rows]
    axes[0].bar(np.arange(len(rows)), [row[1] for row in rows])
    axes[0].axhline(.5, color="black", lw=1, ls="--")
    axes[0].set(xticks=np.arange(len(rows)), xticklabels=labels, ylim=(0, 1),
                xlabel="validation task", ylabel="positive fraction", title="Class balance")
    positive = np.concatenate([row[3] for row in rows])
    negative = np.concatenate([row[2] for row in rows])
    bins = np.arange(min(positive.min(), negative.min()) - .5, max(positive.max(), negative.max()) + 1.5)
    axes[1].hist(negative, bins=bins, alpha=.65, density=True, label="negative")
    axes[1].hist(positive, bins=bins, alpha=.65, density=True, label="positive")
    axes[1].set(xlabel="number of +1 bits", ylabel="density", title="Matched one-count distribution")
    axes[1].legend(frameon=False)
    outputs = _save(fig, "02_data", plots_dir, pdf)
    outputs.extend(_plot_data_examples(split, data_dir, plots_dir, pdf))
    return outputs


def _plot_data_examples(split: dict[str, Any], data_dir: Path, plots_dir: Path,
                        pdf: bool) -> list[str]:
    """Show one real positive/negative sequence and its two motif locations."""
    required = {"x", "y", "a_start", "b_start", "delta"}
    # Prefer a held-out task: it is the data distribution on which the final
    # structural transfer claim is evaluated.
    candidates = [str(task) for task in split.get("test_tasks", [])]
    candidates.extend(str(task) for task in split.get("train_tasks", []))
    selected: tuple[str, dict[str, Any], list[int]] | None = None
    for task in candidates:
        path = data_dir / f"val_{task}.pt"
        if not path.is_file():
            continue
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict) or not required.issubset(payload):
            continue
        y = torch.as_tensor(payload["y"]).flatten()
        positive = torch.nonzero(y == 1, as_tuple=False).flatten()
        negative = torch.nonzero(y == 0, as_tuple=False).flatten()
        if positive.numel() and negative.numel():
            selected = (task, payload, [int(positive[0]), int(negative[0])])
            break
    if selected is None:
        return []

    task, payload, example_indices = selected
    parsed = config.parse_task(task)
    fig, axes = plt.subplots(2, 1, figsize=(12, 3.8), constrained_layout=True)
    for ax, index, label in zip(axes, example_indices, ("positive", "hard negative")):
        sequence = torch.as_tensor(payload["x"])[index].detach().cpu().numpy()
        a_start = int(torch.as_tensor(payload["a_start"])[index])
        b_start = int(torch.as_tensor(payload["b_start"])[index])
        delta = int(torch.as_tensor(payload["delta"])[index])
        ax.imshow(sequence[None, :], cmap="coolwarm", vmin=-1, vmax=1,
                  origin="lower", aspect="auto", extent=(-.5, config.SEQ_LEN - .5, -.5, .5))
        for position, value in enumerate(sequence):
            ax.text(position, 0, f"{int(value):+d}", ha="center", va="center",
                    color="white" if value < 0 else "black", fontsize=9, fontweight="bold")
        for start, color in ((a_start, "tab:green"), (b_start, "tab:orange")):
            for offset in range(config.MOTIF_LEN):
                position = (start + offset) % config.SEQ_LEN
                ax.add_patch(Rectangle((position - .47, -.47), .94, .94, fill=False,
                                       edgecolor=color, linewidth=2.4))
        relation = "target gap" if delta == parsed.gap else "wrong gap"
        ax.set(xticks=np.arange(config.SEQ_LEN), yticks=(), xlim=(-.5, config.SEQ_LEN - .5),
               title=f"{label}: y={int(torch.as_tensor(payload['y'])[index])}, "
                     f"observed delta={delta} ({relation})")
    axes[-1].set_xlabel("circular sequence position")
    fig.legend(handles=(Patch(facecolor="none", edgecolor="tab:green", linewidth=2, label=f"A={parsed.a}"),
                        Patch(facecolor="none", edgecolor="tab:orange", linewidth=2, label=f"B={parsed.b}")),
               loc="outside upper right", frameon=False)
    fig.suptitle(f"Real validation examples: held-out task {task}, target gap={parsed.gap}")
    return _save(fig, "02_data_examples", plots_dir, pdf)


def _candidate_rows(ckpt_root: Path, tasks: Iterable[str]) -> list[tuple[str, float, float, float]]:
    rows = []
    for task in tasks:
        path = ckpt_root / f"task_{task}" / "best10pct_summary.txt"
        if not path.is_file():
            continue
        match = SUMMARY_RE.search(path.read_text())
        if match:
            rows.append((task, *(float(match.group(key)) for key in ("minimum", "median", "maximum"))))
    return rows


def _plot_candidates(split_path: Path, ckpt_root: Path, plots_dir: Path, pdf: bool) -> list[str]:
    if not split_path.is_file():
        raise _stage_missing(f"missing split manifest: {split_path}")
    rows = _candidate_rows(ckpt_root, _read_json(split_path).get("train_tasks", []))
    if not rows:
        raise _stage_missing(f"no best10pct_summary.txt files below {ckpt_root}")
    rows.sort(key=lambda row: (_task_gap(row[0]), row[0]))
    x = np.arange(len(rows))
    minimum = np.array([row[1] for row in rows])
    median = np.array([row[2] for row in rows])
    maximum = np.array([row[3] for row in rows])
    fig, ax = plt.subplots(figsize=(11, 3.8), constrained_layout=True)
    ax.vlines(x, minimum, maximum, color="tab:blue", alpha=.55, lw=1)
    ax.scatter(x, median, s=14, color="tab:blue", label="selected p50")
    ax.scatter(x, minimum, s=10, color="tab:green", label="selected min")
    ax.set(xlabel="meta-train task (ordered by gap)", ylabel="validation BCE",
           title="Candidate-bank selection: top-10% BCE range")
    ax.legend(frameon=False, ncols=2)
    for boundary in np.cumsum([sum(_task_gap(row[0]) == gap for row in rows) for gap in config.GAPS])[:-1]:
        ax.axvline(boundary - .5, color="0.85", lw=.8)
    return _save(fig, "03_candidates", plots_dir, pdf)


def _plot_importance(split_path: Path, ckpt_root: Path, plots_dir: Path, pdf: bool,
                     max_tasks: int, importance_name: str, top_frac: float) -> list[str]:
    if not split_path.is_file():
        raise _stage_missing(f"missing split manifest: {split_path}")
    split = _read_json(split_path)
    tasks = _choose_by_gap([str(x) for x in split.get("train_tasks", [])], max_tasks)
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for task in tasks:
        path = ckpt_root / f"task_{task}" / importance_name
        if not path.is_file():
            continue
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        maps = next((payload[key] for key in ("importance", "maps", "masks")
                     if isinstance(payload.get(key), torch.Tensor)), None)
        if maps is None:
            continue
        maps = maps.float()
        if maps.ndim != 3 or tuple(maps.shape[1:]) != (config.SEQ_LEN, config.H):
            continue
        already_selected = "top_fraction" in payload
        losses = payload.get("val_loss")
        if already_selected:
            chosen = torch.arange(maps.shape[0])
        elif isinstance(losses, torch.Tensor) and losses.numel() == maps.shape[0]:
            n_top = max(1, int(maps.shape[0] * top_frac))
            chosen = losses.flatten().topk(n_top, largest=False).indices
        else:
            continue
        # Match the actual CVAE target transform.  This only permutes the
        # exchangeable hidden columns and preserves all continuous values.
        from models.cvae import canonicalize_hidden_columns
        selected = canonicalize_hidden_columns(maps[chosen])
        grouped[_task_gap(task)].append(selected.mean(dim=0).numpy())
    if not grouped:
        raise _stage_missing(f"no readable {importance_name} artifacts below {ckpt_root}")
    fig, axes = plt.subplots(2, 4, figsize=(10, 5), constrained_layout=True)
    image = None
    for ax, gap in zip(axes.flat, config.GAPS):
        if grouped[gap]:
            image = ax.imshow(np.mean(grouped[gap], axis=0), origin="lower", aspect="auto",
                              vmin=0, vmax=1, cmap="magma")
            ax.set_title(f"gap {gap} (n={len(grouped[gap])})")
        else:
            ax.text(.5, .5, "unavailable", ha="center", va="center")
            ax.set_title(f"gap {gap}")
        ax.set(xticks=(), yticks=())
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=.78, label="mean normalized |W₁|")
    fig.suptitle(f"Top-{top_frac:.0%} continuous importance maps (one mean per gap)")
    return _save(fig, "04_importance", plots_dir, pdf)


def _history_files(run_root: Path) -> list[Path]:
    aggregate_path = run_root / "selection.json"
    if aggregate_path.is_file():
        aggregate = _read_json(aggregate_path)
        selected_files: list[Path] = []
        for selection in aggregate.values() if isinstance(aggregate, dict) else ():
            for row in selection.get("candidates", []) if isinstance(selection, dict) else ():
                run_dir = row.get("run_dir") if isinstance(row, dict) else None
                if not isinstance(run_dir, str):
                    continue
                candidate = (Path(run_dir) / "history.json").resolve()
                try:
                    candidate.relative_to(run_root.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    selected_files.append(candidate)
        if selected_files:
            return sorted(set(selected_files))
    return sorted(path for path in run_root.rglob("history.json") if "plots" not in path.parts)


def _plot_histories(history_root: Path, plots_dir: Path, pdf: bool) -> list[str]:
    series = []
    for path in _history_files(history_root):
        payload = _read_json(path)
        if isinstance(payload, list) and payload and all(isinstance(row, dict) for row in payload):
            if all("epoch" in row and "val" in row for row in payload):
                series.append((str(path.parent.relative_to(history_root)), payload))
    if not series:
        raise _stage_missing(f"no generator history.json below {history_root}")
    # New histories store decomposed rate/distortion dictionaries.  Retain
    # compatibility with the original scalar train/val format for historical
    # runs, while never conflating KL with the total beta-weighted objective.
    nested = any(isinstance(row["val"], dict) for _, records in series for row in records)
    ncols = 3 if nested else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 3.8), constrained_layout=True)
    for name, records in series:
        epoch = [row["epoch"] for row in records]
        if isinstance(records[0]["val"], dict):
            axes[0].plot(epoch, [row["train"]["recon"] for row in records], alpha=.7, label=name)
            axes[1].plot(epoch, [row["val"]["recon"] for row in records], alpha=.7, label=name)
            axes[2].plot(epoch, [row["val"]["kl"] for row in records], alpha=.7, label=name)
        else:
            axes[0].plot(epoch, [row.get("train", np.nan) for row in records], label=name)
            axes[1].plot(epoch, [row["val"] for row in records], label=name)
    if nested:
        axes[0].set(xlabel="epoch", ylabel="BCE sum / sample", title="Train reconstruction")
        axes[1].set(xlabel="epoch", ylabel="BCE sum / sample", title="Internal-val reconstruction")
        axes[2].set(xlabel="epoch", ylabel="nats / sample", title="Internal-val KL")
        axes[2].legend(frameon=False, fontsize=6)
    else:
        axes[0].set(xlabel="epoch", ylabel="objective", title="Generator training")
        axes[1].set(xlabel="epoch", ylabel="objective", title="Generator validation")
        axes[1].legend(frameon=False, fontsize=8)
    return _save(fig, "05_training_histories", plots_dir, pdf)


def _beta_records(value: Any) -> list[dict[str, Any]]:
    """Extract train-validation beta records from the aggregate selector."""
    records = value.get("candidates", []) if isinstance(value, dict) else []
    found: list[dict[str, Any]] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        summary = row.get("summary", {})
        val = summary.get("selected_val") if isinstance(summary, dict) else None
        guard = summary.get("collapse_guard") if isinstance(summary, dict) else None
        if not isinstance(val, dict):
            tail_epochs = summary.get("tail_stability", {}).get("epochs", [])
            final = tail_epochs[-1] if tail_epochs else {}
            val = {"recon": final.get("val_recon"), "kl": final.get("val_kl"),
                   "active_mu_variance_dims": final.get("active_mu_variance_dims", 0)}
        if not isinstance(guard, dict):
            guard = {"passed": False, "observed": {}}
        if row.get("status") != "completed" or not isinstance(row.get("beta"), (float, int)):
            continue
        if not all(isinstance(val.get(key), (float, int)) for key in ("recon", "kl")):
            continue
        observed = guard.get("observed", {})
        found.append({
            "variant": str(row.get("variant", summary.get("variant", "unknown"))),
            "beta": float(row["beta"]),
            "recon": float(val["recon"]),
            "kl": float(val["kl"]),
            "active_dims": int(observed.get("active_mu_variance_dims",
                                             val.get("active_mu_variance_dims", 0))),
            "passed": bool(guard.get("passed", False)),
        })
    return found


def _plot_beta_sweep(gen_dir: Path, plots_dir: Path, pdf: bool) -> list[str]:
    records: list[dict[str, Any]] = []
    sweep = gen_dir / "selection.json"
    if sweep.is_file():
        aggregate = _read_json(sweep)
        for variant, selection in aggregate.items():
            for row in _beta_records(selection):
                row["variant"] = variant
                records.append(row)
    if not records:
        raise _stage_missing(f"no readable aggregate beta sweep at {sweep}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    for variant in sorted({row["variant"] for row in records}):
        rows = sorted((row for row in records if row["variant"] == variant),
                      key=lambda row: row["beta"])
        beta = [row["beta"] for row in rows]
        axes[0].plot(beta, [row["recon"] for row in rows], "o-", label=variant)
        axes[1].plot(beta, [row["kl"] for row in rows], "o-", label=variant)
        axes[2].plot(beta, [row["active_dims"] for row in rows], "o-", label=variant)
        for axis, metric in zip(axes, ("recon", "kl", "active_dims")):
            failed = [row for row in rows if not row["passed"]]
            axis.scatter([row["beta"] for row in failed], [row[metric] for row in failed],
                         marker="x", s=55, color="tab:red", zorder=5)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("beta")
    axes[0].set(ylabel="BCE sum / sample", title="Internal-val reconstruction")
    axes[1].set(ylabel="nats / sample", title="Internal-val KL")
    axes[2].set(ylabel="dimensions", title="Active posterior dimensions")
    axes[0].legend(frameon=False)
    axes[2].text(.02, .98, "red × = collapse guard failed", transform=axes[2].transAxes,
                 va="top", fontsize=8)
    return _save(fig, "06_beta_sweep", plots_dir, pdf)


def _checkpoint_samples(ckpt_root: Path, tasks: Iterable[str], artifact_name: str,
                        top_frac: float) -> tuple[torch.Tensor, list[str]]:
    samples, sample_tasks = [], []
    for task in tasks:
        path = ckpt_root / f"task_{task}" / artifact_name
        if not path.is_file():
            continue
        payload = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(payload, dict):
            continue
        maps = next((payload[key] for key in ("importance", "maps", "masks")
                     if isinstance(payload.get(key), torch.Tensor)), None)
        if maps is None or maps.ndim != 3 or tuple(maps.shape[1:]) != (config.SEQ_LEN, config.H):
            continue
        # Mirror generator training exactly: encode every selected target map,
        # rather than one arbitrary or one-best map per task.
        losses = payload.get("val_loss")
        if "top_fraction" in payload:
            selected = maps.float()
        elif isinstance(losses, torch.Tensor) and losses.numel() == maps.shape[0]:
            n_top = max(1, int(maps.shape[0] * top_frac))
            chosen = losses.flatten().topk(n_top, largest=False).indices
            selected = maps[chosen].float()
        else:
            continue
        samples.extend(selected.unbind(0))
        sample_tasks.extend([task] * len(selected))
    if not samples:
        raise _stage_missing(f"no readable {artifact_name} masks below {ckpt_root}")
    return torch.stack(samples), sample_tasks


def _circular_diagonal_profile(masks: torch.Tensor) -> torch.Tensor:
    """Mean value on every circular diagonal ``(input - hidden) mod n``."""
    values = torch.as_tensor(masks).float()
    if values.ndim < 2 or values.shape[-1] != values.shape[-2]:
        raise ValueError(f"expected [..., n, n] square matrices, got {tuple(values.shape)}")
    n = values.shape[-1]
    row = torch.arange(n, device=values.device)[:, None]
    column = torch.arange(n, device=values.device)[None, :]
    offsets = (row - column) % n
    return torch.stack([values[..., offsets == offset].mean(dim=-1) for offset in range(n)], dim=-1)


def _circular_toeplitz_score(masks: torch.Tensor) -> torch.Tensor:
    """Fraction of matrix variance explained by its circular-Toeplitz projection.

    A score of one means that entries are constant along every circular
    diagonal.  The score is zero when the projection explains no more than the
    global mean.  This is an R2-style diagnostic of the learned canonical
    hidden-column order, not an additional model-selection criterion.
    """
    values = torch.as_tensor(masks).float()
    profile = _circular_diagonal_profile(values)
    n = values.shape[-1]
    row = torch.arange(n, device=values.device)[:, None]
    column = torch.arange(n, device=values.device)[None, :]
    projection = profile[..., (row - column) % n]
    residual = (values - projection).square().sum(dim=(-2, -1))
    total = (values - values.mean(dim=(-2, -1), keepdim=True)).square().sum(dim=(-2, -1))
    score = torch.where(total > torch.finfo(values.dtype).eps, 1.0 - residual / total,
                        torch.ones_like(total))
    return score.clamp(0, 1)


def _align_to_gold(mask: torch.Tensor, gold: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    from evaluation.structural import best_permutation_iou
    diagnostic = best_permutation_iou(mask, gold)
    permutation = torch.as_tensor(diagnostic["permutation"], dtype=torch.long)
    return mask[:, torch.argsort(permutation)], diagnostic


def _load_generator(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise _stage_missing(f"missing generator checkpoint: {checkpoint_path}")
    from models.cvae import CVAE
    payload = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = CVAE(payload.get("mask_dim", config.MASK_DIM), payload.get("latent_dim", 32),
                 payload.get("hidden", 256), payload.get("cond_dim", config.COND_DIM))
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload


@torch.no_grad()
def _topk_from_latent(model, tasks: list[str], z: torch.Tensor) -> torch.Tensor:
    condition = model.condition(tasks, device=z.device)
    scores = torch.sigmoid(model.decode(z, condition))
    top = scores.topk(config.K_ACTIVE, dim=-1).indices
    masks = torch.zeros_like(scores)
    masks.scatter_(1, top, 1.0)
    return masks.reshape(-1, config.SEQ_LEN, config.H)


def _plot_generated_comparison(tasks: list[str], vae_masks: torch.Tensor,
                               cvae_masks: torch.Tensor, plots_dir: Path,
                               pdf: bool) -> list[str]:
    from evaluation.baselines import gold_mask
    fig, axes = plt.subplots(len(tasks), 5, figsize=(12, 18), constrained_layout=True)
    column_titles = ("VAE sample", "VAE aligned", "CVAE sample", "CVAE aligned", "ideal")
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=11)
    for row, (task, vae_mask, cvae_mask) in enumerate(zip(tasks, vae_masks, cvae_masks)):
        gold = gold_mask(task).reshape(config.SEQ_LEN, config.H).float()
        vae_aligned, vae_diag = _align_to_gold(vae_mask, gold)
        cvae_aligned, cvae_diag = _align_to_gold(cvae_mask, gold)
        matrices = (vae_mask, vae_aligned, cvae_mask, cvae_aligned, gold)
        annotations = (
            f"T={float(_circular_toeplitz_score(vae_mask)):.2f}",
            f"IoU={vae_diag['iou']:.2f}",
            f"T={float(_circular_toeplitz_score(cvae_mask)):.2f}",
            f"IoU={cvae_diag['iou']:.2f}",
            "T=1.00",
        )
        for column, (axis, matrix, annotation) in enumerate(zip(axes[row], matrices, annotations)):
            axis.imshow(matrix, origin="lower", cmap="Greys", vmin=0, vmax=1,
                        interpolation="nearest")
            axis.text(.03, .04, annotation, transform=axis.transAxes, fontsize=8,
                      bbox={"boxstyle": "round,pad=.2", "facecolor": "white", "alpha": .82,
                            "edgecolor": "none"})
            axis.set_xticks((0, 5, 10, 15) if row == len(tasks) - 1 else ())
            axis.set_yticks((0, 5, 10, 15) if column == 0 else ())
        axes[row, 0].set_ylabel(f"gap {_task_gap(task)}\ninput")
    fig.supxlabel("hidden unit")
    fig.suptitle("Prior samples, permutation alignment, and ideal support\n"
                 "T = circular-Toeplitz R2 in the learned canonical column order")
    return _save(fig, "07_generator_masks", plots_dir, pdf)


def _random_exact_masks(n: int, generator: torch.Generator) -> torch.Tensor:
    scores = torch.rand(n, config.MASK_DIM, generator=generator)
    top = scores.topk(config.K_ACTIVE, dim=-1).indices
    masks = torch.zeros_like(scores)
    masks.scatter_(1, top, 1)
    return masks.reshape(n, config.SEQ_LEN, config.H)


def _plot_toeplitzness(tasks: list[str], vae_masks: torch.Tensor, cvae_masks: torch.Tensor,
                       conditional_masks: torch.Tensor | None, plots_dir: Path,
                       pdf: bool) -> list[str]:
    """Quantify circular-Toeplitz consistency and expose learned diagonal bands."""
    from evaluation.baselines import gold_mask
    n_gaps, n_samples = len(tasks), vae_masks.shape[1]
    random = _random_exact_masks(n_gaps * n_samples, torch.Generator().manual_seed(20260825))
    random = random.reshape(n_gaps, n_samples, config.SEQ_LEN, config.H)
    ideal = torch.stack([gold_mask(task).reshape(config.SEQ_LEN, config.H).float() for task in tasks])
    x = np.asarray([_task_gap(task) for task in tasks])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    for name, masks, color in (("random exact-96", random, "0.45"),
                               ("VAE", vae_masks.cpu(), "tab:blue"),
                               ("CVAE", cvae_masks.cpu(), "tab:orange")):
        scores = _circular_toeplitz_score(masks).numpy()
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        axes[0].plot(x, mean, "o-", color=color, label=name)
        axes[0].fill_between(x, np.maximum(0, mean - std), np.minimum(1, mean + std),
                             color=color, alpha=.12)
    if conditional_masks is not None:
        axes[0].plot(x, _circular_toeplitz_score(conditional_masks.cpu()).numpy(), "s--",
                     color="tab:green", label="conditional mean")
    axes[0].plot(x, _circular_toeplitz_score(ideal).numpy(), "k:", lw=2, label="ideal")
    axes[0].set(xticks=x, ylim=(0, 1.03), xlabel="gap", ylabel="circular-Toeplitz R2",
                title="Consistency with circular diagonals")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].text(.02, .02, "band = mean +/- 1 SD over samples", transform=axes[0].transAxes,
                 fontsize=7, color="0.35")

    profiles = (_circular_diagonal_profile(vae_masks.cpu()).mean(dim=1),
                _circular_diagonal_profile(cvae_masks.cpu()).mean(dim=1))
    images = []
    for axis, profile, title in zip(axes[1:], profiles, ("VAE diagonal occupancy", "CVAE diagonal occupancy")):
        images.append(axis.imshow(profile, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1,
                                  extent=(-.5, config.SEQ_LEN - .5, x[0] - .5, x[-1] + .5)))
        for task in tasks:
            gold = gold_mask(task).reshape(config.SEQ_LEN, config.H).float()
            offsets = torch.nonzero(_circular_diagonal_profile(gold) > .5,
                                    as_tuple=False).flatten().numpy()
            axis.scatter(offsets, np.full_like(offsets, _task_gap(task)), s=13, facecolors="none",
                         edgecolors="cyan", linewidths=.8)
        axis.set(xticks=np.arange(config.SEQ_LEN), yticks=x, xlabel="circular offset (input - hidden) mod 16",
                 ylabel="gap", title=title)
        axis.tick_params(axis="x", labelsize=7)
    fig.colorbar(images[-1], ax=axes[1:], shrink=.82, label="edge occupancy")
    fig.suptitle("Circular-Toeplitz diagnostics (cyan circles mark ideal diagonal offsets)")
    return _save(fig, "07_generator_toeplitzness", plots_dir, pdf)


def _plot_generator_and_latent(split_path: Path, ckpt_root: Path, gen_dir: Path, plots_dir: Path,
                               pdf: bool, device_name: str, importance_name: str,
                               top_frac: float) -> list[str]:
    # Auto mode still refuses to hide a missing production GPU.  Explicit
    # ``--device cpu`` is supported for lightweight, read-only figure
    # regeneration from already trained checkpoints.
    if device_name not in {"cuda", "cpu"} or (device_name == "cuda" and not torch.cuda.is_available()):
        raise _stage_missing("generator/latent plots need CUDA, or an explicit --device cpu for plotting-only inference")
    try:
        from models.cvae import canonicalize_hidden_columns, task_condition
    except ImportError as error:  # pragma: no cover - only relevant to incomplete installs
        raise _stage_missing(f"cannot import CVAE plotting helpers: {error}") from error
    requested = torch.device(device_name)
    cvae, payload = _load_generator(gen_dir / "cvae" / "best.pt", requested)
    vae, _ = _load_generator(gen_dir / "vae" / "best.pt", requested)
    if vae.latent_dim != cvae.latent_dim:
        raise ValueError("VAE/CVAE latent dimensions differ; cannot make a paired sample plot")
    tasks = [f"A000_B001_G{gap:02d}" for gap in config.GAPS]
    generator = torch.Generator(device=requested).manual_seed(20260824)
    with torch.no_grad():
        z = torch.randn(len(tasks), cvae.latent_dim, device=requested, generator=generator)
        vae_display = _topk_from_latent(vae, tasks, z).cpu()
        cvae_display = _topk_from_latent(cvae, tasks, z).cpu()
    outputs = _plot_generated_comparison(tasks, vae_display, cvae_display, plots_dir, pdf)

    if not split_path.is_file():
        return outputs
    split = _read_json(split_path)
    artifact_name = str(payload.get("importance_name", importance_name))
    checkpoint_top_frac = float(payload.get("top_frac", top_frac))
    if artifact_name != importance_name or abs(checkpoint_top_frac - top_frac) > 1e-12:
        raise ValueError("plot target parameters do not match the promoted CVAE checkpoint")
    maps, source_tasks = _checkpoint_samples(
        ckpt_root, split.get("train_tasks", []), artifact_name, checkpoint_top_frac)
    with torch.no_grad():
        canonical_maps = canonicalize_hidden_columns(maps.to(requested))
        x = canonical_maps.flatten(1)
        c = task_condition(source_tasks, device=requested)
        mu, _ = cvae.encode(x, c)

        n_diagnostic = 64
        diagnostic_tasks = [task for task in tasks for _ in range(n_diagnostic)]
        z = torch.randn(len(diagnostic_tasks), cvae.latent_dim, device=requested, generator=generator)
        vae_diagnostic = _topk_from_latent(vae, diagnostic_tasks, z)
        cvae_diagnostic = _topk_from_latent(cvae, diagnostic_tasks, z)
        vae_diagnostic = vae_diagnostic.reshape(len(tasks), n_diagnostic, config.SEQ_LEN, config.H)
        cvae_diagnostic = cvae_diagnostic.reshape(len(tasks), n_diagnostic, config.SEQ_LEN, config.H)

        conditional = []
        source_gaps = torch.tensor([_task_gap(task) for task in source_tasks], device=requested)
        for gap in config.GAPS:
            mean = canonical_maps[source_gaps == gap].mean(dim=0).flatten()
            indices = mean.topk(config.K_ACTIVE).indices
            mask = torch.zeros_like(mean)
            mask[indices] = 1
            conditional.append(mask.reshape(config.SEQ_LEN, config.H))
        conditional_masks = torch.stack(conditional).cpu()
    outputs.extend(_plot_toeplitzness(tasks, vae_diagnostic, cvae_diagnostic,
                                     conditional_masks, plots_dir, pdf))
    mu = mu.cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    colors = [_task_gap(task) for task in source_tasks]
    scatter = axes[0].scatter(mu[:, 0], mu[:, 1], c=colors, cmap="viridis", s=28)
    fig.colorbar(scatter, ax=axes[0], label="gap")
    axes[0].set(xlabel="posterior μ₁", ylabel="posterior μ₂",
                title="All selected training-map latent means")
    axes[1].hist(np.linalg.norm(mu, axis=1), bins=min(16, max(4, len(mu) // 2)), color="tab:purple")
    axes[1].set(xlabel="||posterior μ||₂", ylabel="count", title="Latent magnitude")
    outputs.extend(_save(fig, "08_latent_posterior", plots_dir, pdf))
    return outputs


def _plot_final_eval(eval_dir: Path, plots_dir: Path, pdf: bool) -> list[str]:
    summary_path = eval_dir / "summary.json"
    if not summary_path.is_file():
        raise _stage_missing(f"missing final summary: {summary_path}")
    methods = _read_json(summary_path).get("methods", {})
    if not isinstance(methods, dict) or not methods:
        raise ValueError("summary.json has no methods")
    names = list(methods)
    acc = [methods[name].get("mean_acc", np.nan) for name in names]
    iou = [methods[name].get("mean_best_permutation_iou", np.nan) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for axis, values, title in zip(axes, (acc, iou), ("Held-out accuracy", "Best-permutation IoU")):
        axis.bar(np.arange(len(names)), values)
        axis.set(xticks=np.arange(len(names)), xticklabels=names, ylabel="mean", title=title)
        axis.tick_params(axis="x", rotation=40)
    outputs = _save(fig, "09_final_eval", plots_dir, pdf)

    results_path = eval_dir / "eval_results.json"
    if not results_path.is_file():
        return outputs
    tasks = _read_json(results_path).get("tasks", {})
    comparisons = [
        ("random_exact96", "CVAE − exact-96 random"),
        ("vae", "CVAE − VAE"),
        ("cvae_wrong_gap", "CVAE − wrong-gap CVAE"),
    ]
    available = [(rhs, label) for rhs, label in comparisons
                 if tasks and all("cvae" in row and rhs in row for row in tasks.values())]
    if not available:
        return outputs
    ordered = sorted(tasks, key=lambda task: (_task_gap(task), task))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True)
    x = np.arange(len(ordered))
    for rhs, label in available:
        acc_delta = [tasks[task]["cvae"]["mean_acc"] - tasks[task][rhs]["mean_acc"]
                     for task in ordered]
        iou_delta = [tasks[task]["cvae"]["mean_best_permutation_iou"]
                     - tasks[task][rhs]["mean_best_permutation_iou"] for task in ordered]
        axes[0].plot(x, acc_delta, "o-", ms=3.5, lw=1, label=label)
        axes[1].plot(x, iou_delta, "o-", ms=3.5, lw=1, label=label)
    labels = [f"g{_task_gap(task)}" for task in ordered]
    for axis, title, ylabel in zip(
            axes, ("Paired held-out accuracy deltas", "Paired held-out structural deltas"),
            ("accuracy delta", "best-permutation IoU delta")):
        axis.axhline(0, color="black", lw=1, ls="--")
        axis.set(xticks=x, xticklabels=labels, xlabel="held-out task (ordered by gap)",
                 ylabel=ylabel, title=title)
        axis.tick_params(axis="x", labelsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    outputs.extend(_save(fig, "10_paired_ood_deltas", plots_dir, pdf))
    return outputs


def build_plots(run_root: Path, *, plots_dir: Path | None = None, split_path: Path | None = None,
                data_dir: Path | None = None, ckpt_root: Path | None = None, gen_dir: Path | None = None,
                eval_dir: Path | None = None, pdf: bool = False, max_tasks: int = 16,
                device: str = "auto", include_generator: bool = True,
                importance_name: str = "importance.pt", top_frac: float = .1) -> dict[str, Any]:
    """Build every available plot and return the persisted provenance manifest."""
    run_root = run_root.resolve()
    plots_dir = (plots_dir or run_root / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    split_path = (split_path or run_root / "split.json").resolve()
    data_dir = (data_dir or _default_data_dir(run_root)).resolve()
    ckpt_root = (ckpt_root or run_root / "checkpoints").resolve()
    gen_dir = (gen_dir or run_root / "generative").resolve()
    eval_dir = (eval_dir or run_root / "eval").resolve()
    device_name = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "unavailable" if device == "auto" else device
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "plots_dir": str(plots_dir),
        "inputs": {"split": str(split_path), "data_dir": str(data_dir), "checkpoint_root": str(ckpt_root),
                   "generative_dir": str(gen_dir), "eval_dir": str(eval_dir)},
        "options": {"pdf": pdf, "max_tasks": max_tasks, "device": device_name,
                    "generator_enabled": include_generator,
                    "importance_name": importance_name, "top_frac": top_frac},
        "stages": {},
    }
    _add_stage(manifest, "split", lambda: _plot_split(split_path, plots_dir, pdf))
    _add_stage(manifest, "data", lambda: _plot_data(split_path, data_dir, plots_dir, pdf, max_tasks))
    _add_stage(manifest, "candidate_selection", lambda: _plot_candidates(split_path, ckpt_root, plots_dir, pdf))
    _add_stage(manifest, "continuous_importance", lambda: _plot_importance(
        split_path, ckpt_root, plots_dir, pdf, max_tasks, importance_name, top_frac))
    _add_stage(manifest, "training_histories", lambda: _plot_histories(gen_dir, plots_dir, pdf))
    _add_stage(manifest, "beta_sweep", lambda: _plot_beta_sweep(gen_dir, plots_dir, pdf))
    if include_generator:
        _add_stage(manifest, "generator_and_latent", lambda: _plot_generator_and_latent(
            split_path, ckpt_root, gen_dir, plots_dir, pdf, device_name,
            importance_name, top_frac))
    else:
        manifest["stages"]["generator_and_latent"] = {
            "status": "skipped", "reason": "disabled by --no-generator", "outputs": []
        }
    _add_stage(manifest, "final_evaluation", lambda: _plot_final_eval(eval_dir, plots_dir, pdf))
    manifest_path = plots_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, required=True, help="split-specific OOD artifact directory")
    result.add_argument("--plots-dir", type=Path, default=None)
    result.add_argument("--split", dest="split_path", type=Path, default=None)
    result.add_argument("--data-dir", type=Path, default=None)
    result.add_argument("--ckpt-root", type=Path, default=None)
    result.add_argument("--gen-dir", type=Path, default=None)
    result.add_argument("--eval-dir", type=Path, default=None)
    result.add_argument("--pdf", action="store_true", help="also save every generated figure as PDF")
    result.add_argument("--max-tasks", type=int, default=16, help="maximum artifacts sampled by data/importance plots")
    result.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    result.add_argument("--importance-name", default="importance.pt")
    result.add_argument("--top-frac", type=float, default=.1)
    result.add_argument("--no-generator", action="store_true", help="skip checkpoint sampling and latent plots")
    result.add_argument("--require-complete", action="store_true",
                        help="exit nonzero unless every stage was generated")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.max_tasks < 1 or not 0 < args.top_frac <= 1:
        raise ValueError("--max-tasks must be positive and --top-frac must lie in (0, 1]")
    manifest = build_plots(args.run_root, plots_dir=args.plots_dir, split_path=args.split_path,
                           data_dir=args.data_dir, ckpt_root=args.ckpt_root, gen_dir=args.gen_dir,
                           eval_dir=args.eval_dir, pdf=args.pdf, max_tasks=args.max_tasks,
                           device=args.device, include_generator=not args.no_generator,
                           importance_name=args.importance_name, top_frac=args.top_frac)
    statuses = {name: stage["status"] for name, stage in manifest["stages"].items()}
    print(f"plots -> {manifest['plots_dir']}")
    print(json.dumps(statuses, sort_keys=True))
    if args.require_complete and any(status != "generated" for status in statuses.values()):
        raise RuntimeError("required plot stage is not generated; inspect plots/manifest.json")


if __name__ == "__main__":
    main()

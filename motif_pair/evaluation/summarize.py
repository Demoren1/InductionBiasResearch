"""Summarize OOD motif-pair evaluation with paired task-level gains."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy import stats


def _mean(xs): return sum(xs) / len(xs)
def _std(xs):
    mu = _mean(xs)
    return math.sqrt(_mean([(x - mu) ** 2 for x in xs]))


def _paired(tasks: dict, lhs: str, rhs: str, metric: str) -> dict:
    deltas = [row[lhs][metric] - row[rhs][metric] for row in tasks.values()]
    mean = _mean(deltas)
    sample_std = stats.tstd(deltas)
    sem = stats.sem(deltas)
    low, high = stats.t.interval(.95, len(deltas) - 1, loc=mean, scale=sem)
    return {
        "comparison": f"{lhs} - {rhs}",
        "metric": metric,
        "n_tasks": len(deltas),
        "mean_delta": mean,
        "sample_std_delta": float(sample_std),
        "ci95_t": [float(low), float(high)],
        "paired_t_pvalue": float(stats.ttest_1samp(deltas, 0).pvalue),
        "positive_tasks": sum(delta > 0 for delta in deltas),
    }


def summarize(payload: dict) -> dict:
    tasks = payload["tasks"]
    method_names = sorted(set.intersection(*(set(x) for x in tasks.values())))
    out = {"provenance": payload["provenance"], "methods": {}}
    for name in method_names:
        rows = [tasks[task][name] for task in tasks]
        out["methods"][name] = {
            "mean_acc": _mean([r["mean_acc"] for r in rows]),
            "std_acc_across_tasks": _std([r["mean_acc"] for r in rows]),
            "mean_bce": _mean([r["mean_bce"] for r in rows]),
            "mean_best_permutation_iou": _mean([r["mean_best_permutation_iou"] for r in rows]),
            "per_task_acc": {task: tasks[task][name]["mean_acc"] for task in tasks},
        }
    primary = "random_exact96"
    if "cvae" in out["methods"] and primary in out["methods"]:
        gains = {task: tasks[task]["cvae"]["mean_acc"] - tasks[task][primary]["mean_acc"] for task in tasks}
        out["primary_transfer"] = {"comparison": f"cvae - {primary}",
                                   "mean_accuracy_gain": _mean(list(gains.values())),
                                   "std_gain_across_tasks": _std(list(gains.values())),
                                   "per_task_gain": gains}
        comparisons = []
        for rhs in (primary, "vae", "cvae_wrong_gap"):
            if rhs not in out["methods"]:
                continue
            comparisons.extend([
                _paired(tasks, "cvae", rhs, "mean_acc"),
                _paired(tasks, "cvae", rhs, "mean_best_permutation_iou"),
            ])
        out["paired_comparisons"] = comparisons
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(); summary = summarize(json.loads(a.results.read_text()))
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(summary, indent=2) + "\n")
    for method, row in summary["methods"].items():
        print(f"{method:20s} acc={row['mean_acc']:.4f} bce={row['mean_bce']:.4f} iou={row['mean_best_permutation_iou']:.3f}")
    if "primary_transfer" in summary: print(f"OOD gain: {summary['primary_transfer']['mean_accuracy_gain']:+.4f}")


if __name__ == "__main__": main()

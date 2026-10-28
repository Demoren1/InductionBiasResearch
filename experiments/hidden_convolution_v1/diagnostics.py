"""Post-freeze teacher comparisons; this module is never imported by solvers."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent


def align_to_teacher(indices, teacher):
    active, target = indices != 0, teacher != 0
    cost = np.not_equal(active[:, :, None], target[:, None, :]).sum(0)
    old, new = linear_sum_assignment(cost)
    order = np.empty(indices.shape[1], dtype=int)
    order[new] = old
    aligned = indices[:, order]
    codes = max(int(indices.max()), int(teacher.max())) + 1
    table = np.zeros((codes, codes), dtype=np.int64)
    np.add.at(table, (aligned.flatten(), teacher.flatten()), 1)
    old, new = linear_sum_assignment(-table[1:, 1:])
    mapping = np.zeros(codes, dtype=int)
    mapping[old + 1] = new + 1
    return mapping[aligned], order


def structure_metrics(indices, teacher):
    aligned, _ = align_to_teacher(indices, teacher)
    active, target = aligned != 0, teacher != 0
    correct = int((active & target).sum())
    precision = correct / max(int(active.sum()), 1)
    recall = correct / max(int(target.sum()), 1)
    codes = max(int(aligned.max()), int(teacher.max())) + 1
    table = np.zeros((codes, codes), dtype=np.int64)
    np.add.at(table, (aligned.flatten(), teacher.flatten()), 1)
    pairs = lambda counts: (counts * (counts - 1) // 2).sum()
    common = int(pairs(table[1:, 1:]))
    predicted_pairs = int(pairs(table[1:].sum(1)))
    true_pairs = int(pairs(table[:, 1:].sum(0)))
    pair_precision = common / max(predicted_pairs, 1)
    pair_recall = common / max(true_pairs, 1)
    return {
        "support_precision": precision,
        "support_recall": recall,
        "support_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "sharing_precision": pair_precision,
        "sharing_recall": pair_recall,
        "sharing_f1": 2
        * pair_precision
        * pair_recall
        / max(pair_precision + pair_recall, 1e-12),
        "correct_active_labels": int(((aligned == teacher) & target).sum()),
        "teacher_active": int(target.sum()),
        "predicted_active": int(active.sum()),
        "active_label_accuracy": float(
            ((aligned == teacher) & target).sum() / max(target.sum(), 1)
        ),
        "exact_support": bool(np.array_equal(active, target)),
        "exact_indexator": bool(np.array_equal(aligned, teacher)),
    }


def source_diagnostics(out):
    bank = np.load(out / "structure_bank.npz")
    teacher = np.load(out / "source_teacher.npz")
    selected = bank["selected"][::2]
    rows = [
        structure_metrics(
            bank["raw_indices"][index], teacher["indices"][bank["task"][index]]
        )
        for index in selected
    ]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
    summary["tasks"] = len(rows)
    if (out / "learned_support.npz").exists():
        mask = np.load(out / "learned_support.npz")["mask"]
        reference = teacher["indices"][0] != 0
        aligned, _ = align_to_teacher(mask.astype(int), reference.astype(int))
        intersection = int(((aligned != 0) & reference).sum())
        summary["common_support"] = {
            "edges": int(mask.sum()),
            "teacher_edges": int(reference.sum()),
            "precision": intersection / max(int(mask.sum()), 1),
            "recall": intersection / int(reference.sum()),
        }
    (out / "source_diagnostic.json").write_text(
        json.dumps({"summary": summary, "per_task": rows}, indent=2)
    )
    return summary


def novelty_diagnostics(out):
    source = np.load(out / "source_teacher.npz")["choices"]
    target = np.load(out / "target_teacher.npz")["choices"]
    bank = np.load(out / "structure_bank.npz")
    train = np.load(out / "prior_split.npz")["train"]
    source = source[np.unique(bank["task"][train])]
    permutations = np.array(list(itertools.permutations(range(4))))
    distances = []
    renamed = permutations[:, source]
    for choices in target:
        distances.append(int((renamed != choices).sum(-1).min()))
    return {
        "target_tasks": len(target),
        "windows": 36,
        "min_changed_windows_vs_any_training_structure": distances,
        "kernel_renaming_allowed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/pilot_v2")
    args = parser.parse_args()
    print(json.dumps(source_diagnostics(args.out), indent=2))


if __name__ == "__main__":
    main()

"""Freeze complete experiments, then score query answers in a separate process."""

import argparse
import csv
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)

METHODS = ("current", "lutq", "yeh_map", "yeh_soft", "msr", "random", "dense")
REAL_DATA = ("omniglot_pilot", "digit_sum_pilot")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expected_predictions(base):
    synthetic = base / "synthetic_confirmation"
    protocol = json.loads((synthetic / "protocol.json").read_text())
    paths = []
    for case in protocol["cases"]:
        for condition in protocol["conditions"]:
            for method in METHODS:
                if condition and method == "dense":
                    continue
                paths.append(
                    synthetic
                    / "targets"
                    / f"{case['name']}_c{condition}"
                    / method
                    / "frozen_predictions.npz"
                )
            if case["inputs"] == 11:
                paths.append(
                    synthetic
                    / "binary_targets"
                    / f"{case['name']}_c{condition}"
                    / "fitted"
                    / "frozen_predictions.npz"
                )
    for dataset in REAL_DATA:
        paths.extend(
            base / dataset / "targets" / method / "frozen_predictions.npz"
            for method in (*METHODS, "msr_native")
        )
        if dataset == "digit_sum_pilot":
            paths.extend(
                base / dataset / "targets" / method / "frozen_predictions.npz"
                for method in ("current_functional", "mask_only", "mask_random")
            )
    return paths


def freeze(base):
    output = base / "report"
    output.mkdir(exist_ok=True)
    path = output / "frozen_results.json"
    if path.exists():
        raise RuntimeError("Results already frozen; do not overwrite the registration")
    state = json.loads((base / "real_fleet.finished.json").read_text())
    if state["completed"] != state["total"] or state["failures"]:
        raise RuntimeError("The unified experiment plan has unfinished/failed jobs")
    timing = json.loads((base / "timing_fleet.finished.json").read_text())
    if timing["completed"] != timing["total"] or timing["failures"]:
        raise RuntimeError("Dedicated unprofiled timing is incomplete")
    functional = json.loads((base / "functional_fleet.finished.json").read_text())
    if functional["completed"] != functional["total"] or functional["failures"]:
        raise RuntimeError(
            "The registered conditional functional ablation is incomplete"
        )
    masks = json.loads((base / "mask_fleet.finished.json").read_text())
    if masks["completed"] != masks["total"] or masks["failures"]:
        raise RuntimeError("The registered connectivity controls are incomplete")
    files = set(expected_predictions(base))
    missing = [str(item) for item in files if not item.exists()]
    if missing:
        raise RuntimeError(f"Missing predictions: {missing}")
    roots = [
        base / "synthetic_confirmation",
        *(base / name for name in REAL_DATA),
        base / "synthetic_pilot",
        base / "dense_cost_measurement",
    ]
    receipts = []
    for root in roots:
        for parent in (
            "receipts",
            "source",
            "competitors",
            "fits",
            "targets",
            "binary_targets",
            "validation_fit",
            "diagnosis",
        ):
            for receipt in (root / parent).rglob("*receipt*.json"):
                record = json.loads(receipt.read_text())
                if record.get("violations"):
                    raise RuntimeError(f"Access violation in {receipt}")
                receipts.append(
                    {
                        "path": str(receipt),
                        "sha256": digest(receipt),
                        "violations": len(record.get("violations", [])),
                    }
                )
        for parent in ("fits", "competitors", "validation_fit"):
            files.update((root / parent).rglob("*selected*.json"))
            files.update((root / parent).rglob("*selected*.npz"))
        files.update((root / "fits").rglob("joint_bank.npz"))
        files.update((root / "fits").rglob("joint_fitted.npz"))
        files.update((root / "dense").rglob("fitted.npz"))
        files.update((root / "dense").rglob("frozen.json"))
        files.update((root / "source").glob("*_t0/learned.npz"))
        if (root / "source" / "dense" / "models.npz").exists():
            files.add(root / "source" / "dense" / "models.npz")
        for parent in ("targets", "binary_targets"):
            files.update((root / parent).rglob("structures.npz"))
            files.update((root / parent).rglob("model.npz"))
            files.update((root / parent).rglob("selected.json"))
            files.update((root / parent).rglob("sizes.json"))
            files.update((root / parent).rglob("ready.json"))
    for prediction in expected_predictions(base):
        metrics = json.loads(prediction.with_name("metrics.json").read_text())
        if metrics["status"] != "complete" or metrics["predictions_sha256"] != digest(
            prediction
        ):
            raise RuntimeError(
                f"Unfinished or changed prediction artifact: {prediction}"
            )
        with np.load(prediction) as archive:
            if any(not np.isfinite(archive[key]).all() for key in archive.files):
                raise FloatingPointError(prediction)
    root = base / "synthetic_confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    for case in protocol["cases"]:
        for panel in ("support.npz", "selection.npz", "query_x.npz"):
            a, b = [
                np.load(root / "targets" / f"{case['name']}_c{c}" / panel)
                for c in (0, 1)
            ]
            if a.files != b.files or any(
                not np.array_equal(a[key], b[key]) for key in a.files
            ):
                raise RuntimeError(
                    "Dense target reuse requires identical target inputs/support observations"
                )
    files.add(base / "real_fleet.json")
    files.add(base / "timing_fleet.json")
    for plan_name in (
        "real_fleet.json",
        "timing_fleet.json",
        "functional_fleet.json",
        "mask_fleet.json",
    ):
        plan = json.loads((base / plan_name).read_text())
        for job in plan["jobs"]:
            files.add(Path(job["metric"]))
            files.add(Path(job["registration"]))
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "evaluation_code_sha256": digest(Path(__file__)),
            "files_sha256": {
                str(item.relative_to(base)): digest(item) for item in sorted(files)
            },
            "access_receipts": receipts,
            "receipt_count": len(receipts),
            "access_violations": 0,
            "target_query_answers_opened": False,
            "structural_truth_opened_for_selection": False,
            "dense_reuse": "Identical target support/selection/query-input arrays verified across source noise conditions",
            "scientific_choices": "All predictors and predictions fixed before separate query/structural scoring",
        },
    )
    print(
        json.dumps(
            {"frozen_files": len(files), "receipts": len(receipts), "violations": 0}
        ),
        flush=True,
    )


def align_index(index, truth, codes):
    best, aligned = index.size + 1, None
    for permutation in itertools.permutations(range(1, codes + 1)):
        for signs in itertools.product((-1, 1), repeat=codes):
            lookup = np.r_[0, np.asarray(permutation) * signs]
            renamed = np.sign(index) * lookup[np.abs(index)]
            cost = (truth[:, :, None] != renamed[:, None]).sum(0)
            left, right = linear_sum_assignment(cost)
            errors = int(cost[left, right].sum())
            if errors < best:
                best = errors
                aligned = renamed[:, right[np.argsort(left)]].copy()
    mask_cost = ((truth != 0)[:, :, None] != (index != 0)[:, None]).sum(0)
    left, right = linear_sum_assignment(mask_cost)
    mask = index[:, right[np.argsort(left)]] != 0
    actual = truth != 0
    tp = int((mask & actual).sum())
    return aligned, {
        "signed_entry_errors": best,
        "exact": best == 0,
        "mask_exact": bool(np.array_equal(mask, actual)),
        "mask_precision": tp / max(int(mask.sum()), 1),
        "mask_recall": tp / max(int(actual.sum()), 1),
        "actual_groups": int(len(np.unique(np.abs(index[index != 0])))),
        "active_connections": int((index != 0).sum()),
    }


def adjusted_rand(a, b):
    if len(a) < 2:
        return None
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    def choose(x):
        return np.sum(x * (x - 1) / 2)

    observed = choose(table)
    row, col, pairs = (
        choose(table.sum(0)),
        choose(table.sum(1)),
        len(a) * (len(a) - 1) / 2,
    )
    expected = row * col / pairs
    denominator = (row + col) / 2 - expected
    return (
        float((observed - expected) / denominator) if abs(denominator) > 1e-12 else 1.0
    )


def score(base, *, published=False):
    output = base / "report"
    published_manifest = base / "published_manifest.json"
    if published_manifest.exists() and not published:
        raise RuntimeError(
            "Use published-score for the compact exported artifact bundle"
        )
    manifest_path = published_manifest if published else output / "frozen_results.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(base / name) != expected:
            raise RuntimeError(f"Frozen result changed before scoring: {name}")
    transfer, recovery, binary = [], [], []
    aligned_exports = {}
    root = base / "synthetic_confirmation"
    protocol = json.loads((root / "protocol.json").read_text())
    for case in protocol["cases"]:
        name, codes = case["name"], case["codes"]
        dataset = "Pattern 0110" if case["inputs"] == 11 else "Synthetic DeepSets"
        targets = np.load(root / "private" / f"{name}_targets.npz")
        answer = np.stack([targets[f"query_y_{task}"] for task in range(8)])
        truth = np.load(root / "private" / f"{name}.npz")["index"]
        canonical = canonical_index(truth, codes)
        for condition in protocol["conditions"]:
            folder = root / "targets" / f"{name}_c{condition}"
            structures = np.load(folder / "structures.npz")
            for method in METHODS:
                model_folder = (
                    root
                    / "targets"
                    / f"{name}_c{0 if method == 'dense' else condition}"
                    / method
                )
                predictions = np.load(model_folder / "frozen_predictions.npz")
                for budget in (64, 256, 1024):
                    for variant, field in (
                        (method, f"prediction_{budget}"),
                        *(
                            (("dense_adam", f"adam_prediction_{budget}"),)
                            if method == "dense"
                            else ()
                        ),
                    ):
                        value = predictions[field].astype(np.float64)
                        error = ((value - answer) ** 2).mean((1, 2))
                        variance = np.var(answer, axis=(1, 2)).clip(1e-12)
                        for task, (mse, nmse) in enumerate(
                            zip(error, error / variance)
                        ):
                            transfer.append(
                                {
                                    "dataset": dataset,
                                    "case": name,
                                    "condition": condition,
                                    "method": variant,
                                    "support_fit": budget,
                                    "support_selection": budget // 4,
                                    "task": task,
                                    "mse": float(mse),
                                    "nmse": float(nmse),
                                    "codes": codes if method != "dense" else 0,
                                    "transfer_mode": "fresh_numeric",
                                }
                            )
                if method in ("current", "lutq", "yeh_map", "random"):
                    aligned, metrics = align_index(structures[method], truth, codes)
                    common = (truth != 0) & (aligned != 0)
                    metrics["sharing_ARI_on_common_active_positions"] = adjusted_rand(
                        np.abs(truth[common]), np.abs(aligned[common])
                    )
                    recovery.append(
                        {
                            "dataset": dataset,
                            "case": name,
                            "condition": condition,
                            "method": method,
                            **metrics,
                        }
                    )
                    aligned_exports[f"{name}_c{condition}_{method}"] = aligned
            bank = np.load(root / "fits" / f"{name}_c{condition}" / "joint_fitted.npz")
            indices = bank["indices"]
            coverage = any(np.array_equal(item, canonical) for item in indices)
            recovery.append(
                {
                    "dataset": dataset,
                    "case": name,
                    "condition": condition,
                    "method": "current_bank",
                    "exact": coverage,
                    "bank_size": len(indices),
                }
            )
            aligned_exports[f"{name}_truth"] = truth
            if dataset == "Pattern 0110":
                order = targets["binary_partition"]
                answer_binary = targets["binary_y"][order[640:]]
                predictions = np.load(
                    root
                    / "binary_targets"
                    / f"{name}_c{condition}"
                    / "fitted"
                    / "frozen_predictions.npz"
                )
                for method in METHODS:
                    logits = predictions[method].reshape(-1)
                    chosen = logits > 0
                    binary.append(
                        {
                            "dataset": dataset,
                            "case": name,
                            "condition": condition,
                            "method": method,
                            "support_fit": 512,
                            "support_selection": 128,
                            "accuracy": float(np.mean(chosen == answer_binary)),
                            "positive_recall": float(chosen[answer_binary == 1].mean()),
                            "negative_recall": float(
                                (~chosen[answer_binary == 0]).mean()
                            ),
                            "binary_CE": float(
                                np.mean(
                                    np.logaddexp(0, logits) - answer_binary * logits
                                )
                            ),
                        }
                    )
    for dataset in REAL_DATA:
        root = base / dataset
        kind = json.loads((root / "protocol.json").read_text())["loss"]
        answer = np.load(root / "private" / "target.npz")["query_y"]
        for method in (
            *METHODS,
            "msr_native",
            *(
                ("current_functional", "mask_only", "mask_random")
                if dataset == "digit_sum_pilot"
                else ()
            ),
        ):
            folder = root / "targets" / method
            value = np.load(folder / "frozen_predictions.npz")["prediction"].astype(
                np.float64
            )
            if method == "msr_native":
                chosen = json.loads((folder / "prepare_selected.json").read_text())
                codes, updates = chosen["selected"]["codes"], 3
            else:
                chosen = json.loads((folder / "selected.json").read_text())
                codes, updates = (
                    chosen["source_selection"]["codes"],
                    chosen["numeric_updates"],
                )
            if kind == "cross_entropy":
                log_partition = np.logaddexp.reduce(value, axis=-1)
                ce = (log_partition - (value * answer).sum(-1)).mean(1)
                accuracy = (value.argmax(-1) == answer.argmax(-1)).mean(1)
                mse = [None] * len(value)
            else:
                mse = ((value - answer) ** 2).mean((1, 2))
                ce, accuracy = [None] * len(value), [None] * len(value)
            for task in range(len(value)):
                transfer.append(
                    {
                        "dataset": "Aug-Omniglot"
                        if kind == "cross_entropy"
                        else "Paired MNIST sums",
                        "case": "one_source_cohort",
                        "condition": 0,
                        "method": method,
                        "support_fit": 25 if kind == "cross_entropy" else 512,
                        "support_selection": 0,
                        "task": task,
                        "mse": None if mse[task] is None else float(mse[task]),
                        "accuracy": None
                        if accuracy[task] is None
                        else float(accuracy[task]),
                        "cross_entropy": None if ce[task] is None else float(ce[task]),
                        "codes": codes,
                        "numeric_updates": updates,
                        "transfer_mode": "learned_numeric_initialization"
                        if method == "msr_native"
                        else "fresh_numeric",
                    }
                )
    write_csv(output / "transfer_per_task.csv", transfer)
    write_csv(output / "structural_recovery.csv", recovery)
    write_csv(output / "binary_classification.csv", binary)
    np.savez_compressed(output / "aligned_indices_for_display.npz", **aligned_exports)
    save_json(
        output / "scoring.json",
        {
            "scored_utc": datetime.now(UTC).isoformat(),
            "evaluation_code_sha256": digest(Path(__file__)),
            "frozen_manifest_sha256": digest(output / "frozen_results.json"),
            "published_bundle_manifest_sha256": digest(published_manifest)
            if published
            else None,
            "transfer_rows": len(transfer),
            "structural_rows": len(recovery),
            "binary_rows": len(binary),
            "teacher_use": "Only after prediction freeze, for query metrics and descriptive structural audit; never candidate/hyperparameter selection",
            "alignment": "Exact signed entry match optimizes code-pair permutation/sign and hidden-unit matching; mask metrics use a separate mask-optimal permutation; sharing ARI uses common active positions under the full signed alignment",
            "bank_coverage": "Descriptive membership of canonical generating I in an already frozen bank; no oracle model selected",
        },
    )
    print(
        json.dumps(
            {
                "scored_transfer_rows": len(transfer),
                "recovery_rows": len(recovery),
                "binary_rows": len(binary),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("freeze", "score", "published-score"), required=True
    )
    args = parser.parse_args()
    if args.phase == "freeze":
        freeze(args.base.resolve())
    else:
        score(args.base.resolve(), published=args.phase == "published-score")


if __name__ == "__main__":
    main()

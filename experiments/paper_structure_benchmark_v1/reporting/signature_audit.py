"""Recompute signature-transfer metrics from compact published predictions."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from experiments.sharing_structure_research_v1.blind_contract import digest


def audit_general(base):
    """Audit all per-task errors without the removed training workspace."""
    plan = json.loads((base / "plan.json").read_text())
    with (base / "transfer_per_task.csv").open() as stream:
        expected = {
            (
                r["case"],
                int(r["condition"]),
                r["method"],
                int(r["budget"]),
                int(r["task"]),
            ): float(r["mse"])
            for r in csv.DictReader(stream)
        }
    checked = 0
    with np.load(base / "scoring_answers.npz") as answers:
        for assignment in plan["assignments"]:
            case, condition, method = [
                assignment[k] for k in ("case", "condition", "method")
            ]
            with np.load(
                base / "predictions" / f"{assignment['job']}.npz"
            ) as predictions:
                for budget in (64, 256, 1024):
                    fields = [(method, f"prediction_{budget}")]
                    if method == "dense":
                        fields.append(("dense_adam", f"adam_prediction_{budget}"))
                    for label, field in fields:
                        losses = (
                            (predictions[field] - answers[f"{case}_query_y"]) ** 2
                        ).mean((1, 2))
                        for task, loss in enumerate(losses):
                            key = case, condition, label, budget, task
                            if not np.isclose(
                                loss, expected[key], rtol=1e-12, atol=1e-15
                            ):
                                raise RuntimeError(f"Query MSE changed: {key}")
                            checked += 1
    if checked != len(expected):
        raise RuntimeError("The compact export has incomplete task coverage")
    return {"dataset": "general", "per_task_metrics_reproduced": checked}


def audit_omniglot(base):
    """Audit accuracy and cross-entropy from every saved query logit."""
    checked = 0
    with np.load(base / "scoring_answers.npz") as targets:
        answers = targets["query_y"]
    with np.load(base / "per_episode.npz") as expected:
        for method in (
            "dense",
            "current",
            "signature",
            "signature_random",
            "signature_k8",
        ):
            with np.load(base / "models" / method / "predictions.npz") as stored:
                prediction = stored["prediction"]
            accuracy = (prediction.argmax(-1) == answers.argmax(-1)).mean(1)
            ce = (logsumexp(prediction, axis=-1) - (prediction * answers).sum(-1)).mean(
                1
            )
            if method == "signature_k8":
                with np.load(base / "k8_per_episode.npz") as reference:
                    target_accuracy, target_ce = (
                        reference["accuracy"],
                        reference["cross_entropy"],
                    )
            else:
                target_accuracy, target_ce = (
                    expected[f"{method}_accuracy"],
                    expected[f"{method}_ce"],
                )
            if not np.array_equal(accuracy, target_accuracy) or not np.array_equal(
                ce, target_ce
            ):
                raise RuntimeError(f"Episode metrics changed: {method}")
            checked += len(accuracy)
    return {"dataset": "omniglot", "episode_pairs_reproduced": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--dataset", choices=("general", "omniglot"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    manifest = json.loads((base / "compact_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        path = (base / name).resolve()
        if not path.is_relative_to(base) or digest(path) != expected:
            raise RuntimeError(f"Compact publication file changed: {name}")
    result = audit_general(base) if args.dataset == "general" else audit_omniglot(base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

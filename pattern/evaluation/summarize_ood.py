"""Summarize held-out pattern evaluation and compute transfer gains."""

import argparse
import json
import math
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def summarize(results: dict, test_patterns: list[str]) -> dict:
    methods = sorted({key.split(":", 1)[1] for key in results})
    summary = {"test_patterns": test_patterns, "methods": {}}
    for method in methods:
        entries = [results[f"{pattern}:{method}"] for pattern in test_patterns
                   if f"{pattern}:{method}" in results]
        if len(entries) != len(test_patterns):
            raise ValueError(
                f"method {method!r} has {len(entries)}/{len(test_patterns)} "
                "held-out pattern results")
        accuracies = [entry["mean_acc"] for entry in entries]
        bces = [entry["mean_bce"] for entry in entries]
        summary["methods"][method] = {
            "mean_acc": mean(accuracies),
            "std_acc_across_tasks": population_std(accuracies),
            "mean_bce": mean(bces),
            "per_pattern_acc": dict(zip(test_patterns, accuracies)),
        }

    cvae = summary["methods"].get("cvae")
    fixed_random = summary["methods"].get("random_exact32")
    if cvae is not None and fixed_random is not None:
        paired = [cvae["per_pattern_acc"][pattern]
                  - fixed_random["per_pattern_acc"][pattern]
                  for pattern in test_patterns]
        summary["primary_transfer"] = {
            "comparison": "cvae - random_exact32",
            "mean_accuracy_gain": mean(paired),
            "std_gain_across_tasks": population_std(paired),
            "per_pattern_gain": dict(zip(test_patterns, paired)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    split = json.loads(args.split.read_text())
    summary = summarize(results, split["test_patterns"])
    summary["split_seed"] = split["split_seed"]
    summary["train_patterns"] = split["train_patterns"]
    args.out.write_text(json.dumps(summary, indent=2) + "\n")

    for method, values in summary["methods"].items():
        print(f"{method:16s} acc={values['mean_acc']:.4f} "
              f"bce={values['mean_bce']:.4f}")
    if "primary_transfer" in summary:
        gain = summary["primary_transfer"]["mean_accuracy_gain"]
        print(f"OOD transfer gain (CVAE - random_exact32): {gain:+.4f}")
    print(f"summary -> {args.out}")


if __name__ == "__main__":
    main()

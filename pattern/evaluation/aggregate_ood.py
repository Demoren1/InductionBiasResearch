"""Aggregate multiple fixed 12/4 OOD split summaries at the split level."""

import argparse
import json
import math
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float:
    center = mean(values)
    return math.sqrt(mean([(value - center) ** 2 for value in values]))


def aggregate(summaries: list[dict]) -> dict:
    if not summaries:
        raise ValueError("at least one summary is required")
    seeds = [summary["split_seed"] for summary in summaries]
    if len(set(seeds)) != len(seeds):
        raise ValueError("split seeds must be unique")

    method_names = set(summaries[0]["methods"])
    for summary in summaries[1:]:
        if set(summary["methods"]) != method_names:
            raise ValueError("all summaries must contain the same methods")

    methods = {}
    for method in sorted(method_names):
        accuracies = [summary["methods"][method]["mean_acc"]
                      for summary in summaries]
        bces = [summary["methods"][method]["mean_bce"]
                for summary in summaries]
        methods[method] = {
            "mean_acc_across_splits": mean(accuracies),
            "std_acc_across_splits": population_std(accuracies),
            "mean_bce_across_splits": mean(bces),
            "per_split_acc": dict(zip(map(str, seeds), accuracies)),
        }

    gains = [summary["primary_transfer"]["mean_accuracy_gain"]
             for summary in summaries]
    return {
        "split_seeds": seeds,
        "n_splits": len(summaries),
        "n_held_out_task_evaluations": sum(
            len(summary["test_patterns"]) for summary in summaries),
        "methods": methods,
        "primary_transfer": {
            "comparison": "cvae - random_exact32",
            "mean_accuracy_gain_across_splits": mean(gains),
            "std_gain_across_splits": population_std(gains),
            "min_split_gain": min(gains),
            "max_split_gain": max(gains),
            "all_splits_positive": all(gain > 0 for gain in gains),
            "per_split_gain": dict(zip(map(str, seeds), gains)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summaries = [json.loads(path.read_text()) for path in args.summaries]
    result = aggregate(summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")

    for method, values in result["methods"].items():
        print(f"{method:16s} "
              f"acc={values['mean_acc_across_splits']:.4f} "
              f"split_std={values['std_acc_across_splits']:.4f} "
              f"bce={values['mean_bce_across_splits']:.4f}")
    transfer = result["primary_transfer"]
    print("OOD transfer gain: "
          f"{transfer['mean_accuracy_gain_across_splits']:+.4f} "
          f"± {transfer['std_gain_across_splits']:.4f} across splits")
    print(f"all splits positive: {transfer['all_splits_positive']}")
    print(f"aggregate -> {args.out}")


if __name__ == "__main__":
    main()

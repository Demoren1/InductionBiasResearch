"""Recompute manuscript metrics from the published answers and predictions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


COMPACT = "compact_experiments_2026-11-18"
WIDE = "omniglot64_2026-11-18"
BUNDLES = (
    (f"{COMPACT}/synthetic", ("results.json",)),
    (f"{COMPACT}/omniglot10", ("results.json",)),
    (f"{COMPACT}/har3", ("results.json",)),
    (WIDE, ("results_with_lutq.json", "clustering_results.json")),
)


def average_by_group(values, group):
    if group is None:
        return values
    return np.stack([values[group == g].mean(0) for g in np.unique(group)])


def halfwidth(values):
    count = len(values)
    return t.ppf(0.975, count - 1) * values.std(ddof=1) / np.sqrt(count)


def assert_metric(actual, expected, context):
    if not np.isclose(actual, expected, atol=1e-9, rtol=1e-7):
        raise ValueError(
            f"{context}: recomputed {actual}, expected {expected}"
        )


def audit(directory, summaries):
    path = directory / "scoring.npz"
    lineage = json.loads((directory / "scoring_lineage.json").read_text())
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != lineage["archive_sha256"]:
        raise ValueError(f"Changed scoring archive: {path}")

    with np.load(path, allow_pickle=False) as data:
        truth = data["truth"]
        group = data.get("subject_group")
        synthetic = directory.name == "synthetic"
        values = {}
        for method in data.files:
            if method in {"truth", "subject_group"}:
                continue
            prediction = data[method]
            if (
                prediction.shape != truth.shape
                or not np.isfinite(prediction).all()
            ):
                raise ValueError(f"Invalid {method} predictions: {path}")
            if synthetic:
                values[method] = np.square(prediction - truth).mean(
                    axis=(1, 2)
                )
            else:
                p = prediction.astype(np.float64)
                accuracy = (p.argmax(-1) == truth.argmax(-1)).mean(-1)
                loss = -(truth * np.log(np.maximum(p, 1e-30))).sum(-1).mean(-1)
                values[method] = average_by_group(
                    np.column_stack((accuracy, loss)),
                    group,
                )
    for filename in summaries:
        result = json.loads((directory / filename).read_text())
        # G34 is retained in some provenance summaries from the old comparison.
        for name, row in result["methods"].items():
            if name == "g34":
                continue
            context = f"{directory.name}/{filename}/{name}"
            if synthetic:
                assert_metric(values[name].mean(), row["mse"], context)
                for i, (value, stored) in enumerate(
                    zip(values[name], row["per_function_mse"], strict=True)
                ):
                    assert_metric(value, stored, f"{context}/function {i}")
                continue
            accuracy, ce = values[name].T
            delta = accuracy - values["asrr"][:, 0]
            assert_metric(accuracy.mean(), row["accuracy"], context)
            assert_metric(ce.mean(), row["ce"], f"{context}/CE")
            assert_metric(
                halfwidth(accuracy),
                row["accuracy_ci95_halfwidth"],
                f"{context}/accuracy CI",
            )
            assert_metric(
                halfwidth(delta),
                row["difference_ci95_halfwidth"],
                f"{context}/paired CI",
            )
        for pair in result.get("paired_differences", []):
            method, reference = pair["method"], pair["reference"]
            delta = values[method][:, 0] - values[reference][:, 0]
            assert_metric(delta.mean(), pair["accuracy_difference"], filename)
            assert_metric(halfwidth(delta), pair["ci95_halfwidth"], filename)
    print(f"{directory}: {len(truth)} cases, {len(values)} methods")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    for bundle, filenames in BUNDLES:
        audit(args.artifacts / bundle, filenames)

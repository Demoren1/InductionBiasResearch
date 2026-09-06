"""Build the final report for the frozen-U and inner-horizon diagnostics.

The builder is intentionally read-only with respect to experimental results.
It refuses to write either output until the frozen-U result and all four
predeclared horizon readouts are present and mutually consistent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import write_json


DEFAULT_ROOT = Path("meta_pattern/outputs/u_diagnosis_20260907")
ARMS = ("before", "h50", "h200", "h500")
BUDGETS = (50, 500, 2000)
DEFAULT_STRATEGY = "default_adam01_mean_restarts"
FIXED_STRATEGIES = (
    ("default_adam01_mean_restarts", "Default Adam .1, mean of 4 starts", 4),
    ("support_selected_restart_adam01", "Support-selected Adam .1 restart", 4),
    ("support_selected_grid", "Support-selected optimizer/restart grid", 20),
)


def _load(path: Path) -> dict[str, Any]:
    with path.open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _required(root: Path) -> dict[str, Path]:
    """Check all readouts first, so a partial run cannot create an output."""
    paths = {arm: root / "readout" / f"{arm}.json" for arm in ARMS}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required horizon readout(s): " + ", ".join(missing))
    fixed = root / "fixed_u" / "SUMMARY.json"
    fixed_shard = root / "fixed_u" / "shard3.json"
    for path in (fixed, fixed_shard):
        if not path.is_file():
            raise FileNotFoundError(f"missing required frozen-U artifact: {path}")
    return {"fixed": fixed, "fixed_shard": fixed_shard, **paths}


def _equal_row(rows: list[dict], *, basis: str, strategy: str, budget: int) -> dict:
    matching = [row for row in rows if row.get("basis") == basis and row.get("strategy") == strategy
                and row.get("budget") == budget and row.get("length") == "equal_length"]
    if len(matching) != 1:
        raise ValueError(f"expected one equal-length row for {basis}/{strategy}/{budget}, found {len(matching)}")
    row = matching[0]
    for name in ("query_accuracy", "query_bce", "support_accuracy", "support_bce"):
        if row.get(name) is None:
            raise ValueError(f"non-finite aggregate {name} for {basis}/{strategy}/{budget}")
    return row


def _verify_source_against_fixed(fixed_source: dict, current_source: dict, arm: str) -> dict:
    """All files in the frozen-U snapshot must exist unchanged in readout."""
    missing = sorted(set(fixed_source) - set(current_source))
    changed = sorted(name for name in fixed_source if name in current_source
                     and fixed_source[name] != current_source[name])
    if missing or changed:
        raise ValueError(f"{arm} source provenance differs from frozen-U; missing={missing}, changed={changed}")
    # The readout module was created after the frozen-U run, hence it is an
    # allowed extra file.  The shared experiment sources above are exact.
    return {"shared_files_verified": len(fixed_source),
            "readout_only_files": sorted(set(current_source) - set(fixed_source))}


def _validate_arm(
    arm: str,
    data: dict,
    fixed: dict,
    fixed_shard: dict,
    root: Path,
) -> dict:
    if data.get("data_hashes") != fixed_shard.get("data_hashes"):
        raise ValueError(f"{arm} data hashes differ from fixed-U shard 3")
    source_info = _verify_source_against_fixed(fixed_shard["source_sha256"], data["source_sha256"], arm)
    if data.get("readout_optimizer_configs") != ["adam_lr0.1_init0.1"]:
        raise ValueError(f"{arm} does not use the fixed Adam .1 readout")
    if data.get("checkpoint_sha256") is None or data["checkpoint_sha256"] != fixed["checkpoint_sha256"]:
        # Before is the shared source checkpoint; continuation checkpoints
        # must differ, so their provenance is checked through protocol below.
        if arm == "before":
            raise ValueError("before readout does not match frozen-U checkpoint")

    expected_horizon = 50 if arm == "before" else int(arm[1:])
    if data.get("horizon") != expected_horizon:
        raise ValueError(f"{arm} has horizon {data.get('horizon')}, expected {expected_horizon}")
    expected_step = 700 if arm == "before" else 750
    if data.get("checkpoint_step") != expected_step:
        raise ValueError(f"{arm} has checkpoint step {data.get('checkpoint_step')}, expected {expected_step}")
    # The default four-start equal-length aggregates must all be finite.
    metrics = []
    for budget in BUDGETS:
        row = _equal_row(data["aggregates"], basis="learned", strategy=DEFAULT_STRATEGY, budget=budget)
        if row["n_patterns"] != 12 or row["n_initializations"] != 48:
            raise ValueError(f"{arm} has unexpected known-validation cardinality at budget {budget}")
        metrics.append({"budget": budget, "query_accuracy": row["query_accuracy"], "query_bce": row["query_bce"],
                        "support_accuracy": row["support_accuracy"], "support_bce": row["support_bce"]})
    if arm == "before":
        return {"arm": arm, "horizon": expected_horizon, "checkpoint_step": expected_step,
                "additional_outer_steps": 0, "outer_optimizer_reset": None,
                "training_seconds": 0.0, "readout_seconds": data.get("seconds"),
                "metrics": metrics, "source_verification": source_info}

    protocol_path = root / "horizon" / arm / "protocol.json"
    done_path = root / "horizon" / arm / "done.json"
    if not protocol_path.is_file() or not done_path.is_file():
        raise FileNotFoundError(f"{arm} is missing continuation provenance or timing artifact")
    protocol, done = _load(protocol_path), _load(done_path)
    if protocol.get("source_checkpoint_sha256") != fixed["checkpoint_sha256"] or protocol.get("source_step") != 700:
        raise ValueError(f"{arm} does not descend from the shared step-700 checkpoint")
    if protocol.get("additional_steps") != 50 or not protocol.get("outer_optimizer_reset"):
        raise ValueError(f"{arm} is not the matched 50-update fresh-Adam continuation")
    if done.get("benchmark_only") or done.get("additional_steps") != 50 or done.get("horizon") != expected_horizon:
        raise ValueError(f"{arm} timing artifact has an incompatible protocol")
    seconds = done.get("seconds")
    if not isinstance(seconds, (int, float)) or not np.isfinite(seconds):
        raise ValueError(f"{arm} has invalid training wall time")
    return {"arm": arm, "horizon": expected_horizon, "checkpoint_step": expected_step,
            "additional_outer_steps": 50, "outer_optimizer_reset": True,
            "training_seconds": float(seconds), "median_step_seconds": done.get("median_step_seconds"),
            "readout_seconds": data.get("seconds"), "metrics": metrics, "source_verification": source_info}


def _pattern_accuracy(data: dict, budget: int = 2000) -> dict[int, dict[str, float]]:
    """Average the four v restarts before treating each pattern as one unit."""
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in data["rows"]:
        if row.get("basis") != "learned" or row.get("config") != "adam_lr0.1_init0.1" or row.get("budget") != budget:
            continue
        score = row.get("query", {})
        value = score.get("accuracy")
        if score.get("nonfinite") or value is None:
            raise ValueError("non-finite query accuracy prevents paired bootstrap")
        grouped.setdefault((row["length"], row["pattern"]), []).append(float(value))
    expected = {3: 2, 4: 2, 6: 4, 8: 4}
    output: dict[int, dict[str, float]] = {}
    for length, n_patterns in expected.items():
        values = {pattern: entries for (candidate_length, pattern), entries in grouped.items() if candidate_length == length}
        if len(values) != n_patterns or any(len(entries) != 4 for entries in values.values()):
            raise ValueError(f"unexpected restart/pattern cardinality for bootstrap length {length}")
        output[length] = {pattern: float(np.mean(entries)) for pattern, entries in values.items()}
    return output


def _paired_bootstrap(reference: dict, candidate: dict, draws: int = 10_000) -> dict:
    """Stratified pattern bootstrap, descriptive only (not over model seeds)."""
    ref, candidate_values = _pattern_accuracy(reference), _pattern_accuracy(candidate)
    rng = np.random.default_rng(20_260_907)
    observed_by_length, samples = [], np.zeros(draws, dtype=np.float64)
    for length in sorted(ref):
        if set(ref[length]) != set(candidate_values[length]):
            raise ValueError(f"bootstrap pattern mismatch at length {length}")
        delta = np.asarray([candidate_values[length][pattern] - ref[length][pattern]
                            for pattern in sorted(ref[length])], dtype=np.float64)
        observed_by_length.append(float(delta.mean()))
        samples += delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    samples /= len(ref)
    return {"budget": 2000, "candidate_minus_h50_query_accuracy": float(np.mean(observed_by_length)),
            "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "draws": draws,
            "unit": "pattern after averaging four v restarts; stratified equally across lengths",
            "interpretation": "descriptive variation across these validation patterns, not uncertainty across outer-model seeds"}


def build(root: Path = DEFAULT_ROOT) -> Path:
    """Validate completed artifacts, then write `results.json` and `RESULTS.md`."""
    root = Path(root)
    paths = _required(root)
    results_json, results_md = root / "results.json", root / "RESULTS.md"
    if results_json.exists() or results_md.exists():
        raise FileExistsError("refusing to overwrite an existing combined report")
    fixed, fixed_shard = _load(paths["fixed"]), _load(paths["fixed_shard"])
    if fixed.get("checkpoint_sha256") != fixed_shard.get("checkpoint_sha256"):
        raise ValueError("fixed-U summary and Adam .1 shard disagree on checkpoint")
    readouts = {arm: _load(paths[arm]) for arm in ARMS}
    horizon = [_validate_arm(arm, readouts[arm], fixed, fixed_shard, root) for arm in ARMS]
    fixed_rows = []
    for basis in ("learned", "ideal"):
        for strategy, label, trajectories in FIXED_STRATEGIES:
            row = _equal_row(fixed["aggregate"], basis=basis, strategy=strategy, budget=2000)
            fixed_rows.append({"basis": basis, "policy": label, "trajectories_per_task": trajectories,
                               "query_accuracy": row["query_accuracy"], "support_accuracy": row["support_accuracy"],
                               "query_bce": row["query_bce"], "support_bce": row["support_bce"]})
    bootstrap = {arm: _paired_bootstrap(readouts["h50"], readouts[arm])
                 for arm in ("before", "h50", "h200", "h500")}
    result = {
        "scope": {"known_validation_lengths": [3, 4, 6, 8], "patterns": 12, "held_lengths_accessed": [],
                  "test_partition_accessed": False},
        "provenance": {"fixed_u_checkpoint_sha256": fixed["checkpoint_sha256"],
                       "data_hashes_match_fixed_u_shard3": True,
                       "source_hashes_match_on_all_fixed_u_snapshot_files": True},
        "fixed_u_2000_steps": fixed_rows,
        "horizon_readouts": horizon,
        "paired_bootstrap_vs_h50": bootstrap,
        "caveats": [
            "One outer-model checkpoint and 12 known-length validation patterns; no held lengths or test partition.",
            "The controls match 50 outer updates, not compute: longer inner horizons cost more per update.",
            "A 50-update continuation is a local diagnostic, not evidence of convergence.",
            "Bootstrap intervals are descriptive across patterns after averaging v restarts, not across model seeds.",
        ],
    }
    # Both writes happen only after every validation, aggregation and bootstrap
    # has succeeded.  A missing h500 therefore leaves no partial report.
    write_json(results_json, result)
    _write_markdown(results_md, result)
    return results_json


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _bce(value: float) -> str:
    return f"{value:.5f}"


def _write_markdown(path: Path, result: dict) -> None:
    lines = ["# U-diagnosis results", "",
             "All readouts use the same 12 hash-selected validation patterns at lengths 3, 4, 6 and 8, "
             "the same support/query examples, and four fresh Adam .1 task-vector initializations. "
             "Lengths 5 and 7 and the test partition were not accessed.", "",
             "## Frozen-U optimization at 2,000 v steps", "",
             "| Frozen U | Adaptation policy | Trajectories/task | Query accuracy | Support accuracy |",
             "|---|---|---:|---:|---:|"]
    for row in result["fixed_u_2000_steps"]:
        lines.append(f"| {row['basis']} | {row['policy']} | {row['trajectories_per_task']} | "
                     f"{_pct(row['query_accuracy'])} | {_pct(row['support_accuracy'])} |")
    lines += ["", "`Support-selected optimizer/restart grid` is a costly adaptation policy (five optimizer/LR settings × four starts), "
              "not an oracle or a theoretical upper bound.  Its selection uses support BCE only.", "",
              "## Matched 50-update horizon continuations", "",
              "Each continuation begins from the same step-700 checkpoint with a fresh outer Adam state and exactly 50 additional outer updates.", "",
              "| Arm | Inner horizon | Outer-training wall time | Query at 50 v steps (accuracy / BCE) | Query at 500 | Query at 2,000 |",
              "|---|---:|---:|---|---|---|"]
    for arm in result["horizon_readouts"]:
        metrics = {row["budget"]: row for row in arm["metrics"]}
        time_text = "—" if arm["arm"] == "before" else f"{arm['training_seconds']:.1f} s"
        def cell(budget: int) -> str:
            row = metrics[budget]
            return f"{_pct(row['query_accuracy'])} / {_bce(row['query_bce'])}"
        lines.append(f"| {arm['arm']} | {arm['horizon']} | {time_text} | {cell(50)} | {cell(500)} | {cell(2000)} |")
    lines += ["", "## Paired 2,000-step query-accuracy changes relative to h50", "",
              "The bootstrap resamples patterns separately within each length after first averaging the four v starts for each pattern. "
              "It describes variation over these patterns, not uncertainty over outer-model seeds.", "",
              "| Arm minus h50 | Change (percentage points) | 95% interval (percentage points) |", "|---|---:|---:|"]
    for arm in ("before", "h50", "h200", "h500"):
        row = result["paired_bootstrap_vs_h50"][arm]
        lines.append(f"| {arm} | {100 * row['candidate_minus_h50_query_accuracy']:.2f} | "
                     f"[{100 * row['ci95'][0]:.2f}, {100 * row['ci95'][1]:.2f}] |")
    lines += ["", "## Limits", ""]
    lines += [f"- {caveat}" for caveat in result["caveats"]]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    path = build(args.root)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()

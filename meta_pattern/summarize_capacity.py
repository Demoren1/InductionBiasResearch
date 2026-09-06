"""Summarize the full-U interpolation capacity study without rerunning it.

The script is intentionally tolerant of a partially completed study.  It
reports what is present, names missing expected measurements, and never fills
an absent value with a value from another model or run.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .common import write_json


HELD_LENGTHS = (5, 7)
PRIMARY_BUDGETS = (50, 500, 2000)
_NAME_SEED = re.compile(r"(?P<name>.+?)_seed(?P<seed>\d+)$")
_STAGE_PREFIXES = (("phase1_", "phase1"), ("extended_", "extension"),
                   ("extension_", "extension"), ("baseline_", "baseline"))


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {"_read_error": str(error)}


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _split_stage_label(value: str, stage: str) -> tuple[str, str]:
    for prefix, mapped in _STAGE_PREFIXES:
        if value.startswith(prefix):
            return mapped, value[len(prefix):]
    return stage, value


def _config_parameter_count(config: dict[str, Any]) -> int | None:
    """Compute parameters from the actual generator configuration."""
    if config.get("method") != "generator":
        return 0 if config.get("method") in {"random", "ideal"} else None
    try:
        seq_len = int(config["seq_len"])
        hidden = int(config["hidden"])
        rank1 = int(config["rank1"])
        rank2 = int(config["rank2"])
        width = int(config["width"])
        depth = int(config["generator_depth"])
    except (KeyError, TypeError, ValueError):
        return None
    if min(seq_len, hidden, rank1, rank2, width, depth) < 1:
        return None
    output_size = (seq_len + 1) * hidden * rank1 + (hidden + 1) * rank2
    return 2 * width + (depth - 1) * (width * width + width) + output_size * width + output_size


def _stage_and_identity(path: Path, data: dict[str, Any]) -> tuple[str, str, int | None]:
    """Recover run identity from the finisher file name or checkpoint path."""
    stage, stem = _split_stage_label(path.stem, "evaluation")
    checkpoint = data.get("checkpoint")
    if isinstance(checkpoint, str):
        candidate = Path(checkpoint).parent.name
        # The extension runner retains the phase label in its checkpoint
        # directory (for example ``extended_small_seed42``).  Normalize it
        # too, otherwise the same architecture is reported under two names.
        checkpoint_stage, candidate = _split_stage_label(candidate, stage)
        if _NAME_SEED.match(candidate):
            stage, stem = checkpoint_stage, candidate
    matched = _NAME_SEED.match(stem)
    if matched:
        return stage, matched.group("name"), int(matched.group("seed"))
    seed = data.get("config", {}).get("seed")
    return stage, stem, int(seed) if isinstance(seed, int) else None


def _expected_evaluation_files(root: Path) -> list[str]:
    """Infer finisher outputs from its job manifests, when they are present."""
    def normalize(path_text: str) -> str:
        path = Path(path_text)
        try:
            return str(path.relative_to(root))
        except ValueError:
            rendered = path.as_posix()
            marker = "evaluation/"
            return rendered[rendered.index(marker):] if marker in rendered else str(path)

    def args_out(job: dict[str, Any]) -> str | None:
        arguments = job.get("args")
        if not isinstance(arguments, list):
            return None
        for index, value in enumerate(arguments[:-1]):
            if value == "--out" and isinstance(arguments[index + 1], str):
                return arguments[index + 1]
        return None

    # Once the finisher has written its dedicated evaluation manifest, it is
    # authoritative.  Its job names are scheduler labels such as
    # ``evaluate_phase1_small_seed42`` and are not output file names.
    evaluation_manifests = sorted(root.glob("*evaluation*jobs*.json"))
    if evaluation_manifests:
        expected = set()
        for path in evaluation_manifests:
            payload = _read_json(path)
            jobs = payload.get("jobs") if isinstance(payload, dict) else payload
            if not isinstance(jobs, list):
                continue
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                explicit = job.get("evaluation") or job.get("evaluation_out") or job.get("out") or args_out(job)
                if isinstance(explicit, str) and explicit.endswith(".json"):
                    expected.add(normalize(explicit))
        if expected:
            return sorted(expected)

    # Before that manifest exists, report the planned phase/extension jobs as
    # a provisional coverage check.  These names are training job names and
    # therefore receive the stage prefix below.
    expected: set[str] = set()
    for path in sorted(root.glob("*jobs*.json")):
        lowered = path.name.lower()
        if not any(token in lowered for token in ("phase1", "extension", "extended", "evaluation", "baseline")):
            continue
        payload = _read_json(path)
        jobs = payload.get("jobs") if isinstance(payload, dict) else payload
        if not isinstance(jobs, list):
            continue
        if "phase1" in lowered:
            prefix = "phase1_"
        elif "extension" in lowered or "extended" in lowered:
            prefix = "extended_"
        elif "baseline" in lowered:
            prefix = "baseline_"
        else:
            prefix = ""
        for job in jobs:
            if not isinstance(job, dict):
                continue
            explicit = job.get("evaluation") or job.get("evaluation_out") or job.get("out") or args_out(job)
            if isinstance(explicit, str) and explicit.endswith(".json"):
                expected.add(normalize(explicit))
                continue
            name = job.get("name")
            if isinstance(name, str):
                name = name if name.startswith(("phase1_", "extended_", "extension_", "baseline_")) else prefix + name
                expected.add(str(Path("evaluation") / f"{name}.json"))
    return sorted(expected)


def _run_state(path: Path) -> str:
    """Mark completion only when a local marker says so; otherwise be explicit."""
    for ancestor in (path.parent, *path.parents):
        marker = ancestor / "done.json"
        if marker.exists():
            payload = _read_json(marker)
            if isinstance(payload, dict) and payload.get("failed"):
                return "failed"
            return "completed"
        finished = ancestor / "finished.json"
        if finished.exists():
            return "completed"
        if ancestor.name in {"phase1", "extension", "extended"}:
            break
    return "no_done_marker"


def _discover_evaluations(root: Path) -> list[dict[str, Any]]:
    locations = sorted(set(root.glob("evaluation*.json")) | set((root / "evaluation").glob("*.json")))
    # Accept custom subdirectories as well, while excluding this script's own
    # compact output on repeated invocations.
    locations += [path for path in root.rglob("evaluation*.json") if path not in locations]
    discovered = []
    for path in sorted(set(locations)):
        data = _read_json(path)
        if not isinstance(data, dict) or "aggregates" not in data:
            continue
        stage, model, seed = _stage_and_identity(path, data)
        config = data.get("config") if isinstance(data.get("config"), dict) else {}
        discovered.append({
            "path": str(path.relative_to(root)), "absolute_path": path, "stage": stage,
            "model": model, "seed": seed, "config": config,
            # The evaluator writes its JSON atomically only after all chunks
            # and budgets complete, so a parseable artifact is itself a
            # completion marker even when it lives under root/evaluation/.
            "parameters": _config_parameter_count(config), "state": "completed", "data": data,
        })
    return discovered


def _validation_identity(path: Path, root: Path) -> tuple[str, str, int | None]:
    relative = path.relative_to(root)
    if any(part in {"extension", "extended"} for part in relative.parts):
        stage = "extension"
    elif any(part == "baseline" for part in relative.parts):
        stage = "baseline"
    else:
        stage = "phase1"
    stage, label = _split_stage_label(path.parent.name, stage)
    matched = _NAME_SEED.match(label)
    if matched:
        return stage, matched.group("name"), int(matched.group("seed"))
    return stage, label, None


def _discover_validation(root: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(root.rglob("validation.jsonl")):
        stage, model, seed = _validation_identity(path, root)
        protocol = _read_json(path.parent / "protocol.json")
        config = protocol.get("config", {}) if isinstance(protocol, dict) else {}
        points, malformed, nonfinite = [], 0, 0
        for line in path.read_text().splitlines():
            try:
                point = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(point, dict):
                malformed += 1
                continue
            bce = point.get("bce")
            if not _finite(bce):
                nonfinite += 1
            points.append({"step": point.get("step"), "bce": float(bce) if _finite(bce) else None,
                           "best": bool(point.get("best", False))})
        finite_points = [point for point in points if point["bce"] is not None]
        entries.append({
            "path": str(path.relative_to(root)), "stage": stage, "model": model, "seed": seed,
            "parameters": _config_parameter_count(config), "config": config,
            "state": _run_state(path), "points": points, "malformed_lines": malformed,
            "nonfinite_points": nonfinite,
            "best_bce": min((point["bce"] for point in finite_points), default=None),
            "last_bce": finite_points[-1]["bce"] if finite_points else None,
        })
    return entries


def _aggregate_rows(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    data = evaluation["data"]
    result = []
    for row in data.get("aggregates", []):
        if not isinstance(row, dict):
            continue
        result.append({
            "stage": evaluation["stage"], "model": evaluation["model"], "seed": evaluation["seed"],
            "parameters": evaluation["parameters"], "state": evaluation["state"],
            "length": row.get("length"), "steps": row.get("steps"),
            "condition": row.get("condition", "correct"),
            "condition_length": row.get("condition_length"), "distribution": row.get("distribution"),
            "n": row.get("n"), "n_finite": row.get("n_finite", row.get("n")),
            "bce": row.get("bce"), "accuracy": row.get("accuracy"),
            "source": evaluation["path"],
        })
    return result


def _bootstrap_mean(values: list[float], seed: int) -> tuple[float, list[float] | None]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(10_000))
    return mean, [samples[249], samples[9749]]


def _paired_condition_deltas(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired correct-minus-neighbour effects, with patterns as bootstrap units."""
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for item in evaluations:
        rows = item["data"].get("rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("length") not in HELD_LENGTHS:
                continue
            if row.get("distribution") != "balanced" or row.get("steps") not in PRIMARY_BUDGETS:
                continue
            if row.get("condition") not in {"correct", "wrong_lower_neighbor", "wrong_upper_neighbor"}:
                continue
            key = (item["stage"], item["model"], int(row["length"]), int(row["steps"]), item["path"])
            grouped[key].append({**row, "seed": item["seed"]})

    # Named factories make the five alignment levels auditable: source,
    # pattern, model seed, condition, then metric samples.
    def metric_samples() -> defaultdict[str, list[float]]:
        return defaultdict(list)

    def conditions() -> defaultdict[str, defaultdict[str, list[float]]]:
        return defaultdict(metric_samples)

    def model_seeds() -> defaultdict[int | None, defaultdict[str, defaultdict[str, list[float]]]]:
        return defaultdict(conditions)

    def patterns() -> defaultdict[str, defaultdict[int | None, defaultdict[str, defaultdict[str, list[float]]]]]:
        return defaultdict(model_seeds)

    def sources() -> defaultdict[str, defaultdict[str, defaultdict[int | None, defaultdict[str, defaultdict[str, list[float]]]]]]:
        return defaultdict(patterns)

    source_pattern = defaultdict(sources)
    nonfinite: dict[tuple[str, str, int, int], int] = defaultdict(int)
    for (stage, model, length, steps, source), rows in grouped.items():
        target = source_pattern[(stage, model, length, steps)][source]
        for row in rows:
            for metric in ("accuracy", "bce"):
                value = row.get(metric)
                if row.get("nonfinite") or not _finite(value):
                    nonfinite[(stage, model, length, steps)] += 1
                    continue
                target[row["pattern"]][row["seed"]][row["condition"]][metric].append(float(value))

    output = []
    for key, sources in sorted(source_pattern.items()):
        stage, model, length, steps = key
        # A stage/model/seed should have one evaluator file.  If it does not,
        # retain the files as independent seed slots rather than silently mix
        # checkpoints.
        pattern_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for source, patterns in sources.items():
            del source
            for pattern, seeds in patterns.items():
                for seed, conditions in seeds.items():
                    del seed
                    for metric in ("accuracy", "bce"):
                        correct = conditions.get("correct", {}).get(metric, [])
                        lower = conditions.get("wrong_lower_neighbor", {}).get(metric, [])
                        upper = conditions.get("wrong_upper_neighbor", {}).get(metric, [])
                        if correct and lower:
                            pattern_values["correct_minus_wrong_lower_neighbor:" + metric][pattern].append(
                                sum(correct) / len(correct) - sum(lower) / len(lower)
                            )
                        if correct and upper:
                            pattern_values["correct_minus_wrong_upper_neighbor:" + metric][pattern].append(
                                sum(correct) / len(correct) - sum(upper) / len(upper)
                            )
                        if correct and lower and upper:
                            pattern_values["correct_minus_mean_neighbors:" + metric][pattern].append(
                                sum(correct) / len(correct) -
                                ((sum(lower) / len(lower) + sum(upper) / len(upper)) / 2.0)
                            )
        for descriptor, values_by_pattern in sorted(pattern_values.items()):
            relation, metric = descriptor.split(":", 1)
            values = [sum(values) / len(values) for _, values in sorted(values_by_pattern.items())]
            mean, interval = _bootstrap_mean(values, seed=20_260_906 + length * 10_000 + steps + len(output))
            output.append({
                "stage": stage, "model": model, "length": length, "steps": steps,
                "distribution": "balanced", "relation": relation, "metric": metric,
                "n_patterns": len(values), "n_nonfinite_rows_excluded": nonfinite[key],
                "mean": mean, "bootstrap_ci95": interval,
                # Do not make a clean-looking paired claim when any matching
                # condition had a non-finite metric: its paired effect was
                # necessarily excluded from the mean above.
                "status": ("incomplete_nonfinite" if nonfinite[key] else
                           "ok" if len(values) >= 2 else "one_pattern_only_no_ci"),
            })
    return output


def _cvae_reference() -> dict[str, Any]:
    """Read the already completed mask-CVAE result as a clearly separate reference."""
    project = Path(__file__).resolve().parents[1]
    path = project / "pattern/outputs/length_interp/hour_20260906_205915/summary.json"
    data = _read_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return {"available": False, "source": str(path), "reason": "reference artifact absent or unreadable"}
    rows = []
    for row in data["rows"]:
        if isinstance(row, dict) and row.get("distribution") == "balanced" and row.get("method") == "cvae":
            rows.append({"length": row.get("length"), "accuracy": row.get("accuracy")})
    return {
        "available": bool(rows), "source": str(path), "balanced_cvae": rows,
        "strict_note": (
            "Mask-CVAE reference only: it uses a different model family and an ordinary MLP optimizer. "
            "The pattern task split/data convention matches, but these values are not a controlled numerical "
            "comparison to full-U adaptation."
        ),
    }


def _fmt_fraction(value: Any) -> str:
    return f"{100.0 * float(value):.2f}%" if _finite(value) else "n/a"


def _fmt_bce(value: Any) -> str:
    return f"{float(value):.5f}" if _finite(value) else "n/a"


def _write_markdown(root: Path, summary: dict[str, Any]) -> None:
    lines = ["# Full-U capacity and interpolation study", "", "This report is generated only from completed artifacts present under this directory.", ""]
    discovery = summary["discovery"]
    lines += ["## Coverage", "", f"- Evaluation artifacts read: {discovery['evaluation_files']}",
              f"- Expected evaluation artifacts from manifests: {len(discovery['expected_evaluation_files'])}",
              f"- Expected evaluation artifacts still absent: {len(discovery['missing_evaluation_files'])}",
              f"- Known-length validation curves read: {discovery['validation_files']}",
              f"- Evaluation artifacts with non-finite aggregate values: {discovery['nonfinite_aggregate_rows']}", ""]
    if discovery["missing"]:
        lines += ["Missing or incomplete evidence:", ""]
        lines += [f"- {item}" for item in discovery["missing"]]
        lines.append("")

    lines += ["## Known-length validation", "", "The meta-objective adapts v for 50 steps on known-length validation tasks. These curves were not selected using held lengths 5 or 7.", "",
              "| Stage | Model | Seed | Parameters | Best BCE | Last BCE | State | Non-finite points |", "|---|---|---:|---:|---:|---:|---|---:|"]
    for row in summary["validation_curves"]:
        lines.append(
            f"| {row['stage']} | {row['model']} | {row['seed'] if row['seed'] is not None else 'n/a'} | "
            f"{row['parameters'] if row['parameters'] is not None else 'n/a'} | {_fmt_bce(row['best_bce'])} | "
            f"{_fmt_bce(row['last_bce'])} | {row['state']} | {row['nonfinite_points']} |"
        )
    lines.append("")

    lines += ["## Held-length balanced evaluation", "", "Primary transfer readout is accuracy/BCE after 500 and 2,000 fresh-v updates; 50 is shown because it matches the meta-training adaptation budget.", "",
              "| Stage | Model | Seed | Params | Length | v steps | Accuracy | BCE | n finite / n | State |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary["held_balanced_correct"]:
        finite = f"{row['n_finite']}/{row['n']}" if row.get("n") is not None else "n/a"
        lines.append(
            f"| {row['stage']} | {row['model']} | {row['seed'] if row['seed'] is not None else 'n/a'} | "
            f"{row['parameters'] if row['parameters'] is not None else 'n/a'} | {row['length']} | {row['steps']} | "
            f"{_fmt_fraction(row['accuracy'])} | {_fmt_bce(row['bce'])} | {finite} | {row['state']} |"
        )
    lines.append("")

    lines += ["## Capacity comparison", "", "Conditional sweep keeps v dimensionality at 20 (rank1=16, rank2=4). `wide_rank` changes it to 40 and is therefore a separate subspace-capacity ablation. `unconditional` has the same architecture as xlarge but removes the length input.", ""]
    for label, names in (("Conditional architecture sweep", {"small", "large", "xlarge"}),
                         ("Separate controls", {"wide_rank", "unconditional"})):
        lines += [f"### {label}", "", "| Stage | Model | Seed | Parameters | Length | 50 / 500 / 2000 balanced accuracy |", "|---|---|---:|---:|---:|---:|"]
        grouped: dict[tuple, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in summary["held_balanced_correct"]:
            if row["model"] in names:
                grouped[(row["stage"], row["model"], row["seed"], row["parameters"], row["length"])][row["steps"]] = row
        for key, values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
            display = " / ".join(_fmt_fraction(values.get(step, {}).get("accuracy")) for step in PRIMARY_BUDGETS)
            lines.append(f"| {key[0]} | {key[1]} | {key[2] if key[2] is not None else 'n/a'} | {key[3] if key[3] is not None else 'n/a'} | {key[4]} | {display} |")
        lines.append("")

    lines += ["## Correct-condition effect", "", "For each pattern, repeats and matching model seeds are averaged before a paired bootstrap over patterns. Positive accuracy differences mean the correct length condition is better; positive BCE differences mean it is worse.", "",
              "| Stage | Model | Length | Steps | Comparison | Metric | Difference | 95% bootstrap CI | Patterns | Excluded non-finite rows | Status |", "|---|---|---:|---:|---|---|---:|---|---:|---:|---|"]
    for row in summary["paired_condition_deltas"]:
        delta = f"{100 * row['mean']:.3f} pp" if row["metric"] == "accuracy" else f"{row['mean']:.6f}"
        if row["bootstrap_ci95"] is None:
            interval = "n/a"
        elif row["metric"] == "accuracy":
            interval = f"[{100 * row['bootstrap_ci95'][0]:.3f}, {100 * row['bootstrap_ci95'][1]:.3f}] pp"
        else:
            interval = f"[{row['bootstrap_ci95'][0]:.6f}, {row['bootstrap_ci95'][1]:.6f}]"
        lines.append(f"| {row['stage']} | {row['model']} | {row['length']} | {row['steps']} | {row['relation']} | {row['metric']} | {delta} | {interval} | {row['n_patterns']} | {row['n_nonfinite_rows_excluded']} | {row['status']} |")
    lines.append("")

    cvae = summary["cvae_reference"]
    lines += ["## Earlier mask-CVAE reference", ""]
    if cvae["available"]:
        lines += ["| Length | Balanced accuracy |", "|---:|---:|"]
        for row in cvae["balanced_cvae"]:
            lines.append(f"| {row['length']} | {_fmt_fraction(row['accuracy'])} |")
        lines += ["", cvae["strict_note"], f"Source: `{cvae['source']}`.", ""]
    else:
        lines += [f"Reference unavailable: {cvae['reason']}.", ""]
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def summarize_capacity(root: Path | str) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Study root does not exist: {root}")
    evaluations = _discover_evaluations(root)
    validation = _discover_validation(root)
    expected_files = _expected_evaluation_files(root)
    actual_files = {item["path"] for item in evaluations}
    aggregate_rows = [row for item in evaluations for row in _aggregate_rows(item)]
    held = [row for row in aggregate_rows if row["length"] in HELD_LENGTHS and row["steps"] in PRIMARY_BUDGETS
            and row["distribution"] == "balanced" and row["condition"] == "correct"]
    held.sort(key=lambda row: (str(row["stage"]), str(row["model"]), str(row["seed"]), row["length"], row["steps"]))
    nonfinite = sum(1 for row in aggregate_rows if not _finite(row["accuracy"]) or not _finite(row["bce"]))
    missing = []
    if not evaluations:
        missing.append("No evaluation*.json artifact found; held-length claims are not available.")
    for length in HELD_LENGTHS:
        for step in PRIMARY_BUDGETS:
            if not any(row["length"] == length and row["steps"] == step for row in held):
                missing.append(f"No correct-condition balanced held metric found for length {length} at {step} v steps.")
    if not validation:
        missing.append("No phase1/extension validation.jsonl curve found.")
    for expected in expected_files:
        if expected not in actual_files:
            missing.append(f"Expected evaluation artifact is absent: {expected}")
    summary = {
        "root": str(root.resolve()),
        "discovery": {
            "evaluation_files": len(evaluations), "validation_files": len(validation),
            "expected_evaluation_files": expected_files,
            "missing_evaluation_files": [name for name in expected_files if name not in actual_files],
            "nonfinite_aggregate_rows": nonfinite, "missing": missing,
        },
        "models": [{key: item[key] for key in ("path", "stage", "model", "seed", "parameters", "state", "config")}
                   for item in evaluations],
        "validation_curves": validation,
        "held_balanced_correct": held,
        "paired_condition_deltas": _paired_condition_deltas(evaluations),
        "cvae_reference": _cvae_reference(),
    }
    write_json(root / "summary.json", summary)
    _write_markdown(root, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_capacity(args.root)
    print(f"WROTE {Path(summary['root']) / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()

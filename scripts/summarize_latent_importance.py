"""Collect paired latent-importance results without retuning any checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean(values) -> float:
    values = list(values)
    if not values:
        raise ValueError("empty aggregate")
    return sum(values) / len(values)


def _group(rows: list[dict], keys: tuple[str, ...], value: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row[value])
    return [dict(zip(keys, key), mean=_mean(values), n=len(values))
            for key, values in sorted(groups.items())]


def summarize(root: Path) -> dict:
    old = Path("meta_pattern/outputs/interpolation_capacity_20260906/adam50/seed42_accuracy/evaluation")
    stage1_seeds = (42, 43, 44, 45)
    stage1_files = {(seed, model):
                    (old / "phase1_small_seed42.json" if seed == 42 and model == "conditional"
                     else root / f"stage1/evaluation/{'unconditional' if model == 'constant' else 'conditional'}_seed{seed}.json")
                    for seed in stage1_seeds for model in ("conditional", "constant")}
    stage1 = {key: _read(path) for key, path in stage1_files.items()}
    hashes = [value["data_hashes"] for value in stage1.values()]
    if any(value != hashes[0] for value in hashes[1:]):
        raise ValueError("stage-1 evaluations did not use identical episodes")
    stage1_rows = []
    for (seed, model), payload in stage1.items():
        for row in payload["rows"]:
            if row["distribution"] == "balanced" and row["condition"] == "correct" and row["steps"] in (50, 500, 2000):
                if row["nonfinite"] or row["bce"] is None:
                    raise ValueError("nonfinite stage-1 row")
                stage1_rows.append({"seed": seed, "model": model, **row})
    stage1_grouped = _group(stage1_rows, ("seed", "model", "length", "steps"), "bce")
    stage1_accuracy = _group(stage1_rows, ("seed", "model", "length", "steps"), "accuracy")
    stage1_index = {(r["seed"], r["model"], r["length"], r["steps"], r["pattern"], r["repeat"]): r
                    for r in stage1_rows}
    paired = []
    for seed in stage1_seeds:
        for length in (5, 7):
            for steps in (50, 500, 2000):
                differences = []
                for key, cond in stage1_index.items():
                    if key[:4] != (seed, "conditional", length, steps):
                        continue
                    const_key = (seed, "constant", length, steps, key[4], key[5])
                    differences.append(cond["bce"] - stage1_index[const_key]["bce"])
                paired.append({"seed": seed, "length": length, "steps": steps,
                               "conditional_minus_constant_bce": _mean(differences),
                               "paired_episodes": len(differences)})

    stage2_payloads = {seed: _read(root / f"stage2/condition_matrix_seed{seed}.json")
                       for seed in stage1_seeds}
    stage2_rows = [{"seed": seed, **row} for seed, payload in stage2_payloads.items()
                   for row in payload["rows"]]
    stage2_grouped = _group(stage2_rows, ("seed", "target_length", "condition_length", "steps"), "test_bce")
    stage2_accuracy = _group(stage2_rows, ("seed", "target_length", "condition_length", "steps"), "test_accuracy")

    stage3_payloads = {seed: _read(root / f"stage3/evaluation/code_seed{seed}.json")
                       for seed in (42, 43)}
    stage3_held = [{"seed": seed, **row} for seed, payload in stage3_payloads.items()
                   for row in payload["held_rows"]]
    stage3_known = [{"seed": seed, **row} for seed, payload in stage3_payloads.items()
                    for row in payload["known_condition_matrix_rows"]]
    stage3_grouped = _group(stage3_held, ("seed", "target_length", "variant", "steps"), "test_bce")
    stage3_accuracy = _group(stage3_held, ("seed", "target_length", "variant", "steps"), "test_accuracy")
    stage3_known_grouped = _group(stage3_known, ("seed", "target_length", "variant", "steps"), "test_bce")

    linear = []
    for path in sorted((root / "stage4").glob("linear_*.json")):
        linear.extend(_read(path)["rows"])
    if not linear:
        raise ValueError("no linear-control results")
    linear_grouped = _group(linear, ("varying_truth", "variable_code"), "test_excess_mse")

    return {
        "stage1": {"bce": stage1_grouped, "accuracy": stage1_accuracy,
                   "paired_conditional_minus_constant_bce": paired,
                   "checkpoints": {f"{seed}_{model}": stage1[(seed, model)]["checkpoint"]
                                   for seed, model in stage1}},
        "stage2": {"bce": stage2_grouped, "accuracy": stage2_accuracy,
                   "structural_projector_distance": {
                       str(seed): stage2_payloads[seed]["structural_projector_distance"]
                       for seed in stage2_payloads}},
        "stage3": {"held_bce": stage3_grouped, "held_accuracy": stage3_accuracy,
                   "known_condition_matrix_bce": stage3_known_grouped,
                   "calibration": {str(seed): stage3_payloads[seed]["calibration"]
                                   for seed in stage3_payloads},
                   "scalar_calibration": {str(seed): stage3_payloads[seed]["scalar_calibration"]
                                          for seed in stage3_payloads}},
        "stage4": {"test_excess_mse": linear_grouped, "rows": linear},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("meta_pattern/outputs/latent_importance_20260919"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summarize(args.root), indent=2) + "\n")


if __name__ == "__main__":
    main()

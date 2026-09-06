"""Finish a fresh capacity study without using held-out lengths for selection.

The phase-one manifest is dispatched first.  Only after *every* manifest job
has a successful ``done.json`` does this script select one conditional
architecture using the mean of the two known-length validation BCE values.
It writes that immutable decision to ``selection.json`` before launching any
interpolation evaluation.

The remainder of the work is also dispatched through the project's restricted
GPU launcher: continuation from phase-one ``latest.pt`` files, fixed random
and ideal-U controls, then exhaustive interpolation evaluation.  It calls the
read-only capacity summarizer after evaluation and then writes a completion
marker.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_GPUS = (0, 4, 5, 6, 7)
CONDITIONAL_ARCHITECTURES = ("small", "large", "xlarge")
SELECTION_SEEDS = (42, 43)
TRAIN_LENGTHS = (3, 4, 6, 8)
EVALUATION_STEPS = (20, 50, 100, 500, 2000)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("train manifest must be a non-empty JSON job list")
    if any(not isinstance(job, dict) or not isinstance(job.get("name"), str)
           or not isinstance(job.get("args"), list) for job in payload):
        raise ValueError("every training job needs string name and argument-list args")
    names = [job["name"] for job in payload]
    if len(names) != len(set(names)):
        raise ValueError("training manifest contains duplicate job names")
    return payload


def _option(args: list[Any], option: str) -> str:
    """Return a required one-value CLI option from a manifest job."""
    try:
        index = args.index(option)
    except ValueError as error:
        raise ValueError(f"manifest job is missing {option}") from error
    if index + 1 >= len(args) or not isinstance(args[index + 1], str):
        raise ValueError(f"manifest job has invalid {option}")
    return args[index + 1]


def _job_output(job: dict[str, Any]) -> Path:
    value = Path(_option(job["args"], "--out"))
    return value if value.is_absolute() else (ROOT / value)


def _require_train_completion(jobs: Iterable[dict[str, Any]], train_out: Path) -> dict[str, Path]:
    """Verify every phase-one artifact was produced under --train-out."""
    train_out = train_out.resolve()
    outputs: dict[str, Path] = {}
    for job in jobs:
        output = _job_output(job).resolve()
        try:
            output.relative_to(train_out)
        except ValueError as error:
            raise ValueError(
                f"phase-one output {output} for {job['name']} is outside --train-out {train_out}"
            ) from error
        required = (output / "done.json", output / "best.pt", output / "latest.pt", output / "protocol.json")
        missing = [path.name for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"phase-one job {job['name']} did not finish; missing {missing} in {output}")
        done = json.loads((output / "done.json").read_text())
        if not isinstance(done, dict) or not math.isfinite(float(done.get("best_val_bce", math.nan))):
            raise RuntimeError(f"phase-one job {job['name']} has no finite known-length validation score")
        outputs[job["name"]] = output
    return outputs


def _load_best(path: Path) -> tuple[dict[str, Any], float]:
    state = torch.load(path / "best.pt", map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not isinstance(state.get("config"), dict):
        raise ValueError(f"invalid best checkpoint in {path}")
    score = float(state.get("val_bce", math.nan))
    if not math.isfinite(score):
        raise ValueError(f"best checkpoint has non-finite validation BCE: {path}")
    return state, score


def _select_architecture(outputs: dict[str, Path]) -> tuple[str, dict[str, Any], dict[str, dict[str, Any]]]:
    """Select exclusively from phase-one best validation checkpoints."""
    scores: dict[str, dict[str, Any]] = {}
    selected_states: dict[str, dict[str, Any]] = {}
    for architecture in CONDITIONAL_ARCHITECTURES:
        per_seed: dict[str, float] = {}
        for seed in SELECTION_SEEDS:
            name = f"{architecture}_seed{seed}"
            if name not in outputs:
                raise RuntimeError(f"required conditional architecture run is absent: {name}")
            state, score = _load_best(outputs[name])
            config = state["config"]
            if config.get("method") != "generator" or config.get("condition_length") is not True:
                raise ValueError(f"{name} is not a conditional generator checkpoint")
            if tuple(config.get("train_lengths", ())) != TRAIN_LENGTHS:
                raise ValueError(f"{name} was not trained at exactly {TRAIN_LENGTHS}")
            per_seed[str(seed)] = score
            selected_states[name] = state
        scores[architecture] = {
            "seed_best_val_bce": per_seed,
            "equal_seed_mean_best_val_bce": sum(per_seed.values()) / len(SELECTION_SEEDS),
        }
    # Stable lexical order makes an exact tie reviewable and independent of
    # dictionary insertion order.  No held-out pattern/data is read here.
    chosen = min(CONDITIONAL_ARCHITECTURES,
                 key=lambda name: (scores[name]["equal_seed_mean_best_val_bce"], name))
    return chosen, scores, selected_states


def _config_to_train_args(config: dict[str, Any], out: Path, resume: Path | None, outer_steps: int) -> list[str]:
    """Render a Config-compatible CLI with every non-default study setting.

    The CLI intentionally exposes only the established fixed 32/32 task
    geometry.  Validate it here instead of silently reconstructing a changed
    architecture from defaults when resuming.
    """
    fixed = {
        "seq_len": 32, "hidden": 32, "lengths": [3, 4, 5, 6, 7, 8],
        "task_split_seed": 42, "input_split_seed": 1729,
        "outer_lr": 0.001, "grad_clip": 5.0,
    }
    for key, value in fixed.items():
        observed = config.get(key)
        if isinstance(value, float):
            if float(observed) != value:
                raise ValueError(f"cannot render nonstandard {key} through meta_pattern.train CLI")
        elif key == "lengths":
            if tuple(observed) != tuple(value):
                raise ValueError(f"cannot render nonstandard {key} through meta_pattern.train CLI")
        elif observed != value:
            raise ValueError(f"cannot render nonstandard {key} through meta_pattern.train CLI")
    if tuple(config.get("train_lengths", ())) != TRAIN_LENGTHS:
        raise ValueError("continuation must preserve train lengths 3,4,6,8")
    if config.get("method") not in {"generator", "random", "ideal"}:
        raise ValueError("only generator/random/ideal are supported in this study finisher")
    args = [
        "-m", "meta_pattern.train", "--out", str(out), "--method", str(config["method"]),
        "--seed", str(config["seed"]), "--outer-steps", str(outer_steps),
        "--inner-steps", str(config["inner_steps"]), "--inner-lr", str(config["inner_lr"]),
        "--inner-optimizer", str(config["inner_optimizer"]), "--init-scale", str(config["init_scale"]),
        "--tasks-per-step", str(config["tasks_per_step"]), "--support-size", str(config["support_size"]),
        "--query-size", str(config["query_size"]), "--batch-size", str(config["batch_size"]),
        "--validate-every", str(config["validate_every"]),
        "--val-tasks-per-length", str(config["val_tasks_per_length"]),
        "--rank1", str(config["rank1"]), "--rank2", str(config["rank2"]),
        "--width", str(config["width"]), "--generator-depth", str(config["generator_depth"]),
        "--train-lengths", *map(str, TRAIN_LENGTHS),
    ]
    if config.get("data_seed") is not None:
        args.extend(("--data-seed", str(config["data_seed"])))
    if config.get("all_unseen_patterns"):
        args.append("--all-unseen-patterns")
    if not config.get("condition_length", True):
        args.append("--unconditional")
    if resume is not None:
        args.extend(("--resume", str(resume)))
    return args


def _write_manifest(path: Path, jobs: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jobs, indent=2) + "\n")


def _dispatch(manifest: Path, out: Path, gpus: list[int]) -> None:
    """Run one finite manifest with the project's authorized-GPU dispatcher."""
    command = [
        sys.executable, str(ROOT / "meta_pattern/scripts/dispatch.py"),
        "--manifest", str(manifest), "--out", str(out), "--gpus", *map(str, gpus),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    status = json.loads((out / "status.json").read_text())
    if not status or any(int(value.get("exit_code", 1)) != 0 for value in status.values()):
        raise RuntimeError(f"dispatch did not complete cleanly: {out}")


def _variant_outputs(outputs: dict[str, Path], variant: str) -> list[tuple[str, Path]]:
    prefix = f"{variant}_seed"
    selected = [(name, output) for name, output in outputs.items() if name.startswith(prefix)]
    if not selected:
        raise RuntimeError(f"phase-one manifest has no {variant!r} checkpoint")
    return sorted(selected, key=lambda item: item[0])


def _evaluation_jobs(checkpoints: Iterable[tuple[str, Path]], root: Path) -> list[dict[str, Any]]:
    jobs = []
    for name, checkpoint in checkpoints:
        jobs.append({
            "name": f"evaluate_{name}",
            "args": [
                "-m", "meta_pattern.evaluate_interpolation", "--checkpoint", str(checkpoint),
                "--out", str(root / "evaluation" / f"{name}.json"),
                "--steps", *map(str, EVALUATION_STEPS), "--repeats", "2",
                "--support-size", "8192", "--test-size", "2048", "--batch-size", "128",
            ],
        })
    return jobs


def finish_capacity_study(root: Path, train_manifest: Path, train_out: Path, gpus: list[int], extend_to: int = 3000) -> Path:
    """Dispatch phase one, freeze selection, extend, control, and evaluate."""
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= set(ALLOWED_GPUS):
        raise ValueError("only distinct user-authorized GPUs 0,4,5,6,7 may be used")
    if extend_to <= 1000:
        raise ValueError("--extend-to must be greater than phase-one's 1000 steps")
    root, train_manifest, train_out = Path(root).resolve(), Path(train_manifest).resolve(), Path(train_out).resolve()
    jobs = _read_manifest(train_manifest)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "finished.json"
    if marker.exists():
        raise FileExistsError(f"study is already marked finished: {marker}")

    _dispatch(train_manifest, root / "phase1_dispatch", gpus)
    outputs = _require_train_completion(jobs, train_out)
    chosen, architecture_scores, states = _select_architecture(outputs)

    selection_path = root / "selection.json"
    if selection_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen selection: {selection_path}")
    selection = {
        "selection_data": "phase-one best checkpoint validation BCE on known train lengths only",
        "heldout_lengths_not_accessed_for_selection": [5, 7],
        "architectures": architecture_scores,
        "selected_conditional_architecture": chosen,
        "tie_break": "architecture name in lexical order",
        "extend_to_outer_steps": extend_to,
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    extension_jobs: list[dict[str, Any]] = []
    extended_checkpoints: list[tuple[str, Path]] = []
    # The chosen conditional model is selected above.  Unconditional and wide
    # rank are controls, retained independently of that outcome.
    for variant in (chosen, "unconditional", "wide_rank"):
        for phase_name, phase_output in _variant_outputs(outputs, variant):
            state, _ = _load_best(phase_output)
            config = state["config"]
            name = f"extended_{phase_name}"
            output = root / "extended" / name
            extension_jobs.append({
                "name": name,
                "args": _config_to_train_args(config, output, phase_output / "latest.pt", extend_to),
            })
            extended_checkpoints.append((name, output / "best.pt"))
    extension_manifest = root / "extension_jobs.json"
    _write_manifest(extension_manifest, extension_jobs)
    _dispatch(extension_manifest, root / "extension_dispatch", gpus)
    _require_train_completion(extension_jobs, root / "extended")

    # Random changes with its seed; ideal U has no trainable basis and hence
    # only one seed is useful.  Both use the selected model's calibrated inner
    # optimizer, learning rate, initialization, train lengths and data policy.
    baseline_jobs: list[dict[str, Any]] = []
    baseline_checkpoints: list[tuple[str, Path]] = []
    reference_config = states[f"{chosen}_seed42"]["config"]
    for method, seeds in (("random", SELECTION_SEEDS), ("ideal", (42,))):
        for seed in seeds:
            config = dict(reference_config)
            config["method"] = method
            config["seed"] = seed
            name = f"baseline_{method}_seed{seed}"
            output = root / "baselines" / name
            # random/ideal train() completes at step zero, so total outer
            # steps is semantically immaterial but kept explicit/reviewable.
            baseline_jobs.append({
                "name": name,
                "args": _config_to_train_args(config, output, None, 0),
            })
            baseline_checkpoints.append((name, output / "best.pt"))
    baseline_manifest = root / "baseline_jobs.json"
    _write_manifest(baseline_manifest, baseline_jobs)
    _dispatch(baseline_manifest, root / "baseline_dispatch", gpus)
    _require_train_completion(baseline_jobs, root / "baselines")

    phase_one_checkpoints = [(f"phase1_{name}", output / "best.pt") for name, output in sorted(outputs.items())]
    evaluation_manifest = root / "evaluation_jobs.json"
    _write_manifest(
        evaluation_manifest,
        _evaluation_jobs([*phase_one_checkpoints, *extended_checkpoints, *baseline_checkpoints], root),
    )
    _dispatch(evaluation_manifest, root / "evaluation_dispatch", gpus)
    # Reporting reads only frozen selection and completed evaluation artifacts;
    # it cannot affect model choice or dispatch any further learning jobs.
    subprocess.run(
        [sys.executable, "-m", "meta_pattern.summarize_capacity", "--root", str(root)],
        cwd=ROOT, check=True,
    )
    marker.write_text(json.dumps({
        "selection": str(selection_path),
        "phase_one_checkpoints": len(phase_one_checkpoints),
        "extended_checkpoints": len(extended_checkpoints),
        "baseline_checkpoints": len(baseline_checkpoints),
        "evaluation_steps": list(EVALUATION_STEPS),
    }, indent=2) + "\n")
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="study directory for dispatches and final artifacts")
    parser.add_argument("--train-manifest", type=Path, required=True, help="fresh phase-one training job manifest")
    parser.add_argument("--train-out", type=Path, required=True, help="directory containing manifest training outputs")
    parser.add_argument("--gpus", nargs="+", type=int, default=list(ALLOWED_GPUS))
    parser.add_argument("--extend-to", type=int, default=3000)
    args = parser.parse_args()
    result = finish_capacity_study(**vars(args))
    print(f"WROTE {result}", flush=True)


if __name__ == "__main__":
    main()

"""Robust replicated hard-mask sampling on matched R=4 pattern VAEs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
REPO_ROOT = PATTERN_ROOT.parent
sys.path.insert(0, str(PATTERN_ROOT))

from evaluation.hard_mask_sampling_robust import evolutionary_search_robust  # noqa: E402
from evaluation.z_star_noise import atomic_save, ci, load_model, sha256_file, write_json  # noqa: E402
from evaluation.z_star_sampling_r4 import analyze_group, full_schedule  # noqa: E402


REFERENCE = PATTERN_ROOT / "outputs/z_star_reachable/r4_multiseed8_20260912"
PAIR_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_sampling_robust_pair_20260913"
FULL_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_sampling_robust_multiseed8_20260913"


@dataclass(frozen=True)
class Settings:
    model_seeds: tuple[int, ...]
    starts_per_group: int = 64
    radius: float = 4.0
    radii_schedule: tuple[tuple[float, ...], ...] = full_schedule()
    screen_steps: int = 400
    refine_steps: int = 2000
    refine_replicates: int = 2
    finalists: int = 4
    final_eval_steps: int = 2000
    min_improvement: float = 1e-4
    seed: int = 20260921


def settings_for(scope: str) -> Settings:
    if scope == "full":
        return Settings(tuple(range(186, 202, 2)))
    if scope == "pair":
        return Settings((186, 188))
    if scope == "smoke":
        return replace(Settings((186,)), starts_per_group=2,
                       radii_schedule=((.25, .5, 1., 2.),), screen_steps=2,
                       refine_steps=3, finalists=2, final_eval_steps=3)
    raise ValueError(scope)


def protocol_payload(reference: Path, settings: Settings, scope: str) -> dict[str, Any]:
    source_protocol = json.loads((reference / "protocol.json").read_text())
    patterns = source_protocol["heldout_patterns"][:1] if scope == "smoke" else source_protocol["heldout_patterns"]
    sources = [Path(__file__), HERE / "hard_mask_sampling_robust.py",
               HERE / "hard_mask_sampling.py", HERE / "single_z_task.py",
               HERE / "z_star_sampling_r4.py", HERE / "z_star_noise.py"]
    artifacts = {}
    for seed in settings.model_seeds:
        oracle = reference / f"seed_{seed}/oracle_same_start.pt"
        initials = reference / f"seed_{seed}/task_0100.pt"
        artifacts[str(seed)] = {
            "oracle": str(oracle), "oracle_sha256": sha256_file(oracle),
            "initials": str(initials), "initials_sha256": sha256_file(initials),
            "checkpoint": source_protocol["checkpoints"][str(seed)]["checkpoint"],
            "checkpoint_sha256": source_protocol["checkpoints"][str(seed)]["checkpoint_sha256"],
        }
    return {
        "experiment": "replicated full-fidelity hard-mask evolutionary latent sampling",
        "scope": scope, "settings": asdict(settings), "patterns": patterns,
        "reference": str(reference), "reference_protocol_sha256": sha256_file(reference / "protocol.json"),
        "parent": source_protocol["parent"], "artifacts": artifacts,
        "groups": ["prior", "oracle_z_star"],
        "selection": (
            "A mutation is accepted only if it changes hard support and beats its paired parent "
            "by min_improvement in every independently trained 2000-step MLP replicate."
        ),
        "gold_access": "post-hoc exact/IoU diagnostics only",
        "independence_unit": "frozen VAE seed",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
    }


def prepare(out: Path, reference: Path, settings: Settings, scope: str) -> None:
    payload = protocol_payload(reference, settings, scope)
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"{path} differs from immutable protocol")
        return
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    write_json(path, payload)


def load_protocol(out: Path) -> tuple[Settings, dict]:
    payload = json.loads((out / "protocol.json").read_text())
    raw = dict(payload["settings"])
    raw["model_seeds"] = tuple(raw["model_seeds"])
    raw["radii_schedule"] = tuple(tuple(row) for row in raw["radii_schedule"])
    return Settings(**raw), payload


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[robust-r4] seed={seed}: complete", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, provenance = load_model(Path(protocol["parent"]), seed, device)
    sources = protocol["artifacts"][str(seed)]
    oracle = torch.load(sources["oracle"], map_location="cpu", weights_only=True)
    reference_task = torch.load(sources["initials"], map_location="cpu", weights_only=True)
    n = settings.starts_per_group
    prior = reference_task["initial_z"][:n].float()
    endpoints = oracle["best_soft"]["z"][:n].float()
    combined = torch.cat([prior, endpoints])
    exact_oracle = oracle["best_soft"]["iou"][:n] == 1
    tasks = {}
    for pattern in protocol["patterns"]:
        path = root / f"task_{pattern}.pt"
        if path.exists():
            result = torch.load(path, map_location="cpu", weights_only=True)
        else:
            result = evolutionary_search_robust(
                model, combined, pattern, device,
                radius=settings.radius, radii_schedule=settings.radii_schedule,
                screen_steps=settings.screen_steps, refine_steps=settings.refine_steps,
                refine_replicates=settings.refine_replicates, finalists=settings.finalists,
                final_eval_steps=settings.final_eval_steps,
                min_improvement=settings.min_improvement, seed=settings.seed + seed,
            )
            result.update({"model_seed": seed, "pattern": pattern, "uses_gold": False,
                           "checkpoint_sha256": provenance["checkpoint_sha256"],
                           "oracle_sha256": sources["oracle_sha256"]})
            atomic_save(path, result)
        tasks[pattern] = result
    summary = {"model_seed": seed, "oracle_exact_count": int(exact_oracle.sum()),
               "tasks": {}, "provenance": provenance}
    for pattern, result in tasks.items():
        summary["tasks"][pattern] = {
            "prior": analyze_group(result, slice(0, n), torch.ones(n, dtype=torch.bool), endpoints),
            "oracle_z_star": analyze_group(result, slice(n, 2 * n), exact_oracle, endpoints),
        }
    write_json(summary_path, summary)
    print(f"[robust-r4] seed={seed}: complete", flush=True)


def aggregate(out: Path) -> dict[str, Any]:
    settings, protocol = load_protocol(out)
    records = [json.loads((out / f"seed_{seed}/summary.json").read_text())
               for seed in settings.model_seeds]
    result: dict[str, Any] = {"protocol": protocol, "groups": {}}
    metrics = ("exact_fraction", "gold_iou", "paired_oracle_distance",
               "task_bce_2000", "task_accuracy_2000")
    for group in protocol["groups"]:
        result["groups"][group] = {}
        for stage in ("initial", "final"):
            nested = []
            for record in records:
                rows = [record["tasks"][pattern][group][stage] for pattern in protocol["patterns"]]
                nested.append({metric: float(np.mean([
                    row[metric] if metric == "exact_fraction" else row[metric]["mean"]
                    for row in rows])) for metric in metrics})
            result["groups"][group][stage] = {
                metric: ci([row[metric] for row in nested]) for metric in metrics
            }
        nested_search = []
        for record in records:
            rows = [record["tasks"][pattern][group]["search"] for pattern in protocol["patterns"]]
            nested_search.append({
                "ever_accepted_fraction": float(np.mean([row["ever_accepted_fraction"] for row in rows])),
                "mean_acceptances_per_start": float(np.mean([row["mean_acceptances_per_start"] for row in rows])),
            })
        result["groups"][group]["search"] = {
            metric: ci([row[metric] for row in nested_search]) for metric in nested_search[0]
        }
    return result


def fmt(value: dict) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}; {value['ci95'][1]:.3f}]"


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    lines = ["# Pattern R=4: robust hard-mask sampling", "",
             "Mutation is accepted only if it beats the paired parent in both independent "
             "2000-step MLP fits. Gold is post-hoc only.", "",
             "| Старт | Стадия | Exact | Gold IoU | Fresh accuracy | Fresh BCE | L2 до oracle |",
             "|---|---|---:|---:|---:|---:|---:|"]
    labels = {"prior": "Prior", "oracle_z_star": "Oracle z*"}
    for group, label in labels.items():
        for stage in ("initial", "final"):
            row = summary["groups"][group][stage]
            lines.append(f"| {label} | {stage} | {fmt(row['exact_fraction'])} | "
                         f"{fmt(row['gold_iou'])} | {fmt(row['task_accuracy_2000'])} | "
                         f"{fmt(row['task_bce_2000'])} | {fmt(row['paired_oracle_distance'])} |")
    lines += ["", "| Старт | Ever accepted | Acceptances/start |", "|---|---:|---:|"]
    for group, label in labels.items():
        row = summary["groups"][group]["search"]
        lines.append(f"| {label} | {fmt(row['ever_accepted_fraction'])} | "
                     f"{fmt(row['mean_acceptances_per_start'])} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"[robust-r4] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--scope", choices=("smoke", "pair", "full"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    default_out = PAIR_OUT if args.scope == "pair" else FULL_OUT
    if args.scope == "smoke":
        default_out = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_sampling_robust_smoke"
    out = (args.out or default_out).resolve()
    reference = args.reference.resolve()
    requested = settings_for(args.scope)
    if args.stage == "prepare":
        prepare(out, reference, requested, args.scope)
        print(f"[robust-r4] protocol -> {out / 'protocol.json'}", flush=True)
        return
    settings, protocol = load_protocol(out)
    if settings != requested or protocol["scope"] != args.scope:
        raise ValueError("CLI scope differs from saved protocol")
    if args.stage == "run":
        if args.model_seed is None:
            parser.error("--model-seed is required")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        run_seed(out, args.model_seed, args.device)
    else:
        report(out)


if __name__ == "__main__":
    main()

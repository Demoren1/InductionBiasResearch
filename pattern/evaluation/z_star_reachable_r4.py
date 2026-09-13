"""Matched R=4 test: Gold oracle and task-z from the same latent starts.

For every frozen VAE, the experiment projects 64 saved prior starts into the
R=4 ball.  A Gold-guided oracle and the label-only task-z search receive those
same starts.  Task-z is additionally launched from the oracle endpoints, which
tests whether an already ideal latent is stable under the task objective.
"""

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

from data.generate import ideal_mask  # noqa: E402
from evaluation.oracle_ideal import optimize_ideal  # noqa: E402
from evaluation.single_z_task import optimize_task_z  # noqa: E402
from evaluation.z_star_noise import (  # noqa: E402
    _hard_stats,
    atomic_save,
    checkpoint_paths,
    ci,
    load_model,
    project,
    sha256_file,
    write_json,
)


DEFAULT_PARENT = PATTERN_ROOT / "outputs/decoder_agreement/multiseed32_tuned_20260912"
DEFAULT_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_multiseed8_20260912"


@dataclass(frozen=True)
class Settings:
    model_seeds: tuple[int, ...] = tuple(range(186, 202, 2))
    starts: int = 64
    radius: float = 4.0
    temperature: float = 0.5
    oracle_steps: int = 2000
    oracle_lr: float = 0.03
    task_outer_steps: int = 30
    task_warmup_steps: int = 300
    task_grad_steps: int = 100
    task_z_lr: float = 0.05
    task_seed: int = 20260912

    @classmethod
    def smoke(cls) -> "Settings":
        return replace(cls(), model_seeds=(186,), starts=4, oracle_steps=300,
                       task_outer_steps=1, task_warmup_steps=1, task_grad_steps=1)


def protocol_payload(parent: Path, settings: Settings, smoke: bool) -> dict[str, Any]:
    split_path = PATTERN_ROOT / "outputs/ood/split_seed_42/split.json"
    split = json.loads(split_path.read_text())
    checkpoints = {}
    for seed in settings.model_seeds:
        checkpoint, metadata, agreement = checkpoint_paths(parent, seed)
        checkpoints[str(seed)] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "metadata_sha256": sha256_file(metadata),
            "agreement_initials": str(agreement),
            "agreement_initials_sha256": sha256_file(agreement),
        }
    sources = [Path(__file__), HERE / "z_star_noise.py", HERE / "oracle_ideal.py",
               HERE / "single_z_task.py", HERE / "decoder_agreement.py"]
    return {
        "experiment": "matched oracle-vs-task-z reachability inside radius 4",
        "settings": asdict(settings), "smoke": smoke,
        "parent": str(parent), "checkpoints": checkpoints,
        "heldout_patterns": split["test_patterns"],
        "task_split": str(split_path), "task_split_sha256": sha256_file(split_path),
        "initialization": (
            "the first 64 saved agreement initial_z1 codes, projected into the R=4 ball"
        ),
        "groups": {
            "same_start": "task-z begins at the same projected prior code as the Gold oracle",
            "oracle_endpoint": "task-z begins at that start's Gold-oracle best-soft endpoint",
        },
        "gold_access": "only oracle construction and post-hoc mask/distance diagnostics",
        "task_z_access": "task labels and search validation only; no Gold mask",
        "independence_unit": "frozen VAE seed; tasks and starts are nested within seed",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
    }


def prepare(out: Path, parent: Path, settings: Settings, smoke: bool) -> None:
    payload = protocol_payload(parent, settings, smoke)
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"{path} differs from the requested immutable protocol")
        return
    out.mkdir(parents=True)
    (out / "logs").mkdir()
    write_json(path, payload)


def load_protocol(out: Path) -> tuple[Settings, dict]:
    payload = json.loads((out / "protocol.json").read_text())
    values = dict(payload["settings"])
    values["model_seeds"] = tuple(values["model_seeds"])
    return Settings(**values), payload


def describe(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().cpu()
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "values": values.tolist()}


def analyze_group(codes: torch.Tensor, masks: torch.Tensor, group_slice: slice,
                  paired_oracle: torch.Tensor, exact_oracle: torch.Tensor,
                  target: torch.Tensor) -> dict[str, Any]:
    z = codes[group_slice].float().cpu()
    group_masks = masks[group_slice].cpu()
    iou, exact = _hard_stats(group_masks, target)
    nearest = torch.cdist(z, paired_oracle[exact_oracle]).min(dim=1).values
    paired = (z - paired_oracle).norm(dim=1)
    eligible_iou = iou[exact_oracle]
    eligible_exact = exact[exact_oracle]
    return {
        "count": len(z), "gold_iou": describe(iou),
        "exact_count": int(exact.sum()), "exact_fraction": float(exact.float().mean()),
        "eligible_count": int(exact_oracle.sum()),
        "eligible_gold_iou": describe(eligible_iou),
        "eligible_exact_count": int(eligible_exact.sum()),
        "eligible_exact_fraction": float(eligible_exact.float().mean()),
        "distance_to_paired_oracle": describe(paired),
        "distance_to_nearest_exact_oracle": describe(nearest),
    }


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[reachable-r4] seed={seed}: complete", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, provenance = load_model(Path(protocol["parent"]), seed, device)
    agreement_path = Path(protocol["checkpoints"][str(seed)]["agreement_initials"])
    agreement = torch.load(agreement_path, map_location="cpu", weights_only=True)
    same_start = project(agreement["initial_z1"][:settings.starts].float(), settings.radius)

    oracle_path = root / "oracle_same_start.pt"
    if oracle_path.exists():
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
    else:
        oracle = optimize_ideal(
            model, same_start.to(device), ideal_mask().float().to(device),
            steps=settings.oracle_steps, lr=settings.oracle_lr,
            temperature=settings.temperature, radius=settings.radius,
        )
        oracle.update({"model_seed": seed, "uses_gold": True,
                       "checkpoint_sha256": provenance["checkpoint_sha256"],
                       "agreement_initials_sha256": sha256_file(agreement_path)})
        atomic_save(oracle_path, oracle)
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)

    endpoints = oracle["best_soft"]["z"].float()
    exact_oracle = oracle["best_soft"]["iou"] == 1
    if not exact_oracle.any():
        raise RuntimeError(f"seed {seed}: matched Gold oracle found no exact endpoint")
    combined = torch.cat([same_start, endpoints])
    task_results = {}
    for pattern in protocol["heldout_patterns"]:
        path = root / f"task_{pattern}.pt"
        if path.exists():
            result = torch.load(path, map_location="cpu", weights_only=True)
        else:
            result = optimize_task_z(
                model, combined, pattern, device,
                outer_steps=settings.task_outer_steps,
                warmup_steps=settings.task_warmup_steps,
                grad_steps=settings.task_grad_steps,
                z_lr=settings.task_z_lr, radius=settings.radius,
                temperature=settings.temperature, seed=settings.task_seed,
            )
            result.update({"model_seed": seed, "pattern": pattern, "uses_gold": False,
                           "checkpoint_sha256": provenance["checkpoint_sha256"],
                           "oracle_sha256": sha256_file(oracle_path)})
            atomic_save(path, result)
        task_results[pattern] = result

    target = ideal_mask().float()
    groups = {"same_start": slice(0, settings.starts),
              "oracle_endpoint": slice(settings.starts, 2 * settings.starts)}
    stages = {"initial": ("initial_z", "initial_masks"),
              "final": ("final_z", "final_masks"),
              "best_query": ("best_val_z", "best_val_masks")}
    summary = {
        "model_seed": seed,
        "oracle": {
            "starts": settings.starts,
            "exact_count": int(exact_oracle.sum()),
            "exact_fraction": float(exact_oracle.float().mean()),
            "gold_iou": describe(oracle["best_soft"]["iou"]),
            "z_norm": describe(endpoints.norm(dim=1)),
        },
        "tasks": {},
        "provenance": {**provenance, "oracle_sha256": sha256_file(oracle_path)},
    }
    for pattern, result in task_results.items():
        summary["tasks"][pattern] = {}
        for group_name, group_slice in groups.items():
            summary["tasks"][pattern][group_name] = {
                stage: analyze_group(result[z_key], result[mask_key], group_slice,
                                     endpoints, exact_oracle, target)
                for stage, (z_key, mask_key) in stages.items()
            }
    write_json(summary_path, summary)
    print(f"[reachable-r4] seed={seed}: {int(exact_oracle.sum())}/{settings.starts} exact oracle", flush=True)


def aggregate(out: Path) -> dict[str, Any]:
    settings, protocol = load_protocol(out)
    records = [json.loads((out / f"seed_{seed}/summary.json").read_text())
               for seed in settings.model_seeds]
    payload: dict[str, Any] = {
        "protocol": protocol,
        "oracle": {
            "exact_fraction": ci([row["oracle"]["exact_fraction"] for row in records]),
            "gold_iou": ci([row["oracle"]["gold_iou"]["mean"] for row in records]),
            "z_norm": ci([row["oracle"]["z_norm"]["mean"] for row in records]),
        },
        "groups": {},
    }
    for group in ("same_start", "oracle_endpoint"):
        payload["groups"][group] = {}
        for stage in ("initial", "final", "best_query"):
            per_seed = []
            for record in records:
                rows = [record["tasks"][pattern][group][stage]
                        for pattern in protocol["heldout_patterns"]]
                per_seed.append({
                    "exact_fraction": float(np.mean([row["exact_fraction"] for row in rows])),
                    "eligible_exact_fraction": float(np.mean([
                        row["eligible_exact_fraction"] for row in rows])),
                    "gold_iou": float(np.mean([row["gold_iou"]["mean"] for row in rows])),
                    "paired_distance": float(np.mean([
                        row["distance_to_paired_oracle"]["mean"] for row in rows])),
                    "nearest_exact_distance": float(np.mean([
                        row["distance_to_nearest_exact_oracle"]["mean"] for row in rows])),
                })
            payload["groups"][group][stage] = {
                metric: ci([row[metric] for row in per_seed]) for metric in per_seed[0]
            }
    return payload


def fmt(value: dict[str, Any]) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}; {value['ci95'][1]:.3f}]"


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    lines = [
        "# Pattern-8: matched reachability при R=4", "",
        "Gold-oracle и task-aware single-z получают одни и те же 64 prior-старта, "
        "спроецированные в шар R=4. Дополнительно task-z стартует непосредственно из "
        "Gold-oracle endpoint для проверки устойчивости уже найденного z*.", "",
        f"Gold-oracle на тех же стартах: exact **{fmt(summary['oracle']['exact_fraction'])}**, "
        f"IoU **{fmt(summary['oracle']['gold_iou'])}**, норма z **{fmt(summary['oracle']['z_norm'])}**.", "",
        "В таблице exact считается только среди тех пар стартов, для которых matched Gold-oracle "
        "действительно нашёл точную hard-маску.", "",
        "| Старт task-z | Стадия | Exact ideal | Gold IoU | L2 до paired oracle z* | L2 до ближайшего exact oracle |",
        "|---|---|---:|---:|---:|---:|",
    ]
    labels = {"same_start": "Тот же prior-старт", "oracle_endpoint": "Уже в z*"}
    for group, label in labels.items():
        for stage, stage_label in (("initial", "до"), ("final", "final"),
                                   ("best_query", "best-query")):
            row = summary["groups"][group][stage]
            lines.append(
                f"| {label} | {stage_label} | {fmt(row['eligible_exact_fraction'])} | "
                f"{fmt(row['gold_iou'])} | {fmt(row['paired_distance'])} | "
                f"{fmt(row['nearest_exact_distance'])} |"
            )
    lines += ["", "ДИ рассчитаны по 8 независимо обученным frozen VAE; задачи и 64 старта "
              "вложены внутрь VAE seed. Gold не используется task-z оптимизатором.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(f"[reachable-r4] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out, parent = args.out.resolve(), args.parent.resolve()
    requested = Settings.smoke() if args.smoke else Settings()
    if args.stage == "prepare":
        prepare(out, parent, requested, args.smoke)
        print(f"[reachable-r4] protocol -> {out / 'protocol.json'}", flush=True)
        return
    settings, _ = load_protocol(out)
    if settings != requested:
        raise ValueError("CLI smoke/full mode differs from saved protocol")
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

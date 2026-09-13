"""Multiseed R=4 hard-mask evolutionary search from prior codes and oracle z*."""

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
from evaluation.hard_mask_sampling import evolutionary_search  # noqa: E402
from evaluation.z_star_noise import (  # noqa: E402
    _hard_stats,
    atomic_save,
    ci,
    load_model,
    sha256_file,
    write_json,
)


REFERENCE = PATTERN_ROOT / "outputs/z_star_reachable/r4_multiseed8_20260912"
DEFAULT_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_sampling_multiseed8_20260912"


def full_schedule() -> tuple[tuple[float, ...], ...]:
    broad = (.5,) * 4 + (1.,) * 4 + (1.5,) * 4 + (2.,) * 4
    medium = (.25,) * 4 + (.5,) * 4 + (1.,) * 4 + (1.5,) * 4
    local = (.1,) * 4 + (.25,) * 4 + (.5,) * 4 + (1.,) * 4
    return (broad, broad, medium, medium, medium, local, local, local)


@dataclass(frozen=True)
class Settings:
    model_seeds: tuple[int, ...] = tuple(range(186, 202, 2))
    starts_per_group: int = 64
    radius: float = 4.0
    radii_schedule: tuple[tuple[float, ...], ...] = full_schedule()
    screen_steps: int = 400
    refine_steps: int = 1000
    finalists: int = 4
    final_eval_steps: int = 2000
    seed: int = 20260920

    @classmethod
    def smoke(cls) -> "Settings":
        return replace(cls(), model_seeds=(186,), starts_per_group=2,
                       radii_schedule=((.25, .5, 1., 2.),),
                       screen_steps=2, refine_steps=3, finalists=2,
                       final_eval_steps=3)


def protocol_payload(reference: Path, settings: Settings, smoke: bool) -> dict[str, Any]:
    source_protocol = json.loads((reference / "protocol.json").read_text())
    patterns = source_protocol["heldout_patterns"][:1] if smoke else source_protocol["heldout_patterns"]
    sources = [Path(__file__), HERE / "hard_mask_sampling.py", HERE / "single_z_task.py",
               HERE / "z_star_noise.py", HERE / "decoder_agreement.py"]
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
        "experiment": "hard-mask evolutionary latent sampling versus gradient task-z",
        "settings": asdict(settings), "smoke": smoke, "patterns": patterns,
        "reference": str(reference),
        "reference_protocol_sha256": sha256_file(reference / "protocol.json"),
        "parent": source_protocol["parent"], "artifacts": artifacts,
        "groups": ["prior", "oracle_z_star"],
        "selection": (
            "At every generation parent and mutations are evaluated with paired MLP initialization/data; "
            "the parent is retained unless a mutation has lower binary-mask validation BCE."
        ),
        "gold_access": "Gold is excluded from search and used only for post-hoc exact/IoU diagnostics",
        "independence_unit": "frozen VAE seed; tasks, starts, and proposals are nested",
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
    }


def prepare(out: Path, reference: Path, settings: Settings, smoke: bool) -> None:
    payload = protocol_payload(reference, settings, smoke)
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


def describe(values: torch.Tensor) -> dict[str, Any]:
    values = values.float().cpu()
    return {"mean": float(values.mean()), "min": float(values.min()),
            "max": float(values.max()), "values": values.tolist()}


def analyze_group(result: dict, group_slice: slice, eligible: torch.Tensor,
                  paired_oracle: torch.Tensor) -> dict:
    target = ideal_mask().float()
    rows = {}
    for stage in ("initial", "final"):
        z = result[f"{stage}_z"][group_slice][eligible].float()
        masks = result[f"{stage}_masks"][group_slice][eligible]
        iou, exact = _hard_stats(masks, target)
        eval_bce = result[f"final_eval_{stage}_bce"][group_slice][eligible]
        eval_accuracy = result[f"final_eval_{stage}_accuracy"][group_slice][eligible]
        rows[stage] = {
            "count": len(z), "exact_count": int(exact.sum()),
            "exact_fraction": float(exact.float().mean()), "gold_iou": describe(iou),
            "paired_oracle_distance": describe((z - paired_oracle[eligible]).norm(dim=1)),
            "task_bce_2000": describe(eval_bce), "task_accuracy_2000": describe(eval_accuracy),
        }
    accepted = torch.stack([generation["accepted"][group_slice][eligible]
                            for generation in result["history"]])
    different = [generation["candidate_different_support_fraction"]
                 for generation in result["history"]]
    unique = [float(np.mean(generation["unique_mask_counts"][group_slice]))
              for generation in result["history"]]
    rows["search"] = {
        "accepted_fraction_per_generation": accepted.float().mean(1).tolist(),
        "ever_accepted_fraction": float(accepted.any(0).float().mean()),
        "mean_acceptances_per_start": float(accepted.sum(0).float().mean()),
        "candidate_different_support_fraction_all_parents": different,
        "mean_unique_masks_per_generation": unique,
    }
    return rows


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[sampling-r4] seed={seed}: complete", flush=True)
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
            result = evolutionary_search(
                model, combined, pattern, device,
                radius=settings.radius, radii_schedule=settings.radii_schedule,
                screen_steps=settings.screen_steps, refine_steps=settings.refine_steps,
                finalists=settings.finalists, final_eval_steps=settings.final_eval_steps,
                seed=settings.seed + seed,
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
    print(f"[sampling-r4] seed={seed}: complete", flush=True)


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
            per_seed = []
            for record in records:
                rows = [record["tasks"][pattern][group][stage]
                        for pattern in protocol["patterns"]]
                per_seed.append({
                    metric: float(np.mean([
                        row[metric] if metric == "exact_fraction" else row[metric]["mean"]
                        for row in rows])) for metric in metrics
                })
            result["groups"][group][stage] = {
                metric: ci([row[metric] for row in per_seed]) for metric in metrics
            }
        per_seed_search = []
        for record in records:
            rows = [record["tasks"][pattern][group]["search"] for pattern in protocol["patterns"]]
            per_seed_search.append({
                "ever_accepted_fraction": float(np.mean([row["ever_accepted_fraction"] for row in rows])),
                "mean_acceptances_per_start": float(np.mean([row["mean_acceptances_per_start"] for row in rows])),
            })
        result["groups"][group]["search"] = {
            metric: ci([row[metric] for row in per_seed_search])
            for metric in per_seed_search[0]
        }
    return result


def fmt(value: dict) -> str:
    return f"{value['mean']:.3f} [{value['ci95'][0]:.3f}; {value['ci95'][1]:.3f}]"


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    labels = {"prior": "Prior", "oracle_z_star": "Oracle z*"}
    colors = {"prior": "#347f8b", "oracle_z_star": "#c76e42"}
    for group in labels:
        before = summary["groups"][group]["initial"]
        after = summary["groups"][group]["final"]
        axes[0].plot([0, 1], [before["exact_fraction"]["mean"], after["exact_fraction"]["mean"]],
                     marker="o", label=labels[group], color=colors[group])
        axes[1].plot([0, 1], [before["gold_iou"]["mean"], after["gold_iou"]["mean"]],
                     marker="o", color=colors[group])
        axes[2].plot([0, 1], [before["task_accuracy_2000"]["mean"],
                             after["task_accuracy_2000"]["mean"]],
                     marker="o", color=colors[group])
    axes[0].set(ylabel="Exact ideal fraction", ylim=(-.02, 1.02))
    axes[1].set(ylabel="Gold IoU", ylim=(.45, 1.02))
    axes[2].set(ylabel="Fresh-MLP test accuracy", ylim=(.8, 1.0))
    for axis in axes:
        axis.set(xticks=[0, 1], xticklabels=["Initial", "Sampling final"])
        axis.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle("Hard-mask evolutionary sampling in frozen VAE latent space (R=4)")
    fig.savefig(out / "sampling_summary.png", dpi=170)
    fig.savefig(out / "sampling_summary.pdf")
    plt.close(fig)

    lines = ["# Pattern R=4: hard-mask evolutionary sampling", "",
             "Frozen VAE, 64 starts per group, 16 mutations, 8 generations. Selection uses only "
             "paired binary-mask MLP validation BCE; Gold is post-hoc.", "",
             "| Старт | Стадия | Exact ideal | Gold IoU | Fresh MLP accuracy | Fresh MLP BCE | L2 до paired oracle |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for group, label in labels.items():
        for stage in ("initial", "final"):
            row = summary["groups"][group][stage]
            lines.append(f"| {label} | {stage} | {fmt(row['exact_fraction'])} | "
                         f"{fmt(row['gold_iou'])} | {fmt(row['task_accuracy_2000'])} | "
                         f"{fmt(row['task_bce_2000'])} | {fmt(row['paired_oracle_distance'])} |")
    lines += ["", "| Старт | Хотя бы одно принятое изменение | Принято изменений на старт |",
              "|---|---:|---:|"]
    for group, label in labels.items():
        row = summary["groups"][group]["search"]
        lines.append(f"| {label} | {fmt(row['ever_accepted_fraction'])} | "
                     f"{fmt(row['mean_acceptances_per_start'])} |")
    lines += ["", "![Sampling summary](sampling_summary.png)", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print(f"[sampling-r4] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out, reference = args.out.resolve(), args.reference.resolve()
    requested = Settings.smoke() if args.smoke else Settings()
    if args.stage == "prepare":
        prepare(out, reference, requested, args.smoke)
        print(f"[sampling-r4] protocol -> {out / 'protocol.json'}", flush=True)
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

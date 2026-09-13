"""Robust hard-mask latent sampling on all 32 saved VAE pairs (64 decoders)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
REPO_ROOT = PATTERN_ROOT.parent
sys.path.insert(0, str(PATTERN_ROOT))

from data.generate import ideal_mask  # noqa: E402
from evaluation.hard_mask_sampling_robust import evolutionary_search_robust  # noqa: E402
from evaluation.oracle_ideal import optimize_ideal  # noqa: E402
from evaluation.z_star_noise import (  # noqa: E402
    atomic_save,
    ci,
    sha256_file,
    write_json,
)
from evaluation.z_star_sampling_r4 import analyze_group, full_schedule  # noqa: E402
from models.cvae import CVAE  # noqa: E402
import config  # noqa: E402


DEFAULT_PARENT = PATTERN_ROOT / "outputs/decoder_agreement/multiseed32_tuned_20260912"
DEFAULT_OUT = PATTERN_ROOT / "outputs/z_star_reachable/r4_hard_sampling_robust_pairs32_20260913"


@dataclass(frozen=True)
class Settings:
    pair_starts: tuple[int, ...] = tuple(range(186, 250, 2))
    starts_per_group: int = 64
    oracle_steps: int = 2000
    oracle_lr: float = 0.03
    temperature: float = 0.5
    radius: float = 4.0
    radii_schedule: tuple[tuple[float, ...], ...] = full_schedule()
    screen_steps: int = 400
    refine_steps: int = 2000
    refine_replicates: int = 2
    finalists: int = 4
    final_eval_steps: int = 2000
    min_improvement: float = 1e-4
    seed: int = 20260922

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((seed, seed + 1) for seed in self.pair_starts)

    @property
    def model_seeds(self) -> tuple[int, ...]:
        return tuple(seed for pair in self.pairs for seed in pair)


def _paths(parent: Path, seed: int) -> tuple[Path, Path, Path, str]:
    first = seed if seed % 2 == 0 else seed - 1
    pair = parent / f"pair_{first}_{first + 1}"
    index = "1" if seed == first else "2"
    return (
        pair / f"vae_{seed}/cvae_best.pt",
        pair / f"vae_{seed}/cvae_meta.pt",
        pair / "optimization.pt",
        index,
    )


def _load_model(parent: Path, seed: int, device: torch.device) -> tuple[CVAE, dict]:
    checkpoint, metadata_path, initials_path, index = _paths(parent, seed)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    checkpoint_hash = sha256_file(checkpoint)
    if metadata["seed"] != seed or metadata["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError(f"invalid checkpoint provenance for seed {seed}")
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    return model, {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "agreement_initials": str(initials_path),
        "agreement_initials_sha256": sha256_file(initials_path),
        "initial_z_key": f"initial_z{index}",
    }


def _protocol(parent: Path, settings: Settings) -> dict[str, Any]:
    split_path = PATTERN_ROOT / "outputs/ood/split_seed_42/split.json"
    patterns = json.loads(split_path.read_text())["test_patterns"]
    artifacts = {}
    for seed in settings.model_seeds:
        checkpoint, metadata, initials, index = _paths(parent, seed)
        if not checkpoint.is_file() or not metadata.is_file() or not initials.is_file():
            raise FileNotFoundError(f"incomplete saved decoder for seed {seed}")
        artifacts[str(seed)] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "metadata": str(metadata), "metadata_sha256": sha256_file(metadata),
            "initials": str(initials), "initials_sha256": sha256_file(initials),
            "initial_z_key": f"initial_z{index}",
        }
    sources = [Path(__file__), HERE / "hard_mask_sampling_robust.py",
               HERE / "hard_mask_sampling.py", HERE / "oracle_ideal.py",
               HERE / "z_star_sampling_r4.py"]
    return {
        "experiment": "R=4 robust hard-mask sampling on 32 VAE pairs / 64 frozen decoders",
        "settings": asdict(settings), "pairs": [list(pair) for pair in settings.pairs],
        "model_seeds": list(settings.model_seeds), "patterns": patterns,
        "parent": str(parent), "artifacts": artifacts,
        "oracle": {
            "uses_gold": True,
            "objective": "MSE to analytic ideal soft-top-32 after Hungarian column alignment",
            "selection": "best soft objective; only exact hard-mask endpoints enter z-star group",
        },
        "sampling": {
            "uses_gold": False,
            "selection": "support-changing mutation beats parent by >1e-4 in both replicated 2000-step MLP fits",
            "final_evaluation": "independent paired 2000-step MLP fit",
        },
        "independence_unit": "VAE pair; two decoders, tasks and latent starts are nested",
        "task_split": str(split_path), "task_split_sha256": sha256_file(split_path),
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in sources},
    }


def prepare(out: Path, parent: Path, settings: Settings) -> None:
    payload = _protocol(parent, settings)
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise FileExistsError(f"{path} differs from immutable protocol")
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    write_json(path, payload)


def load_protocol(out: Path) -> tuple[Settings, dict]:
    payload = json.loads((out / "protocol.json").read_text())
    raw = dict(payload["settings"])
    raw["pair_starts"] = tuple(raw["pair_starts"])
    raw["radii_schedule"] = tuple(tuple(row) for row in raw["radii_schedule"])
    return Settings(**raw), payload


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    if seed not in settings.model_seeds:
        raise ValueError(f"seed {seed} is absent from protocol")
    root = out / f"seed_{seed}"
    summary_path = root / "summary.json"
    if summary_path.exists():
        print(f"[pairs32] seed={seed}: complete", flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, provenance = _load_model(Path(protocol["parent"]), seed, device)
    initials_record = torch.load(provenance["agreement_initials"], map_location="cpu", weights_only=True)
    initial_z = initials_record[provenance["initial_z_key"]][:settings.starts_per_group].float().to(device)

    oracle_path = root / "oracle_same_start.pt"
    if oracle_path.exists():
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
    else:
        oracle = optimize_ideal(
            model, initial_z, ideal_mask().to(device), steps=settings.oracle_steps,
            lr=settings.oracle_lr, temperature=settings.temperature, radius=settings.radius,
        )
        oracle["model_seed"] = seed
        oracle["provenance"] = provenance
        atomic_save(oracle_path, oracle)
    prior = initial_z.cpu()
    endpoints = oracle["best_soft"]["z"][:settings.starts_per_group].float()
    eligible = oracle["best_soft"]["iou"][:settings.starts_per_group] == 1
    combined = torch.cat([prior, endpoints])

    tasks = {}
    for pattern in protocol["patterns"]:
        path = root / f"task_{pattern}.pt"
        if path.exists():
            result = torch.load(path, map_location="cpu", weights_only=True)
        else:
            result = evolutionary_search_robust(
                model, combined, pattern, device, radius=settings.radius,
                radii_schedule=settings.radii_schedule, screen_steps=settings.screen_steps,
                refine_steps=settings.refine_steps, refine_replicates=settings.refine_replicates,
                finalists=settings.finalists, final_eval_steps=settings.final_eval_steps,
                min_improvement=settings.min_improvement, seed=settings.seed + seed,
            )
            result.update({"model_seed": seed, "pattern": pattern, "uses_gold": False,
                           "checkpoint_sha256": provenance["checkpoint_sha256"],
                           "oracle_sha256": sha256_file(oracle_path)})
            atomic_save(path, result)
        tasks[pattern] = result

    n = settings.starts_per_group
    summary = {"model_seed": seed, "oracle_exact_count": int(eligible.sum()),
               "tasks": {}, "provenance": provenance}
    for pattern, result in tasks.items():
        summary["tasks"][pattern] = {
            "prior": analyze_group(result, slice(0, n), torch.ones(n, dtype=torch.bool), endpoints),
            "oracle_z_star": analyze_group(result, slice(n, 2 * n), eligible, endpoints),
        }
    write_json(summary_path, summary)
    print(f"[pairs32] seed={seed}: complete; exact oracle={int(eligible.sum())}/{n}", flush=True)


def _decoder_nested(record: dict, patterns: list[str], group: str, stage: str,
                    metrics: tuple[str, ...]) -> dict[str, float]:
    rows = [record["tasks"][pattern][group][stage] for pattern in patterns]
    return {metric: float(np.mean([
        row[metric] if metric == "exact_fraction" else row[metric]["mean"] for row in rows
    ])) for metric in metrics}


def aggregate(out: Path) -> dict[str, Any]:
    settings, protocol = load_protocol(out)
    records = {seed: json.loads((out / f"seed_{seed}/summary.json").read_text())
               for seed in settings.model_seeds}
    metrics = ("exact_fraction", "gold_iou", "paired_oracle_distance",
               "task_bce_2000", "task_accuracy_2000")
    result: dict[str, Any] = {"protocol": protocol, "groups": {}, "oracle": {}}
    result["oracle"]["exact_endpoints"] = {
        "count": sum(record["oracle_exact_count"] for record in records.values()),
        "total": len(records) * settings.starts_per_group,
    }
    result["oracle"]["exact_endpoints"]["fraction"] = (
        result["oracle"]["exact_endpoints"]["count"] / result["oracle"]["exact_endpoints"]["total"])
    for group in ("prior", "oracle_z_star"):
        result["groups"][group] = {}
        for stage in ("initial", "final"):
            per_pair = []
            for pair in settings.pairs:
                decoder_rows = [_decoder_nested(records[seed], protocol["patterns"], group, stage, metrics)
                                for seed in pair]
                per_pair.append({metric: float(np.mean([row[metric] for row in decoder_rows]))
                                 for metric in metrics})
            result["groups"][group][stage] = {
                metric: ci([row[metric] for row in per_pair]) for metric in metrics
            }
        per_pair_search = []
        for pair in settings.pairs:
            decoder_rows = []
            for seed in pair:
                rows = [records[seed]["tasks"][pattern][group]["search"]
                        for pattern in protocol["patterns"]]
                decoder_rows.append({
                    "ever_accepted_fraction": float(np.mean([row["ever_accepted_fraction"] for row in rows])),
                    "mean_acceptances_per_start": float(np.mean([row["mean_acceptances_per_start"] for row in rows])),
                })
            per_pair_search.append({key: float(np.mean([row[key] for row in decoder_rows]))
                                    for key in decoder_rows[0]})
        result["groups"][group]["search"] = {
            key: ci([row[key] for row in per_pair_search]) for key in per_pair_search[0]
        }
    return result


def _fmt(value: dict) -> str:
    return f"{value['mean']:.4f} [{value['ci95'][0]:.4f}; {value['ci95'][1]:.4f}]"


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    rows = [
        "# Robust sampling на 32 парах VAE (pattern-8)", "",
        "Дата: 2026-09-13.", "",
        "Проверены 32 независимо обученные пары (64 замороженных декодера). "
        "Доверительные интервалы построены по 32 парам; два декодера, четыре held-out задачи "
        "и latent-старты вложены внутрь пары.", "",
        "Gold используется только для построения диагностического `z*` и постфактум-метрик. "
        "Сам sampling видит только качество MLP с жёсткой exact-32 маской.", "",
        "| Старт | Стадия | Exact ideal | Gold IoU | Test accuracy | Test BCE | L2 до paired z* |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"prior": "Prior", "oracle_z_star": "Oracle z*"}
    for group, label in labels.items():
        for stage in ("initial", "final"):
            item = summary["groups"][group][stage]
            rows.append(f"| {label} | {stage} | {_fmt(item['exact_fraction'])} | "
                        f"{_fmt(item['gold_iou'])} | {_fmt(item['task_accuracy_2000'])} | "
                        f"{_fmt(item['task_bce_2000'])} | {_fmt(item['paired_oracle_distance'])} |")
    rows += ["", "| Старт | Хотя бы одно принятие | Принято мутаций / start |",
             "|---|---:|---:|"]
    for group, label in labels.items():
        item = summary["groups"][group]["search"]
        rows.append(f"| {label} | {_fmt(item['ever_accepted_fraction'])} | "
                    f"{_fmt(item['mean_acceptances_per_start'])} |")
    exact = summary["oracle"]["exact_endpoints"]
    rows += ["", "## Контроль достижимости", "",
             f"Oracle нашёл exact ideal для **{exact['count']}/{exact['total']}** стартов "
             f"({exact['fraction']:.2%}) при `R=4`.", "",
             "## Вывод", "",
             "Вывод заполняется автоматически после завершения расчёта с опорой на "
             "`summary.json`; сырые тензоры каждого поиска сохранены в `seed_*/task_*.pt`.", ""]
    (out / "RESULTS.md").write_text("\n".join(rows))
    print(f"[pairs32] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    out, parent = args.out.resolve(), args.parent.resolve()
    if args.stage == "prepare":
        prepare(out, parent, Settings())
        print(f"[pairs32] protocol -> {out / 'protocol.json'}", flush=True)
        return
    if args.device == "cuda" and not torch.cuda.is_available() and args.stage == "run":
        raise RuntimeError("CUDA unavailable")
    if args.stage == "run":
        if args.model_seed is None:
            parser.error("--model-seed is required")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        run_seed(out, args.model_seed, args.device)
    else:
        report(out)


if __name__ == "__main__":
    main()

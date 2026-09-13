"""Can task-aware single-z optimization recover an oracle ideal latent?

For each frozen pattern-8 VAE, first obtain a reference set of latent codes by
directly minimizing distance to the analytic ideal mask.  Then run the existing
target-aware single-z procedure from ordinary prior starts and from controlled
L2 perturbations around fixed exact oracle anchors.  Gold is used only by the
oracle construction and post-hoc structural diagnostics, never by task-z.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats

HERE = Path(__file__).resolve().parent
PATTERN_ROOT = HERE.parent
REPO_ROOT = PATTERN_ROOT.parent
sys.path.insert(0, str(PATTERN_ROOT))

import config  # noqa: E402
from data.generate import ideal_mask  # noqa: E402
from evaluation.decoder_agreement import align_columns  # noqa: E402
from evaluation.oracle_ideal import optimize_ideal  # noqa: E402
from evaluation.single_z_task import optimize_task_z  # noqa: E402
from models.cvae import CVAE  # noqa: E402


DEFAULT_PARENT = PATTERN_ROOT / "outputs/decoder_agreement/multiseed32_tuned_20260912"
DEFAULT_OUT = PATTERN_ROOT / "outputs/z_star_noise/multiseed8_20260912"


@dataclass(frozen=True)
class ExperimentConfig:
    model_seeds: tuple[int, ...] = tuple(range(186, 202, 2))
    oracle_starts: int = 128
    oracle_steps: int = 2000
    oracle_lr: float = 0.03
    temperature: float = 0.5
    latent_radius: float = 12.0
    anchors: int = 8
    directions_per_anchor: int = 4
    noise_radii: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    prior_starts: int = 64
    task_outer_steps: int = 30
    task_warmup_steps: int = 300
    task_grad_steps: int = 100
    task_z_lr: float = 0.05
    task_seed: int = 20260912
    oracle_seed: int = 20260913
    noise_seed: int = 20260914

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_seeds", tuple(self.model_seeds))
        object.__setattr__(self, "noise_radii", tuple(float(value) for value in self.noise_radii))
        if len(set(self.model_seeds)) != len(self.model_seeds) or not self.model_seeds:
            raise ValueError("model seeds must be nonempty and unique")
        if any(seed % 2 for seed in self.model_seeds):
            raise ValueError("this suite uses the first (even-seed) decoder from each saved pair")
        counts = (self.oracle_starts, self.oracle_steps, self.anchors, self.directions_per_anchor,
                  self.prior_starts, self.task_outer_steps)
        if min(counts) < 1 or self.task_warmup_steps < 0 or self.task_grad_steps < 0:
            raise ValueError("invalid search counts")
        if self.anchors > self.oracle_starts:
            raise ValueError("anchors cannot exceed oracle starts")
        if not self.noise_radii or min(self.noise_radii) < 0 or len(set(self.noise_radii)) != len(self.noise_radii):
            raise ValueError("noise radii must be distinct and nonnegative")
        if min(self.oracle_lr, self.temperature, self.latent_radius, self.task_z_lr) <= 0:
            raise ValueError("learning rates, temperature, and radius must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def smoke(cls) -> "ExperimentConfig":
        return replace(cls(), model_seeds=(186,), oracle_starts=4, oracle_steps=300,
                       anchors=1, directions_per_anchor=1, noise_radii=(0.0, 0.5),
                       prior_starts=2, task_outer_steps=1, task_warmup_steps=1,
                       task_grad_steps=1)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little")


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def checkpoint_paths(parent: Path, seed: int) -> tuple[Path, Path, Path]:
    pair = parent / f"pair_{seed}_{seed + 1}"
    return (pair / f"vae_{seed}/cvae_best.pt", pair / f"vae_{seed}/cvae_meta.pt",
            pair / "optimization.pt")


def load_model(parent: Path, seed: int, device: torch.device) -> tuple[CVAE, dict]:
    checkpoint, metadata_path, agreement_path = checkpoint_paths(parent, seed)
    metadata = torch.load(metadata_path, map_location="cpu", weights_only=True)
    if metadata["seed"] != seed or metadata["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError(f"invalid checkpoint provenance for VAE seed {seed}")
    model = CVAE(config.MASK_DIM, config.LATENT_DIM, config.CVAE_HIDDEN)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval().requires_grad_(False)
    return model, {
        "seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
        "metadata": str(metadata_path), "metadata_sha256": sha256_file(metadata_path),
        "agreement_initials": str(agreement_path), "agreement_initials_sha256": sha256_file(agreement_path),
    }


def source_hashes() -> dict[str, str]:
    paths = (Path(__file__), HERE / "oracle_ideal.py", HERE / "single_z_task.py",
             HERE / "decoder_agreement.py")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def make_protocol(parent: Path, settings: ExperimentConfig, smoke: bool) -> dict:
    checkpoints = {}
    for seed in settings.model_seeds:
        checkpoint, metadata, agreement = checkpoint_paths(parent, seed)
        for path in (checkpoint, metadata, agreement):
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoints[str(seed)] = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "metadata_sha256": sha256_file(metadata), "agreement_initials_sha256": sha256_file(agreement),
        }
    split_path = PATTERN_ROOT / "outputs/ood/split_seed_42/split.json"
    split = json.loads(split_path.read_text())
    return {
        "experiment": "task-z recovery from controlled L2 neighborhoods of oracle ideal z-star",
        "settings": settings.to_dict(), "smoke": smoke,
        "parent": str(parent), "checkpoints": checkpoints,
        "heldout_patterns": split["test_patterns"], "task_split": str(split_path),
        "task_split_sha256": sha256_file(split_path),
        "oracle": {
            "uses_gold": True,
            "objective": "MSE to analytic ideal soft-top-32 mask after Hungarian column matching",
            "anchor_selection": "first exact best-soft oracle solutions in fixed start order",
        },
        "task_z": {
            "uses_gold": False, "uses_target_labels": True,
            "procedure": "existing direct-gradient single-z optimization with fresh inner MLPs",
            "selection": "final iterate and minimum search-validation BCE are both retained",
        },
        "primary_outcome": "exact ideal hard-mask recovery modulo hidden-column permutation",
        "distance_outcomes": [
            "Euclidean distance to the source z-star",
            "Euclidean distance to the nearest exact oracle solution found for the same decoder",
            "latent displacement from the task-z initialization",
        ],
        "independence_unit": "frozen VAE training seed; anchors, directions, tasks, and starts are nested",
        "source_sha256": source_hashes(),
    }


def prepare(out: Path, parent: Path, settings: ExperimentConfig, smoke: bool) -> None:
    payload = make_protocol(parent, settings, smoke)
    protocol = out / "protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text()) != payload:
            raise FileExistsError(f"{protocol} exists with a different immutable protocol")
        return
    out.mkdir(parents=True, exist_ok=False)
    (out / "logs").mkdir()
    write_json(protocol, payload)


def load_protocol(out: Path) -> tuple[ExperimentConfig, dict]:
    protocol = json.loads((out / "protocol.json").read_text())
    settings = ExperimentConfig(**protocol["settings"])
    return settings, protocol


def project(latent: torch.Tensor, radius: float) -> torch.Tensor:
    norm = latent.norm(dim=1, keepdim=True).clamp_min(torch.finfo(latent.dtype).tiny)
    return latent * (radius / norm).clamp(max=1.0)


def build_initials(anchors: torch.Tensor, prior: torch.Tensor, settings: ExperimentConfig,
                   seed: int) -> tuple[torch.Tensor, dict]:
    """Concatenate prior controls and exact-L2 perturbations in fixed order."""
    anchors = anchors[:settings.anchors].cpu()
    prior = prior[:settings.prior_starts].cpu()
    generator = torch.Generator().manual_seed(seed_for("noise", settings.noise_seed, seed))
    pieces = [prior]
    groups: dict[str, dict] = {
        "prior": {"start": 0, "stop": len(prior), "source_anchor": [-1] * len(prior),
                  "requested_radius": None},
    }
    cursor = len(prior)
    for radius in settings.noise_radii:
        values, parents = [], []
        for anchor_index, anchor in enumerate(anchors):
            for _ in range(settings.directions_per_anchor):
                direction = torch.randn(anchor.shape, generator=generator)
                direction /= direction.norm().clamp_min(torch.finfo(direction.dtype).tiny)
                values.append(anchor + radius * direction)
                parents.append(anchor_index)
        noisy = project(torch.stack(values), settings.latent_radius)
        key = f"noise_{radius:g}"
        groups[key] = {"start": cursor, "stop": cursor + len(noisy),
                       "source_anchor": parents, "requested_radius": radius,
                       "actual_source_distance": torch.stack([
                           (value - anchors[parent]).norm() for value, parent in zip(noisy, parents)
                       ]).tolist()}
        pieces.append(noisy)
        cursor += len(noisy)
    return torch.cat(pieces), {"groups": groups, "anchors": anchors}


def _hard_stats(masks: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    target_batch = target.unsqueeze(0).expand(len(masks), -1, -1).to(masks)
    aligned = align_columns(target_batch.float(), masks.float())
    intersection = (target_batch * aligned).sum((1, 2))
    iou = intersection / (64 - intersection)
    exact = (aligned == target_batch).all(2).all(1)
    return iou.cpu(), exact.cpu()


def _nearest_distance(values: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    return torch.cdist(values.float().cpu(), references.float().cpu()).min(dim=1).values


def _describe(values: torch.Tensor) -> dict:
    values = values.float().cpu()
    return {"mean": float(values.mean()), "std": float(values.std(unbiased=False)),
            "min": float(values.min()), "max": float(values.max()), "values": values.tolist()}


def analyze_stage(codes: torch.Tensor, masks: torch.Tensor, initial_codes: torch.Tensor,
                  references: torch.Tensor, anchors: torch.Tensor, group: dict,
                  target: torch.Tensor) -> dict:
    start, stop = group["start"], group["stop"]
    z = codes[start:stop].cpu()
    z_initial = initial_codes[start:stop].cpu()
    iou, exact = _hard_stats(masks[start:stop], target)
    nearest = _nearest_distance(z, references)
    result = {
        "count": len(z), "gold_iou": _describe(iou), "exact_fraction": float(exact.float().mean()),
        "exact_count": int(exact.sum()), "nearest_oracle_distance": _describe(nearest),
        "displacement_from_initial": _describe((z - z_initial).norm(dim=1)),
        "z_norm": _describe(z.norm(dim=1)),
    }
    parents = group["source_anchor"]
    if parents and parents[0] >= 0:
        parent_z = anchors[torch.tensor(parents)]
        distance = (z - parent_z).norm(dim=1)
        initial_distance = (z_initial - parent_z).norm(dim=1)
        result["source_z_star_distance"] = _describe(distance)
        result["initial_source_z_star_distance"] = _describe(initial_distance)
        result["fraction_moved_closer_to_source"] = float((distance < initial_distance - 1e-7).float().mean())
        exact_distance = distance[exact]
        result["exact_source_distance"] = (_describe(exact_distance) if len(exact_distance) else None)
    return result


def summarize_seed(out: Path, seed: int, model: CVAE, oracle: dict, setup: dict,
                   task_results: dict[str, dict], settings: ExperimentConfig) -> dict:
    target = ideal_mask().float()
    oracle_exact = oracle["best_soft"]["iou"] == 1
    references = oracle["best_soft"]["z"][oracle_exact]
    anchors = setup["anchors"]
    initial_codes = setup["initials"]
    with torch.no_grad():
        logits = model.decode(initial_codes.to(next(model.parameters()).device),
                              initial_codes.new_empty(len(initial_codes), 0).to(next(model.parameters()).device))
        initial_masks = torch.zeros_like(logits).scatter(1, logits.topk(32, dim=1).indices, 1.0)
        initial_masks = initial_masks.reshape(-1, 8, 8).cpu()
    summary = {
        "model_seed": seed,
        "oracle": {
            "starts": settings.oracle_starts, "exact_best_soft": int(oracle_exact.sum()),
            "exact_fraction": float(oracle_exact.float().mean()),
            "best_soft_iou": _describe(oracle["best_soft"]["iou"]),
            "exact_z_norm": _describe(references.norm(dim=1)),
            "all_best_soft_z_norm": _describe(oracle["best_soft"]["z"].norm(dim=1)),
            "at_radius_fraction": float((oracle["best_soft"]["z"].norm(dim=1)
                                         >= settings.latent_radius - 1e-5).float().mean()),
        },
        "groups": setup["groups"], "tasks": {},
    }
    for pattern, result in task_results.items():
        stages = {
            "initial": (initial_codes, initial_masks),
            "final": (result["final_z"], result["final_masks"]),
            "best_query": (result["best_val_z"], result["best_val_masks"]),
        }
        summary["tasks"][pattern] = {}
        for group_name, group in setup["groups"].items():
            summary["tasks"][pattern][group_name] = {
                stage: analyze_stage(codes, masks, initial_codes, references, anchors, group, target)
                for stage, (codes, masks) in stages.items()
            }
    return summary


def run_seed(out: Path, seed: int, device_name: str) -> None:
    settings, protocol = load_protocol(out)
    if seed not in settings.model_seeds:
        raise ValueError(f"seed {seed} is absent from protocol")
    seed_root = out / f"seed_{seed}"
    summary_path = seed_root / "summary.json"
    if summary_path.exists():
        print(f"[z-star] seed={seed}: completed artifact exists", flush=True)
        return
    seed_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    model, provenance = load_model(Path(protocol["parent"]), seed, device)
    target = ideal_mask().float().to(device)
    oracle_path = seed_root / "oracle.pt"
    if oracle_path.exists():
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
        if oracle["checkpoint_sha256"] != provenance["checkpoint_sha256"]:
            raise ValueError("saved oracle uses a different decoder")
    else:
        generator = torch.Generator(device=device).manual_seed(seed_for("oracle-starts", settings.oracle_seed, seed))
        initial = torch.randn(settings.oracle_starts, config.LATENT_DIM, generator=generator, device=device)
        oracle = optimize_ideal(
            model, initial, target, steps=settings.oracle_steps, lr=settings.oracle_lr,
            temperature=settings.temperature, radius=settings.latent_radius)
        oracle.update({"model_seed": seed, "checkpoint_sha256": provenance["checkpoint_sha256"],
                       "uses_gold": True})
        atomic_save(oracle_path, oracle)
        oracle = torch.load(oracle_path, map_location="cpu", weights_only=True)
    exact = oracle["best_soft"]["iou"] == 1
    if int(exact.sum()) < settings.anchors:
        raise RuntimeError(f"seed {seed}: only {int(exact.sum())} exact oracle solutions; need {settings.anchors}")
    exact_z = oracle["best_soft"]["z"][exact]
    anchors = exact_z[:settings.anchors]
    _, _, agreement_path = checkpoint_paths(Path(protocol["parent"]), seed)
    agreement = torch.load(agreement_path, map_location="cpu", weights_only=True)
    prior = agreement["initial_z1"]
    if len(prior) < settings.prior_starts:
        raise ValueError("saved agreement artifact has too few prior starts")
    initials, construction = build_initials(anchors, prior, settings, seed)
    setup = {"initials": initials, "anchors": construction["anchors"],
             "groups": construction["groups"], "oracle_sha256": sha256_file(oracle_path)}
    setup_path = seed_root / "initials.pt"
    if setup_path.exists():
        saved = torch.load(setup_path, map_location="cpu", weights_only=True)
        if not torch.equal(saved["initials"], setup["initials"]):
            raise ValueError("saved noisy initials differ from regenerated protocol")
        setup = saved
    else:
        atomic_save(setup_path, setup)

    task_results = {}
    for pattern in protocol["heldout_patterns"]:
        task_path = seed_root / f"task_{pattern}.pt"
        if task_path.exists():
            result = torch.load(task_path, map_location="cpu", weights_only=True)
            if result["oracle_sha256"] != sha256_file(oracle_path):
                raise ValueError(f"{task_path} uses a different oracle")
        else:
            result = optimize_task_z(
                model, initials, pattern, device, outer_steps=settings.task_outer_steps,
                warmup_steps=settings.task_warmup_steps, grad_steps=settings.task_grad_steps,
                z_lr=settings.task_z_lr, radius=settings.latent_radius,
                temperature=settings.temperature, seed=settings.task_seed)
            result.update({
                "model_seed": seed, "pattern": pattern, "uses_gold": False,
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "oracle_sha256": sha256_file(oracle_path), "initials_sha256": sha256_file(setup_path),
            })
            atomic_save(task_path, result)
        task_results[pattern] = result
    summary = summarize_seed(out, seed, model, oracle, setup, task_results, settings)
    summary["provenance"] = {**provenance, "oracle_sha256": sha256_file(oracle_path),
                             "initials_sha256": sha256_file(setup_path)}
    write_json(summary_path, summary)
    print(f"[z-star] seed={seed}: complete -> {summary_path}", flush=True)


def ci(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or not len(array):
        raise ValueError("CI input must be finite and nonempty")
    mean = float(array.mean())
    if len(array) == 1:
        low = high = mean
    else:
        half = float(stats.t.ppf(0.975, len(array) - 1) * array.std(ddof=1) / math.sqrt(len(array)))
        low, high = mean - half, mean + half
    return {"mean": mean, "ci95": [low, high], "n": len(array), "values": array.tolist()}


def aggregate(out: Path) -> dict:
    settings, protocol = load_protocol(out)
    records = [json.loads((out / f"seed_{seed}/summary.json").read_text())
               for seed in settings.model_seeds]
    result: dict[str, Any] = {
        "protocol": protocol, "model_count": len(records), "oracle": {},
        "noise": {}, "prior": {}, "per_task": {},
    }
    for metric in ("exact_fraction", "at_radius_fraction"):
        result["oracle"][metric] = ci([record["oracle"][metric] for record in records])
    for metric in ("exact_z_norm", "best_soft_iou"):
        result["oracle"][metric] = ci([record["oracle"][metric]["mean"] for record in records])

    patterns = protocol["heldout_patterns"]
    for radius in settings.noise_radii:
        group = f"noise_{radius:g}"
        result["noise"][group] = {}
        for stage in ("initial", "final", "best_query"):
            per_model = []
            for record in records:
                rows = [record["tasks"][pattern][group][stage] for pattern in patterns]
                per_model.append({
                    "exact_fraction": float(np.mean([row["exact_fraction"] for row in rows])),
                    "gold_iou": float(np.mean([row["gold_iou"]["mean"] for row in rows])),
                    "source_distance": float(np.mean([row["source_z_star_distance"]["mean"] for row in rows])),
                    "nearest_distance": float(np.mean([row["nearest_oracle_distance"]["mean"] for row in rows])),
                    "moved_closer": float(np.mean([row["fraction_moved_closer_to_source"] for row in rows])),
                })
            result["noise"][group][stage] = {
                key: ci([row[key] for row in per_model]) for key in per_model[0]
            }
    for stage in ("initial", "final", "best_query"):
        per_model = []
        for record in records:
            rows = [record["tasks"][pattern]["prior"][stage] for pattern in patterns]
            per_model.append({
                "exact_fraction": float(np.mean([row["exact_fraction"] for row in rows])),
                "gold_iou": float(np.mean([row["gold_iou"]["mean"] for row in rows])),
                "nearest_distance": float(np.mean([row["nearest_oracle_distance"]["mean"] for row in rows])),
                "displacement": float(np.mean([row["displacement_from_initial"]["mean"] for row in rows])),
            })
        result["prior"][stage] = {key: ci([row[key] for row in per_model]) for key in per_model[0]}
    for pattern in patterns:
        result["per_task"][pattern] = {}
        for group in ("prior", *(f"noise_{radius:g}" for radius in settings.noise_radii)):
            result["per_task"][pattern][group] = {
                stage: {
                    metric: ci([record["tasks"][pattern][group][stage][metric]
                                if metric == "exact_fraction" else
                                record["tasks"][pattern][group][stage][metric]["mean"]
                                for record in records])
                    for metric in ("exact_fraction", "gold_iou", "nearest_oracle_distance")
                } for stage in ("initial", "final", "best_query")
            }
    return result


def fmt(value: dict, digits: int = 3, scale: float = 1.0) -> str:
    low, high = value["ci95"]
    return f"{scale * value['mean']:.{digits}f} [{scale * low:.{digits}f}; {scale * high:.{digits}f}]"


def render_report(out: Path, summary: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = ExperimentConfig(**summary["protocol"]["settings"])
    radii = list(settings.noise_radii)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for stage, label, color in (("initial", "Before task-z", "#777777"),
                                ("final", "Task-z final", "#337f8c"),
                                ("best_query", "Task-z best query", "#c26d3a")):
        exact = [summary["noise"][f"noise_{radius:g}"][stage]["exact_fraction"]["mean"] for radius in radii]
        iou = [summary["noise"][f"noise_{radius:g}"][stage]["gold_iou"]["mean"] for radius in radii]
        distance = [summary["noise"][f"noise_{radius:g}"][stage]["source_distance"]["mean"] for radius in radii]
        axes[0].plot(radii, exact, marker="o", label=label, color=color)
        axes[1].plot(radii, iou, marker="o", label=label, color=color)
        axes[2].plot(radii, distance, marker="o", label=label, color=color)
    axes[0].set(ylabel="Exact ideal fraction", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Gold IoU", ylim=(0, 1.02))
    axes[2].set(ylabel="L2 distance to source z*")
    for axis in axes:
        axis.set(xlabel="Initial perturbation L2 radius")
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Task-aware single-z optimization around oracle z*")
    fig.savefig(out / "noise_recovery.png", dpi=170)
    fig.savefig(out / "noise_recovery.pdf")
    plt.close(fig)

    lines = [
        "# Pattern-8: восстанавливает ли single-z оптимизация oracle latent z*?", "",
        f"Проверено **{summary['model_count']} независимо обученных frozen VAE** и четыре held-out pattern-задачи. "
        "Для каждого decoder сначала отдельно построено множество oracle-решений прямой оптимизацией к идеальной маске; "
        "Gold явно используется только на этом шаге и в post-hoc диагностике.", "",
        "## Oracle z*", "",
        f"Из {settings.oracle_starts} стартов на decoder доля точных ideal hard-масок: "
        f"**{fmt(summary['oracle']['exact_fraction'])}**. Средняя норма точных z*: "
        f"**{fmt(summary['oracle']['exact_z_norm'])}**; доля решений на границе radius={settings.latent_radius:g}: "
        f"**{fmt(summary['oracle']['at_radius_fraction'])}**.", "",
        "Точные z* не обязаны быть единственными: один decoder может отображать разные latent-коды в одну и ту же "
        "hard-маску, а скрытые колонки эквивалентны с точностью до перестановки. Поэтому восстановление маски является "
        "первичным критерием, а евклидово расстояние до конкретного z* — диагностикой.", "",
        "## Шум вокруг z*", "",
        "Каждый шум задаётся именно L2-расстоянием в 32-мерном latent, до проекции в общий шар. "
        f"На каждом radius используются {settings.anchors} фиксированных exact-z* × "
        f"{settings.directions_per_anchor} направления на decoder. Затем запускается прежняя task-aware direct-gradient "
        "оптимизация одного z; она видит метки задачи, но не Gold.", "",
        "| L2 noise | Exact до task-z | Exact после, final | Exact, best-query | Gold IoU final | Distance до исходного z*, final | Доля движений к z* |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for radius in radii:
        row = summary["noise"][f"noise_{radius:g}"]
        lines.append(f"| {radius:g} | {fmt(row['initial']['exact_fraction'])} | "
                     f"{fmt(row['final']['exact_fraction'])} | {fmt(row['best_query']['exact_fraction'])} | "
                     f"{fmt(row['final']['gold_iou'])} | {fmt(row['final']['source_distance'])} | "
                     f"{fmt(row['final']['moved_closer'])} |")
    lines += ["", "## Обычные prior-старты", "",
              "| Стадия | Exact ideal | Gold IoU | Distance до ближайшего oracle z* | Смещение от старта |",
              "|---|---:|---:|---:|---:|"]
    for stage, label in (("initial", "До task-z"), ("final", "Final"), ("best_query", "Best query")):
        row = summary["prior"][stage]
        lines.append(f"| {label} | {fmt(row['exact_fraction'])} | {fmt(row['gold_iou'])} | "
                     f"{fmt(row['nearest_distance'])} | {fmt(row['displacement'])} |")
    lines += ["", "![Recovery around z-star](noise_recovery.png)", "",
              "## Как читать результат", "",
              "- `radius=0` проверяет, сохраняет ли task-loss оптимизация уже идеальный decoder output или уводит его.",
              "- Строки с ненулевым radius показывают размер локального basin идеальной hard-маски и способность task-z вернуться в него.",
              "- Prior-блок отвечает, находит ли тот же алгоритм идеальную маску без подсказки близким z*.",
              "- Малое расстояние до одного z* не требуется для успеха: другой удалённый latent может декодироваться в тот же ideal support.",
              "- ДИ рассчитаны по VAE seeds; задачи, anchors, направления и latent-старты вложены внутрь seed.", "",
              "Числа по каждому decoder, задаче, старту и расстоянию сохранены в `seed_*/summary.json` и `.pt`-артефактах. "
              "После полного запуска этот автоматический отчёт следует дополнить содержательной интерпретацией.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


def report(out: Path) -> None:
    summary = aggregate(out)
    write_json(out / "summary.json", summary)
    render_report(out, summary)
    print(f"[z-star] report -> {out / 'RESULTS.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--stage", choices=("prepare", "run", "report"), required=True)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out, parent = args.out.resolve(), args.parent.resolve()
    settings = ExperimentConfig.smoke() if args.smoke else ExperimentConfig()
    if args.stage == "prepare":
        prepare(out, parent, settings, args.smoke)
        print(f"[z-star] protocol -> {out / 'protocol.json'}", flush=True)
        return
    saved_settings, _ = load_protocol(out)
    if saved_settings != settings:
        raise ValueError("CLI smoke/full mode differs from the saved protocol")
    if args.stage == "run":
        if args.model_seed is None:
            parser.error("--model-seed is required for --stage run")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        run_seed(out, args.model_seed, args.device)
    else:
        report(out)


if __name__ == "__main__":
    main()

"""Diagnose optimization of fresh task vectors in two frozen parameter bases.

This is a validation-only diagnostic.  It uses a fixed, hash-selected subset
of validation patterns at lengths 3, 4, 6, and 8 and never imports the held
length evaluation.  For a learned frozen ``U`` and the analytic ``U`` it
compares several optimizers and four independent initializations of ``v``.

Selection is deliberately by *support* BCE only.  Query metrics are read only
after the restart or grid point has been selected, so the report answers
whether a fixed basis contains a good solution that is merely hard to find.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from .calibrate import _batched_logits, _selected_tasks, batched_adapt_v
from .common import dataset, seed_for, setup, source_hashes, write_json
from .config import Config
from .evaluate_interpolation import _file_sha256, load_interpolation_checkpoint
from .models import UTuple, ideal_u


KNOWN_LENGTHS = (3, 4, 6, 8)
BUDGETS = (50, 500, 2000)
SUPPORT_SIZE = 8192
QUERY_SIZE = 2048
BATCH_SIZE = 128
RESTARTS = 4
DIAGNOSIS_SEED = 20_260_907
CHECKPOINT = Path("meta_pattern/outputs/interpolation_capacity_20260906/adam50/phase1/small_seed42/best.pt")


@dataclass(frozen=True)
class VConfig:
    optimizer: str
    lr: float
    init_scale: float = 0.1

    @property
    def identifier(self) -> str:
        return f"{self.optimizer}_lr{self.lr:g}_init{self.init_scale:g}"


def grid() -> tuple[VConfig, ...]:
    return tuple([VConfig("adam", lr) for lr in (0.003, 0.01, 0.03, 0.1)] + [VConfig("sgd", 1.0)])


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        value = value.detach().contiguous().cpu()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float | bool | None]:
    if not bool(torch.isfinite(logits).all()):
        return {"bce": None, "accuracy": None, "nonfinite": True}
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    if not bool(torch.isfinite(bce)):
        return {"bce": None, "accuracy": None, "nonfinite": True}
    return {
        "bce": float(bce.item()),
        "accuracy": float(((logits > 0) == labels.bool()).float().mean().item()),
        "nonfinite": False,
    }


def _episodes(config: Config, length: int, device: str) -> tuple[list[dict], dict[str, dict[str, str]]]:
    """Materialize fixed data once; restarts and bases receive identical data."""
    result, hashes = [], {}
    for task in _selected_tasks(42, "val", length, limit=4):
        base = seed_for("u-diagnosis", DIAGNOSIS_SEED, task.pattern)
        support = dataset(config, task, SUPPORT_SIZE, seed_for(base, "support"), "support", device)
        query = dataset(config, task, QUERY_SIZE, seed_for(base, "query"), "query", device)
        hashes[task.pattern] = {
            "support": _tensor_sha256(support["ids"], support["y"]),
            "query": _tensor_sha256(query["ids"], query["y"]),
        }
        result.append({
            "pattern": task.pattern,
            "support_x": support["x"], "support_y": support["y"],
            "query_x": query["x"], "query_y": query["y"],
            "v_seeds": [seed_for(base, "v", restart) for restart in range(RESTARTS)],
        })
    return result, hashes


def _finite_metric(row: dict, side: str) -> bool:
    score = row[side]
    return not score["nonfinite"] and score["bce"] is not None and score["accuracy"] is not None


def choose_by_support(rows: Sequence[dict]) -> dict | None:
    """Choose a finite trajectory by support BCE, with deterministic ties.

    ``query`` is intentionally never accessed here.  Keeping the rule in a
    standalone function makes accidental test-set selection easy to detect.
    """
    candidates = [row for row in rows if _finite_metric(row, "support")]
    if not candidates:
        return None
    return min(candidates, key=lambda row: (float(row["support"]["bce"]), row["config_index"], row["restart"]))


def _selection_rows(rows: Sequence[dict]) -> list[dict]:
    """Return per-task selected query outcomes for each pre-declared rule."""
    groups: dict[tuple[str, int, str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["basis"], row["budget"], row["pattern"], row["length"]), []).append(row)
    selected = []
    default_config = "adam_lr0.1_init0.1"
    for (basis, budget, pattern, length), group in sorted(groups.items()):
        default = sorted((row for row in group if row["config"] == default_config), key=lambda row: row["restart"])
        if len(default) != RESTARTS:
            raise ValueError(f"missing default trajectories for {basis}, k={length}, {pattern}, budget={budget}")
        for label, candidates in (
            ("default_adam01_single_restart0", [row for row in default if row["restart"] == 0]),
            ("support_selected_restart_adam01", default),
            ("support_selected_grid", group),
        ):
            chosen = choose_by_support(candidates)
            selected.append({
                "basis": basis, "budget": budget, "length": length, "pattern": pattern,
                "strategy": label, "selected": None if chosen is None else {
                    "config": chosen["config"], "config_index": chosen["config_index"],
                    "restart": chosen["restart"], "support": chosen["support"], "query": chosen["query"],
                },
            })
        # The initialization average is a descriptive baseline, not a
        # selection.  A single failed restart makes its aggregate invalid.
        selected.append({
            "basis": basis, "budget": budget, "length": length, "pattern": pattern,
            "strategy": "default_adam01_mean_restarts", "selected": [{
                "config": row["config"], "config_index": row["config_index"], "restart": row["restart"],
                "support": row["support"], "query": row["query"],
            } for row in default],
        })
    return selected


def _aggregate_selection(selected: Sequence[dict]) -> list[dict]:
    """Aggregate query metrics, disqualifying any selected non-finite case."""
    grouped: dict[tuple[str, str, int, int], list[dict]] = {}
    for row in selected:
        grouped.setdefault((row["basis"], row["strategy"], row["budget"], row["length"]), []).append(row)
    result = []
    for (basis, strategy, budget, length), group in sorted(grouped.items()):
        query_metrics, support_metrics = [], []
        nonfinite = 0
        selected_configs: dict[str, int] = {}
        selected_restarts: dict[str, int] = {}
        for row in group:
            choices = row["selected"] if isinstance(row["selected"], list) else [row["selected"]]
            if not choices or choices[0] is None:
                nonfinite += 1
                continue
            for choice in choices:
                if not _finite_metric(choice, "query") or not _finite_metric(choice, "support"):
                    nonfinite += 1
                    continue
                query_metrics.append(choice["query"])
                support_metrics.append(choice["support"])
                selected_configs[choice["config"]] = selected_configs.get(choice["config"], 0) + 1
                selected_restarts[str(choice["restart"])] = selected_restarts.get(str(choice["restart"]), 0) + 1
        expected = len(group) * (RESTARTS if strategy == "default_adam01_mean_restarts" else 1)
        complete = nonfinite == 0 and len(query_metrics) == expected
        result.append({
            "basis": basis, "strategy": strategy, "budget": budget, "length": length,
            "n_patterns": len(group), "n_initializations": expected,
            "n_finite": len(query_metrics), "nonfinite_selected": nonfinite,
            "query_bce": (sum(float(x["bce"]) for x in query_metrics) / len(query_metrics)) if complete else None,
            "query_accuracy": (sum(float(x["accuracy"]) for x in query_metrics) / len(query_metrics)) if complete else None,
            "support_bce": (sum(float(x["bce"]) for x in support_metrics) / len(support_metrics)) if complete else None,
            "support_accuracy": (sum(float(x["accuracy"]) for x in support_metrics) / len(support_metrics)) if complete else None,
            "selected_config_counts": selected_configs,
            "selected_restart_counts": selected_restarts,
        })
    # Equal-length average stays undefined if any constituent length failed.
    for basis in sorted({row["basis"] for row in result}):
        for strategy in sorted({row["strategy"] for row in result}):
            for budget in BUDGETS:
                pieces = [row for row in result if row["basis"] == basis and row["strategy"] == strategy and row["budget"] == budget]
                if len(pieces) != len(KNOWN_LENGTHS):
                    continue
                complete = all(row["query_bce"] is not None for row in pieces)
                result.append({
                    "basis": basis, "strategy": strategy, "budget": budget, "length": "equal_length",
                    "n_patterns": sum(row["n_patterns"] for row in pieces),
                    "n_initializations": sum(row["n_initializations"] for row in pieces),
                    "n_finite": sum(row["n_finite"] for row in pieces),
                    "nonfinite_selected": sum(row["nonfinite_selected"] for row in pieces),
                    "query_bce": sum(float(row["query_bce"]) for row in pieces) / len(pieces) if complete else None,
                    "query_accuracy": sum(float(row["query_accuracy"]) for row in pieces) / len(pieces) if complete else None,
                    "support_bce": sum(float(row["support_bce"]) for row in pieces) / len(pieces) if complete else None,
                    "support_accuracy": sum(float(row["support_accuracy"]) for row in pieces) / len(pieces) if complete else None,
                    "selected_config_counts": {}, "selected_restart_counts": {},
                })
    return result


def _differences(aggregate: Sequence[dict]) -> list[dict]:
    lookup = {(x["strategy"], x["budget"], x["length"], x["basis"]): x for x in aggregate}
    output = []
    for strategy in sorted({row["strategy"] for row in aggregate}):
        for budget in BUDGETS:
            for length in (*KNOWN_LENGTHS, "equal_length"):
                learned, ideal = lookup.get((strategy, budget, length, "learned")), lookup.get((strategy, budget, length, "ideal"))
                if learned is None or ideal is None or learned["query_bce"] is None or ideal["query_bce"] is None:
                    delta_bce = delta_acc = None
                else:
                    delta_bce = float(learned["query_bce"] - ideal["query_bce"])
                    delta_acc = float(learned["query_accuracy"] - ideal["query_accuracy"])
                output.append({"strategy": strategy, "budget": budget, "length": length,
                               "learned_minus_ideal_query_bce": delta_bce,
                               "learned_minus_ideal_query_accuracy": delta_acc})
    return output


def run_shard(out: Path, shard: int, shards: int, device: str, checkpoint: Path = CHECKPOINT) -> Path:
    configs = grid()
    if shards != len(configs) or not 0 <= shard < shards:
        raise ValueError(f"use exactly {len(configs)} shards numbered 0..{len(configs) - 1}")
    target = out / f"shard{shard}.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    out.mkdir(parents=True, exist_ok=True)
    setup(DIAGNOSIS_SEED, device)
    checkpoint_state, config, model = load_interpolation_checkpoint(checkpoint, device)
    del checkpoint_state
    selected_config = configs[shard]
    started = time.monotonic()
    rows, data_hashes, task_manifest = [], {}, {}
    for length in KNOWN_LENGTHS:
        episodes, hashes = _episodes(config, length, device)
        data_hashes[str(length)] = hashes
        task_manifest[str(length)] = [episode["pattern"] for episode in episodes]
        count = len(episodes) * RESTARTS
        support_x = torch.stack([episode["support_x"] for episode in episodes for _ in range(RESTARTS)])
        support_y = torch.stack([episode["support_y"] for episode in episodes for _ in range(RESTARTS)])
        query_x = torch.stack([episode["query_x"] for episode in episodes for _ in range(RESTARTS)])
        query_y = torch.stack([episode["query_y"] for episode in episodes for _ in range(RESTARTS)])
        v_seeds = torch.tensor([seed for episode in episodes for seed in episode["v_seeds"]], dtype=torch.int64)
        metadata = [(episode, restart) for episode in episodes for restart in range(RESTARTS)]
        with torch.no_grad():
            learned_u: UTuple = tuple(value.detach() for value in model(length))
        bases: dict[str, UTuple] = {
            "learned": learned_u,
            "ideal": ideal_u(length, seq_len=config.seq_len, hidden=config.hidden,
                             rank1=config.rank1, rank2=config.rank2, device=device),
        }
        for basis_name, u in bases.items():
            snapshots = batched_adapt_v(
                u, support_x, support_y, steps=max(BUDGETS),
                lrs=torch.full((count,), selected_config.lr, device=device), seeds=v_seeds,
                optimizers=[selected_config.optimizer] * count,
                init_scales=torch.full((count,), selected_config.init_scale, device=device),
                batch_size=BATCH_SIZE, checkpoints=BUDGETS,
            )
            for budget in BUDGETS:
                with torch.no_grad():
                    support_scores = [_metrics(logit, y) for logit, y in zip(_batched_logits(support_x, u, snapshots[budget]), support_y)]
                    query_scores = [_metrics(logit, y) for logit, y in zip(_batched_logits(query_x, u, snapshots[budget]), query_y)]
                for (episode, restart), support_score, query_score in zip(metadata, support_scores, query_scores):
                    rows.append({
                        "basis": basis_name, "config": selected_config.identifier, "config_index": shard,
                        "optimizer": selected_config.optimizer, "lr": selected_config.lr,
                        "init_scale": selected_config.init_scale, "length": length,
                        "pattern": episode["pattern"], "restart": restart,
                        "v_seed": episode["v_seeds"][restart], "budget": budget,
                        "support": support_score, "query": query_score,
                    })
        print(f"FIXED_U shard={shard} config={selected_config.identifier} length={length} seconds={time.monotonic() - started:.1f}", flush=True)
    result = {
        "protocol": {"known_lengths": list(KNOWN_LENGTHS), "forbidden_lengths": [5, 7],
                     "task_split": "val", "task_selection": "calibrate._selected_tasks(42, 'val', k, limit=4)",
                     "data_seed_rule": "seed_for('u-diagnosis', 20260907, pattern)",
                     "support_size": SUPPORT_SIZE, "query_size": QUERY_SIZE, "batch_size": BATCH_SIZE,
                     "budgets": list(BUDGETS), "restarts": RESTARTS,
                     "selection": "support BCE only; query never used for restart/grid selection"},
        "shard": shard, "shards": shards, "config": asdict(selected_config) | {"identifier": selected_config.identifier},
        "all_grid": [asdict(item) | {"identifier": item.identifier} for item in configs],
        "checkpoint": str(checkpoint), "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_config": config.to_dict(), "selected_tasks": task_manifest,
        "data_hashes": data_hashes, "source_sha256": source_hashes(),
        "seconds": time.monotonic() - started, "rows": rows,
    }
    write_json(target, result)
    return target


def summarize(out: Path) -> Path:
    shards = []
    for index in range(len(grid())):
        path = out / f"shard{index}.json"
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")
        shards.append(json.loads(path.read_text()))
    reference = shards[0]
    for shard in shards[1:]:
        for field in ("protocol", "checkpoint_sha256", "selected_tasks", "data_hashes", "source_sha256"):
            if shard[field] != reference[field]:
                raise ValueError(f"shards disagree on {field}")
    rows = [row for shard in shards for row in shard["rows"]]
    expected = len(grid()) * 2 * sum(min(4, len(_selected_tasks(42, "val", k, limit=4))) for k in KNOWN_LENGTHS) * RESTARTS * len(BUDGETS)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rows, found {len(rows)}")
    selected = _selection_rows(rows)
    aggregate = _aggregate_selection(selected)
    failures = []
    for basis in ("learned", "ideal"):
        for config in grid():
            for budget in BUDGETS:
                values = [row for row in rows if row["basis"] == basis and row["config"] == config.identifier and row["budget"] == budget]
                failures.append({"basis": basis, "config": config.identifier, "budget": budget,
                                 "n": len(values), "nonfinite_support": sum(not _finite_metric(x, "support") for x in values),
                                 "nonfinite_query": sum(not _finite_metric(x, "query") for x in values)})
    result = {
        "protocol": reference["protocol"], "checkpoint": reference["checkpoint"],
        "checkpoint_sha256": reference["checkpoint_sha256"], "checkpoint_config": reference["checkpoint_config"],
        "selected_tasks": reference["selected_tasks"], "data_hashes": reference["data_hashes"],
        "source_sha256": reference["source_sha256"], "all_grid": reference["all_grid"],
        "raw_rows": len(rows), "selection_rows": selected, "aggregate": aggregate,
        "learned_minus_ideal": _differences(aggregate), "failure_counts": failures,
    }
    summary_json = out / "SUMMARY.json"
    if summary_json.exists():
        raise FileExistsError(f"refusing to overwrite {summary_json}")
    write_json(summary_json, result)
    _write_markdown(out / "SUMMARY.md", result)
    return summary_json


def _fmt(value: float | None, percent: bool = False) -> str:
    if value is None:
        return "NONFINITE"
    return f"{100 * value:.2f}%" if percent else f"{value:.5f}"


def _write_markdown(path: Path, result: dict) -> None:
    lines = ["# Frozen-U optimization diagnosis", "",
             "Fresh `v` is fitted only on support data for frozen learned or analytic U. "
             "Restart/grid selection is by support BCE only; query is reported afterwards.", "",
             "No held lengths (5, 7) or test partition were accessed.", "",
             "## Query metrics by selection rule", "",
             "| Basis | Rule | Steps | Length | Query BCE | Query accuracy | Support BCE | Nonfinite selected |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["aggregate"]:
        lines.append(f"| {row['basis']} | {row['strategy']} | {row['budget']} | {row['length']} | "
                     f"{_fmt(row['query_bce'])} | {_fmt(row['query_accuracy'], True)} | {_fmt(row['support_bce'])} | {row['nonfinite_selected']} |")
    lines += ["", "## Learned minus analytic query metric", "",
              "| Rule | Steps | Length | Δ BCE | Δ accuracy |", "|---|---:|---:|---:|---:|"]
    for row in result["learned_minus_ideal"]:
        lines.append(f"| {row['strategy']} | {row['budget']} | {row['length']} | "
                     f"{_fmt(row['learned_minus_ideal_query_bce'])} | {_fmt(row['learned_minus_ideal_query_accuracy'], True)} |")
    lines += ["", "## Non-finite trajectory counts", "",
              "| Basis | Optimizer | Steps | Cases | Support nonfinite | Query nonfinite |",
              "|---|---|---:|---:|---:|---:|"]
    for row in result["failure_counts"]:
        lines.append(f"| {row['basis']} | {row['config']} | {row['budget']} | {row['n']} | {row['nonfinite_support']} | {row['nonfinite_query']} |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--shards", type=int, default=len(grid()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        if args.shard is not None:
            parser.error("--summarize does not take --shard")
        path = summarize(args.out)
    else:
        if args.shard is None:
            parser.error("--shard is required unless --summarize is used")
        path = run_shard(args.out, args.shard, args.shards, args.device, args.checkpoint)
    print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()

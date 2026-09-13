from __future__ import annotations

import hashlib
import itertools
import json
import random
from pathlib import Path
from typing import Any

import torch

from meta_pattern.data import PatternTask, sample_dataset

from .config import Config


ROOT = Path(__file__).resolve().parents[2]


def seed_for(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_save(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    result = {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(here.rglob("*.py"))}
    dependencies = (
        ROOT / "meta_pattern/data.py",
        ROOT / "pattern/length_interp/bank.py",
        ROOT / "pattern/length_interp/mlp.py",
        ROOT / "pattern/models/cvae.py",
        ROOT / "pattern/evaluation/decoder_agreement.py",
    )
    result.update({str(path.relative_to(ROOT)): sha256_file(path) for path in dependencies})
    return result


def _orbit(pattern: str) -> tuple[str, ...]:
    complement = "".join("1" if bit == "0" else "0" for bit in pattern)
    return tuple(sorted({pattern, pattern[::-1], complement, complement[::-1]}))


def task_split(config: Config) -> dict[str, list[str]]:
    """Deterministic 24/8 orbit-safe OOD split, optionally capped for smoke."""
    patterns = {format(value, f"0{config.pattern_len}b") for value in range(2 ** config.pattern_len)}
    orbits = []
    while patterns:
        orbit = _orbit(min(patterns))
        orbits.append(orbit)
        patterns.difference_update(orbit)
    random.Random(config.task_split_seed).shuffle(orbits)
    candidates = [combo for width in range(1, len(orbits) + 1)
                  for combo in itertools.combinations(range(len(orbits)), width)
                  if sum(len(orbits[index]) for index in combo) == 8]
    if not candidates:
        raise AssertionError("could not construct an eight-pattern held-out orbit set")
    chosen = set(candidates[0])
    test = sorted(pattern for index in chosen for pattern in orbits[index])
    train = sorted(pattern for index, orbit in enumerate(orbits) if index not in chosen for pattern in orbit)
    if len(train) != 24 or len(test) != 8 or set(train) & set(test):
        raise AssertionError("invalid fixed-k5 task split")
    if config.train_task_limit:
        train = sorted(random.Random(seed_for("train-cap", config.task_split_seed)).sample(
            train, min(config.train_task_limit, len(train))))
    if config.test_task_limit:
        test = sorted(random.Random(seed_for("test-cap", config.task_split_seed)).sample(
            test, min(config.test_task_limit, len(test))))
    return {"train_patterns": train, "test_patterns": test}


def load_protocol(out: str | Path) -> tuple[Config, dict]:
    payload = json.loads((Path(out) / "protocol.json").read_text())
    config = Config(**payload["config"])
    if payload["task_split"] != task_split(config):
        raise ValueError("saved task split differs from the deterministic protocol")
    return config, payload


def bank_path(out: str | Path, pattern: str) -> Path:
    return Path(out) / "bank" / "k5" / f"pattern_{pattern}.pt"


def pair_dir(out: str | Path, pair: tuple[int, int]) -> Path:
    return Path(out) / "pairs" / f"pair_{pair[0]}_{pair[1]}"


def ideal_mask(config: Config) -> torch.Tensor:
    mask = torch.zeros(config.seq_len, config.hidden)
    windows = config.seq_len - config.pattern_len + 1
    for column in range(config.hidden):
        start = column % windows
        mask[start:start + config.pattern_len, column] = 1
    if int(mask.sum()) != config.k_active:
        raise AssertionError("ideal support has the wrong cardinality")
    return mask


def prepare_task_data(out: str | Path, config: Config, patterns: list[str]) -> None:
    """Create one shared, disjoint search/evaluation data artifact per OOD task."""
    root = Path(out) / "task_data"
    root.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        destination = root / f"pattern_{pattern}.pt"
        if destination.exists():
            record = torch.load(destination, map_location="cpu", weights_only=True)
            expected = {
                "pattern": pattern,
                "input_split_seed": config.input_split_seed,
                "task_support_size": config.task_support_size,
                "eval_support_size": config.eval_support_size,
                "task_search_val_size": config.task_search_val_size,
                "eval_test_size": config.eval_test_size,
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"incompatible task data artifact: {destination}")
            continue
        task = PatternTask(pattern)
        support = sample_dataset(
            task, config.task_support_size + config.eval_support_size,
            seed=seed_for("combined-support", config.input_split_seed, pattern), split="support",
            split_seed=config.input_split_seed, balanced=True, seq_len=config.seq_len,
        )
        query = sample_dataset(
            task, config.task_search_val_size,
            seed=seed_for("task-query", config.input_split_seed, pattern), split="query",
            split_seed=config.input_split_seed, balanced=True, seq_len=config.seq_len,
        )
        test = sample_dataset(
            task, config.eval_test_size,
            seed=seed_for("final-test", config.input_split_seed, pattern), split="test",
            split_seed=config.input_split_seed, balanced=True, seq_len=config.seq_len,
        )
        cut = config.task_support_size
        record = {
            "pattern": pattern,
            "input_split_seed": config.input_split_seed,
            "task_support_size": config.task_support_size,
            "eval_support_size": config.eval_support_size,
            "task_search_val_size": config.task_search_val_size,
            "eval_test_size": config.eval_test_size,
            "task_support": {key: value[:cut] for key, value in support.items()},
            "eval_support": {key: value[cut:] for key, value in support.items()},
            "task_query": query,
            "eval_test": test,
        }
        all_ids = [record[name]["ids"] for name in ("task_support", "eval_support", "task_query", "eval_test")]
        if sum(value.numel() for value in all_ids) != torch.unique(torch.cat(all_ids)).numel():
            raise AssertionError("search/evaluation input IDs are not disjoint")
        record["ids_sha256"] = {name: sha256_tensor(record[name]["ids"])
                                  for name in ("task_support", "eval_support", "task_query", "eval_test")}
        atomic_save(destination, record)


def protocol_payload(config: Config, smoke: bool) -> dict:
    split = task_split(config)
    return {
        "experiment": "fixed-k5 pattern-32 agreement across independent unconditional VAEs",
        "config": config.to_dict(),
        "task_split": split,
        "smoke": smoke,
        "mask": {"shape": [32, 32], "k_active": 160,
                 "hidden_columns_equivalent_up_to_permutation": True},
        "searches": {
            "agreement": "label-free MSE between soft-top-160 decoder masks after Hungarian column matching",
            "individual": "target-label direct-gradient single-z adaptation on the first VAE of every pair",
            "random_latent_pair": (
                f"same {config.random_proposals} decoded latent-pair states per start, "
                "selected by agreement loss"),
            "random_exact_k": "uniform exact-160 binary mask",
        },
        "selection_barriers": {
            "agreement": "no held-out inputs, labels, or ideal support",
            "individual": "held-out support/query labels allowed; final test and ideal support excluded",
            "gold": "ideal support used only in post-hoc metrics and plots",
        },
        "independence_unit": "VAE pair; latent starts, tasks, and evaluation repeats are nested",
        "source_sha256": source_hashes(),
    }

"""Run nested agreement searches for 3–10 pattern-specific VAEs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from evaluation.multi_decoder_agreement import optimize_multi_agreement  # noqa: E402
from evaluation.run_cross_pattern_agreement import PAIRS, ROOT as MODEL_ROOT, load_model  # noqa: E402
from evaluation.train_agreement_vaes import sha256_file  # noqa: E402


ROOT = config.OUTPUTS / "multi_pattern_agreement_20260921"
SIZES = tuple(range(3, 11))
N_STARTS = 64
STEPS = 2000
LR = .03
TEMPERATURE = .5
RADIUS = 12.


def group_patterns(group_index: int) -> tuple[str, ...]:
    """Nested ten-pattern group whose first pair is an earlier experiment."""
    first = PAIRS[group_index]
    second_pair = PAIRS[(group_index + 1) % len(PAIRS)]
    third_side = group_index % 2
    third_pair = PAIRS[(group_index + 2) % len(PAIRS)]
    fourth_pair = PAIRS[(group_index + 3) % len(PAIRS)]
    fifth_pair = PAIRS[(group_index + 4) % len(PAIRS)]
    return (*first, second_pair[third_side], second_pair[1 - third_side],
            third_pair[third_side], third_pair[1 - third_side],
            fourth_pair[third_side], fourth_pair[1 - third_side],
            fifth_pair[third_side], fifth_pair[1 - third_side])


def locate(pattern: str) -> tuple[int, int]:
    matches = [(i, side) for i, pair in enumerate(PAIRS)
               for side, candidate in enumerate(pair) if candidate == pattern]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one trained location for {pattern}: {matches}")
    return matches[0]


def run_one(group_index: int, size: int, replicate: int,
            device: torch.device) -> None:
    patterns = group_patterns(group_index)[:size]
    folder = ROOT / f"group_{group_index:02d}" / f"n{size}" / f"rep_{replicate}"
    folder.mkdir(parents=True, exist_ok=True)
    models, records = [], []
    for pattern in patterns:
        pair_index, side = locate(pattern)
        model, record = load_model(MODEL_ROOT, pair_index, replicate, side, device)
        models.append(model)
        records.append(record)
    # Same seed across sizes preserves the initial latents of all shared models.
    seed = 20260921 + group_index * 100 + replicate
    protocol = {
        "group_index": group_index, "patterns": list(patterns), "size": size,
        "replicate": replicate, "models": records, "n_starts": N_STARTS,
        "steps": STEPS, "lr": LR, "temperature": TEMPERATURE,
        "radius": RADIUS, "k_active": config.K_ACTIVE, "seed": seed,
        "objective": "variance around mean of soft exact-32 masks aligned to first decoder",
        "selection": "minimum ensemble soft variance per start, including initialization",
        "gold_or_task_data_used": False,
    }
    protocol_path = folder / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise FileExistsError(f"different protocol already exists: {protocol_path}")
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    result_path = folder / "optimization.pt"
    if not result_path.exists():
        result = optimize_multi_agreement(
            models, n_starts=N_STARTS, steps=STEPS, lr=LR, seed=seed,
            temperature=TEMPERATURE, radius=RADIUS, device=device,
            k=config.K_ACTIVE,
        )
        torch.save(result, result_path)
    provenance = {
        "protocol_sha256": sha256_file(protocol_path),
        "optimization_sha256": sha256_file(result_path),
    }
    provenance_path = folder / "provenance.json"
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
        raise FileExistsError(f"changed artifact in {folder}")
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"[multi-pattern] group={group_index} size={size} rep={replicate} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--size", type=int, choices=SIZES, required=True)
    parser.add_argument("--replicate", type=int, choices=range(4), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 0 <= args.group_index < len(PAIRS):
        raise ValueError("group-index out of range")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    run_one(args.group_index, args.size, args.replicate, torch.device(args.device))


if __name__ == "__main__":
    main()

"""Pattern-specific VAE agreement: eight pattern pairs, four seeds each.

Run one pair per process/GPU. Train and search stages never read gold masks or
task test labels. Evaluation is a separate, post-hoc stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from evaluation.decoder_agreement import optimize_agreement  # noqa: E402
from evaluation.run_decoder_agreement import eval_one_task, random_search  # noqa: E402
from evaluation.train_agreement_vaes import (  # noqa: E402
    BATCH_SIZE, BETA, HIDDEN, LATENT_DIM, LOSS, LR, MAP_SPLIT_SEED,
    REDUCTION, VAL_FRACTION, sha256_file, sha256_json, train_one,
)
from models.cvae import CVAE, make_loaders  # noqa: E402


ROOT = config.OUTPUTS / "cross_pattern_agreement_20260919"
PAIRS = (
    ("0000", "0001"), ("0010", "0011"), ("1100", "1101"),
    ("1110", "1111"), ("0100", "1011"), ("0101", "1010"),
    ("0110", "1001"), ("0111", "1000"),
)
TOP_FRAC = 0.1
EPOCHS = 160
SEARCH_STEPS = 2000
SEARCH_STARTS = 64
SEARCH_LR = .03
TEMPERATURE = .5
RADIUS = 12.


def path_for(root: Path, pair_index: int) -> Path:
    a, b = PAIRS[pair_index]
    return root / f"pair_{pair_index:02d}_{a}_{b}"


def model_seed(pair_index: int, replicate: int, side: int) -> int:
    return 30_000 + pair_index * 100 + replicate * 2 + side


def checked_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f"different protocol already exists: {path}")
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def bank_data(root: Path, pattern: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    original = config.pattern_dir(pattern) / "importance.pt"
    bank_dir = root / "bank" / f"pattern_{pattern}"
    bank_protocol_path = bank_dir / "protocol.json"
    protocol = json.loads(bank_protocol_path.read_text())
    expected_chunks = (protocol["n_extra"] + protocol["chunk_size"] - 1) // protocol["chunk_size"]
    chunks = [bank_dir / f"chunk_{i:03d}.pt" for i in range(expected_chunks)]
    if not all(path.exists() for path in chunks):
        raise FileNotFoundError(f"incomplete additional bank for {pattern}")
    old = torch.load(original, map_location="cpu", weights_only=True)
    parts = [old] + [torch.load(path, map_location="cpu", weights_only=True) for path in chunks]
    if any(part["pattern"] != pattern for part in parts):
        raise ValueError("importance bank pattern mismatch")
    maps = torch.cat([part["importance"] for part in parts])
    losses = torch.cat([part["val_loss"] for part in parts])
    if len(maps) != 2000 + protocol["n_extra"]:
        raise ValueError("unexpected importance-bank size")
    indices = torch.argsort(losses, stable=True)[:round(len(maps) * TOP_FRAC)]
    x = maps[indices].reshape(len(indices), -1).float().contiguous()
    y = config.pattern_to_pm1(pattern).float().repeat(len(x), 1)
    provenance = {
        "pattern": pattern, "n_total": len(maps), "n_selected": len(x),
        "top_frac": TOP_FRAC, "selection": "lowest stored validation BCE",
        "selected_worst_bce": float(losses[indices].max()),
        "selected_mean_bce": float(losses[indices].mean()),
        "files_sha256": {str(path.resolve()): sha256_file(path)
                         for path in [original, bank_protocol_path, *chunks]},
        "selected_tensor_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
    }
    return x, y, provenance


def training_stage(root: Path, pair_index: int, device: torch.device) -> None:
    pair_root = path_for(root, pair_index)
    a, b = PAIRS[pair_index]
    pair_root.mkdir(parents=True, exist_ok=True)
    for side, pattern in enumerate((a, b)):
        x, y, provenance = bank_data(root, pattern)
        _, _, train_idx, val_idx = make_loaders(
            x, y, VAL_FRACTION, BATCH_SIZE, MAP_SPLIT_SEED)
        partition = {"map_split_seed": MAP_SPLIT_SEED,
                     "val_fraction": VAL_FRACTION,
                     "train_indices": train_idx.tolist(),
                     "val_indices": val_idx.tolist(),
                     "loader_order_seed": MAP_SPLIT_SEED}
        expected_base = {
            "patterns": [pattern], "map_split_seed": MAP_SPLIT_SEED,
            "loader_order_seed": MAP_SPLIT_SEED,
            "partition_sha256": sha256_json(partition),
            "dataset_files_sha256": provenance["files_sha256"],
            "dataset_selected_tensor_sha256": provenance["selected_tensor_sha256"],
            "latent_dim": LATENT_DIM, "hidden": HIDDEN, "beta": BETA,
            "loss": LOSS, "reduction": REDUCTION, "top_frac": TOP_FRAC,
            "importance_name": "original plus independent additional bank",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "val_fraction": VAL_FRACTION,
            "per_pattern": [{"pattern": pattern, "n_masks": len(x), "sparsity": 0.}],
        }
        model_root = pair_root / "models" / pattern
        model_root.mkdir(parents=True, exist_ok=True)
        checked_json(model_root / "data_provenance.json", provenance)
        for replicate in range(4):
            seed = model_seed(pair_index, replicate, side)
            result = train_one(seed, x, y, {**expected_base, "seed": seed},
                               model_root, device)
            print(f"[cross-pattern] pair={pair_index} {pattern} "
                  f"rep={replicate} val={result['best_val']:.5f}", flush=True)


def load_model(root: Path, pair_index: int, replicate: int,
               side: int, device: torch.device) -> tuple[CVAE, dict]:
    pattern = PAIRS[pair_index][side]
    seed = model_seed(pair_index, replicate, side)
    folder = path_for(root, pair_index) / "models" / pattern / f"vae_{seed}"
    ckpt = folder / "cvae_best.pt"
    meta = torch.load(folder / "cvae_meta.pt", map_location="cpu", weights_only=True)
    if meta["patterns"] != [pattern] or meta["seed"] != seed:
        raise ValueError(f"wrong VAE metadata in {folder}")
    if meta["checkpoint_sha256"] != sha256_file(ckpt):
        raise ValueError(f"VAE checkpoint hash mismatch in {folder}")
    model = CVAE(config.MASK_DIM, LATENT_DIM, HIDDEN).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval().requires_grad_(False)
    return model, {"pattern": pattern, "seed": seed,
                   "checkpoint": str(ckpt.resolve()),
                   "checkpoint_sha256": sha256_file(ckpt),
                   "validation_loss": float(meta["best_val_loss"]),
                   "noncollapse": meta["noncollapse_diagnostics"]}


def search_one(root: Path, pair_index: int, name: str,
               left: tuple[int, int], right: tuple[int, int],
               device: torch.device, random_control: bool) -> None:
    pair_root = path_for(root, pair_index)
    out = pair_root / name
    out.mkdir(parents=True, exist_ok=True)
    models, provenance = [], []
    for rep, side in (left, right):
        model, record = load_model(root, pair_index, rep, side, device)
        models.append(model)
        provenance.append(record)
    protocol = {
        "comparison": name,
        "models": provenance,
        "n_starts": SEARCH_STARTS, "steps": SEARCH_STEPS,
        "lr": SEARCH_LR, "temperature": TEMPERATURE, "radius": RADIUS,
        "seed": 20260919 + (int(name.split("_")[-1]) if name.startswith("cross_") else 0),
        "k_active": config.K_ACTIVE,
        "objective": "soft exact-32 MSE after detached Hungarian column matching",
        "selection": "minimum soft agreement per latent start, including initialization",
        "random_control": random_control,
        "gold_or_task_data_used": False,
    }
    checked_json(out / "protocol.json", protocol)
    result_path = out / "optimization.pt"
    if result_path.exists():
        print(f"[cross-pattern] verified existing {name}: {result_path}", flush=True)
    else:
        result = optimize_agreement(*models, n_starts=SEARCH_STARTS,
                                    steps=SEARCH_STEPS, lr=SEARCH_LR,
                                    seed=protocol["seed"], temperature=TEMPERATURE,
                                    radius=RADIUS, device=device, k=config.K_ACTIVE)
        torch.save(result, result_path)
    if random_control:
        random_path = out / "random_search.pt"
        if not random_path.exists():
            random_result = random_search(models, {
                "n_starts": SEARCH_STARTS, "steps": SEARCH_STEPS,
                "seed": protocol["seed"], "temperature": TEMPERATURE,
                "latent_radius": RADIUS,
            }, device)
            torch.save(random_result, random_path)
    hashes = {p.name: sha256_file(p) for p in out.glob("*.pt")}
    checked_json(out / "search_provenance.json", {
        "protocol_sha256": sha256_file(out / "protocol.json"),
        "artifact_sha256": hashes,
    })
    print(f"[cross-pattern] pair={pair_index} completed {name}", flush=True)


def search_stage(root: Path, pair_index: int, device: torch.device) -> None:
    for rep in range(4):
        search_one(root, pair_index, f"cross_{rep}", (rep, 0), (rep, 1),
                   device, random_control=True)
    for side, label in enumerate("ab"):
        for first, second in ((0, 1), (2, 3)):
            search_one(root, pair_index, f"within_{label}_{first}{second}",
                       (first, side), (second, side), device,
                       random_control=False)


def evaluation_stage(root: Path, pair_index: int, device: torch.device) -> None:
    from data.generate import ideal_mask  # post-hoc only
    from models.mlp import generate_fixed_sparsity_masks

    pair_root = path_for(root, pair_index)
    for rep in range(4):
        out = pair_root / f"cross_{rep}"
        if not (out / "optimization.pt").exists():
            raise FileNotFoundError(f"missing frozen search result in {out}")
        result = torch.load(out / "optimization.pt", map_location="cpu", weights_only=True)
        random = torch.load(out / "random_search.pt", map_location="cpu", weights_only=True)
        for side, pattern in enumerate(PAIRS[pair_index]):
            target = out / f"task_{pattern}.json"
            if target.exists():
                continue
            n = 16  # fixed first starts; no selection by gold or downstream score
            own = side + 1
            partner = 2 - side
            masks = {
                "own_initial": result[f"initial_masks{own}"][:n],
                "own_optimized": result[f"final_masks{own}"][:n],
                "partner_initial": result[f"initial_masks{partner}"][:n],
                "partner_optimized": result[f"final_masks{partner}"][:n],
                "random_search": random[f"masks{own}"][:n],
                "random_exact32": generate_fixed_sparsity_masks(
                    n, 8, 8, 32, 20260919 + rep),
                "ideal": ideal_mask().float().unsqueeze(0).expand(n, -1, -1),
            }
            scores = eval_one_task(masks, pattern, {
                "n_masks_per_method": n, "seed": 20260919 + rep,
                "steps": 2000, "batch_size": 128,
            }, device)
            target.write_text(json.dumps(scores, indent=2, allow_nan=False) + "\n")
            print(f"[cross-pattern] evaluated pair={pair_index} rep={rep} "
                  f"task={pattern}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--pair-index", type=int, required=True)
    parser.add_argument("--stage", choices=["train", "search", "evaluate", "all"], default="all")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if not 0 <= args.pair_index < len(PAIRS):
        raise ValueError("pair-index out of range")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    checked_json(args.root / "suite_protocol.json", {
        "experiment": "agreement between VAEs trained on different individual patterns",
        "pairs": [list(pair) for pair in PAIRS],
        "pair_selection": "four Hamming-distance-1 and four complementary pairs; all 16 patterns once",
        "replicates_per_pair": 4,
        "model_seeds": "30000 + 100*pair_index + 2*replicate + side",
        "maps_per_pattern": 800,
        "vae_epochs": EPOCHS,
        "search_starts": SEARCH_STARTS, "search_steps": SEARCH_STEPS,
        "within_pattern_controls": "two disjoint model-seed comparisons per pattern pair side",
        "analysis_unit": "pattern pair; model-seed replicates nested within pair",
    })
    if args.stage in ("train", "all"):
        training_stage(args.root, args.pair_index, device)
    if args.stage in ("search", "all"):
        search_stage(args.root, args.pair_index, device)
    if args.stage in ("evaluate", "all"):
        evaluation_stage(args.root, args.pair_index, device)


if __name__ == "__main__":
    main()

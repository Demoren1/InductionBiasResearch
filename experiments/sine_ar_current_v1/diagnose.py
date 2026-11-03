"""Diagnose bank filtering and temperature without changing the saved run."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.sine_ar_current_v1.run import solve, tasks
from pattern.autoregressive.model import IndexMADE

BASE = Path(__file__).resolve().parent
PROTOCOL = BASE / "diagnosis_protocol.json"
PLAN = json.loads(PROTOCOL.read_text())
SOURCE = BASE / PLAN["source_run"]
OUTPUT = BASE / PLAN["output"]
FIT_PLAN = json.loads((SOURCE / "protocol.json").read_text())


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tag(threshold):
    return f"threshold_{threshold:g}"


def codes(masks):
    return np.asarray(masks, dtype=np.int64) @ (1 << np.arange(16))


def pair_codes():
    return np.asarray(
        [
            [(1 << (2 * a)) + (1 << (2 * (b + 4) + 1)) for b in range(4)]
            for a in range(4)
        ]
    )


def load_bank(threshold):
    with np.load(OUTPUT / tag(threshold) / "bank.npz") as saved:
        return dict(saved)


def load_model(threshold, index, device):
    model = IndexMADE(16, 2, PLAN["prior_width"]).to(device)
    path = OUTPUT / tag(threshold) / f"prior_{index}.pt"
    model.load_state_dict(
        torch.load(path, weights_only=True, map_location=device)
    )
    return model.eval().requires_grad_(False)


def audit():
    OUTPUT.mkdir(exist_ok=False)
    shutil.copy2(PROTOCOL, OUTPUT / "protocol.json")
    shutil.copy2(__file__, OUTPUT / "diagnose_source.py")
    chunks = []
    source_hashes = {}
    for path in sorted((SOURCE / "source").glob("task_*.npz")):
        with np.load(path) as saved:
            chunks.append({k: saved[k] for k in ["masks", "query", "heldout"]})
        source_hashes[str(path.relative_to(BASE))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    arrays = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in chunks[0]
    }
    masks, query, heldout = (
        arrays["masks"],
        arrays["query"],
        arrays["heldout"],
    )
    pairs = pair_codes()
    rows = []
    for threshold in PLAN["threshold_audit"]:
        accepted = query < threshold
        unique, counts = np.unique(
            masks[accepted & ~heldout], axis=0, return_counts=True
        )
        probabilities = counts / counts.sum()
        bank_codes = codes(unique)
        pair_counts = np.zeros((4, 4), dtype=int)
        for code, count in zip(bank_codes, counts):
            pair_counts[pairs == code] = count
        row = {
            "threshold": threshold,
            "accepted": int(accepted.sum()),
            "train": int((accepted & ~heldout).sum()),
            "validation": int((accepted & heldout).sum()),
            "unique_train_masks": len(unique),
            "effective_masks": float(
                np.exp(-(probabilities * np.log(probabilities)).sum())
            ),
            "entropy_per_token": float(
                -(probabilities * np.log(probabilities)).sum() / 16
            ),
            "mean_source_query": float(query[accepted].mean()),
            "train_pair_counts": pair_counts.tolist(),
            "novel_pairs_already_in_bank": int(
                sum(pair_counts[a, (a + 3) % 4] > 0 for a in range(4))
            ),
        }
        rows.append(row)
        if threshold in PLAN["trained_thresholds"]:
            folder = OUTPUT / tag(threshold)
            folder.mkdir()
            np.savez_compressed(
                folder / "bank.npz",
                masks=masks[accepted],
                validation=heldout[accepted],
                source_query=query[accepted],
            )
        print(json.dumps(row), flush=True)
    unique, counts = np.unique(masks, axis=0, return_counts=True)
    dump(
        OUTPUT / "bank_audit.json",
        {
            "source_attempts": len(masks),
            "unique_proposed_masks": len(unique),
            "proposal_count_range": [int(counts.min()), int(counts.max())],
            "thresholds": rows,
            "source_sha256": source_hashes,
            "old_calibration_npz": sorted(
                p.name for p in (SOURCE / "calibration").glob("*.npz")
            ),
            "saving_bug": (
                "Path.with_suffix truncated decimal-temperature names, overwriting "
                "calibration NPZs. In-memory selection scores and target outputs "
                "were unaffected. The maintained save_fit now appends .npz."
            ),
        },
    )


@torch.no_grad()
def distribution(model, threshold, index, device):
    """Exact probabilities of every binary mask, including invalid ones."""
    integers = np.arange(1 << 16)
    masks = ((integers[:, None] >> np.arange(16)) & 1).astype(np.uint8)
    tensor = torch.as_tensor(masks, dtype=torch.long, device=device)
    differences = []
    for batch in tensor.split(4096):
        logits = model(batch)
        differences.append((logits[..., 1] - logits[..., 0]).cpu().numpy())
    delta = np.concatenate(differences).astype(np.float64)
    used = masks.cumsum(1) - masks
    forced_zero = used == 2
    forced_one = 2 - used == 16 - np.arange(16)
    pairs = pair_codes()
    a, b = np.indices((4, 4))
    train = np.isin((b - a) % 4, [0, 1])
    calibration = (b - a) % 4 == 2
    novel = (b - a) % 4 == 3
    rows = []
    all_probabilities = []
    for temperature in PLAN["diagnostic_temperatures"]:
        log_conditional = -np.logaddexp(
            0, np.where(masks, -delta, delta) / temperature
        )
        for method in ["ar", "ar_budget2"]:
            values = log_conditional.copy()
            if method == "ar_budget2":
                values = np.where(
                    forced_zero, np.where(masks == 0, 0, -np.inf), values
                )
                values = np.where(
                    forced_one, np.where(masks == 1, 0, -np.inf), values
                )
            probability = np.exp(values.sum(1))
            pair_probability = probability[pairs]
            row = {
                "threshold": threshold,
                "prior_index": index,
                "method": method,
                "temperature": temperature,
                "total_mass": float(probability.sum()),
                "strict_source_mass": float(pair_probability[train].sum()),
                "calibration_pair_mass": float(
                    pair_probability[calibration].sum()
                ),
                "novel_pair_mass": float(pair_probability[novel].sum()),
                "two_edge_mass": float(probability[masks.sum(1) == 2].sum()),
                "valid_pair_mass": float(pair_probability.sum()),
                "novel_exact_hit_32": float(
                    (1 - (1 - pair_probability[novel]) ** 32).mean()
                ),
                "pair_probabilities": pair_probability.tolist(),
                "edge_count_probabilities": np.bincount(
                    masks.sum(1).astype(int), weights=probability, minlength=17
                ).tolist(),
            }
            rows.append(row)
            all_probabilities.append(probability.astype(np.float32))
    folder = OUTPUT / tag(threshold)
    np.savez_compressed(
        folder / f"distribution_{index}.npz",
        probabilities=np.asarray(all_probabilities),
    )
    dump(folder / f"distribution_{index}.json", rows)
    torch.manual_seed(51001)
    ordinary = model.sample(2048, 2)
    torch.manual_seed(51001)
    cached = model.sample_cached(2048, 2)
    return {
        "threshold": threshold,
        "prior_index": index,
        "cached_sample_different_rows": int((ordinary != cached).any(1).sum()),
        "sample_rows": len(ordinary),
        "max_mass_normalization_error": max(
            abs(row["total_mass"] - 1) for row in rows
        ),
    }


def train(device):
    sampler_diagnostics = []
    for threshold in PLAN["trained_thresholds"]:
        bank = load_bank(threshold)
        unique, inverse = np.unique(bank["masks"], axis=0, return_inverse=True)
        train_count = np.bincount(
            inverse[~bank["validation"]], minlength=len(unique)
        )
        val_count = np.bincount(
            inverse[bank["validation"]], minlength=len(unique)
        )
        values = torch.as_tensor(unique, dtype=torch.long, device=device)
        train_weight = torch.as_tensor(
            train_count / train_count.sum(), dtype=torch.float32, device=device
        )
        val_weight = torch.as_tensor(
            val_count / val_count.sum(), dtype=torch.float32, device=device
        )
        folder = OUTPUT / tag(threshold)
        for index, seed in enumerate(PLAN["prior_seeds"]):
            if threshold == 0.1:
                for suffix in [".pt", "_history.json"]:
                    shutil.copy2(
                        SOURCE / f"prior_{index}{suffix}",
                        folder / f"prior_{index}{suffix}",
                    )
                model = load_model(threshold, index, device)
            else:
                torch.manual_seed(seed)
                model = IndexMADE(16, 2, PLAN["prior_width"]).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=PLAN["prior_learning_rate"]
                )
                history, best, selected = [], float("inf"), None
                for step in range(PLAN["prior_steps"] + 1):
                    loss = (
                        F.cross_entropy(
                            model(values).flatten(0, 1),
                            values.flatten(),
                            reduction="none",
                        )
                        .reshape(len(unique), 16)
                        .mean(1)
                    )
                    train_loss = (loss * train_weight).sum()
                    if step % 25 == 0:
                        val_loss = float((loss.detach() * val_weight).sum())
                        history.append(
                            {
                                "step": step,
                                "train_nll": float(train_loss.detach()),
                                "validation_nll": val_loss,
                            }
                        )
                        if val_loss < best:
                            best = val_loss
                            selected = copy.deepcopy(model.state_dict())
                    if step == PLAN["prior_steps"]:
                        break
                    optimizer.zero_grad(set_to_none=True)
                    train_loss.backward()
                    optimizer.step()
                model.load_state_dict(selected)
                model.eval().requires_grad_(False)
                torch.save(selected, folder / f"prior_{index}.pt")
                dump(folder / f"prior_{index}_history.json", history)
            diagnostic = distribution(model, threshold, index, device)
            sampler_diagnostics.append(diagnostic)
            print(json.dumps(diagnostic), flush=True)
    dump(OUTPUT / "sampler_diagnostics.json", sampler_diagnostics)


def calibrate(device):
    specs = json.loads((SOURCE / "calibration_tasks.json").read_text())
    published = json.loads((SOURCE / "selection_frozen.json").read_text())
    original_scores = {r["temperature"]: r for r in published["ar_scores"]}
    rows = []
    for threshold in PLAN["trained_thresholds"]:
        bank = load_bank(threshold)
        folder = OUTPUT / tag(threshold) / "calibration"
        folder.mkdir()
        models = [load_model(threshold, i, device) for i in range(3)]
        for temperature in PLAN["temperatures"]:
            if threshold == 0.1 and temperature in original_scores:
                scores = original_scores[temperature]["seed_query"]
                reused = True
            else:
                scores = []
                reused = False
                for index, model in enumerate(models):
                    result, _ = solve(
                        FIT_PLAN,
                        specs,
                        "ar",
                        FIT_PLAN["calibration_searches"],
                        FIT_PLAN["seed"] + 60000 + index * 1000,
                        bank,
                        model,
                        temperature,
                        FIT_PLAN["structured_sgd"],
                        device,
                    )
                    np.savez_compressed(
                        folder / f"t{temperature}_seed{index}.npz", **result
                    )
                    scores.append(float(result["selected_query"].mean()))
            row = {
                "threshold": threshold,
                "temperature": temperature,
                "mean_query": float(np.mean(scores)),
                "seed_query": scores,
                "reused_published_score": reused,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    selected = {
        "strict": min(
            (r for r in rows if r["threshold"] == 0.1),
            key=lambda r: r["mean_query"],
        ),
        "relaxed": min(
            (r for r in rows if r["threshold"] > 0.1),
            key=lambda r: r["mean_query"],
        ),
        "overall": min(rows, key=lambda r: r["mean_query"]),
        "frozen_before_new_functions": True,
    }
    dump(OUTPUT / "calibration.json", rows)
    dump(OUTPUT / "selection_frozen.json", selected)
    print(json.dumps({"selection": selected}), flush=True)


def evaluate(device):
    selected = json.loads((OUTPUT / "selection_frozen.json").read_text())
    old = json.loads((SOURCE / "selection_frozen.json").read_text())
    relaxed = selected["relaxed"]["threshold"]
    cases = [
        ("strict_t2", "ar", 0.1, 2),
        ("strict_retuned", "ar", 0.1, selected["strict"]["temperature"]),
        ("relaxed_t2", "ar", relaxed, 2),
        (
            "relaxed_selected",
            "ar",
            relaxed,
            selected["relaxed"]["temperature"],
        ),
        ("relaxed_replay", "replay", relaxed, 1),
        ("strict_frequency", "frequency", 0.1, 1),
        ("relaxed_frequency", "frequency", relaxed, 1),
        ("uniform", "uniform", 0.1, 1),
        ("dense_sgd", "dense_sgd", 0.1, 1),
        ("dense_adam", "dense_adam", 0.1, 1),
        ("oracle", "oracle", 0.1, 1),
    ]
    dump(OUTPUT / "evaluation_cases.json", cases)
    for condition, remainders, searches in [
        ("novel", [3], FIT_PLAN["novel_searches_per_task_seed"]),
        ("familiar", [0, 1], FIT_PLAN["familiar_searches_per_task_seed"]),
    ]:
        specs = tasks(PLAN["new_function_seeds"], remainders)
        folder = OUTPUT / condition
        folder.mkdir()
        dump(folder / "tasks.json", specs)
        for index in range(3):
            for name, method, threshold, temperature in cases:
                bank = load_bank(threshold)
                model = load_model(threshold, index, device)
                config = (
                    old["dense"][method.removeprefix("dense_")]
                    if method.startswith("dense_")
                    else FIT_PLAN["structured_sgd"]
                )
                result, _ = solve(
                    FIT_PLAN,
                    specs,
                    method,
                    searches,
                    PLAN["evaluation_seed"] + index * 1000,
                    bank,
                    model,
                    temperature,
                    config,
                    device,
                )
                np.savez_compressed(
                    folder / f"{name}_seed{index}.npz", **result
                )
                print(
                    json.dumps(
                        {
                            "condition": condition,
                            "method": name,
                            "prior_index": index,
                            "success": int(
                                (result["selected_population"] < 0.1).sum()
                            ),
                            "searches": result["selected_population"].size,
                            "mean_mse": float(
                                result["selected_population"].mean()
                            ),
                        }
                    ),
                    flush=True,
                )
    dump(OUTPUT / "COMPLETE.json", {"complete": True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["audit", "train", "calibrate", "evaluate"],
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "audit":
        audit()
    else:
        {"train": train, "calibrate": calibrate, "evaluate": evaluate}[
            args.phase
        ](args.device)


if __name__ == "__main__":
    main()

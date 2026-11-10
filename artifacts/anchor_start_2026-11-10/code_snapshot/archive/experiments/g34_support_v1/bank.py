"""Release fixed support or enforce full support in the original projector."""

import json
import time
from datetime import UTC, datetime

import numpy as np

from experiments.sharing_structure_research_v1 import blind_joint_bank as original
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import numpy_predict


def assign(initial, initial_mask, normal, uncertainty, codes, mode, steps=40):
    """Use the original numerical step; change only admissible assignments."""
    index = initial.copy()
    index[~initial_mask] = 0
    positions = np.flatnonzero(initial_mask)
    if len(positions) < codes:
        return None
    missing = index.ravel()[positions] == 0
    index.ravel()[positions[missing]] = np.resize(
        np.arange(1, codes + 1), missing.sum()
    )
    if mode == "full":
        missing = np.flatnonzero(index == 0)
        index.ravel()[missing] = np.resize(
            np.arange(1, codes + 1), len(missing)
        )
    initial_support = index != 0
    signed = np.r_[np.arange(1, codes + 1), -np.arange(1, codes + 1)]
    symbols = np.r_[0, signed] if mode == "free" else signed
    q_history, flips, updates = [], 0, 0
    for _ in range(steps):
        code, scales = original.scale_projection(index, normal, uncertainty, codes)
        q_history.append(original.projection_score(
            index, code, scales, normal, uncertainty
        ))
        centers = np.c_[code, -code]
        if mode == "free":
            centers = np.c_[np.zeros(len(code)), centers]
        target = scales[:, None] * normal
        deviation = (
            np.maximum(uncertainty, 0.0003)
            * np.maximum(np.abs(scales), 0.01)
        )
        standardized = (
            (target[..., None] - centers[:, None, None])
            / deviation[:, None, :, None]
        ) ** 2
        costs = np.mean(4 * np.log1p(standardized / 4), axis=0)
        updated = symbols[costs.argmin(-1)].astype(np.int16)
        # Keep the original nonempty-code repair; zero has no occupancy quota.
        for label in range(1, codes + 1):
            if np.any(np.abs(updated) == label):
                continue
            counts = np.bincount(np.abs(updated).ravel(), minlength=codes + 1)
            allowed = (updated == 0) | (counts[np.abs(updated)] > 1)
            if not allowed.any():
                return None
            plus = int(np.flatnonzero(symbols == label)[0])
            minus = int(np.flatnonzero(symbols == -label)[0])
            local = np.minimum(costs[..., plus], costs[..., minus])
            selected = int(np.where(allowed, local, np.inf).argmin())
            flat = costs.reshape(-1, len(symbols))
            updated.ravel()[selected] = (
                label if flat[selected, plus] <= flat[selected, minus] else -label
            )
        flips += int(np.sum((index == 0) != (updated == 0)))
        updates += 1
        if np.array_equal(index, updated):
            break
        index = updated
    code, scales = original.scale_projection(index, normal, uncertainty, codes)
    score = original.projection_score(index, code, scales, normal, uncertainty)
    q_history.append(score)
    diagnostic = {
        "initial_active": int(initial_support.sum()),
        "final_active": int((index != 0).sum()),
        "final_support_changes": int(np.sum(initial_support != (index != 0))),
        "support_flip_events": flips,
        "assignment_updates": updates,
        "q": q_history,
    }
    return index, code, scales, score, diagnostic


def propose(root, case, condition, mode, anchors=4, starts=16):
    sizes = load_sizes(root, case)
    codes = sizes["codes"]
    folder = root / "fits" / f"{case}_c{condition}"
    base = dict(np.load(folder / "fitted.npz"))
    tasks = base["parameters"].shape[1]
    source = [
        dict(np.load(root / "source" / f"{case}_c{condition}_t{t}" / "learned.npz"))
        for t in range(tasks)
    ]
    ranked = base["validation_mse"].mean(1).argsort()[:anchors]
    rng = np.random.default_rng(33670019)
    bank, origin, runs = {}, {}, []
    started = time.monotonic()
    for number in ranked:
        anchor = base["indices"][number]
        mappings = [
            original.align_source(
                anchor, base["parameters"][number, t, :codes],
                source[t]["normal"], source[t]["uncertainty"],
            )
            for t in range(tasks)
        ]
        normal = np.stack([s["normal"][:, m] for s, m in zip(source, mappings)])
        uncertainty = np.stack([
            s["uncertainty"][m] for s, m in zip(source, mappings)
        ])
        evidence = np.abs(normal) > np.maximum(3 * uncertainty[:, None], 0.002)
        for votes in (0, 1, 2, 3):
            initial_mask = (
                anchor != 0 if votes == 0
                else evidence.sum(0) >= min(votes, tasks)
            )
            seeds = [anchor]
            for observed, mapping in zip(source, mappings):
                for key in [k for k in observed if k.startswith("index_")][:2]:
                    seeds.append(observed[key][:, mapping])
            while len(seeds) < starts:
                label = rng.integers(1, codes + 1, size=initial_mask.shape)
                sign = np.where(normal[len(seeds) % tasks] >= 0, 1, -1)
                seeds.append((label * sign).astype(np.int16))
            for seed_number, seed in enumerate(seeds):
                result = assign(
                    seed, initial_mask, normal, uncertainty, codes, mode
                )
                if result is None:
                    continue
                index, code, scales, score, diagnostic = result
                runs.append({
                    "anchor": int(number), "votes": votes,
                    "seed": seed_number, **diagnostic,
                })
                if not np.isfinite(score):
                    continue
                for reference in range(tasks):
                    oriented = index * np.where(scales[reference] >= 0, 1, -1)[None]
                    canonical = canonical_index(oriented.astype(np.int16), codes)
                    key = canonical.tobytes()
                    if key in bank and bank[key][0] <= score:
                        continue
                    bank[key] = (score, canonical, np.abs(code))
                    origin[key] = {
                        "anchor": int(number), "votes": votes,
                        "seed": seed_number, "reference": reference,
                    }
        print(json.dumps({
            "case": case, "mode": mode, "anchor": int(number),
            "unique": len(bank), "seconds": time.monotonic() - started,
        }), flush=True)
    if not bank:
        raise ValueError("No complete support-ablation proposals")
    keys = sorted(bank, key=lambda key: bank[key][0])
    np.savez_compressed(
        folder / "joint_bank.npz",
        indices=np.stack([bank[k][1] for k in keys]),
        score=np.array([bank[k][0] for k in keys]),
        code_guesses=np.stack([bank[k][2] for k in keys]),
    )
    save_json(folder / "joint_bank.json", {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "seconds": time.monotonic() - started, "anchors": ranked.tolist(),
        "starts_minimum": starts, "mode": mode,
        "origins": [origin[k] for k in keys], "teacher_access": False,
        "sha256": digest(folder / "joint_bank.npz"),
    })
    save_json(folder / "optimization.json", {"mode": mode, "runs": runs})


def select(root, case, condition, mode):
    """Use the same independent meta observations and enforce the full control."""
    if mode == "free":
        original.select(root, case, condition)
        return
    sizes = load_sizes(root, case)
    folder = root / "fits" / f"{case}_c{condition}"
    fitted = dict(np.load(folder / "joint_fitted.npz"))
    meta = dict(np.load(root / "public" / f"{case}_c{condition}" / "meta.npz"))
    tasks = fitted["parameters"].shape[1]
    risks, predictions = [], []
    for index, parameters in zip(fitted["indices"], fitted["parameters"]):
        values = np.stack([
            numpy_predict(
                meta[f"x_{t}"], index, parameters[t],
                sizes["codes"], sizes["outputs"],
            )
            for t in range(tasks)
        ])
        target = np.stack([meta[f"y_{t}"] for t in range(tasks)])
        predictions.append(values)
        risks.append(float(np.mean((values - target) ** 2)))
    allowed = np.flatnonzero(np.all(fitted["indices"] != 0, axis=(1, 2)))
    if not len(allowed):
        raise ValueError("The full-support control has no admissible candidate")
    chosen = int(allowed[np.argmin(np.array(risks)[allowed])])
    np.savez_compressed(
        folder / "joint_selected.npz",
        joint_functional_index=fitted["indices"][chosen],
        joint_functional_parameters=fitted["parameters"][chosen],
        joint_functional_meta=predictions[chosen],
    )
    save_json(folder / "joint_selected.json", {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "methods": {"joint_functional": {
            "candidate": chosen, "meta_mse": risks[chosen],
            "joint_geometry_score": float(fitted["joint_geometry_score"][chosen]),
        }},
        "all_meta_mse": risks, "eligible_full_candidates": allowed.tolist(),
        "teacher_access": False,
        "sha256": digest(folder / "joint_selected.npz"),
    })

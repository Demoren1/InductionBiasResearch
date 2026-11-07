"""Explore sharing separately after freezing a source-selected DARTS mask."""

import argparse
import itertools
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from .data import BASE, digest, save_json, teacher_predict
from .evaluate import align_indices
from .model import score
from .train import fit, save_fit


def run(parent, device):
    output = parent.parent / "deepsets_mask_then_sharing"
    output.mkdir(exist_ok=False)
    plan = json.loads((parent / "protocol.json").read_text())
    spec = json.loads((parent / "specification.json").read_text())
    choice = json.loads((parent / "architecture_selection.json").read_text())
    with np.load(parent / "confirmation/hard_fits.npz") as archive:
        mask = archive["indices"][choice["primary"]["row"]] != 0
    shutil.copy2(parent / "source.npz", output / "source.npz")
    save_json(output / "specification.json", spec)
    save_json(
        output / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "status": "exploratory follow-up after the first test had been inspected",
            "reason": "Direct signed-DARTS improved the source score but left unused code groups. Test a generic separation of connectivity and numeric grouping; no teacher assignment enters this continuation.",
            "source_mask_parent": str(parent),
            "source_mask_row": choice["primary"]["row"],
            "mask_sha256": digest(parent / "confirmation/hard_fits.npz"),
            "steps": [
                "Freeze the binary mask from the previously source-selected categorical I.",
                "Fit independent weights within that frozen mask, shared b/a/d, four starts per source-only setting.",
                "For each configuration choose its numerical start/checkpoint by query, then one-dimensional k-means of absolute active weights into the known m base groups; signs become signed I.",
                "Freeze each proposed integer I and refit only C/b/a/d. Select by independent selection MSE.",
                "Evaluate 32 new codebooks at the source-selected numerical settings and a new held-out input draw. Do not describe this as pure soft DARTS or as an untouched benchmark.",
            ],
        },
    )
    snapshots = output / "code_snapshot"
    snapshots.mkdir()
    for name in ("mask_sharing.py", "train.py", "model.py"):
        shutil.copy2(BASE / name, snapshots / name)
    rows = [
        {"numeric_lr": lr, "radius": radius, "seed": seed}
        for lr, radius, seed in itertools.product(
            [0.003, 0.01, 0.03], [0.5, 1.5], [91, 92]
        )
    ]
    numeric = fit(
        output, plan, spec, rows, device, mask=np.repeat(mask[None], len(rows), axis=0)
    )
    save_fit(output / "masked_numeric", numeric, rows)
    proposals, inherited = [], []
    codes = spec["codes"]
    for number, row in enumerate(rows):
        restart = int(numeric["query_mse"][number].argmin())
        weights = numeric["weights"][number, restart]
        values = np.abs(weights[mask])[:, None]
        clusters = KMeans(
            n_clusters=codes, n_init=16, random_state=plan["seed"] + number
        ).fit(values)
        order = np.argsort(clusters.cluster_centers_[:, 0])
        labels = np.empty(codes, dtype=int)
        labels[order] = np.arange(1, codes + 1)
        index = np.zeros_like(mask, dtype=np.int16)
        index[mask] = labels[clusters.labels_] * np.sign(weights[mask]).astype(int)
        proposals.append(index)
        inherited.append(
            np.r_[
                clusters.cluster_centers_[order, 0],
                numeric["numeric"][number, restart, -3:],
            ]
        )
    initial = np.array(inherited)[:, None]
    result = fit(
        output,
        plan,
        spec,
        rows,
        device,
        indices=np.array(proposals),
        initial=initial,
        restarts=1,
    )
    result["indices"] = np.array(proposals)
    save_fit(output / "hard_sharing", result, rows)
    selected = int(result["selection_mse"][:, 0].argmin())
    save_json(
        output / "selection.json",
        {
            "row": selected,
            "settings": rows[selected],
            "selection_mse": float(result["selection_mse"][selected, 0]),
            "selected_indexator": proposals[selected].tolist(),
        },
    )
    fresh_rows = [{**rows[selected], "seed": 93, "aligned_initial_codes": False}]
    fresh = fit(
        output,
        plan,
        spec,
        fresh_rows,
        device,
        indices=np.array(proposals[selected])[None],
        restarts=32,
    )
    save_fit(output / "fresh", fresh, fresh_rows)
    fresh_restart = int(fresh["query_mse"][0].argmin())
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source_row": selected,
            "fresh_restart": fresh_restart,
            "sha256": {
                name: digest(output / name)
                for name in (
                    "source.npz",
                    "registration.json",
                    "selection.json",
                    "hard_sharing.npz",
                    "fresh.npz",
                )
            },
        },
    )
    with np.load(parent / "teacher_audit_only.npz") as archive:
        truth = dict(archive)
    rng = np.random.default_rng(plan["seed"] + 1900000)
    x = rng.normal(size=(8192, spec["inputs"])).astype(np.float32)
    y = teacher_predict(x, truth, spec)
    x_tensor, y_tensor = torch.tensor(x, device=device), torch.tensor(y, device=device)
    inherited_mse = float(
        score(
            x_tensor,
            y_tensor,
            torch.tensor(result["weights"][selected : selected + 1], device=device),
            torch.tensor(result["numeric"][selected : selected + 1], device=device),
            spec,
        )[0, 0]
    )
    fresh_mse = (
        score(
            x_tensor,
            y_tensor,
            torch.tensor(fresh["weights"], device=device),
            torch.tensor(fresh["numeric"], device=device),
            spec,
        )
        .cpu()
        .numpy()[0]
    )
    recovery, aligned, columns = align_indices(
        proposals[selected], truth["indices"], codes
    )
    np.savez_compressed(
        output / "selected_states.npz",
        indices=proposals[selected],
        aligned_indices=aligned,
        aligned_columns=columns,
        true_indices=truth["indices"],
        learned_inherited_weights=result["weights"][selected, 0],
        learned_inherited_numeric=result["numeric"][selected, 0],
        learned_fresh_weights=fresh["weights"][0, fresh_restart],
        learned_fresh_numeric=fresh["numeric"][0, fresh_restart],
    )
    summary = {
        "task": spec,
        "method": "DARTS mask -> independent masked weights -> signed k-means -> fixed-I codebook fit",
        "exploratory": True,
        "test_seed": plan["seed"] + 1900000,
        "recovery": recovery,
        "inherited_test_mse": inherited_mse,
        "inherited_test_nmse": inherited_mse / spec["target_variance"],
        "fresh_test_mse": float(fresh_mse[fresh_restart]),
        "fresh_test_nmse": float(fresh_mse[fresh_restart] / spec["target_variance"]),
        "fresh_successes": int((fresh_mse / spec["target_variance"] < 0.01).sum()),
        "fresh_restarts": 32,
    }
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run(args.parent, torch.device("cuda:0"))


if __name__ == "__main__":
    main()

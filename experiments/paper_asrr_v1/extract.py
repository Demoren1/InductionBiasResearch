"""Construct source-only aSRR candidates at the manuscript's fixed budgets."""

import argparse
import json
import shutil
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)
from experiments.srr_approx_v1.core import (
    canonical,
    input_forest,
    local_forest,
    signed_forest,
    synchronize_profiles,
)
from experiments.srr_approx_v1.projection import project_chunked

from .prepare import GRID, OUTPUT, PREVIOUS, ROOT


def source_profiles(path, dimensions, hidden):
    with np.load(path) as data:
        ranking = (
            data["numerical_cross_entropy"].min(0).argsort()
            if "numerical_cross_entropy" in data
            else np.arange(len(data["parameters"]))
        )
        selected = (
            data["parameters"][ranking, : dimensions * hidden]
            .reshape(-1, dimensions, hidden)
            .astype(float)
        )
        ensemble = (
            data["ensemble_parameters"][ranking, :, : dimensions * hidden]
            .reshape(len(selected), -1, dimensions, hidden)
            .astype(float)
            if "ensemble_parameters" in data
            else selected[:, None]
        )
    normal = selected / np.maximum(
        np.linalg.norm(selected, axis=1, keepdims=True), 1e-12
    )
    matched, means, errors = [], [], []
    for reference, choices in zip(normal, ensemble):
        aligned = []
        for weight in choices:
            unit = weight / np.maximum(np.linalg.norm(weight, axis=0), 1e-12)
            left, right = linear_sum_assignment(-reference.T @ unit)
            aligned.append(unit[:, right[np.argsort(left)]])
        aligned = np.stack(aligned)
        left, right = linear_sum_assignment(-normal[0].T @ reference)
        order = right[np.argsort(left)]
        matched.append(reference[:, order])
        means.append(aligned.mean(0)[:, order])
        error = np.sqrt(np.mean((aligned - reference) ** 2, axis=(0, 1)))
        errors.append(np.maximum(error[order], 0.0003))
    return np.stack(matched), np.stack(means), np.stack(errors), ranking


def run(dataset, device):
    torch.set_num_threads(2)
    root, output = (ROOT / dataset).resolve(), (OUTPUT / dataset).resolve()
    boundary = StudyBoundary(root)
    boundary.install()
    started = time.perf_counter()
    sizes = json.loads((root / "protocol.json").read_text())
    budgets = (4,) if dataset.startswith("synthetic") else GRID
    auxiliary = 24 if dataset.startswith("synthetic") else 256
    selected, mean, error, ranking = source_profiles(
        root / "source/dense/models.npz", sizes["inputs"], sizes["hidden"]
    )
    folder = root / "source/asrr"
    folder.mkdir()
    np.savez_compressed(
        folder / "profiles.npz",
        selected=selected,
        consensus=mean,
        uncertainty=error,
        ranking=ranking,
    )
    global_bank, global_trace = signed_forest(
        mean, sorted(set((*budgets, auxiliary))), device=device
    )
    synchronized = synchronize_profiles(global_bank[auxiliary], mean, error)
    sync_bank, sync_trace = signed_forest(synchronized, budgets, device=device)
    families = {"global": global_bank, "synchronized": sync_bank}
    if max(budgets) >= sizes["hidden"]:
        families.update(
            local=local_forest(mean, budgets),
            input=input_forest(mean, budgets),
            selected_input=input_forest(selected, budgets),
        )
    arrays, rows, seen = {}, [], {m: set() for m in budgets}
    for codes in budgets:
        candidates = [
            (family, bank[codes], None)
            for family, bank in families.items()
            if codes in bank
        ]
        for family in ("global", "synchronized"):
            refined, trace = project_chunked(
                families[family][codes], mean, error, steps=12, device=device
            )
            candidates.append((f"{family}_a1", refined, trace))
        for family, index, trace in candidates:
            index = canonical(index)
            if int(np.abs(index).max()) != codes or np.any(index == 0):
                raise RuntimeError(
                    f"Invalid aSRR budget/support: {family}/{codes}"
                )
            if index.tobytes() in seen[codes]:
                continue
            seen[codes].add(index.tobytes())
            name = f"{family}_m{codes}"
            arrays[name] = index
            rows.append(
                {
                    "name": name,
                    "family": family,
                    "codes": codes,
                    "anchor": int(ranking[0]),
                    "projection": trace,
                }
            )
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "budget": codes,
                    "candidates": len(seen[codes]),
                }
            ),
            flush=True,
        )
    np.savez_compressed(folder / "candidates.npz", **arrays)
    save_json(folder / "candidates.json", {"candidates": rows})
    np.savez_compressed(
        folder / "trees.npz",
        global_tree=global_trace["tree"],
        synchronized_tree=sync_trace["tree"],
    )
    source_hash = digest(root / "source/dense/models.npz")
    expected = None
    if not dataset.startswith("synthetic"):
        expected = json.loads(
            (PREVIOUS / dataset / "pair/registration.json").read_text()
        )["previous_dense_sha256"]
    save_json(
        output / "extraction.json",
        {
            "source_sha256": source_hash,
            "original_source_sha256": expected,
            "identical_source_archive": source_hash == expected
            if expected
            else None,
            "candidates": rows,
            "seconds": time.perf_counter() - started,
            "candidate_sha256": digest(folder / "candidates.npz"),
        },
    )
    for name in ("candidates.npz", "candidates.json", "profiles.npz"):
        shutil.copy2(folder / name, output / name)
    boundary.save(output / "extraction_access.json", "complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(args.dataset, args.device)

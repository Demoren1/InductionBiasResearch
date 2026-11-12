"""Select code budget and adaptation rule, preserving geometric top-one."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    fit_adam,
    signed_basis,
)
from experiments.paper_structure_benchmark_v1.fitting.real_validation import (
    choose_source,
)
from experiments.paper_structure_benchmark_v1.protocol.real_boundary import (
    StudyBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json


def run(root, output, method, device):
    torch.set_num_threads(2)
    if method != "dense" and not (root / "source_choices_frozen.json").exists():
        raise RuntimeError("Source structures must be frozen first")
    boundary = StudyBoundary(root, "validation")
    boundary.install()
    folder = root / "validation_fit" / method
    folder.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    h, o = protocol["hidden"], protocol["outputs"]
    if method == "dense":
        budgets = [0]
    elif method == "signature":
        budgets = json.loads((output / "geometric_choices_frozen.json").read_text())[
            "available_budgets"
        ]
    elif method == "signature_random":
        selection = json.loads(
            (root / "validation_fit/signature/selected.json").read_text()
        )
        budgets = [selection["source_selection"]["codes"]]
    else:
        budgets = protocol["independent_code_grid"]
    save_json(
        folder / "registered.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "method": method,
            "code_sha256": digest(Path(__file__)),
            "code_grid": budgets,
            "max_numeric_steps": 2000,
            "choice": "Mean development CE over 128 tasks and both fresh restarts; choose budget, global LR, activity quantile, checkpoint",
            "signature_structure_choice": "First geometric candidate already frozen; never rerank its bank on source M or development tasks",
        },
    )
    data = dict(np.load(root / "validation" / "validation.npz"))
    x, y, vx, vy = [
        torch.tensor(data[key], dtype=torch.float32, device=device)
        for key in ("support_x", "support_y", "query_x", "query_y")
    ]
    x, vx = x / 255, vx / 255
    records, curves, chosen = [], {}, None
    started = time.perf_counter()
    for codes in budgets:
        if method == "signature":
            origin = root / "fits" / f"signature_k{codes}" / "selected.npz"
            arrays = np.load(origin)
            index = arrays["index"]
            basis = torch.tensor(arrays["basis"], device=device, dtype=torch.float32)
            source = {
                "codes": codes,
                "selection": "First geometric signature candidate",
                "structure_sha256": digest(origin),
            }
        elif method == "signature_random":
            origin = root / "validation_fit/signature/model.npz"
            index = np.load(origin)["index"].copy()
            index = (
                np.random.default_rng(261112991)
                .permutation(index.ravel())
                .reshape(index.shape)
            )
            basis = signed_basis(index, codes, device, torch.float32)
            basis /= basis.square().sum(-1).mean().sqrt().clamp_min(1e-12)
            source = {
                "codes": codes,
                "selection": "Shuffle selected signature I; preserve density, group counts and sign counts",
                "matched_structure_sha256": digest(origin),
            }
        else:
            basis, index, source = choose_source(root, method, [codes], device)
        _, trace = fit_adam(
            x, y, vx, vy, basis, h, o, loss="cross_entropy", steps=2000, seed=769003
        )
        mean = (
            trace["validation_history"]
            .mean(-1)
            .reshape(len(trace["steps"]), 9, 2)
            .mean(-1)
        )
        checkpoint, scenario = np.unravel_index(mean.argmin(), mean.shape)
        record = {
            "codes": codes,
            "validation_mean_loss": float(mean[checkpoint, scenario]),
            "numeric_updates": int(trace["steps"][checkpoint]),
            "global_setting": trace["settings"][2 * scenario],
            "source_selection": source,
        }
        records.append(record)
        for key, value in trace.items():
            if isinstance(value, np.ndarray):
                curves[f"k{codes}_{key}"] = value
        if (
            chosen is None
            or record["validation_mean_loss"] < chosen["validation_mean_loss"]
        ):
            chosen = record
            selected_basis = np.empty((0,)) if basis is None else basis.cpu().numpy()
            selected_index = (
                np.empty((0,), dtype=np.int16) if index is None else index.copy()
            )
        print(json.dumps({"method": method, **record}), flush=True)
    if chosen is None:
        raise RuntimeError("Signature algorithm produced no complete partition")
    np.savez_compressed(
        folder / "model.npz", basis=selected_basis, index=selected_index
    )
    np.savez_compressed(folder / "all_curves.npz", **curves)
    save_json(
        folder / "selected.json",
        {
            "method": method,
            "global_setting": chosen["global_setting"],
            "numeric_updates": chosen["numeric_updates"],
            "support_count": 25,
            "validation_mean_loss": chosen["validation_mean_loss"],
            "source_selection": chosen["source_selection"],
            "all_budget_choices": records,
            "model_sha256": digest(folder / "model.npz"),
            "target_query_used": False,
        },
    )
    save_json(
        folder / "metrics.json",
        {"status": "complete", "wall_seconds": time.perf_counter() - started},
    )
    boundary.save(folder / "access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.output.resolve(),
        args.method,
        torch.device(args.device),
    )


if __name__ == "__main__":
    main()

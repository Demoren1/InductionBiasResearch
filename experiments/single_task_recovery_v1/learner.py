"""Freeze a direct-ratio bank, then fit and select on source observations."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1 import (
    blind_orientation_refine as orientation,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    load_sizes,
    save_json,
)
from experiments.sharing_structure_research_v1.blind_numeric import (
    fit_batch,
    numpy_predict,
)
from experiments.sharing_structure_research_v1.blind_runtime import (
    ObservationBoundary,
)

from .ratios import ratio_proposals


def extract(root, case):
    boundary = ObservationBoundary(root, case, 0)
    boundary.phase = "ratio_proposals"
    boundary.install()
    sizes = load_sizes(root, case)
    folder = root / "ratio" / f"{case}_c0"
    folder.mkdir(exist_ok=False)
    source = dict(np.load(root / "source" / f"{case}_c0_t0" / "learned.npz"))
    save_json(
        folder / "registered.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "source_sha256": digest(
                root / "source" / f"{case}_c0_t0" / "learned.npz"
            ),
            "algorithm_sha256": digest(Path(__file__).with_name("ratios.py")),
            "sources": 1,
            "bank_limit": 32,
            "thresholds": "0 plus 48 geometric values from 1e-12 to 0.2",
            "completion": (
                "Retain exactly four groups; report partial partitions"
            ),
        },
    )
    bank, diagnostics = ratio_proposals(source, sizes["codes"])
    shape = (0, sizes["inputs"], sizes["hidden"])
    np.savez_compressed(
        folder / "candidates.npz",
        indices=np.stack([item["index"] for item in bank])
        if bank
        else np.empty(shape, dtype=np.int16),
        scores=np.array([item["score"] for item in bank]),
        code_guesses=np.stack([item["code_guesses"] for item in bank])
        if bank
        else np.empty((0, 1, sizes["codes"])),
    )
    save_json(
        folder / "candidates.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": digest(folder / "candidates.npz"),
            "origins": [item["origin"] for item in bank],
            "diagnostics": diagnostics,
            "teacher_access": False,
        },
    )
    boundary.save(root / "receipts" / f"{case}_ratio_extract.json", "complete")
    print(json.dumps({"case": case, **diagnostics}), flush=True)


def refit(root, case, device):
    torch.set_num_threads(2)
    boundary = ObservationBoundary(root, case, 0)
    boundary.phase = "ratio_numeric"
    boundary.install()
    sizes = load_sizes(root, case)
    codes, outputs = sizes["codes"], sizes["outputs"]
    folder = root / "ratio" / f"{case}_c0"
    bank = dict(np.load(folder / "candidates.npz"))
    source = dict(np.load(root / "source" / f"{case}_c0_t0" / "learned.npz"))
    train, validation = [
        dict(np.load(root / "public" / f"{case}_c0" / f"{name}.npz"))
        for name in ("fixed_train", "selection")
    ]
    x, y, vx, vy = (
        train["x_0"],
        train["y_0"],
        validation["x_0"],
        validation["y_0"],
    )
    started = time.perf_counter()
    states, losses = [], []
    for number, index in enumerate(bank["indices"]):
        initial = orientation.initial_states(
            index,
            {"codes_joint": bank["code_guesses"][number, 0], **source},
            codes,
            x,
            y,
            vx,
            vy,
            device,
            keep=4,
        )
        fitted, trace = fit_batch(
            np.repeat(index[None], 4, axis=0),
            initial,
            *[np.repeat(value[None], 4, axis=0) for value in (x, y, vx, vy)],
            codes,
            device,
            steps=48,
        )
        chosen = int(trace["validation_mse"].argmin())
        states.append(fitted[chosen])
        losses.append(float(trace["validation_mse"][chosen]))
        print(
            json.dumps(
                {
                    "case": case,
                    "ratio_refit": number + 1,
                    "total": len(bank["indices"]),
                }
            ),
            flush=True,
        )
    boundary.role, boundary.phase = "select", "ratio_selection"
    meta = dict(np.load(root / "public" / f"{case}_c0" / "meta.npz"))
    risks = [
        float(
            np.mean(
                (
                    numpy_predict(meta["x_0"], index, state, codes, outputs)
                    - meta["y_0"]
                )
                ** 2
            )
        )
        for index, state in zip(bank["indices"], states)
    ]
    selected = int(np.argmin(risks)) if risks else None
    parameter_count = codes + sizes["hidden"] * (1 + outputs) + outputs
    np.savez_compressed(
        folder / "fitted.npz",
        indices=bank["indices"],
        parameters=np.array(states).reshape(-1, parameter_count),
        validation_mse=np.array(losses),
        meta_mse=np.array(risks),
    )
    save_json(
        folder / "selected.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected": selected,
            "geometric_first": 0 if risks else None,
            "all_meta_mse": risks,
            "all_validation_mse": losses,
            "seconds": time.perf_counter() - started,
            "candidate_count": len(risks),
            "starts": 4,
            "gn_steps": 48,
            "sha256": digest(folder / "fitted.npz"),
            "teacher_access": False,
        },
    )
    boundary.save(root / "receipts" / f"{case}_ratio_refit.json", "complete")
    print(
        json.dumps(
            {
                "case": case,
                "selected": selected,
                "meta_mse": risks[selected] if risks else None,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--case", choices=("case_04", "case_05"), required=True
    )
    parser.add_argument("--phase", choices=("extract", "refit"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.phase == "extract":
        extract(args.root.resolve(), args.case)
    else:
        refit(args.root.resolve(), args.case, torch.device(args.device))


if __name__ == "__main__":
    main()

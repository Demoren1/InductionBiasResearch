"""Select the native MSR model without discarding its learned initialization."""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json
from ..fitting.adaptation import losses
from ..methods.bilevel import predict
from .real_boundary import StudyBoundary


def run(root):
    torch.set_num_threads(2)
    boundary = StudyBoundary(root, "select")
    boundary.install()
    folder = root / "targets" / "msr_native"
    folder.mkdir(exist_ok=False)
    protocol = json.loads((root / "protocol.json").read_text())
    h, o, kind = protocol["hidden"], protocol["outputs"], protocol["loss"]
    panel = dict(np.load(root / "public" / "structure.npz"))
    x, y = torch.tensor(panel["x"], dtype=torch.float32) / 255, torch.tensor(panel["y"])
    save_json(
        folder / "prepare_registered.json",
        {
            "code_sha256": digest(Path(__file__)),
            "selection": "Native source-M loss after three learned-SGD steps; no common fresh refit",
            "transfer": "Auxiliary native MSR regime: transfer both U and learned numeric initialization/rates",
            "not_structure_only": True,
        },
    )
    tick = time.perf_counter()
    records, best, chosen = [], float("inf"), None
    for codes in protocol["independent_code_grid"]:
        path = root / "source" / f"msr_k{codes}" / "native.npz"
        native = np.load(path)
        numeric = torch.tensor(native["source_parameters"]).clone()
        center, scale = (
            torch.tensor(native["normalization_center"]),
            torch.tensor(native["normalization_scale"]),
        )
        numeric[..., codes + h : -o] *= scale[None, :, None]
        numeric[..., -o:] = numeric[..., -o:] * scale[None, :, None] + center[None]
        basis = torch.tensor(native["basis"])
        errors = []
        with torch.no_grad():
            for begin in range(0, len(basis), 2):
                value = predict(
                    x, basis[begin : begin + 2], numeric[begin : begin + 2], h, o
                )
                errors.extend(losses(value, y, kind).mean(1).tolist())
        scenario = int(np.argmin(errors))
        records.append(
            {
                "codes": codes,
                "source_M_loss_per_configuration": errors,
                "native_sha256": digest(path),
            }
        )
        if errors[scenario] < best:
            best = errors[scenario]
            chosen = {
                "basis": native["basis"][scenario],
                "initial": native["native_initialization"][scenario],
                "rates": native["native_rates"][scenario],
            }
            selected = {
                "codes": codes,
                "scenario": scenario,
                "source_M_loss": best,
                "native_sha256": digest(path),
            }
    np.savez_compressed(folder / "model.npz", **chosen)
    save_json(
        folder / "prepare_selected.json",
        {
            "status": "complete",
            "model_sha256": digest(folder / "model.npz"),
            "selected": selected,
            "all_candidates": records,
            "target_steps": 3,
            "transferred_numeric_initialization": True,
            "wall_seconds": time.perf_counter() - tick,
        },
    )
    shutil.copy2(__file__, folder / "prepare_executed_code.py")
    boundary.save(folder / "prepare_access_receipt.json", "complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()

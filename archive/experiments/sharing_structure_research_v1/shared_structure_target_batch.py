"""Measured execution and exact-task reuse for frozen-I target adaptation."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .shared_structure_numeric import fit_codes as serial_fit
from .shared_structure_numeric import fresh_codes
from .shared_structure_numeric_batch import fit_codes as batch_fit


def load_job(root, plan, job_id, device):
    phase = root / "transfer"
    path = phase / "banks" / f"job_{job_id:03d}"
    bank = json.loads(path.with_suffix(".json").read_text())
    data_path = phase / "data" / f"family_{bank['family_id']:03d}.npz"
    frozen = json.loads((phase / "data_frozen.json").read_text())
    if digest(data_path) != frozen["sha256"][str(data_path.relative_to(phase))]:
        raise RuntimeError("Frozen target observations changed")
    with np.load(path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    with np.load(data_path) as arrays:
        x, vx, y, vy = [
            torch.tensor(arrays[name], device=device)
            for name in (
                "support_x",
                "validation_x",
                f"support_y_{bank['noise']}",
                f"validation_y_{bank['noise']}",
            )
        ]
    family = plan["families"][bank["family_id"]]["family"]
    initial = torch.stack(
        [
            fresh_codes(plan, family, plan["sources"] + task, device)
            for task in range(plan["targets"])
        ]
    )
    identity = {
        "family": family,
        "noise": bank["noise"],
        "data_sha256": digest(data_path),
        "initial_C_sha256": hashlib.sha256(initial.cpu().numpy().tobytes()).hexdigest(),
        "numeric_code_sha256": digest(
            Path(__file__).with_name("shared_structure_numeric.py")
        ),
        "batch_code_sha256": digest(
            Path(__file__).with_name("shared_structure_numeric_batch.py")
        ),
        "settings": {
            key: plan[key]
            for key in (
                "bias",
                "hidden",
                "codes",
                "axis_code_max",
                "numeric_code_steps",
                "numeric_ridge",
                "numeric_line_search",
                "numeric_chunk",
            )
        },
    }
    return bank, indices, (x, y, vx, vy, initial), identity


def profile(root, plan, device):
    destination = root / "transfer/execution_measurement.json"
    if destination.exists():
        raise RuntimeError("Target execution measurement already exists")
    bank, indices, data, identity = load_job(root, plan, 9, device)
    indices = indices[:4]
    x, y, vx, vy, initial = data
    serial_fit(indices[0], x, y[0], vx, vy[0], initial[0], plan, max_steps=1)
    batch_fit(indices[:1], x, y[:1], vx, vy[:1], initial[:1], plan, max_steps=1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    codes, histories = [], []
    for index in indices:
        group_codes, group_histories = [], []
        for task in range(plan["targets"]):
            code, history = serial_fit(
                index, x, y[task], vx, vy[task], initial[task], plan
            )
            group_codes.append(code)
            group_histories.append(history)
        codes.append(torch.stack(group_codes))
        histories.append(group_histories)
    torch.cuda.synchronize()
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batched, batched_histories, _ = batch_fit(indices, x, y, vx, vy, initial, plan)
    torch.cuda.synchronize()
    batch_seconds = time.perf_counter() - started
    score_error = max(
        float(
            np.max(
                np.abs(
                    np.array(histories[b][t]["best_validation_per_start"])
                    - np.array(batched_histories[b][t]["best_validation_per_start"])
                )
                / (1 + np.abs(histories[b][t]["best_validation_per_start"]))
            )
        )
        for b in range(len(indices))
        for t in range(plan["targets"])
    )
    code_error = float((torch.stack(codes) - batched).abs().max())
    result = {
        "job_id": bank["job_id"],
        "indexators": len(indices),
        "tasks": plan["targets"],
        "signed_starts": initial.shape[1],
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "speedup": serial_seconds / batch_seconds,
        "maximum_relative_per_start_validation_difference": score_error,
        "maximum_absolute_selected_C_difference": code_error,
        "parity_passed": score_error < 1e-8 and code_error < 1e-6,
        "backend": "batch" if batch_seconds < serial_seconds else "serial",
        "numeric_identity": identity,
        "phase_protocol_sha256": digest(root / "transfer/protocol.json"),
        "execution_code_sha256": digest(Path(__file__)),
        "target_test_labels_opened": False,
        "candidate_retained": False,
        "scope": "Synchronized wall time on the same frozen target bank including the separate true-I numerical control. No teacher C, test labels, source-I changes or retained diagnostic fits. Initialization/runtime warmup excluded; current own bi-level workers may coexist.",
    }
    save_json(destination, result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["parity_passed"]:
        raise ArithmeticError("Diagnose target numerical differences before adoption")


def fit_job(root, plan, job_id, device, batch_size, reuse_root):
    phase = root / "transfer"
    measurement = json.loads((phase / "execution_measurement.json").read_text())
    if not measurement["parity_passed"]:
        raise RuntimeError("Unaccepted numerical execution")
    bank, indices, data, identity = load_job(root, plan, job_id, device)
    identity_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    path = phase / "banks" / f"job_{job_id:03d}"
    destination = phase / "fits" / f"job_{job_id:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Target job already completed")
    destination.mkdir(exist_ok=True)
    phase_hash = digest(phase / "protocol.json")
    reusable = None
    if reuse_root is not None:
        reusable = json.loads((reuse_root / "transfer/fit_frozen.json").read_text())
    pending, artefacts = [], []
    for number, candidate in enumerate(bank["candidates"]):
        output = destination / candidate["signature"]
        artefacts.append(str(output.relative_to(phase)))
        if output.with_suffix(".json").exists():
            saved = json.loads(output.with_suffix(".json").read_text())
            if (
                saved["phase_protocol_sha256"] != phase_hash
                or saved["numeric_identity_sha256"] != identity_hash
                or saved["arrays_sha256"] != digest(output.with_suffix(".npz"))
            ):
                raise RuntimeError(
                    "Saved target fit changed or belongs to another phase"
                )
            continue
        source = (
            None
            if reuse_root is None
            else (
                reuse_root
                / "transfer/fits"
                / f"job_{job_id:03d}"
                / candidate["signature"]
            )
        )
        if source is None or not source.with_suffix(".json").exists():
            pending.append(number)
            continue
        for suffix in (".json", ".npz"):
            saved = source.with_suffix(suffix)
            if (
                digest(saved)
                != reusable["sha256"][str(saved.relative_to(reuse_root / "transfer"))]
            ):
                raise RuntimeError("Reusable fixed-I target fit changed after freeze")
        old = json.loads(source.with_suffix(".json").read_text())
        if old["numeric_identity_sha256"] != identity_hash:
            raise RuntimeError(
                "Reuse requires identical target data, starts and numeric objective"
            )
        shutil.copy2(source.with_suffix(".npz"), output.with_suffix(".npz"))
        save_json(
            output.with_suffix(".json"),
            {
                **old,
                "candidate": number,
                "phase_protocol_sha256": phase_hash,
                "seconds": 0.0,
                "reused_identical_target_fit": {
                    "source": str(source),
                    "source_metadata_sha256": digest(source.with_suffix(".json")),
                    "history_seconds_describe_original_computation": True,
                },
            },
        )
    x, y, vx, vy, initial = data
    for begin in range(0, len(pending), batch_size):
        numbers = pending[begin : begin + batch_size]
        started = time.monotonic()
        if measurement["backend"] == "batch":
            codes, histories, _ = batch_fit(
                indices[numbers], x, y, vx, vy, initial, plan
            )
        else:
            all_codes, histories = [], []
            for number in numbers:
                local_codes, local_histories = [], []
                for task in range(plan["targets"]):
                    code, history = serial_fit(
                        indices[number], x, y[task], vx, vy[task], initial[task], plan
                    )
                    local_codes.append(code)
                    local_histories.append(history)
                all_codes.append(torch.stack(local_codes))
                histories.append(local_histories)
            codes = torch.stack(all_codes)
        seconds = time.monotonic() - started
        for local, number in enumerate(numbers):
            signature = bank["candidates"][number]["signature"]
            output = destination / signature
            np.savez_compressed(
                output.with_suffix(".npz"),
                index=indices[number],
                codes=codes[local].cpu().numpy(),
            )
            save_json(
                output.with_suffix(".json"),
                {
                    "candidate": number,
                    "signature": signature,
                    "numeric_fits": histories[local],
                    "numeric_C_transferred": False,
                    "test_labels_opened": False,
                    "phase_protocol_sha256": phase_hash,
                    "data_sha256": identity["data_sha256"],
                    "numeric_identity_sha256": identity_hash,
                    "arrays_sha256": digest(output.with_suffix(".npz")),
                    "seconds": seconds / len(numbers),
                    "batch": {
                        "candidates": numbers,
                        "elapsed_seconds": seconds,
                        "amortized": True,
                        "backend": measurement["backend"],
                    },
                },
            )
        print(
            json.dumps(
                {"target_job": job_id, "candidates": numbers, "batch_seconds": seconds}
            ),
            flush=True,
        )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": job_id,
            "fit_artefacts": artefacts,
            "bank_sha256": digest(path.with_suffix(".json")),
            "source_I_changed": False,
            "test_labels_opened": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("profile", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reuse-root", type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}:
        raise RuntimeError("Only explicitly selected GPU0--3 is permitted")
    torch.set_num_threads(1)
    phase = json.loads((args.root / "transfer/protocol.json").read_text())
    if phase["code_sha256"][Path(__file__).name] != digest(Path(__file__)):
        raise RuntimeError("Target execution code changed after registration")
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "profile":
        profile(args.root, plan, args.device)
    else:
        for job_id in range(
            args.shard, len(plan["families"]) * len(plan["noise"]), args.shards
        ):
            fit_job(
                args.root, plan, job_id, args.device, args.batch_size, args.reuse_root
            )


if __name__ == "__main__":
    main()

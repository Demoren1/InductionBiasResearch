"""Resume G24 numeric work with the actual frozen fit-panel data contract."""

import argparse
import json
import os
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .joint_prefix_cpu_continuation import cpu_workers
from .shared_structure_numeric import fit_codes as serial_fit
from .shared_structure_numeric import fresh_codes
from .shared_structure_numeric_batch import fit_codes as batch_fit
from .shared_structure_pipeline import check_code, pid_identity, run_stage, state


def load_fit_inputs(root, plan, case):
    bank_path = root / "common/banks" / f"job_{case:03d}"
    bank = json.loads(bank_path.with_suffix(".json").read_text())
    registered = json.loads((root / "numeric_io_continuation.json").read_text())
    for suffix in (".json", ".npz"):
        path = bank_path.with_suffix(suffix)
        if digest(path) != registered["bank_sha256"][str(path.relative_to(root))]:
            raise RuntimeError("Frozen search bank changed")
    source = root / "data" / f"family_{bank['family_id']:03d}_source.npz"
    frozen = json.loads((root / "data_frozen.json").read_text())
    if digest(source) != frozen["sha256"][str(source.relative_to(root))]:
        raise RuntimeError("Frozen fit observations changed")
    with np.load(source) as arrays:
        data = tuple(
            torch.tensor(arrays[name])
            for name in (
                "train_x",
                f"train_y_{bank['noise']}",
                "fit_validation_x",
                f"fit_validation_y_{bank['noise']}",
            )
        )
    with np.load(bank_path.with_suffix(".npz")) as arrays:
        indices = arrays["indices"]
    initial = torch.stack(
        [
            fresh_codes(
                plan, plan["families"][bank["family_id"]]["family"], task, "cpu"
            )
            for task in range(plan["sources"])
        ]
    )
    return bank, indices, (*data, initial)


def fit_candidate(index, data, plan, backend):
    x, y, vx, vy, initial = data
    if backend == "batch":
        codes, history, _ = batch_fit(index[None], x, y, vx, vy, initial, plan)
        return codes[0], history[0]
    values = [
        serial_fit(index, x, y[task], vx, vy[task], initial[task], plan)
        for task in range(plan["sources"])
    ]
    return torch.stack([value[0] for value in values]), [value[1] for value in values]


def measure(root, plan):
    _, indices, data = load_fit_inputs(root, plan, 9)
    x, y, vx, vy, initial = data
    serial_fit(indices[0], x, y[0], vx, vy[0], initial[0], plan, max_steps=1)
    batch_fit(indices[:1], x, y, vx, vy, initial, plan, max_steps=1)
    started = time.perf_counter()
    serial_code, serial_history = fit_candidate(indices[0], data, plan, "serial")
    serial_seconds = time.perf_counter() - started
    started = time.perf_counter()
    batch_code, batch_history = fit_candidate(indices[0], data, plan, "batch")
    batch_seconds = time.perf_counter() - started
    code_error = float((serial_code - batch_code).abs().max())
    score_error = max(
        float(
            np.max(
                np.abs(
                    np.array(a["best_validation_per_start"])
                    - np.array(b["best_validation_per_start"])
                )
                / (1 + np.abs(a["best_validation_per_start"]))
            )
        )
        for a, b in zip(serial_history, batch_history, strict=True)
    )
    parity = code_error < 1e-6 and score_error < 1e-8
    result = {
        "device": "cpu",
        "indexators": 1,
        "tasks": plan["sources"],
        "serial_seconds": serial_seconds,
        "batch_seconds": batch_seconds,
        "serial_over_batch": serial_seconds / batch_seconds,
        "maximum_C_difference": code_error,
        "maximum_relative_per_start_validation_difference": score_error,
        "parity_passed": parity,
        "backend": "batch" if parity and batch_seconds < serial_seconds else "serial",
        "candidate_retained": False,
        "meta_or_teacher_opened": False,
        "continuation_sha256": digest(root / "numeric_io_continuation.json"),
    }
    save_json(root / "cpu_numeric_measurement.json", result)
    print(json.dumps(result), flush=True)


def fit_case(root, plan, case):
    bank, indices, data = load_fit_inputs(root, plan, case)
    phase_hash = digest(root / "common/protocol.json")
    destination = root / "common/fits" / f"job_{case:03d}"
    if destination.with_suffix(".json").exists():
        raise RuntimeError("Fit case already completed")
    destination.mkdir(exist_ok=True)
    measured = json.loads((root / "cpu_numeric_measurement.json").read_text())
    artifacts = []
    for number, candidate in enumerate(bank["candidates"]):
        output = destination / candidate["signature"]
        artifacts.append(str(output.relative_to(root / "common")))
        if output.with_suffix(".json").exists():
            saved = json.loads(output.with_suffix(".json").read_text())
            if saved["phase_protocol_sha256"] != phase_hash or saved[
                "arrays_sha256"
            ] != digest(output.with_suffix(".npz")):
                raise RuntimeError("Existing numeric fit changed")
            continue
        started = time.monotonic()
        code, history = fit_candidate(indices[number], data, plan, measured["backend"])
        np.savez_compressed(
            output.with_suffix(".npz"), index=indices[number], codes=code.numpy()
        )
        save_json(
            output.with_suffix(".json"),
            {
                "job_id": case,
                "candidate": number,
                "signature": candidate["signature"],
                "numeric_fits": history,
                "meta_labels_opened": False,
                "phase_protocol_sha256": phase_hash,
                "arrays_sha256": digest(output.with_suffix(".npz")),
                "seconds": time.monotonic() - started,
                "numeric_execution": {
                    "device": "cpu",
                    "backend": measured["backend"],
                    "input_contract": "frozen train and new H only; axis grid unnecessary",
                },
            },
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "candidate": number,
                    "of": len(indices),
                    "seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    save_json(
        destination.with_suffix(".json"),
        {
            "job_id": case,
            "fit_artefacts": artifacts,
            "bank_sha256": digest(root / "common/banks" / f"job_{case:03d}.json"),
            "phase_protocol_sha256": phase_hash,
            "meta_labels_opened": False,
        },
    )


def run(root):
    if (root / "numeric_io_continuation.json").exists():
        raise RuntimeError(
            "Numeric continuation already registered; inspect its session"
        )
    source = json.loads((root / "protocol.json").read_text())
    previous = json.loads((root / "cpu_continuation.json").read_text())
    check_code(previous)
    stage = json.loads((root / "comparison_pipeline_state.json").read_text())
    if (
        pid_identity(previous["supervisor"]["pid"]) == previous["supervisor"]
        or stage["stage"] != "g24_cpu_continuation_stopped"
    ):
        raise RuntimeError("Previous continuation is not confirmed terminal/stopped")
    if (
        list((root / "common/fits").glob("job_*"))
        or (root / "common/fit_frozen.json").exists()
    ):
        raise RuntimeError("Numeric work already exists; inspect before resuming")
    bank_paths = sorted((root / "common/banks").glob("job_*.*"))
    if len(bank_paths) != 2 * len(source["families"]) * len(source["noise"]):
        raise RuntimeError("Incomplete frozen search banks")
    protocol = {
        **previous,
        "registered_utc": datetime.now(UTC).isoformat(),
        "code_sha256": {
            **previous["code_sha256"],
            Path(__file__).name: digest(Path(__file__)),
        },
        "previous_terminal_supervisor": previous["supervisor"],
        "supervisor": pid_identity(os.getpid()),
        "bank_sha256": {
            str(path.relative_to(root)): digest(path) for path in bank_paths
        },
        "search_freeze_sha256": digest(root / "extract_frozen.json"),
        "reason": "All300scientific search outputs preserved. Previous numeric stage stopped before fitting because legacy whole-source loader expected unused axis_t/axis_vt. Read only the four actual frozen fit arrays, call unchanged pure GN APIs. Do not modify data, replay search, change seeds/objective/checkpoints or open meta early.",
    }
    save_json(root / "numeric_io_continuation.json", protocol)
    shutil.copy2(Path(__file__), root / "numeric_io_continuation_snapshot.py")
    torch.set_num_threads(1)
    state(root, "cpu_numeric_paired_measurement_after_input_adapter")
    measure(root, source)
    path = root / "common/protocol.json"
    phase = json.loads(path.read_text())
    phase["cpu_numeric_execution"] = {
        "measurement_sha256": digest(root / "cpu_numeric_measurement.json"),
        "continuation_sha256": digest(root / "numeric_io_continuation.json"),
        "input_arrays": ["train_x", "train_y", "fit_validation_x", "fit_validation_y"],
        "scientific_settings_unchanged": True,
    }
    phase["code_sha256"][Path(__file__).name] = digest(Path(__file__))
    save_json(path, phase)
    shutil.copy2(
        Path(__file__), root / "common/executed_code_snapshot" / Path(__file__).name
    )
    cpu_workers(root, protocol, "joint_prefix_fit_continuation", "fit")
    run_stage(root, protocol, "joint_prefix_benchmark", "freeze_fit")
    cpu_workers(root, protocol, "joint_prefix_benchmark", "select", ("--device", "cpu"))
    for action in ("freeze_selection", "audit"):
        run_stage(root, protocol, "joint_prefix_benchmark", action)
    run_stage(root, protocol, "joint_prefix_report")
    state(
        root,
        "g24_diagnosis_report_ready",
        report=str(root / "report/REPORT.md"),
        overall_goal_complete=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU continuation must hide CUDA devices")
    root = args.root.resolve()
    if args.action == "run":
        try:
            run(root)
        except BaseException as error:
            state(
                root,
                "g24_numeric_io_continuation_stopped",
                error=repr(error),
                automatically_restarted=False,
            )
            raise
    else:
        torch.set_num_threads(1)
        check_code(json.loads((root / "numeric_io_continuation.json").read_text()))
        plan = json.loads((root / "protocol.json").read_text())
        for case in range(
            args.shard, len(plan["families"]) * len(plan["noise"]), args.shards
        ):
            fit_case(root, plan, case)


if __name__ == "__main__":
    main()

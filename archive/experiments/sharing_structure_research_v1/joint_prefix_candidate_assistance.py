"""Balance a large CPU fit case without losing its in-flight numerical fit."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json

from .joint_prefix_fit_continuation import fit_candidate, load_fit_inputs
from .shared_structure_pipeline import PACKAGE, check_code, pid_identity


def fit_one(root, case, candidate):
    plan = json.loads((root / "protocol.json").read_text())
    check_code(json.loads((root / "numeric_io_continuation.json").read_text()))
    bank, indices, data = load_fit_inputs(root, plan, case)
    output = (
        root
        / "common/fits"
        / f"job_{case:03d}"
        / bank["candidates"][candidate]["signature"]
    )
    if output.with_suffix(".json").exists():
        raise RuntimeError("Helper assignment overlaps a completed numerical fit")
    phase_hash = digest(root / "common/protocol.json")
    measured = json.loads((root / "cpu_numeric_measurement.json").read_text())
    started = time.monotonic()
    code, history = fit_candidate(indices[candidate], data, plan, measured["backend"])
    temporary = output.with_suffix(".partial.npz")
    np.savez_compressed(temporary, index=indices[candidate], codes=code.numpy())
    temporary.replace(output.with_suffix(".npz"))
    metadata = {
        "job_id": case,
        "candidate": candidate,
        "signature": bank["candidates"][candidate]["signature"],
        "numeric_fits": history,
        "meta_labels_opened": False,
        "phase_protocol_sha256": phase_hash,
        "arrays_sha256": digest(output.with_suffix(".npz")),
        "seconds": time.monotonic() - started,
        "numeric_execution": {
            "device": "cpu",
            "backend": measured["backend"],
            "helper_provenance_sha256": digest(root / "candidate_assistance_jobs.json"),
        },
    }
    temporary = output.with_suffix(".partial.json")
    save_json(temporary, metadata)
    temporary.replace(output.with_suffix(".json"))
    print(
        json.dumps(
            {"case": case, "candidate": candidate, "seconds": metadata["seconds"]}
        ),
        flush=True,
    )


def coordinate(root, case, pid):
    path = root / "candidate_assistance.json"
    if path.exists():
        raise RuntimeError("Assistance already registered; inspect its session")
    identity = pid_identity(pid)
    command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    required = (
        b"experiments.sharing_structure_research_v1.joint_prefix_fit_continuation",
        b"fit",
        str(root).encode(),
    )
    if (
        identity is None
        or Path(f"/proc/{pid}").stat().st_uid != os.getuid()
        or any(item not in command for item in required)
    ):
        raise RuntimeError("Target is not this user's verified G24 numeric worker")
    if (root / "common/fits" / f"job_{case:03d}.json").exists():
        print("Case already complete; no process was paused", flush=True)
        return
    save_json(
        path,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "supervisor": pid_identity(os.getpid()),
            "paused_worker": identity,
            "case": case,
            "code_sha256": digest(Path(__file__)),
            "phase_protocol_sha256": digest(root / "common/protocol.json"),
            "policy": "Briefly SIGSTOP only this exact own CPU worker, preserving its in-flight fit in RAM. Protect its first unfinished candidate. Helpers compute only subsequent unstarted independent I with unchanged pure GN/data/starts. Never modify bank/data/phase. Finally SIGCONT the original worker; it verifies/reuses completed helper fits. No lost numerical progress or GPU use.",
        },
    )
    shutil.copy2(Path(__file__), root / "candidate_assistance_snapshot.py")
    os.kill(pid, signal.SIGSTOP)
    processes, handles = [], []
    try:
        bank = json.loads((root / "common/banks" / f"job_{case:03d}.json").read_text())
        directory = root / "common/fits" / f"job_{case:03d}"
        first_missing = next(
            (
                n
                for n, candidate in enumerate(bank["candidates"])
                if not (directory / (candidate["signature"] + ".json")).exists()
            ),
            len(bank["candidates"]),
        )
        pending = list(range(first_missing + 1, len(bank["candidates"])))
        save_json(
            root / "candidate_assistance_jobs.json",
            {
                "case": case,
                "protected_in_flight_candidate": first_missing,
                "assigned_candidates": pending,
                "registration_sha256": digest(path),
                "new_meta_or_teacher_opened": False,
            },
        )
        print(
            json.dumps({"protected": first_missing, "parallel_candidates": pending}),
            flush=True,
        )
        for candidate in pending:
            handle = (root / "logs" / f"candidate_helper_{case}_{candidate}.log").open(
                "w"
            )
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        PACKAGE + "joint_prefix_candidate_assistance",
                        "fit",
                        "--root",
                        str(root),
                        "--case",
                        str(case),
                        "--candidate",
                        str(candidate),
                    ],
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "OMP_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                    },
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            )
        while any(process.poll() is None for process in processes):
            time.sleep(5)
        save_json(
            root / "candidate_assistance_terminal.json",
            {
                "workers": [
                    {"pid": process.pid, "exit_code": process.returncode}
                    for process in processes
                ]
            },
        )
        if any(process.returncode for process in processes):
            raise RuntimeError("Candidate helper stopped; original worker will resume")
    finally:
        for process in processes:
            if process.poll() is None:
                process.wait()
        for handle in handles:
            handle.close()
        if pid_identity(pid) == identity:
            os.kill(pid, signal.SIGCONT)
        print("Original numerical worker resumed with its in-flight state", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("coordinate", "fit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--candidate", type=int)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU helpers must hide CUDA devices")
    torch.set_num_threads(1)
    root = args.root.resolve()
    if args.action == "fit":
        expected = json.loads((root / "candidate_assistance.json").read_text())
        if digest(Path(__file__)) != expected["code_sha256"]:
            raise RuntimeError("Registered helper code changed")
        fit_one(root, args.case, args.candidate)
    else:
        coordinate(root, args.case, args.pid)


if __name__ == "__main__":
    main()

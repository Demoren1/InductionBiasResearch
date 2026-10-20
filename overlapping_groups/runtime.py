"""CUDA allocation and bounded subprocess scheduling shared by launchers."""

import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch

from .io import REPOSITORY, save_json


def require_cuda(value):
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("All training must run on CUDA")
    torch.cuda.set_device(device)
    return device


def configure_cuda(value, memory_gib):
    device = require_cuda(value)
    total = torch.cuda.get_device_properties(device).total_memory
    if not 2 < memory_gib <= total / 1024**3:
        raise ValueError("GPU budget must exceed 2 GiB and fit the device")
    torch.cuda.set_per_process_memory_fraction(
        (memory_gib - 2) * 1024**3 / total, device
    )
    return device


def cuda_devices(value):
    devices = value.split(",")
    if len(set(devices)) != len(devices) or any(
        not device.startswith("cuda:") or not device[5:].isdigit()
        for device in devices
    ):
        raise ValueError("Distinct CUDA devices are required")
    return devices


def run_jobs(jobs, root, devices, workers_per_device=1):
    """Each job is (name, module, arguments); append a CUDA device per slot."""
    if workers_per_device < 1:
        raise ValueError("workers-per-device must be positive")
    root = Path(root)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pending = deque(jobs)
    expected = len(pending)
    slots = [
        (device, index)
        for index in range(workers_per_device)
        for device in devices
    ]
    running, outcomes = {}, []
    started = time.monotonic()
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    try:
        while pending or running:
            for slot in slots:
                if slot in running or not pending:
                    continue
                name, module, arguments = pending.popleft()
                command = [
                    sys.executable,
                    "-u",
                    "-m",
                    module,
                    *map(str, arguments),
                    "--device",
                    slot[0],
                ]
                log = (logs / f"{name}.log").open("a", buffering=1)
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=REPOSITORY,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                except BaseException:
                    log.close()
                    raise
                running[slot] = (name, process, log)
                print(
                    json.dumps({"started": name, "device": slot[0]}),
                    flush=True,
                )
            for slot, (name, process, log) in list(running.items()):
                if process.poll() is None:
                    continue
                log.close()
                outcomes.append(
                    {"run": name, "returncode": process.returncode}
                )
                del running[slot]
                print(json.dumps(outcomes[-1]), flush=True)
                if process.returncode:
                    raise RuntimeError(
                        f"Worker failed; inspect {logs / (name + '.log')}"
                    )
            if running:
                time.sleep(1)
    finally:
        for _, process, _ in running.values():
            if process.poll() is None:
                process.terminate()
        for _, process, log in running.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        save_json(
            root / "launch_results.json",
            {
                "expected": expected,
                "outcomes": outcomes,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    return outcomes

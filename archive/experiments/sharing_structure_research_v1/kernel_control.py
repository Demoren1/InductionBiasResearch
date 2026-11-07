"""G2 known control: discrete signed kernel partition without rank truncation."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .gradient_spectrum import BASE, observations

KERNELS = ("raw", "corrected", "cross")
STARTS = ("ols_profiles", "ols_noise_debiased_spectrum")


def basis(index, codes):
    labels = index.flatten().long()
    return F.one_hot(labels.abs(), codes + 1)[:, 1:].double() * labels.sign()[:, None]


def partition_score(index, matrix, codes):
    assigned = basis(index, codes)
    counts = assigned.square().sum(0)
    return ((assigned * (matrix @ assigned)).sum(0) / counts.clamp_min(1)).sum()


@torch.no_grad()
def coordinate_partition(index, matrix, codes, steps=2048):
    """Greedy best exact coordinate gain; zero is a fixed unused group."""
    current = index.clone().flatten().long()
    diagonal = matrix.diagonal()
    trace, scores = [current.cpu().numpy().astype(np.int16)], []
    previous = partition_score(current, matrix, codes)
    scores.append(float(previous))
    labels = torch.arange(-codes, codes + 1, device=matrix.device)
    for _ in range(steps):
        assigned = basis(current, codes)
        counts = assigned.square().sum(0)
        product = matrix @ assigned
        quadratic = (assigned * product).sum(0)
        contribution = quadratic / counts.clamp_min(1)
        old_group = (current.abs() - 1).clamp_min(0)
        rows = torch.arange(len(current), device=matrix.device)
        removed_q = (
            quadratic[old_group]
            - 2 * current.sign() * product[rows, old_group]
            + diagonal
        )
        removed_n = counts[old_group] - 1
        remove_gain = (removed_q / removed_n.clamp_min(1)) * (
            removed_n > 0
        ) - contribution[old_group]
        remove_gain = torch.where(current != 0, remove_gain, 0)
        gains = remove_gain[:, None].expand(-1, len(labels)).clone()
        for column, label in enumerate(labels.tolist()):
            if label == 0:
                continue
            group, sign = abs(label) - 1, 1 if label > 0 else -1
            add_gain = (quadratic[group] + 2 * sign * product[:, group] + diagonal) / (
                counts[group] + 1
            ) - contribution[group]
            gains[:, column] += add_gain
            same = current.abs() == abs(label)
            delta = sign - current.sign()
            flip_gain = (
                2 * delta * product[:, group] + delta.square() * diagonal
            ) / counts[group].clamp_min(1)
            gains[:, column] = torch.where(same, flip_gain, gains[:, column])
        winner = int(gains.argmax())
        position, column = divmod(winner, len(labels))
        gain = gains[position, column]
        if float(gain) <= 1e-12:
            break
        current[position] = labels[column]
        value = partition_score(current, matrix, codes)
        if not torch.allclose(value - previous, gain, atol=1e-9, rtol=1e-7):
            raise FloatingPointError(
                "Analytical hard-partition gain disagrees with score"
            )
        if float(value) < float(previous) - 1e-10:
            raise FloatingPointError("Hard-partition objective decreased")
        previous = value
        trace.append(current.cpu().numpy().astype(np.int16))
        scores.append(float(value))
    return current.reshape(index.shape), np.stack(trace), scores


def register(root):
    path = root / "kernel_control_registration.json"
    if path.exists():
        raise RuntimeError("Kernel control already registered")
    save_json(
        path,
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "purpose": "Known kernel-partition coordinate baseline, not a proposed novel method. All six previously inspected G1 hard cases, diagnostic only.",
            "kernels": KERNELS,
            "starts": STARTS,
            "maximum_steps": 2048,
            "stop_gain": 1e-12,
            "selection": "Each kernel selects its two-start result using its own score only; no teacher selection.",
            "cases": "condition100, weak1/0.3/0.03, noise1,128tasks,8192examples",
            "theory_sha256": digest(BASE / "THEORY.md"),
            "code_sha256": digest(Path(__file__)),
        },
    )
    (root / "kernel_control").mkdir(exist_ok=False)


@torch.no_grad()
def run_family(root, family, device):
    plan = json.loads((root / "protocol.json").read_text())
    registration = root / "kernel_control_registration.json"
    if not registration.exists():
        raise RuntimeError("Register before running")
    destination = root / "kernel_control" / f"family_{family}.json"
    if destination.exists():
        raise RuntimeError("Family already completed")
    started = time.perf_counter()
    truth, x, y, _, _ = observations(family, 100.0, "weak", 1.0, 8192, plan, device)
    n, tasks = x.shape[1], x.shape[0]
    fitted = torch.linalg.lstsq(x, y).solution.transpose(-1, -2)
    profiles = fitted.flatten(1).T
    raw = profiles @ profiles.T / tasks
    residual = torch.einsum("tni,toi->tno", x, fitted) - y
    variance = residual.square().sum(1) / (n - x.shape[-1])
    gram = x.transpose(-1, -2) @ x
    inverse_gram = torch.linalg.inv(gram)
    blocks = torch.einsum("to,tij->oij", variance, inverse_gram) / tasks
    corrected = raw - torch.block_diag(*blocks.unbind(0))
    parts = [
        torch.linalg.lstsq(x[:, part], y[:, part])
        .solution.transpose(-1, -2)
        .flatten(1)
        .T
        for part in (slice(0, n // 2), slice(n // 2, n))
    ]
    matrices = {
        "raw": raw,
        "corrected": corrected,
        "cross": (parts[0] @ parts[1].T + parts[1] @ parts[0].T) / (2 * tasks),
    }
    arrays = {f"kernel_{name}": value.cpu().numpy() for name, value in matrices.items()}
    arrays["ols_profiles"] = profiles.cpu().numpy()
    arrays["gram"] = gram.cpu().numpy()
    arrays["cross_yx"] = (y.transpose(-1, -2) @ x).cpu().numpy()
    arrays["noise_variance"] = variance.cpu().numpy()
    arrays["truth_audit_only"] = truth.cpu().numpy()
    records = []
    with np.load(root / "showcases" / f"family_{family}_n8192.npz") as archive:
        for kernel, matrix in matrices.items():
            for source in STARTS:
                initial = torch.tensor(archive[f"{source}_indices"], device=device)
                index, trace, scores = coordinate_partition(
                    initial, matrix, plan["codes"]
                )
                recovery, aligned = alignment(
                    index.cpu().numpy(), truth.cpu().numpy(), plan["codes"]
                )
                key = f"{kernel}__{source}"
                arrays[f"{key}_indices"] = index.cpu().numpy()
                arrays[f"{key}_aligned_indices"] = aligned
                arrays[f"{key}_trace"] = trace
                recall = [
                    alignment(
                        step.reshape(truth.shape), truth.cpu().numpy(), plan["codes"]
                    )[0]
                    for step in trace
                ]
                records.append(
                    {
                        "kernel": kernel,
                        "start": source,
                        "scores": scores,
                        "recovery_trace": recall,
                        "recovery": recovery,
                        "truth_score_audit": float(
                            partition_score(truth, matrix, plan["codes"])
                        ),
                    }
                )
    selected = {
        kernel: max(
            (r for r in records if r["kernel"] == kernel),
            key=lambda r: r["scores"][-1],
        )["start"]
        for kernel in KERNELS
    }
    np.savez_compressed(root / "kernel_control" / f"family_{family}.npz", **arrays)
    save_json(
        destination,
        {
            "family": family,
            "registration_sha256": digest(registration),
            "records": records,
            "selected_start": selected,
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        family,
        {
            f"{r['kernel']}__{r['start']}": r["recovery"]["categorical_errors"]
            for r in records
        },
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--families", type=int, nargs="+")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
    else:
        for family in args.families:
            run_family(args.root, family, args.device)


if __name__ == "__main__":
    main()

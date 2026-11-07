"""G2 diagnostic: full structure score versus its top-K relaxation."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.linear import index_basis

from .gradient_spectrum import BASE, METHODS, observations


def score(index, matrix, codes):
    basis = index_basis(index[None], codes)[0].double().flatten(0, 1)
    count = basis.square().sum(0)
    return ((basis * (matrix @ basis)).sum(0) / count.clamp_min(1)).sum()


@torch.no_grad()
def audit(root, device):
    plan = json.loads((root / "protocol.json").read_text())
    registration = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "purpose": "Compare full noise-corrected structure scores to top-K decoded candidate scores in all six preexisting hard cases. Diagnostic only; teacher never participates in candidate generation.",
        "cases": "condition100, weak1/0.3/0.03, noise1,128tasks,8192examples",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "candidates": ["truth_audit_only", *METHODS],
    }
    path = root / "kernel_audit_registration.json"
    if path.exists():
        raise RuntimeError(
            "This diagnostic is already registered; use its saved results"
        )
    save_json(path, registration)
    records = []
    for family in plan["families"]:
        truth, x, y, _, weights = observations(
            family, 100.0, "weak", 1.0, 8192, plan, device
        )
        n, tasks = x.shape[1], x.shape[0]
        fitted = torch.linalg.lstsq(x, y).solution.transpose(-1, -2)
        profiles = fitted.flatten(1).T
        raw = profiles @ profiles.T / tasks
        residual = torch.einsum("tni,toi->tno", x, fitted) - y
        variance = residual.square().sum(1) / (n - x.shape[-1])
        inverse_gram = torch.linalg.inv(x.transpose(-1, -2) @ x)
        noise_blocks = torch.einsum("to,tij->oij", variance, inverse_gram) / tasks
        corrected = raw - torch.block_diag(*noise_blocks.unbind(0))
        parts = []
        for selected in (slice(0, n // 2), slice(n // 2, n)):
            parts.append(
                torch.linalg.lstsq(x[:, selected], y[:, selected])
                .solution.transpose(-1, -2)
                .flatten(1)
                .T
            )
        cross = (parts[0] @ parts[1].T + parts[1] @ parts[0].T) / (2 * tasks)
        w = weights.flatten(1).T
        signal = w @ w.T / tasks
        values, vectors = torch.linalg.eigh(corrected)
        top = vectors[:, -plan["codes"] :]
        truncated = (top * values[-plan["codes"] :].clamp_min(0)[None]) @ top.T
        matrices = {
            "raw": raw,
            "corrected": corrected,
            "cross": cross,
            "corrected_topK": truncated,
            "true_signal_audit": signal,
        }
        true_scores = {
            name: float(score(truth, matrix, plan["codes"]))
            for name, matrix in matrices.items()
        }
        with np.load(root / "showcases" / f"family_{family}_n8192.npz") as archive:
            for method in METHODS:
                candidate = torch.tensor(archive[f"{method}_indices"], device=device)
                scores = {
                    name: float(score(candidate, matrix, plan["codes"]))
                    for name, matrix in matrices.items()
                }
                records.append(
                    {
                        "family": family,
                        "method": method,
                        "candidate_scores": scores,
                        "truth_scores": true_scores,
                        "truth_minus_candidate": {
                            name: true_scores[name] - value
                            for name, value in scores.items()
                        },
                    }
                )
        print(
            "kernel audit",
            family,
            {
                method: next(
                    r["truth_minus_candidate"]["corrected"]
                    for r in records
                    if r["family"] == family and r["method"] == method
                )
                for method in ("ols_profiles", "ols_pca", "ols_noise_debiased_spectrum")
            },
            flush=True,
        )
    save_json(
        root / "kernel_audit.json",
        {
            "registration_sha256": digest(path),
            "records": records,
            "scope": "Oracle used for post-hoc objective diagnosis only, no new method or heldout claim.",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    audit(args.root, args.device)


if __name__ == "__main__":
    main()

"""G14: opened-cohort diagnosis of a conditional gradient/GLS spectrum.

This is an exploratory post-selection diagnostic, not a confidence certificate.
Only the separate audit reads teacher artifacts, after the whole cohort freezes.
"""

import argparse
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment

from .competitor_models import (
    adapted_stats,
    signed_basis,
    stats_loss,
    validation_loss,
)
from .competitor_pilot import regression_validation, save_results
from .contrasts import noise_whitened_basis
from .gaussian_stats import fit_stats
from .gradient_spectrum import decode
from .interval_pilot import weighted_signed_dp

BASE = Path(__file__).resolve().parent
SPECTRA = ("pca", "corrected_pca", "whitened_pca")
METHODS = (
    "coarse_native",
    "dense_profiles",
    *(f"dense_{name}" for name in SPECTRA),
    *(f"subset_{name}" for name in SPECTRA),
    *(f"restricted_{name}" for name in SPECTRA),
)


def register(root, coarse):
    original = json.loads((coarse / "protocol.json").read_text())
    specification = json.loads((coarse / "confirmation_registration.json").read_text())
    frozen = coarse / "confirmation_models_frozen.json"
    if not frozen.exists():
        raise RuntimeError("Coarse source models must already be frozen")
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        **{
            key: original[key]
            for key in ("seed", "inputs", "outputs", "codes", "tasks", "g4_root")
        },
        "stage": "G14 conditional efficient-score diagnosis on already-opened G4; exploratory post-selection only",
        "registered_utc": datetime.now(UTC).isoformat(),
        "coarse_root": str(coarse.resolve()),
        "coarse_cohort_sha256": digest(frozen),
        "jobs": specification["jobs"],
        "methods": list(METHODS),
        "decoder_restarts": 32,
        "decoder_steps": 100,
        "readout": "Rank-one: exact scalar signed DP with fixed zero, no given support/size. Rank>1: same signed vector Lloyd, 32 own-distortion restarts. PCA readouts explicitly additional to native coarse-I.",
        "selection": "All individual methods retained. Separate observed source-validation selectors: coarse+three restricted spectra versus four unrestricted profile/spectral controls; same32 full-C source functions as G4. No teacher-based restart/method selection.",
        "assumptions": "Conditional Gaussian covariance exact only for fixed correct coarse equalities. Reusing coarse selection data makes this a post-selection point diagnostic; no F/Wald guarantee and no independent confirmation claimed.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": {
            name: digest(BASE / name)
            for name in (
                "residual_spectrum.py",
                "competitor_models.py",
                "competitor_pilot.py",
                "contrasts.py",
                "gaussian_stats.py",
                "gradient_spectrum.py",
                "interval_pilot.py",
            )
        },
        "checks": "No software tests. Necessary scientific/provenance guards, lint, and figures only.",
    }
    save_json(root / "protocol.json", plan)
    shutil.copy2(BASE / "THEORY.md", root / "theory_at_registration.md")
    snapshot = root / "executed_code_snapshot"
    snapshot.mkdir()
    for name in plan["code_sha256"]:
        shutil.copy2(BASE / name, snapshot / name)
    (root / "models").mkdir()


def canonical_nonzero(index):
    flat = index.flatten()
    labels = sorted(set(flat.abs().tolist()) - {0})
    result = torch.zeros_like(flat)
    for new, old in enumerate(labels, 1):
        result[flat.abs() == old] = new * flat[flat.abs() == old].sign()
    return result, len(labels)


def restricted_profiles(stats, flat, groups, variance):
    """Full nuisance-aware least squares, retaining every coarse-zero position."""
    tasks, outputs, inputs = stats["cross"].shape
    remaining = torch.where(flat == 0)[0]
    size = flat.numel()
    old_basis = signed_basis(flat, groups)
    identity = torch.eye(size, device=flat.device, dtype=torch.float64)
    basis = torch.cat((old_basis, identity[:, remaining]), -1)
    blocks = torch.eye(outputs, device=flat.device, dtype=torch.float64)
    full_hessian = torch.einsum("ab,tij->taibj", blocks, stats["gram"])
    full_hessian = full_hessian.reshape(tasks, size, size)
    reduced = basis.T @ full_hessian @ basis
    cross = stats["cross"].flatten(1) @ basis
    coefficient = torch.linalg.solve(reduced, cross[..., None])[..., 0]
    covariance = torch.linalg.inv(reduced)[:, groups:, groups:]
    raw = (coefficient @ basis.T).reshape(tasks, outputs, inputs)
    return (
        coefficient[:, groups:].T,
        variance * covariance.mean(0),
        raw,
        remaining,
    )


def spectra(profiles, noise, codes):
    raw = profiles @ profiles.T / profiles.shape[1]
    values, vectors = torch.linalg.eigh((raw + raw.T) * 0.5)
    corrected = (raw + raw.T) * 0.5 - noise
    corrected_values, corrected_vectors = torch.linalg.eigh(corrected)
    whitened, whitened_values = noise_whitened_basis(profiles, noise, codes)
    return {
        "pca": (vectors[:, -codes:], values),
        "corrected_pca": (corrected_vectors[:, -codes:], corrected_values),
        "whitened_pca": (whitened, whitened_values),
    }


def decode_embedding(embedding, codes, plan, seed):
    if codes == 1 and embedding.shape[1] == 1:
        indices, _ = weighted_signed_dp(embedding.T, torch.ones_like(embedding.T), 1)
        return torch.tensor(indices[0], device=embedding.device).long()
    return decode(embedding, {**plan, "codes": codes}, seed)[0]


@torch.no_grad()
def run_case(root, plan, job_id, device):
    started = time.monotonic()
    job = plan["jobs"][job_id]
    coarse_root = Path(plan["coarse_root"])
    coarse_path = coarse_root / "confirmation" / f"job_{job_id:03d}" / "model.npz"
    old_freeze = json.loads(
        (coarse_root / "confirmation_models_frozen.json").read_text()
    )
    if (
        digest(coarse_path)
        != old_freeze["sha256"][str(coarse_path.relative_to(coarse_root))]
    ):
        raise RuntimeError("Frozen coarse model changed")
    source = (
        Path(plan["g4_root"])
        / "confirmation"
        / f"case_{job['source_case']:03d}"
        / f"source_n{job['n']}.npz"
    )
    old_meta = json.loads(coarse_path.with_suffix(".json").read_text())
    if digest(source) != old_meta["source_sha256"]:
        raise RuntimeError("Frozen source statistics changed")
    with np.load(coarse_path) as archive:
        initial = torch.tensor(archive["selected_index"], device=device).long()
    with np.load(source) as archive:
        stats = {
            name: torch.tensor(archive[name], device=device) for name in archive.files
        }
    fitted, residual, inverse = fit_stats(stats)
    variance = residual.sum() / (
        plan["tasks"] * plan["outputs"] * (job["n"] - plan["inputs"])
    )
    noise = torch.block_diag(*[variance * inverse.mean(0)] * plan["outputs"])
    profiles = fitted.flatten(1).T
    flat, groups = canonical_nonzero(initial)
    missing = plan["codes"] - groups
    if missing < 0:
        raise ValueError("Native model exceeds known codebook budget")
    seed = plan["seed"] + job["family"]["id"] + job["n"]
    candidates = {"coarse_native": flat.reshape(initial.shape)}
    diagnostic_arrays = {"dense_ols": fitted.cpu().numpy()}
    candidates["dense_profiles"] = decode_embedding(
        profiles, plan["codes"], plan, seed
    ).reshape(initial.shape)
    for name, (embedding, spectrum) in spectra(profiles, noise, plan["codes"]).items():
        candidates[f"dense_{name}"] = decode_embedding(
            embedding, plan["codes"], plan, seed
        ).reshape(initial.shape)
        diagnostic_arrays[f"dense_{name}_spectrum"] = spectrum.cpu().numpy()
    details = {"coarse_groups": groups, "missing_budget": missing}
    if missing:
        conditional, covariance, raw, remaining = restricted_profiles(
            stats, flat, groups, variance
        )
        unrestricted_covariance = noise[remaining][:, remaining]
        diagnostic_arrays["restricted_raw"] = raw.cpu().numpy()
        diagnostic_arrays["remaining_positions"] = remaining.cpu().numpy()
        diagnostic_arrays["restricted_marginal_variance"] = (
            covariance.diagonal().cpu().numpy()
        )
        diagnostic_arrays["subset_marginal_variance"] = (
            unrestricted_covariance.diagonal().cpu().numpy()
        )
        details["median_variance_ratio"] = float(
            (covariance.diagonal() / unrestricted_covariance.diagonal()).median()
        )
        for mode, values, matrix in (
            ("subset", profiles[remaining], unrestricted_covariance),
            ("restricted", conditional, covariance),
        ):
            for name, (embedding, spectrum) in spectra(values, matrix, missing).items():
                selected = decode_embedding(embedding, missing, plan, seed)
                complete = flat.clone()
                complete[remaining] = (selected.abs() + groups) * selected.sign()
                candidates[f"{mode}_{name}"] = complete.reshape(initial.shape)
                diagnostic_arrays[f"{mode}_{name}_spectrum"] = spectrum.cpu().numpy()
                if missing == 1:
                    diagnostic_arrays[f"{mode}_{name}_physical_vector"] = (
                        embedding[:, 0].cpu().numpy()
                    )
    else:
        details["median_variance_ratio"] = None
        for mode in ("subset", "restricted"):
            for name in SPECTRA:
                candidates[f"{mode}_{name}"] = candidates["coarse_native"]
    indices = torch.stack([candidates[name] for name in METHODS])
    validation = regression_validation(
        job["family"]["id"], job["family"]["kind"], plan, device
    )
    basis = signed_basis(indices, plan["codes"])
    coefficients = adapted_stats(basis, stats, job["n"])
    weights = torch.einsum("soik,stk->stoi", basis, coefficients)
    selection = validation_loss(basis, validation)
    for selector, names in (
        ("residual_selected", ("coarse_native", *(f"restricted_{x}" for x in SPECTRA))),
        ("dense_selected", ("dense_profiles", *(f"dense_{x}" for x in SPECTRA))),
    ):
        rows = [METHODS.index(name) for name in names]
        chosen = rows[int(selection[rows].argmin())]
        details[selector] = METHODS[chosen]
    if not torch.isfinite(weights).all() or not torch.isfinite(selection).all():
        raise FloatingPointError("Nonfinite scientific estimator")
    result = {
        "job": job,
        "indices": indices.cpu().numpy().astype(np.int16),
        "weights": weights.cpu().numpy(),
        "selection_mse": selection.cpu().numpy(),
        "source_mse": stats_loss(weights, stats, job["n"]).cpu().numpy(),
        "details": details,
        "diagnostics": diagnostic_arrays,
        "source_sha256": digest(source),
        "coarse_sha256": digest(coarse_path),
        "protocol_sha256": digest(root / "protocol.json"),
        "seconds": time.monotonic() - started,
    }
    save_results(root / "models", f"job_{job_id:03d}", result)
    print(
        json.dumps(
            {"finished": job_id, "groups": groups, "seconds": result["seconds"]}
        ),
        flush=True,
    )


def freeze(root, plan):
    hashes = {}
    for job_id in range(len(plan["jobs"])):
        for suffix in ("npz", "json"):
            path = root / "models" / f"job_{job_id:03d}.{suffix}"
            hashes[str(path.relative_to(root))] = digest(path)
    path = root / "models_frozen.json"
    if path.exists():
        raise RuntimeError("Cohort already frozen")
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": hashes,
        },
    )


def audit(root, plan):
    frozen = json.loads((root / "models_frozen.json").read_text())
    output = root / "audit.json"
    if output.exists():
        raise RuntimeError("Cohort already audited")
    records = []
    for job_id, job in enumerate(plan["jobs"]):
        path = root / "models" / f"job_{job_id:03d}.npz"
        for candidate in (path, path.with_suffix(".json")):
            if digest(candidate) != frozen["sha256"][str(candidate.relative_to(root))]:
                raise RuntimeError("Frozen diagnostic model changed")
        meta = json.loads(path.with_suffix(".json").read_text())
        truth_path = (
            Path(plan["g4_root"])
            / "confirmation"
            / f"case_{job['source_case']:03d}"
            / "teacher_audit_only.npz"
        )
        with np.load(truth_path) as archive:
            truth = archive["indices"]
        aligned = []
        case_records = []
        with np.load(path) as archive:
            indices = archive["indices"]
            for row, name in enumerate(METHODS):
                recovery, current = alignment(indices[row], truth, plan["codes"])
                aligned.append(current)
                case_records.append(
                    {
                        "job_id": job_id,
                        "job": job,
                        "method": name,
                        "recovery": recovery,
                        "validation_mse": float(archive["selection_mse"][row]),
                        "source_mse": float(archive["source_mse"][row]),
                    }
                )
            records.extend(case_records)
            for selector in ("residual_selected", "dense_selected"):
                chosen = meta["details"][selector]
                records.append(
                    {
                        **case_records[METHODS.index(chosen)],
                        "method": selector,
                        "chosen_method": chosen,
                    }
                )
        np.savez_compressed(
            root / "models" / f"job_{job_id:03d}_audit_only.npz",
            truth=truth,
            aligned=np.stack(aligned),
        )
    save_json(
        output,
        {
            "audited_utc": datetime.now(UTC).isoformat(),
            "cohort_frozen_sha256": digest(root / "models_frozen.json"),
            "records": records,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "freeze", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coarse", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root, args.coarse)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.action == "freeze":
        freeze(args.root, plan)
    elif args.action == "audit":
        audit(args.root, plan)
    else:
        for job_id in range(args.shard, len(plan["jobs"]), args.shards):
            path = args.root / "models" / f"job_{job_id:03d}.json"
            if path.exists():
                raise RuntimeError("Refusing to overwrite completed experiment")
            run_case(args.root, plan, job_id, args.device)


if __name__ == "__main__":
    main()

"""Freeze, independent structural confirmation, and fresh-C transfer for G4."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment, subspace_error

from .competitor_models import (
    adapted_stats,
    signed_basis,
    train_quantized,
    train_relaxed,
)
from .competitor_pilot import METHODS, regression_validation, save_results
from .contrasts import noise_whitened_basis, signed_graph
from .gaussian_stats import fit_stats, nested_observations
from .gradient_spectrum import BASE, decode, spectral
from .projected_contrasts import crossfit_graph


def register(root):
    plan = json.loads((root / "protocol.json").read_text())
    frozen = root / "hpo_frozen.json"
    if not frozen.exists():
        raise RuntimeError("Tune and freeze competitors first")
    path = root / "confirmation_registration.json"
    if path.exists():
        raise RuntimeError("Confirmation already registered")
    confirmation = {
        "registered_utc": datetime.now(UTC).isoformat(),
        "hpo_sha256": digest(frozen),
        "seed": plan["seed"],
        "families": [
            {"id": family, "kind": kind}
            for kind, prefix in (("conv", 7100), ("signed", 7200))
            for family in range(prefix + 1, prefix + 7)
        ],
        "observations": [512, 2048, 8192, 16384, 32768, 65536, 131072],
        "alpha": 0.05,
        "regimes": [
            {
                "name": "primary",
                "condition": 100.0,
                "noise": 1.0,
                "excitation": "weak",
                "train_at": [8192, 131072],
            },
            {
                "name": "low_noise",
                "condition": 100.0,
                "noise": 0.2,
                "excitation": "weak",
                "train_at": [131072],
            },
            {
                "name": "high_noise",
                "condition": 100.0,
                "noise": 2.0,
                "excitation": "weak",
                "train_at": [131072],
            },
        ],
        "seeds": [31, 32, 33],
        "methods": [
            *METHODS,
            "ols_profiles",
            "ols_pca",
            "ols_pca_scores",
            "corrected_pca",
            "noise_whitened_pca",
            "signed_contrast_graph",
            "codebook_crossfit",
            "whitened_pca_direction_crossfit",
        ],
        "main_method": "codebook_crossfit",
        "source_validation": "Same separate observed full-C/isotropic/noise0.1 source-validation functions as tuning; gradients do not use these. Available to all algorithms for selection. Point spectral estimators and contrast methods have no validation-selected hyperparameters and ignore these extra labels. They are counted separately from main weak-C data. This is an optimizer/recovery benchmark, not a claim of information-theoretic impossibility from all provided data.",
        "confirmation_policy": "Save all models and all source-statistic hashes, then freeze cohort before reading teacher for structural scoring or generating fresh coefficient target tasks. No correction based on confirmation geometry metrics.",
        "transfer": {
            "tasks": 100,
            "support": [1, 4, 16, 64],
            "noise": 0.1,
            "codes": "Fresh independent standard Gaussian, not weak source values",
            "query": "Exact isotropic population clean prediction MSE, computed analytically after new support-fit C; no target query used for selection.",
        },
        "executed_sha256": {
            name: digest(BASE / name)
            for name in (
                "confirmation.py",
                "competitor_models.py",
                "competitor_pilot.py",
                "contrasts.py",
                "gaussian_stats.py",
                "projected_contrasts.py",
                "gradient_spectrum.py",
            )
        },
    }
    confirmation["jobs"] = [
        {"family": family, **regime}
        for family in confirmation["families"]
        for regime in confirmation["regimes"]
    ]
    save_json(path, confirmation)
    (root / "confirmation").mkdir()


def run_case(root, job_id, device):
    search_plan = json.loads((root / "protocol.json").read_text())
    spec = json.loads((root / "confirmation_registration.json").read_text())
    frozen = json.loads((root / "hpo_frozen.json").read_text())
    job = spec["jobs"][job_id]
    folder = root / "confirmation" / f"case_{job_id:03d}"
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    plan = {**search_plan, **{name: spec[name] for name in ("observations", "alpha")}}
    family = job["family"]
    validation = regression_validation(family["id"], family["kind"], plan, device)
    records, arrays, first = [], {}, {}
    observed = nested_observations(
        family["id"],
        family["kind"],
        job["condition"],
        job["excitation"],
        job["noise"],
        plan,
        device,
    )
    for n, truth, stats, halves in observed:
        stats_path = folder / f"source_n{n}.npz"
        if stats_path.exists():
            with np.load(stats_path) as archive:
                stats = {
                    name: torch.tensor(archive[name], device=device)
                    for name in archive.files
                }
        fitted, residual, inverse = fit_stats(stats)
        variance = residual.sum() / (
            plan["tasks"] * plan["outputs"] * (n - plan["inputs"])
        )
        profiles = fitted.flatten(1).T
        raw = profiles @ profiles.T / plan["tasks"]
        noise = torch.block_diag(*[inverse.mean(0) * variance] * plan["outputs"])
        eigenvalues, vectors = torch.linalg.eigh(raw)
        values = {
            "ols_profiles": profiles,
            "ols_pca": vectors[:, -plan["codes"] :],
            "ols_pca_scores": vectors[:, -plan["codes"] :]
            * eigenvalues[-plan["codes"] :].clamp_min(0).sqrt()[None],
            "corrected_pca": spectral(raw - noise, plan["codes"])[0],
            "noise_whitened_pca": noise_whitened_basis(profiles, noise, plan["codes"])[
                0
            ],
        }
        for name, embedding in values.items():
            index, _, _ = decode(embedding, plan, plan["seed"] + family["id"] + n)
            arrays[f"{name}__n{n}"] = index.cpu().numpy().reshape(truth.shape)
        graph_index, graph, _ = signed_graph(
            fitted,
            inverse,
            variance,
            plan["tasks"] * plan["outputs"] * (n - plan["inputs"]),
            plan["codes"],
            spec["alpha"] / len(spec["observations"]),
        )
        graph_results = {"signed_contrast_graph": (graph_index, graph)}
        for name, mode in (
            ("codebook_crossfit", "codebook"),
            ("whitened_pca_direction_crossfit", "whitened_pca"),
        ):
            index, metadata, _ = crossfit_graph(
                stats,
                halves,
                n,
                plan,
                plan["seed"] + family["id"] + n,
                direction_mode=mode,
            )
            graph_results[name] = index, metadata
        for name, (index, metadata) in graph_results.items():
            if index is not None:
                arrays[f"{name}__n{n}"] = index
                first.setdefault(name, n)
            records.append({"n": n, "method": name, "graph": metadata})
        if n in job["train_at"]:
            if not stats_path.exists():
                np.savez_compressed(
                    stats_path,
                    **{name: value.cpu().numpy() for name, value in stats.items()},
                )
            local_plan = {**plan, "n": n}
            for method in METHODS:
                representations = (
                    ["native_or_decoded", "continuous"]
                    if method in ("yeh", "msr_exact")
                    else ["native_or_decoded"]
                )
                selected_rows = sorted(
                    {
                        frozen["choices"][f"{method}__{name}"]["row"]
                        for name in representations
                    }
                )
                rows = [plan["candidates"][method][row] for row in selected_rows]
                for seed in spec["seeds"]:
                    prefix = f"{method}_seed{seed}_n{n}"
                    saved_path = folder / f"{prefix}.json"
                    if saved_path.exists():
                        metadata = json.loads(saved_path.read_text())
                        if (
                            digest(saved_path.with_suffix(".npz"))
                            != metadata["result_sha256"]
                        ):
                            raise RuntimeError(
                                f"Saved model checksum mismatch: {saved_path}"
                            )
                        print("resume saved model", job_id, prefix, flush=True)
                        continue
                    local_plan["checkpoint_every"] = (
                        5 if method == "lc" else plan["checkpoint_every"]
                    )
                    function = (
                        train_relaxed
                        if method in ("yeh", "msr_exact")
                        else train_quantized
                    )
                    result = function(
                        method,
                        rows,
                        stats,
                        validation,
                        local_plan,
                        seed,
                        plan["steps"][method],
                    )
                    result["selected_original_rows"] = np.array(selected_rows)
                    save_results(folder, prefix, result)
        print(
            "confirmation",
            job_id,
            family["id"],
            job["name"],
            n,
            "graph_components",
            {
                name: metadata["components"]
                for name, (_, metadata) in graph_results.items()
            },
            flush=True,
        )
    np.savez_compressed(folder / "statistical_indices.npz", **arrays)
    # Only data synthesis has used this index so far. Scoring occurs after freeze.
    np.savez_compressed(folder / "teacher_audit_only.npz", indices=truth.cpu().numpy())
    save_json(
        folder / "complete.json",
        {
            "job": job,
            "graphs": records,
            "first_return": first,
            "registration_sha256": digest(root / "confirmation_registration.json"),
            "output_sha256": {
                path.name: digest(path) for path in sorted(folder.glob("*.npz"))
            },
        },
    )


def freeze(root):
    spec = json.loads((root / "confirmation_registration.json").read_text())
    dense = root / "dense_prior_frozen.json"
    if not dense.exists():
        raise RuntimeError("Freeze the observation-only dense ridge baseline first")
    hashes = {dense.name: digest(dense)}
    for name in ("protocol.json", "hpo_frozen.json", "confirmation_registration.json"):
        hashes[name] = digest(root / name)
    if (root / "fusion_registration.json").exists():
        fusion = root / "fusion_models_frozen.json"
        state = json.loads(fusion.read_text())
        hashes.update(state["sha256"])
        hashes[fusion.name] = digest(fusion)
        hashes["fusion_hpo_frozen.json"] = digest(root / "fusion_hpo_frozen.json")
    for job_id in range(len(spec["jobs"])):
        folder = root / "confirmation" / f"case_{job_id:03d}"
        state = json.loads((folder / "complete.json").read_text())
        hashes[str((folder / "complete.json").relative_to(root))] = digest(
            folder / "complete.json"
        )
        for name, expected in state["output_sha256"].items():
            path = folder / name
            if digest(path) != expected:
                raise RuntimeError(f"Model changed before freeze: {path}")
            hashes[str(path.relative_to(root))] = expected
            metadata = path.with_suffix(".json")
            if metadata.exists():
                hashes[str(metadata.relative_to(root))] = digest(metadata)
    destination = root / "confirmation_models_frozen.json"
    if destination.exists():
        raise RuntimeError("Models already frozen")
    save_json(
        destination,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "sha256": hashes,
            "registration_sha256": digest(root / "confirmation_registration.json"),
        },
    )


@torch.no_grad()
def transfer(index_or_basis, truth, spec, seed, device, dense_ridge=None):
    index = torch.tensor(truth, device=device)
    codes = spec["codes"]
    reference = signed_basis(index, codes)
    generator = torch.Generator(device=device).manual_seed(seed)
    tasks, maximum = spec["transfer"]["tasks"], max(spec["transfer"]["support"])
    coefficient = torch.randn(
        (tasks, codes), generator=generator, device=device, dtype=torch.float64
    )
    weights = torch.einsum("oik,tk->toi", reference, coefficient)
    x = torch.randn(
        (tasks, maximum, truth.shape[1]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    y = x @ weights.transpose(-1, -2)
    y += spec["transfer"]["noise"] * torch.randn(
        y.shape, generator=generator, device=device, dtype=y.dtype
    )
    basis = None
    if index_or_basis is not None:
        basis = torch.tensor(index_or_basis, device=device, dtype=torch.float64)
        if basis.ndim == 2:
            basis = signed_basis(basis.long(), codes)
    result = {}
    for n in spec["transfer"]["support"]:
        sx, sy = x[:, :n], y[:, :n]
        stats = {
            "gram": sx.transpose(-1, -2) @ sx,
            "cross": sy.transpose(-1, -2) @ sx,
            "square_y": sy.square().sum(1),
        }
        if basis is None:
            matrix = stats["gram"] + dense_ridge * torch.eye(
                truth.shape[1], device=device, dtype=x.dtype
            )
            predicted = torch.linalg.solve(
                matrix, stats["cross"].transpose(-1, -2)
            ).transpose(-1, -2)
        else:
            fitted = adapted_stats(basis[None], stats, n)[0]
            predicted = torch.einsum("oik,tk->toi", basis, fitted)
        errors = (predicted - weights).square().sum(-1).mean(-1)
        result[str(n)] = {
            "mean_mse": float(errors.mean()),
            "median_mse": float(errors.median()),
            "per_task": errors.cpu().tolist(),
        }
    return result


@torch.no_grad()
def freeze_dense(root):
    """Empirical Bayes scalar ridge, using source-validation observations only."""
    path = root / "dense_prior_frozen.json"
    if path.exists():
        raise RuntimeError("Dense baseline already frozen")
    if (root / "confirmation_models_frozen.json").exists():
        raise RuntimeError("Add the baseline before model and target freeze")
    spec = json.loads((root / "confirmation_registration.json").read_text())
    plan = json.loads((root / "protocol.json").read_text())
    rows = {}
    for family in spec["families"]:
        validation = regression_validation(family["id"], family["kind"], plan, "cpu")
        fitted, residual, inverse = fit_stats(validation["query"])
        variance = residual.mean() / (validation["query_n"] - plan["inputs"])
        prior = (
            fitted.square().mean()
            - variance * inverse.diagonal(dim1=-2, dim2=-1).mean()
        )
        ridge = float(variance / prior.clamp_min(1e-8))
        rows[str(family["id"])] = {
            "ridge": ridge,
            "estimated_noise_variance": float(variance),
            "estimated_isotropic_weight_variance": float(prior),
        }
    save_json(
        path,
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "reason": "Strong unrestricted linear numerical baseline. Scalar empirical Bayes ridge lambda=sigma_hat^2/tau_hat^2; tau_hat debiases mean squared dense OLS coefficients using residual noise and empirical inverse Gram. Estimates use only the same source-validation query observations; no teacher or target. Same lambda for all target support budgets, all numerical weights fitted anew.",
            "choices": rows,
            "confirmation_registration_sha256": digest(
                root / "confirmation_registration.json"
            ),
        },
    )


def evaluate_case(root, job_id, device):
    frozen = root / "confirmation_models_frozen.json"
    if not frozen.exists():
        raise RuntimeError(
            "Freeze the full cohort before structural audit or target tasks"
        )
    spec = json.loads((root / "confirmation_registration.json").read_text())
    plan = json.loads((root / "protocol.json").read_text())
    choices = json.loads((root / "hpo_frozen.json").read_text())["choices"]
    folder = root / "confirmation" / f"case_{job_id:03d}"
    if (folder / "audit.json").exists():
        raise RuntimeError("Case already audited")
    state = json.loads((folder / "complete.json").read_text())
    dense_ridge = json.loads((root / "dense_prior_frozen.json").read_text())["choices"][
        str(state["job"]["family"]["id"])
    ]["ridge"]
    with np.load(folder / "teacher_audit_only.npz") as archive:
        truth = archive["indices"]
    records = []
    with np.load(folder / "statistical_indices.npz") as archive:
        for key in archive.files:
            method, n = key.split("__n")
            recovery, _ = alignment(archive[key], truth, plan["codes"])
            record = {"method": method, "n": int(n), "seed": None, "recovery": recovery}
            if (
                int(n) in state["job"]["train_at"]
                or method == "codebook_crossfit"
                and int(n) == state["first_return"].get(method)
            ):
                record["transfer"] = transfer(
                    archive[key],
                    truth,
                    {**spec, "codes": plan["codes"]},
                    761105000 + state["job"]["family"]["id"],
                    device,
                )
            records.append(record)
    for n in state["job"]["train_at"]:
        for method in METHODS:
            representations = (
                ["native_or_decoded", "continuous"]
                if method in ("yeh", "msr_exact")
                else ["native_or_decoded"]
            )
            for seed in spec["seeds"]:
                with np.load(folder / f"{method}_seed{seed}_n{n}.npz") as archive:
                    rows = archive["selected_original_rows"].tolist()
                    for representation in representations:
                        selected = rows.index(
                            choices[f"{method}__{representation}"]["row"]
                        )
                        name = f"{method}__{representation}"
                        record = {
                            "method": name,
                            "n": n,
                            "seed": seed,
                            "validation_mse": float(
                                archive[f"{representation}__selection_mse"][selected]
                            ),
                            "selected_step": int(
                                archive[f"{representation}__selected_step"][selected]
                            ),
                        }
                        if representation == "native_or_decoded":
                            model = archive[f"{representation}__indices"][selected]
                            record["recovery"], _ = alignment(
                                model, truth, plan["codes"]
                            )
                        else:
                            model = archive[f"{representation}__basis"][selected]
                            value, rank = subspace_error(
                                model,
                                signed_basis(
                                    torch.tensor(truth), plan["codes"]
                                ).numpy(),
                            )
                            record["subspace_error"], record["rank"] = value, rank
                        record["transfer"] = transfer(
                            model,
                            truth,
                            {**spec, "codes": plan["codes"]},
                            761105000 + state["job"]["family"]["id"],
                            device,
                        )
                        records.append(record)
        records.append(
            {
                "method": "oracle_audit",
                "n": n,
                "seed": None,
                "transfer": transfer(
                    truth,
                    truth,
                    {**spec, "codes": plan["codes"]},
                    761105000 + state["job"]["family"]["id"],
                    device,
                ),
            }
        )
        records.append(
            {
                "method": "dense_empirical_bayes_ridge",
                "n": n,
                "seed": None,
                "ridge": dense_ridge,
                "transfer": transfer(
                    None,
                    truth,
                    {**spec, "codes": plan["codes"]},
                    761105000 + state["job"]["family"]["id"],
                    device,
                    dense_ridge=dense_ridge,
                ),
            }
        )
        if n == 131072:
            for method in ("wsr_prox", "oscar_prox"):
                metadata_path = folder / f"fusion_{method}.json"
                if not metadata_path.exists():
                    continue
                metadata = json.loads(metadata_path.read_text())
                record = {"method": method, "n": n, "seed": None, "recovery": None}
                if metadata["returned_index"]:
                    with np.load(metadata_path.with_suffix(".npz")) as archive:
                        model = archive["indices"][0]
                    record["recovery"], _ = alignment(model, truth, plan["codes"])
                    record["transfer"] = transfer(
                        model,
                        truth,
                        {**spec, "codes": plan["codes"]},
                        761105000 + state["job"]["family"]["id"],
                        device,
                    )
                records.append(record)
    save_json(
        folder / "audit.json",
        {
            "job": state["job"],
            "records": records,
            "first_return": state["first_return"],
            "frozen_sha256": digest(frozen),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("register", "run", "dense_freeze", "freeze", "evaluate")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--jobs", type=int, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
    elif args.action == "dense_freeze":
        freeze_dense(args.root)
    elif args.action == "freeze":
        freeze(args.root)
    else:
        spec = json.loads((args.root / "confirmation_registration.json").read_text())
        function = run_case if args.action == "run" else evaluate_case
        jobs = (
            args.jobs
            if args.jobs is not None
            else range(args.shard, len(spec["jobs"]), args.shards)
        )
        for job in jobs:
            function(args.root, job, args.device)


if __name__ == "__main__":
    main()

"""G6 diagnostic: observed Hermite moments of the original nonlinear DeepSets.

The whitening/tensor reconstruction is a known-method control, not a new
algorithm claim. The teacher is used for data generation and post-freeze audit.
"""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from experiments.darts_structured_tasks_v1.codebook_projection import (
    exact_signed_projection,
)
from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.darts_structured_tasks_v1.evaluate import align_indices
from experiments.sharing_benchmark_v1.linear import Adam

BASE = Path(__file__).resolve().parent


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G6 exploratory nonlinear observability diagnostic; no new method or competitive confirmation claim",
        "registered_utc": datetime.now(UTC).isoformat(),
        "seed": 2611056817,
        "families": [8401, 8402, 8403, 8404],
        "functions": ["original_codes", "fresh_gaussian_codes"],
        "points": 8,
        "features": 2,
        "point_hidden": 2,
        "codes": 4,
        "inputs": 16,
        "hidden": 16,
        "bias": 0.15,
        "amplitude": 1.0,
        "noise": [0.0, 0.2],
        "observations": [16384, 131072, 1048576, 8388608],
        "affine_pilot": 8192,
        "validation": 8192,
        "chunk": 32768,
        "moment_variants": ["raw", "affine_residual"],
        "reconstruction_sketches": 8,
        "dense": {
            "n": 1048576,
            "steps": 3000,
            "batch": 1024,
            "rates": [0.001, 0.003, 0.01],
            "seeds": [31, 32, 33],
            "checkpoint_every": 100,
            "selection": "Separate observed noisy validation loss, then best restart/rate selected with no teacher. Dense first-layer weights, one shared bias, amplitude and offset; no sharing prior during training.",
        },
        "known_to_reconstruction": "Standard Gaussian inputs, shape H=D, common positive bias and common positive readout; no values of bias/readout or true weights/indices/roles. Moment reconstruction recovers W/b. No teacher autograd queries.",
        "projection": "Additional exact signed scalar DP with fixed zero and at most4 codes, zero sparsity charge. Explicit postprocessing control, not a native indexator learner.",
        "selection": "Eight random matrix sketches chosen by observed whitened tensor off-diagonal residual. No true I used; no change to algorithm after diagnostic metrics.",
        "audit": "Per-function outputs frozen before opening population moment formulas and teacher indices for metrics; these are diagnostic oracle controls, not learning data. Hidden columns and signed code names aligned only for audit.",
        "data": "Nested IID Gaussian input/label prefixes, generated in chunks; raw samples are not written to disk. Independent pilot affine fit, independent validation. Both noise levels share input and unit Gaussian noise. No software tests.",
        "theory_sha256": digest(BASE / "THEORY.md"),
        "code_sha256": digest(Path(__file__)),
    }
    save_json(root / "protocol.json", plan)


def make_teacher(family, function, plan, device):
    rng = np.random.default_rng(plan["seed"] + family)
    permutation = rng.permutation(plan["inputs"])
    index = np.zeros((plan["inputs"], plan["hidden"]), dtype=np.int16)
    for point in range(plan["points"]):
        rows = permutation[point * 2 : (point + 1) * 2]
        columns = np.arange(point * 2, (point + 1) * 2)
        index[np.ix_(rows, columns)] = np.arange(1, 5).reshape(2, 2)
    codes = np.array([0.8, -1.1, 1.3, 0.45])
    if function == "fresh_gaussian_codes":
        codes = rng.normal(size=plan["codes"])
    weights = torch.tensor(np.r_[0.0, codes][index], device=device)
    return weights, index, codes


@torch.no_grad()
def labels(x, weights, plan):
    return (
        (x @ weights + plan["bias"]).relu().sum(-1)
        * plan["amplitude"]
        / math.sqrt(plan["hidden"])
    )


def moment_sums(x, y):
    """Accumulate two noise levels and both raw/control-variate targets."""
    size = x.shape[-1]
    first = y.T @ x
    second = torch.einsum("na,ni,nj->aij", y, x, x)
    identity = torch.eye(size, dtype=x.dtype, device=x.device)
    second -= y.sum(0)[:, None, None] * identity
    products = (x[:, :, None] * x[:, None, :]).flatten(1)
    third = (products.T @ (y[:, :, None] * x[:, None]).flatten(1)).reshape(
        size, size, y.shape[1], size
    )
    third = third.permute(2, 0, 1, 3)
    third -= (
        first[:, :, None, None] * identity[None, None]
        + first[:, None, :, None] * identity[None, :, None]
        + first[:, None, None, :] * identity[None, :, :, None]
    )
    return first, second, third


@torch.no_grad()
def reconstruct(second, third, sketches):
    eigenvalues, eigenvectors = torch.linalg.eigh(second)
    info = {
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
    }
    if eigenvalues[0] <= eigenvalues[-1] * 1e-10:
        return None, {**info, "reason": "Second moment is not positive definite"}
    square = (eigenvectors * eigenvalues.sqrt()) @ eigenvectors.T
    inverse = (eigenvectors / eigenvalues.sqrt()) @ eigenvectors.T
    whitened = torch.einsum("ia,jb,kc,ijk->abc", inverse, inverse, inverse, third)
    best = None
    for sketch in sketches.T:
        contracted = torch.einsum("ijk,k->ij", whitened, sketch)
        _, orthogonal = torch.linalg.eigh(contracted)
        diagonalized = torch.einsum(
            "ia,jb,kc,ijk->abc", orthogonal, orthogonal, orthogonal, whitened
        )
        diagonal = torch.einsum("iii->i", diagonalized)
        off_diagonal = (
            diagonalized.square().sum() - diagonal.square().sum()
        ).clamp_min(0) / whitened.square().sum().clamp_min(1e-30)
        if best is None or off_diagonal < best[0]:
            best = off_diagonal, orthogonal
    off_diagonal, orthogonal = best
    direction = square @ orthogonal
    direction /= direction.norm(dim=0, keepdim=True)
    dual = inverse @ orthogonal
    contraction = torch.einsum("ijk,ia,ja->ka", third, dual, dual)
    ratio = (contraction * direction).sum(0)
    if ratio.abs().min() < 1e-12:
        return None, {**info, "reason": "Vanishing third-order scale"}
    weights = -direction / ratio
    return weights, {
        **info,
        "tensor_off_diagonal_fraction": float(off_diagonal),
        "reconstructed_norm": float(weights.norm()),
    }


@torch.no_grad()
def project(weights, codes):
    labels, centers, quantized = exact_signed_projection(
        weights[None], codes, weights.new_zeros(1), weights.new_ones(1)
    )
    index = torch.where(labels > codes, codes - labels, labels)[0]
    return index.cpu().numpy().astype(np.int16), centers[0], quantized[0]


def train_dense(x, y, validation, plan, seed):
    settings, device = plan["dense"], x.device
    generator = torch.Generator(device=device).manual_seed(seed + 1007)
    combinations = [
        (rate, restart) for rate in settings["rates"] for restart in settings["seeds"]
    ]
    count = len(combinations)
    initial = []
    for _, restart in combinations:
        local = torch.Generator(device=device).manual_seed(seed + restart * 100003)
        initial.append(
            0.1
            * torch.randn(
                (plan["inputs"], plan["hidden"]), generator=local, device=device
            )
        )
    weights = torch.stack(initial).requires_grad_()
    numeric = torch.tensor(
        [[0.1, 1.0, float(y.mean())] for _ in combinations], device=device
    ).requires_grad_()
    optimizer = Adam(
        [weights, numeric], weights.new_tensor([c[0] for c in combinations])
    )
    best = weights.new_full((count,), torch.inf)
    saved_w, saved_numeric = weights.detach().clone(), numeric.detach().clone()
    history = []

    def predict(inputs):
        hidden = torch.einsum("nd,sdh->snh", inputs, weights)
        return (hidden + numeric[:, None, :1]).relu().sum(-1) * numeric[
            :, 1, None
        ] / math.sqrt(plan["hidden"]) + numeric[:, 2, None]

    vx, vy = validation
    for step in range(settings["steps"] + 1):
        if step % settings["checkpoint_every"] == 0:
            with torch.no_grad():
                loss = (predict(vx) - vy).square().mean(-1)
                improved = loss < best
                saved_w[improved] = weights[improved]
                saved_numeric[improved] = numeric[improved]
                best = torch.minimum(best, loss)
                history.append({"step": step, "validation_mse": loss.tolist()})
        if step == settings["steps"]:
            break
        positions = torch.randint(
            len(x), (settings["batch"],), generator=generator, device=device
        )
        loss = (predict(x[positions]) - y[positions]).square().mean(-1)
        optimizer.step(torch.autograd.grad(loss.sum(), [weights, numeric]))
    selected = int(best.argmin())
    return {
        "weights": saved_w.cpu().numpy(),
        "numeric": saved_numeric.cpu().numpy(),
        "validation_mse": best.cpu().tolist(),
        "selected": selected,
        "settings": combinations,
        "history": history,
    }


@torch.no_grad()
def population_moments(weights, plan):
    norm = weights.norm(dim=0)
    directions = weights / norm
    density = torch.exp(-0.5 * (plan["bias"] / norm).square()) / math.sqrt(2 * math.pi)
    coefficient = plan["amplitude"] * norm * density / math.sqrt(plan["hidden"])
    second = (directions * coefficient) @ directions.T
    third = torch.einsum(
        "ij,kj,lj,j->ikl",
        directions,
        directions,
        directions,
        -coefficient * plan["bias"] / norm,
    )
    return second, third


def weight_error(estimate, truth):
    cost = ((estimate[:, :, None] - truth[:, None, :]) ** 2).sum(0)
    source, target = linear_sum_assignment(cost)
    return float(cost[source, target].sum() / np.square(truth).sum())


def run_function(root, plan, family, function, device):
    folder = root / f"family_{family}" / function
    folder.mkdir(parents=True, exist_ok=False)
    seed = plan["seed"] + family + 100000 * (function != "original_codes")
    generator = torch.Generator(device=device).manual_seed(seed)
    weights, truth, codes = make_teacher(family, function, plan, device)
    dimensions = plan["inputs"]
    # The independent affine pilot supplies observed control variates only.
    pilot_x = torch.randn(
        (plan["affine_pilot"], dimensions),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    pilot_clean = labels(pilot_x, weights, plan)
    pilot_noise = torch.randn(
        len(pilot_x), generator=generator, device=device, dtype=torch.float64
    )
    pilot_y = pilot_clean[:, None] + pilot_noise[:, None] * weights.new_tensor(
        plan["noise"]
    )
    design = torch.cat((torch.ones_like(pilot_x[:, :1]), pilot_x), -1)
    affine = torch.linalg.lstsq(design, pilot_y).solution
    validation_x = torch.randn(
        (plan["validation"], dimensions),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    validation_clean = labels(validation_x, weights, plan)
    validation_noise = torch.randn(
        len(validation_x), generator=generator, device=device, dtype=torch.float64
    )
    validation_y = validation_clean[:, None] + validation_noise[
        :, None
    ] * weights.new_tensor(plan["noise"])
    sketches = torch.randn(
        (dimensions, plan["reconstruction_sketches"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    cumulative = [
        weights.new_zeros((4, dimensions)),
        weights.new_zeros((4, dimensions, dimensions)),
        weights.new_zeros((4, dimensions, dimensions, dimensions)),
    ]
    records, arrays, stored_x, stored_y = [], {}, [], []
    count, started = 0, time.perf_counter()
    for n in plan["observations"]:
        while count < n:
            batch = min(plan["chunk"], n - count)
            x = torch.randn(
                (batch, dimensions),
                generator=generator,
                device=device,
                dtype=torch.float64,
            )
            clean = labels(x, weights, plan)
            noise = torch.randn(
                batch, generator=generator, device=device, dtype=torch.float64
            )
            y = clean[:, None] + noise[:, None] * weights.new_tensor(plan["noise"])
            predicted = torch.cat((torch.ones_like(x[:, :1]), x), -1) @ affine
            all_y = torch.cat((y, y - predicted), -1)
            for total, increment in zip(cumulative, moment_sums(x, all_y)):
                total.add_(increment)
            if count < plan["dense"]["n"]:
                stored_x.append(x.float())
                stored_y.append(y.float())
            count += batch
        for variant_id, variant in enumerate(plan["moment_variants"]):
            for noise_id, noise in enumerate(plan["noise"]):
                row = variant_id * 2 + noise_id
                first, second, third = [total[row] / n for total in cumulative]
                reconstructed, info = reconstruct(second, third, sketches)
                key = f"{variant}__noise{noise}__n{n}"
                for name, value in (
                    ("first", first),
                    ("second", second),
                    ("third", third),
                ):
                    arrays[f"{key}__{name}"] = value.cpu().numpy()
                record = {"method": variant, "noise": noise, "n": n, **info, "key": key}
                if reconstructed is not None:
                    index, centers, quantized = project(reconstructed, plan["codes"])
                    arrays[f"{key}__weights_over_bias"] = reconstructed.cpu().numpy()
                    arrays[f"{key}__indices"] = index
                    arrays[f"{key}__centers"] = centers.cpu().numpy()
                    arrays[f"{key}__quantized"] = quantized.cpu().numpy()
                records.append(record)
        print(
            json.dumps(
                {
                    "family": family,
                    "function": function,
                    "moment_n": n,
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
    dense_x, dense_y = torch.cat(stored_x), torch.cat(stored_y)
    del stored_x, stored_y
    for noise_id, noise in enumerate(plan["noise"]):
        result = train_dense(
            dense_x,
            dense_y[:, noise_id],
            (validation_x.float(), validation_y[:, noise_id].float()),
            plan,
            seed,
        )
        selected = result["selected"]
        dense_w = torch.tensor(
            result["weights"][selected], device=device, dtype=torch.float64
        )
        index, centers, quantized = project(dense_w, plan["codes"])
        key = f"dense_adam__noise{noise}__n{plan['dense']['n']}"
        arrays[f"{key}__weights"] = result.pop("weights")
        arrays[f"{key}__numeric"] = result.pop("numeric")
        arrays[f"{key}__indices"] = index
        arrays[f"{key}__quantized"] = quantized.cpu().numpy()
        records.append(
            {
                "method": "dense_adam",
                "noise": noise,
                "n": plan["dense"]["n"],
                "key": key,
                **result,
            }
        )
    np.savez_compressed(folder / "learner_outputs.npz", **arrays)
    save_json(folder / "learner_records.json", {"records": records})
    save_json(
        folder / "learner_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "sha256": {
                name: digest(folder / name)
                for name in ("learner_outputs.npz", "learner_records.json")
            },
        },
    )
    # Only now open the teacher for structural and numerical diagnosis.
    population_second, population_third = population_moments(weights, plan)
    reconstructed, info = reconstruct(population_second, population_third, sketches)
    if reconstructed is None:
        raise RuntimeError("Population identifiability diagnostic unexpectedly failed")
    index, _, _ = project(reconstructed, plan["codes"])
    population_recovery, population_aligned, _ = align_indices(
        index, truth, plan["codes"]
    )
    audit_arrays = {
        "truth_indices": truth,
        "truth_weights": weights.cpu().numpy(),
        "truth_codes": codes,
        "population_second": population_second.cpu().numpy(),
        "population_third": population_third.cpu().numpy(),
        "population_reconstruction": reconstructed.cpu().numpy(),
        "population_aligned_indices": population_aligned,
    }
    for record in records:
        key = record["key"]
        if f"{key}__indices" in arrays:
            recovery, aligned, _ = align_indices(
                arrays[f"{key}__indices"], truth, plan["codes"]
            )
            record["recovery"] = recovery
            audit_arrays[f"{key}__aligned_indices"] = aligned
        if record["method"] == "dense_adam":
            selected = record["selected"]
            heads = arrays[f"{key}__numeric"][selected]
            fitted = arrays[f"{key}__weights"][selected]
            record["weight_relative_squared_error"] = weight_error(
                fitted * heads[1], weights.cpu().numpy() * plan["amplitude"]
            )
            record["selected_validation_mse"] = float(
                record["validation_mse"][selected]
            )
            prediction = (
                validation_x.float() @ torch.tensor(fitted, device=device)
                + float(heads[0])
            ).relu().sum(-1) * float(heads[1]) / math.sqrt(plan["hidden"]) + float(
                heads[2]
            )
            record["clean_validation_nmse_audit"] = float(
                (prediction - validation_clean).square().mean() / validation_clean.var()
            )
        else:
            record["relative_second_error"] = float(
                (
                    torch.tensor(arrays[f"{key}__second"], device=device)
                    - population_second
                ).norm()
                / population_second.norm()
            )
            record["relative_third_error"] = float(
                (
                    torch.tensor(arrays[f"{key}__third"], device=device)
                    - population_third
                ).norm()
                / population_third.norm()
            )
            if f"{key}__weights_over_bias" in arrays:
                record["weight_relative_squared_error"] = weight_error(
                    arrays[f"{key}__weights_over_bias"],
                    weights.cpu().numpy() / plan["bias"],
                )
    np.savez_compressed(folder / "audit_only.npz", **audit_arrays)
    save_json(
        folder / "audit.json",
        {
            "family": family,
            "function": function,
            "population_oracle_only": {
                **info,
                "recovery": population_recovery,
                "weight_relative_squared_error": weight_error(
                    reconstructed.cpu().numpy(), weights.cpu().numpy() / plan["bias"]
                ),
            },
            "records": records,
            "frozen_sha256": digest(folder / "learner_frozen.json"),
            "seconds": time.perf_counter() - started,
        },
    )
    print(
        json.dumps(
            {
                "family": family,
                "function": function,
                "complete": True,
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action == "register":
        register(args.root)
        return
    plan = json.loads((args.root / "protocol.json").read_text())
    if args.family not in plan["families"]:
        raise ValueError("Family must be registered")
    for function in plan["functions"]:
        run_function(args.root, plan, args.family, function, args.device)


if __name__ == "__main__":
    main()

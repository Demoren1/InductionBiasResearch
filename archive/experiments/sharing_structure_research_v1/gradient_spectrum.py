"""G1: observation-only gradient/spectral diagnostics, not a proposed new method."""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.darts_structured_tasks_v1.data import digest, save_json
from experiments.sharing_benchmark_v1.evaluate import alignment
from experiments.sharing_benchmark_v1.linear import index_basis

BASE = Path(__file__).resolve().parent
METHODS = (
    "gradient_profiles",
    "gradient_pca",
    "population_preconditioned_profiles",
    "ols_profiles",
    "ols_pca",
    "ols_pca_scores",
    "gradient_cross_spectrum",
    "ols_cross_spectrum",
    "ols_noise_debiased_spectrum",
)


def register(root):
    root.mkdir(parents=True, exist_ok=False)
    plan = {
        "stage": "G1 diagnostic, not confirmation or novelty claim",
        "seed": 2611053307,
        "families": [4101, 4102, 4103, 4201, 4202, 4203],
        "inputs": 16,
        "outputs": 14,
        "codes": 3,
        "observations": [32, 128, 512, 2048],
        "source_tasks": [8, 32, 128],
        "input_condition": [1.0, 100.0],
        "code_excitation": ["full", "weak"],
        "noise": [0.0, 0.2, 1.0],
        "additional": [
            {
                "excitation": "collision",
                "condition": 1.0,
                "noise": 0.0,
                "tasks": 128,
                "n": 2048,
            },
            {
                "excitation": "weak",
                "condition": 100.0,
                "noise": 1.0,
                "tasks": 128,
                "n": 8192,
            },
        ],
        "decoder_restarts": 8,
        "decoder_steps": 64,
        "methods": list(METHODS),
        "dictionary": "Fixed zero plus three vector centers and their negatives; no known active count.",
        "families_description": {
            "4101_4103": "Randomly permuted valid convolution, 14 entries per code.",
            "4201_4203": "Arbitrary signed sharing with hidden group sizes48/12/3 and161 zeros. Counts are teacher sampling only, never passed to decoder.",
        },
        "data": "Nested source task/sample prefixes per family/condition/excitation. Same X and Gaussian noise realization across noise levels. Two source halves are independent observations with the same task C. No teacher derivatives or code values enter estimators.",
        "scope": "All algorithms here are diagnostic baselines, including covariance correction. Population covariance preconditioning is an oracle-design diagnostic, explicitly excluded from fair production comparison.",
        "identifiability_control": "collision has c2=c1 for every source task. Its true distinction is unidentifiable from arbitrarily many observations of those functions.",
        "checks": "No software tests. Check empirical gradient preconditioning equals OLS and OLS gradient vanishes, then scientific metrics and plots.",
    }
    save_json(root / "protocol.json", plan)
    save_json(
        root / "registration.json",
        {
            "registered_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(root / "protocol.json"),
            "theory_sha256": digest(BASE / "THEORY.md"),
            "code_sha256": digest(Path(__file__)),
        },
    )


def teacher_index(family, plan, device):
    rng = np.random.default_rng(plan["seed"] + family)
    index = np.zeros((plan["outputs"], plan["inputs"]), dtype=np.int16)
    if family < 4200:
        for output in range(plan["outputs"]):
            index[output, output : output + plan["codes"]] = np.arange(
                1, plan["codes"] + 1
            )
        index = index[rng.permutation(len(index))][:, rng.permutation(index.shape[1])]
    else:
        positions = rng.permutation(index.size)
        start = 0
        for code, count in enumerate((48, 12, 3), 1):
            signs = rng.choice((-1, 1), size=count)
            index.flat[positions[start : start + count]] = code * signs
            start += count
    return torch.tensor(index, device=device).long()


def observations(family, condition, excitation, noise, maximum, plan, device):
    truth = teacher_index(family, plan, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(plan["seed"] + family * 100003 + 19)
    if condition == 1:
        transform = torch.eye(plan["inputs"], device=device, dtype=torch.float64)
    else:
        random = torch.randn(
            (plan["inputs"], plan["inputs"]),
            generator=generator,
            device=device,
            dtype=torch.float64,
        )
        q, _ = torch.linalg.qr(random)
        eigenvalues = torch.logspace(
            0, np.log10(condition), plan["inputs"], device=device, dtype=torch.float64
        )
        eigenvalues /= eigenvalues.mean()
        transform = q * eigenvalues.sqrt()[None]
    covariance = transform @ transform.T
    generator.manual_seed(plan["seed"] + family * 100003 + 23)
    centers = torch.randn(
        (128, plan["codes"]), generator=generator, device=device, dtype=torch.float64
    )
    if excitation == "weak":
        centers *= centers.new_tensor((1.0, 0.3, 0.03))
    elif excitation == "collision":
        centers[:, 1] = centers[:, 0]
    basis = index_basis(truth[None], plan["codes"])[0].double()
    weights = torch.einsum("oik,tk->toi", basis, centers)
    generator.manual_seed(plan["seed"] + family * 100003 + 29)
    raw = torch.randn(
        (128, maximum, plan["inputs"]),
        generator=generator,
        device=device,
        dtype=torch.float64,
    )
    x = raw @ transform.T
    clean = torch.einsum("tni,toi->tno", x, weights)
    generator.manual_seed(plan["seed"] + family * 100003 + 31)
    disturbance = torch.randn(
        clean.shape, generator=generator, device=device, dtype=torch.float64
    )
    return truth, x, clean + noise * disturbance, covariance, weights


def spectral(matrix, codes):
    eigenvalues, vectors = torch.linalg.eigh((matrix + matrix.T) * 0.5)
    return vectors[:, -codes:], eigenvalues[-codes:], eigenvalues


def profile_spectral(values, codes):
    covariance = values @ values.T / values.shape[1]
    vectors, eigenvalues, spectrum = spectral(covariance, codes)
    return vectors, vectors * eigenvalues.clamp_min(0).sqrt()[None], spectrum


def decode(values, plan, seed):
    """Signed vector Lloyd with a fixed zero; choose restart by its own fit."""
    values = values / np.sqrt(values.shape[-1])
    generator = torch.Generator(device=values.device).manual_seed(seed)
    restarts, codes = plan["decoder_restarts"], plan["codes"]
    centers = values.new_zeros((restarts, codes, values.shape[-1]))
    distance = values.square().sum(-1).expand(restarts, -1).clone()
    for code in range(codes):
        position = torch.multinomial(distance + 1e-30, 1, generator=generator)[:, 0]
        centers[:, code] = values[position]
        positive = (values[None] - centers[:, code, None]).square().sum(-1)
        negative = (values[None] + centers[:, code, None]).square().sum(-1)
        distance = torch.minimum(distance, torch.minimum(positive, negative))
    labels, label_trace, loss_trace = None, [], []
    for _ in range(plan["decoder_steps"]):
        product = torch.einsum("ef,rkf->rek", values, centers)
        norm = centers.square().sum(-1)[:, None]
        costs = torch.cat(
            (product[..., :1] * 0, norm - 2 * product, norm + 2 * product), -1
        )
        selected = costs.argmin(-1)
        current = torch.where(selected > codes, codes - selected, selected)
        assigned = F.one_hot(current.abs().long(), codes + 1)[..., 1:].to(values.dtype)
        assigned *= current.sign()[..., None]
        counts = assigned.abs().sum(1)
        sums = torch.einsum("rek,ef->rkf", assigned, values)
        means = sums / counts.clamp_min(1)[..., None]
        centers = torch.where((counts > 0)[..., None], means, centers)
        reconstruction = torch.einsum("rek,rkf->ref", assigned, centers)
        loss = (values[None] - reconstruction).square().sum(-1).mean(-1)
        label_trace.append(current.cpu().numpy().astype(np.int16))
        loss_trace.append(loss.cpu().numpy())
        if labels is not None and torch.equal(labels, current):
            break
        labels = current
    winner = int(loss.argmin())
    return (
        current[winner],
        np.stack(label_trace)[:, winner],
        np.stack(loss_trace)[:, winner],
    )


def subspace_metrics(vectors, truth, codes):
    target = index_basis(truth[None], codes)[0].double().reshape(-1, codes)
    counts = target.square().sum(0)
    orthogonal = target / counts.sqrt()[None]
    overlap = orthogonal.T @ vectors
    left, singular, right = torch.linalg.svd(overlap)
    aligned = vectors @ (left @ right).T
    row_error = (aligned - orthogonal).norm(dim=1)
    return {
        "subspace_error": float(
            ((codes - singular.square().sum()).clamp_min(0) / codes).sqrt()
        ),
        "minimum_subspace_cosine": float(singular.min()),
        "max_aligned_row_error": float(row_error.max()),
        "sufficient_row_error_bound": float(counts.max().rsqrt() / 4),
    }


@torch.no_grad()
def run_case(truth, x, y, covariance, weights, plan, key, root):
    started = time.perf_counter()
    tasks, n, inputs = x.shape
    gram = x.transpose(-1, -2) @ x
    cross = y.transpose(-1, -2) @ x
    gradient = cross / n
    estimate = torch.linalg.lstsq(x, y).solution.transpose(-1, -2)
    normal = torch.linalg.solve(gram, cross.transpose(-1, -2)).transpose(-1, -2)
    relative = float((normal - estimate).norm() / estimate.norm().clamp_min(1e-12))
    stationarity = estimate @ gram / n - gradient
    stationary_relative = float(stationarity.norm() / gradient.norm().clamp_min(1e-12))
    if relative > 1e-7 or stationary_relative > 1e-7:
        raise FloatingPointError("Linear gradient/normal-equation identity failed")
    raw = gradient.flatten(1).T
    ols = estimate.flatten(1).T
    preconditioned = (
        torch.linalg.solve(covariance, gradient.transpose(-1, -2))
        .transpose(-1, -2)
        .flatten(1)
        .T
    )
    raw_u, _, raw_spectrum = profile_spectral(raw, plan["codes"])
    ols_u, ols_scores, ols_spectrum = profile_spectral(ols, plan["codes"])
    pre_u, _, pre_spectrum = profile_spectral(preconditioned, plan["codes"])
    gradients, estimates = [], []
    for split in (slice(0, n // 2), slice(n // 2, n)):
        local_x, local_y = x[:, split], y[:, split]
        gradients.append(
            (local_y.transpose(-1, -2) @ local_x / local_x.shape[1]).flatten(1).T
        )
        estimates.append(
            torch.linalg.lstsq(local_x, local_y).solution.transpose(-1, -2).flatten(1).T
        )
    raw_cross = (gradients[0] @ gradients[1].T + gradients[1] @ gradients[0].T) / (
        2 * tasks
    )
    ols_cross = (estimates[0] @ estimates[1].T + estimates[1] @ estimates[0].T) / (
        2 * tasks
    )
    raw_cv, _, raw_cv_spectrum = spectral(raw_cross, plan["codes"])
    ols_cv, _, ols_cv_spectrum = spectral(ols_cross, plan["codes"])
    residual = torch.einsum("tni,toi->tno", x, estimate) - y
    variance = residual.square().sum(1) / (n - inputs)
    inverse_gram = torch.linalg.inv(gram)
    noise_blocks = torch.einsum("to,tij->oij", variance, inverse_gram) / tasks
    noise_covariance = torch.block_diag(*noise_blocks.unbind(0))
    debiased, _, debiased_spectrum = spectral(
        ols @ ols.T / tasks - noise_covariance, plan["codes"]
    )
    embeddings = {
        "gradient_profiles": (raw, raw_u, raw_spectrum),
        "gradient_pca": (raw_u, raw_u, raw_spectrum),
        "population_preconditioned_profiles": (preconditioned, pre_u, pre_spectrum),
        "ols_profiles": (ols, ols_u, ols_spectrum),
        "ols_pca": (ols_u, ols_u, ols_spectrum),
        "ols_pca_scores": (ols_scores, ols_u, ols_spectrum),
        "gradient_cross_spectrum": (raw_cv, raw_cv, raw_cv_spectrum),
        "ols_cross_spectrum": (ols_cv, ols_cv, ols_cv_spectrum),
        "ols_noise_debiased_spectrum": (debiased, debiased, debiased_spectrum),
    }
    records, showcase = (
        [],
        {
            "truth": truth.cpu().numpy(),
            "source_codes_weight_matrix": weights.cpu().numpy(),
        },
    )
    truth_numpy = truth.cpu().numpy()
    for method, (embedding, basis, spectrum) in embeddings.items():
        method_started = time.perf_counter()
        index, trace, losses = decode(embedding, plan, plan["seed"] + key["family"])
        index = index.reshape_as(truth)
        recovery, aligned = alignment(index.cpu().numpy(), truth_numpy, plan["codes"])
        recall_trace = []
        for labels in trace:
            predicted = labels.reshape(truth_numpy.shape) != 0
            actual = truth_numpy != 0
            recall_trace.append(float((predicted & actual).sum() / actual.sum()))
        record = {
            **key,
            "method": method,
            "recovery": recovery,
            "spectral": subspace_metrics(basis, truth, plan["codes"]),
            "largest_eigenvalues": spectrum[-6:].flip(0).cpu().tolist(),
            "spectral_gap_at_K": float(
                spectrum[-plan["codes"]] - spectrum[-plan["codes"] - 1]
            ),
            "fit_loss_by_step": losses.tolist(),
            "mask_recall_by_step": recall_trace,
            "decode_seconds": time.perf_counter() - method_started,
            "normal_equation_relative_error": relative,
            "optimum_to_initial_gradient_norm": stationary_relative,
        }
        records.append(record)
        showcase[f"{method}_indices"] = index.cpu().numpy()
        showcase[f"{method}_aligned_indices"] = aligned
        showcase[f"{method}_embedding"] = embedding.cpu().numpy()
        showcase[f"{method}_spectrum"] = spectrum.cpu().numpy()
    if (
        key["tasks"] == 128
        and key["n"] >= 2048
        and key["noise"] == 1.0
        and key["excitation"] == "weak"
        and key["condition"] == 100
    ):
        output = root / "showcases"
        output.mkdir(exist_ok=True)
        np.savez_compressed(output / f"family_{key['family']}_n{n}.npz", **showcase)
    return records, time.perf_counter() - started


def jobs(plan):
    cases = []
    for family, condition, excitation, noise in itertools.product(
        plan["families"],
        plan["input_condition"],
        plan["code_excitation"],
        plan["noise"],
    ):
        cases.append(
            {
                "family": family,
                "condition": condition,
                "excitation": excitation,
                "noise": noise,
                "sizes": list(
                    itertools.product(plan["source_tasks"], plan["observations"])
                ),
            }
        )
    for family, extra in itertools.product(plan["families"], plan["additional"]):
        cases.append(
            {
                "family": family,
                "condition": extra["condition"],
                "excitation": extra["excitation"],
                "noise": extra["noise"],
                "sizes": [(extra["tasks"], extra["n"])],
            }
        )
    return cases


def worker(root, shard, shards, device):
    plan = json.loads((root / "protocol.json").read_text())
    output = root / "cases"
    output.mkdir(exist_ok=True)
    for number, job in list(enumerate(jobs(plan)))[shard::shards]:
        path = output / f"job_{number:03d}.json"
        if path.exists():
            continue
        maximum = max(8192, max(n for _, n in job["sizes"]))
        truth, x, y, covariance, weights = observations(
            job["family"],
            job["condition"],
            job["excitation"],
            job["noise"],
            maximum,
            plan,
            device,
        )
        records, elapsed = [], 0.0
        for tasks, n in job["sizes"]:
            key = {name: value for name, value in job.items() if name != "sizes"}
            key.update(tasks=tasks, n=n)
            values, seconds = run_case(
                truth,
                x[:tasks, :n],
                y[:tasks, :n],
                covariance,
                weights[:tasks],
                plan,
                key,
                root,
            )
            records.extend(values)
            elapsed += seconds
        save_json(
            path,
            {
                "job": job,
                "records": records,
                "seconds": elapsed,
                "protocol_sha256": digest(root / "protocol.json"),
                "code_sha256": digest(Path(__file__)),
            },
        )
        print(
            "finished job",
            number,
            job["family"],
            job["condition"],
            job["excitation"],
            job["noise"],
            "cases",
            len(job["sizes"]),
            "seconds",
            round(elapsed, 2),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=("register", "run"), required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == "register":
        register(args.root)
    else:
        worker(args.root, args.shard, args.shards, args.device)


if __name__ == "__main__":
    main()

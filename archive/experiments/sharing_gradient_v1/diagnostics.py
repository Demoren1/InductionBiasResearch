"""Compare projection proxies on identical observation-only candidate pools."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.darts_structured_tasks_v1.data import save_json
from experiments.sharing_benchmark_v1.linear import (
    adaptation_errors,
    adapted,
    index_basis,
)

from .algorithms import metric_projection, source_statistics, weights_from_codes
from .run import cohort, result_path


@torch.no_grad()
def projection_audit(root, folder, data, family, seed):
    plan = json.loads((root / "protocol.json").read_text())
    choices = json.loads((root / "projection_frozen.json").read_text())["choices"]
    _, _, gram, cross, _ = source_statistics(data)
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype)
    results = {}
    for method in ("z_hessian", "z_taylor"):
        selected = choices[method]["settings"]
        start = selected["configuration"] * plan["restarts"]
        stop = start + plan["restarts"]
        path = result_path(
            folder, "pilot_projection", plan["pilot_task_budget"], method, seed
        )
        with np.load(path) as archive:
            values = torch.tensor(
                archive["final_values"][start:stop], device=gram.device
            )
            index = torch.tensor(
                archive["final_indices"][start:stop], device=gram.device
            ).long()
            centers = torch.tensor(
                archive["final_codes"][start:stop], device=gram.device
            )
        alpha = selected["rho_final"] / (2 * gram.shape[-1])
        metric = (gram + alpha * eye)[None].expand(len(values), -1, -1, -1)
        target = (
            torch.einsum("btoi,btij->btoj", values, metric)
            if method == "z_hessian"
            else cross[None] + alpha * values
        )
        before = weights_from_codes(index, centers)
        _, _, after = metric_projection(
            index.clone(), centers.clone(), metric, target, plan["projection_sweeps"]
        )

        def objective(weights, metric=metric, target=target):
            return (
                weights * torch.einsum("btoi,btij->btoj", weights, metric)
                - 2 * weights * target
            ).mean((1, 2, 3))

        initial, final = objective(before), objective(after)
        increased = float((final - initial).max())
        if increased > 1e-8 * max(1, float(initial.abs().max())):
            raise FloatingPointError(
                "Local full-metric projector increased its own quadratic objective"
            )
        results[method] = {
            "initial_objective": initial.tolist(),
            "final_objective": final.tolist(),
            "max_increase": increased,
        }
    return results


@torch.no_grad()
def candidate_scores(pool, reference, data, codes):
    ref = torch.tensor(reference, device=data["train_support_x"].device).double()
    xq, yq = data["train_query_x"].double(), data["train_query_y"].double()
    residual = torch.einsum("tni,toi->tno", xq, ref) - yq
    baseline = residual.square().mean()
    accumulated = {
        name: []
        for name in (
            "euclidean_projected_codes",
            "euclidean_support_codes",
            "diagonal_hessian",
            "full_hessian",
            "gradient_term",
            "taylor",
            "adapted_query",
            "validation",
        )
    }
    for start in range(0, len(pool), 32):
        index = torch.tensor(pool[start : start + 32], device=ref.device)
        basis = index_basis(index, codes).double()
        centers = adapted(
            basis, data["train_support_x"].double(), data["train_support_y"].double()
        )
        weights = torch.einsum("boik,btk->btoi", basis, centers)
        difference = weights - ref[None]
        changed = torch.einsum("tni,btoi->btno", xq, difference)
        hessian = changed.square().mean((1, 2, 3))
        gradient = 2 * (residual[None] * changed).mean((1, 2, 3))
        actual = (residual[None] + changed).square().mean((1, 2, 3))
        fitted_centers = torch.einsum("boik,toi->btk", basis, ref)
        fitted_centers /= basis.square().sum((1, 2))[:, None].clamp_min(1)
        fitted_weights = torch.einsum("boik,btk->btoi", basis, fitted_centers)
        values = {
            "euclidean_projected_codes": (fitted_weights - ref[None])
            .square()
            .sum((2, 3))
            .mean(1)
            / ref.shape[1],
            "euclidean_support_codes": difference.square().sum((2, 3)).mean(1)
            / ref.shape[1],
            "diagonal_hessian": (
                difference.square() * xq.square().mean(1)[None, :, None]
            )
            .sum((2, 3))
            .mean(1)
            / ref.shape[1],
            "full_hessian": hessian,
            "gradient_term": gradient,
            "taylor": baseline + gradient + hessian,
            "adapted_query": actual,
            "validation": adaptation_errors(
                basis,
                data["selection_support_x"],
                data["selection_support_y"],
                data["selection_query_x"],
                data["selection_query_y"],
            ).mean(-1),
        }
        for name, value in values.items():
            accumulated[name].append(value.cpu().numpy())
    arrays = {name: np.concatenate(parts) for name, parts in accumulated.items()}
    gap = float(np.max(np.abs(arrays["taylor"] - arrays["adapted_query"])))
    if gap > 1e-9 * max(1, float(arrays["adapted_query"].max())):
        raise FloatingPointError(
            "Full Taylor expression does not equal linear squared loss"
        )
    summary = {
        "candidates": len(pool),
        "dense_query_mse": float(baseline),
        "max_taylor_error": gap,
        "metrics": {},
    }
    for name in (
        "euclidean_projected_codes",
        "euclidean_support_codes",
        "diagonal_hessian",
        "full_hessian",
        "taylor",
    ):
        score = arrays[name]
        chosen = int(score.argmin())
        summary["metrics"][name] = {
            "spearman_with_adapted_query": float(
                spearmanr(score, arrays["adapted_query"]).statistic
            ),
            "chosen_adapted_query_mse": float(arrays["adapted_query"][chosen]),
            "chosen_validation_mse": float(arrays["validation"][chosen]),
            "chosen_candidate": chosen,
        }
    return arrays, summary


def diagnose(root, shard, shards, device):
    plan = json.loads((root / "protocol.json").read_text())
    selected = json.loads((root / "projection_frozen.json").read_text())["choices"][
        "z_euclidean"
    ]["settings"]["configuration"]
    jobs = list(itertools.product(plan["pilot_families"], plan["pilot_seeds"]))
    for family, seed in jobs[shard::shards]:
        folder = root / "hidden" / f"family_{family}"
        tasks = plan["pilot_task_budget"]
        source = result_path(folder, "pilot_projection", tasks, "z_euclidean", seed)
        with np.load(source) as archive:
            best = archive["indices"][selected]
            checkpoints = archive["checkpoint_indices"].reshape(-1, *best.shape)
            references = {
                "initial": archive["source_fit_reference"],
                "final": archive["final_latent"][selected * plan["restarts"]],
            }
        with np.load(
            result_path(folder, "pilot_meta", tasks, "meta_full", seed)
        ) as archive:
            checkpoints = np.concatenate(
                (checkpoints, archive["checkpoint_indices"].reshape(-1, *best.shape))
            )
        rng = np.random.default_rng(plan["seed"] + family + seed)
        perturbations = np.repeat(best[None], 1024, axis=0)
        for candidate in perturbations:
            for _ in range(int(rng.integers(1, 5))):
                candidate.flat[int(rng.integers(candidate.size))] = int(
                    rng.integers(-3, 4)
                )
        pool = np.unique(np.concatenate((checkpoints, perturbations)), axis=0)
        data = cohort(folder, tasks, device)
        output = root / "diagnostics"
        output.mkdir(exist_ok=True)
        summaries = {}
        saved = {"candidate_indices": pool}
        for label, ref in references.items():
            arrays, summary = candidate_scores(pool, ref, data, plan["hidden"]["codes"])
            saved.update({f"{label}_{name}": value for name, value in arrays.items()})
            summaries[label] = summary
        np.savez_compressed(output / f"family_{family}_seed_{seed}.npz", **saved)
        save_json(
            output / f"family_{family}_seed_{seed}.json",
            {
                "family": family,
                "seed": seed,
                "source_tasks": tasks,
                "identical_candidate_pool": True,
                "teacher_used": False,
                "states": summaries,
                "local_projector_audit": projection_audit(
                    root, folder, data, family, seed
                ),
            },
        )
        print("diagnosed proxies", family, seed, len(pool), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    diagnose(args.root, args.shard, args.shards, args.device)


if __name__ == "__main__":
    main()

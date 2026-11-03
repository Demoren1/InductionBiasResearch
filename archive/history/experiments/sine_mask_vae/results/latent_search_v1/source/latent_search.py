"""Optimize latent codes through SGD while keeping the reference VAE frozen.

Run with ``python3 -m sine_mask_vae.latent_search --help``. This is a separate
experiment: the reference bank, checkpoints and published v1 are read-only.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .core import population_loss, save_json, task, teacher_vectors
from .models import MaskVAE, hard_mask
from .settings import DIRECTORY, digest, families


METHODS = ("gradient_z", "random_z", "prior_sampling")


@dataclass(frozen=True)
class LatentSearchOptions:
    restarts: int
    rounds: int
    final_restarts: int
    inner_steps: int
    batch: int
    query_examples: int
    weight_lr: float
    initial_scale: float
    latent_lr: float
    temperature: float
    gradient_clip: float

    @property
    def fits_per_search(self):
        return self.restarts * self.rounds + self.final_restarts


def relaxed_mask(scores, temperature):
    """Hard global top-2 forward, sequential RelaxedTopK derivative backward."""
    logits = scores.flatten(-2)
    relaxed = torch.zeros_like(logits)
    for _ in range(2):
        probabilities = torch.softmax(logits / temperature, dim=-1)
        relaxed = relaxed + probabilities
        logits = logits + torch.log((1 - probabilities).clamp_min(1e-6))
    soft = relaxed.reshape_as(scores)
    return hard_mask(scores) + (soft - soft.detach())


def examples(spec, count, generator, device):
    x = torch.randn(count, 8, generator=generator, device=device)
    low, high, small = teacher_vectors(spec, device)
    return x, small * torch.sin(x @ low) + torch.sin(x @ high)


def predictions(x, weights, amplitude):
    return amplitude[:, None] * torch.sin(
        torch.einsum("bi,nij->nbj", x, weights)
    )


def inner_sgd(mask, initial, batches, targets, query, query_targets, options):
    """Unroll the exact sine-network SGD updates, differentiably when needed.

    The analytic parameter gradients equal autograd of MSE over batch and the
    two outputs. Autograd differentiates the complete update trajectory w.r.t.
    the mask and z; no population-loss formula is used for optimization.
    """
    raw = initial
    amplitude = torch.ones(len(mask), 2, device=mask.device)
    for x, y in zip(batches, targets):
        angles = torch.einsum("bi,nij->nbj", x, raw * mask)
        sine = angles.sin()
        errors = amplitude[:, None] * sine - y
        grad_weights = torch.einsum(
            "bi,nbj->nij", x, errors * amplitude[:, None] * angles.cos()
        ) / len(x)
        grad_amplitude = (errors * sine).mean(1)
        raw = raw - options.weight_lr * grad_weights * mask
        amplitude = amplitude - options.weight_lr * grad_amplitude
    weights = raw * mask
    train_loss = (
        predictions(batches[-1], weights, amplitude) - targets[-1]
    ).square().mean((1, 2))
    query_loss = (
        predictions(query, weights, amplitude) - query_targets
    ).square().mean((1, 2))
    return train_loss, query_loss, weights, amplitude


def load_decoder(reference, variant, seed, device):
    records = json.loads((reference / "models.json").read_text())
    record = next(
        r for r in records if r["variant"] == variant and r["seed"] == seed
    )
    path = reference / record["checkpoint"]
    if digest(path) != record["sha256"]:
        raise RuntimeError(f"Reference checkpoint changed: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = MaskVAE(state["tasks"])
    model.load_state_dict(state["state"])
    return model.to(device).eval().requires_grad_(False), record


def run_search(model, spec, seed, trials, method, options, device):
    """Run paired searches; only gradient_z backpropagates through training."""
    started = time.perf_counter()
    data_rng = torch.Generator(device=device).manual_seed(seed + 700001)
    x, y = examples(
        spec, options.inner_steps * options.batch, data_rng, device
    )
    batches = x.reshape(options.inner_steps, options.batch, 8)
    targets = y.reshape(options.inner_steps, options.batch, 2)
    query, query_targets = examples(
        spec, options.query_examples, data_rng, device
    )
    # An independent empirical sample selects checkpoints and final candidates.
    selection_x, selection_y = examples(
        spec, options.query_examples, data_rng, device
    )
    latent_rng = torch.Generator(device=device).manual_seed(seed + 3)
    weight_rng = torch.Generator(device=device).manual_seed(seed)
    per_round = (
        options.fits_per_search // options.rounds
        if method == "prior_sampling"
        else options.restarts
    )
    count = trials * per_round
    initial = options.initial_scale * torch.randn(
        count, 8, 2, generator=weight_rng, device=device
    )
    latent = torch.randn(count, 16, generator=latent_rng, device=device)
    if method == "gradient_z":
        latent.requires_grad_()
        optimizer = torch.optim.Adam([latent], lr=options.latent_lr)
    best_empirical = torch.full((trials,), float("inf"), device=device)
    best_population = best_empirical.clone()
    best_mask = torch.zeros(trials, 8, 2, device=device)
    oracle = torch.zeros(8, 2, device=device)
    oracle[spec["rows"], torch.arange(2, device=device)] = 1
    history, all_masks, all_empirical, all_population = [], [], [], []
    initial_latent = latent.detach().clone()

    def retain(mask, weights, amplitude):
        nonlocal best_empirical, best_population, best_mask
        with torch.no_grad():
            empirical = (
                predictions(selection_x, weights, amplitude) - selection_y
            ).square().mean((1, 2))
            population = population_loss(weights, amplitude, spec)
            candidates = len(mask) // trials
            empirical = empirical.reshape(trials, candidates)
            chosen = empirical.argmin(1)
            rows = torch.arange(trials, device=device)
            values = empirical[rows, chosen]
            improved = values < best_empirical
            best_empirical = torch.minimum(best_empirical, values)
            selected_population = population.reshape(trials, candidates)[
                rows, chosen
            ]
            best_population = torch.where(
                improved, selected_population, best_population
            )
            selected_mask = mask.detach().reshape(trials, candidates, 8, 2)[
                rows, chosen
            ]
            best_mask = torch.where(
                improved[:, None, None], selected_mask, best_mask
            )
            all_masks.append(mask.detach().cpu().numpy())
            all_empirical.append(empirical.cpu().numpy())
            all_population.append(
                population.reshape(trials, candidates).cpu().numpy()
            )
        return population

    for step in range(options.rounds):
        if step and method in ("random_z", "prior_sampling"):
            latent = torch.randn(
                count, 16, generator=latent_rng, device=device
            )
        if step and method == "prior_sampling":
            initial = options.initial_scale * torch.randn(
                count, 8, 2, generator=weight_rng, device=device
            )
        differentiable = method == "gradient_z" and step < options.rounds - 1
        with nullcontext() if differentiable else torch.no_grad():
            scores = model.decode(latent)
            mask = (
                relaxed_mask(scores, options.temperature)
                if differentiable
                else hard_mask(scores)
            )
            train_loss, query_loss, weights, amplitude = inner_sgd(
                mask, initial, batches, targets, query, query_targets, options
            )
        population = retain(mask, weights, amplitude)
        record = {
            "round": step,
            "network_fits": (step + 1) * per_round,
            "train_mse": float(train_loss.detach().mean()),
            "query_mse": float(query_loss.detach().mean()),
            "candidate_population_mse": float(population.mean()),
            "candidate_recall_at_2": float(
                (mask.detach() * oracle).sum((-1, -2)).mean() / 2
            ),
            "selected_population_mse": float(best_population.mean()),
            "selected_recall_at_2": float(
                (best_mask * oracle).sum((-1, -2)).mean() / 2
            ),
            "selected_success_rate": float((best_population < 0.1).float().mean()),
        }
        if differentiable:
            optimizer.zero_grad(set_to_none=True)
            query_loss.sum().backward()
            gradients = latent.grad
            if gradients is None or not torch.isfinite(gradients).all():
                raise RuntimeError("Non-finite or missing latent gradient")
            norms = gradients.norm(dim=-1)
            record["gradient_norm_mean"] = float(norms.mean())
            record["nonzero_gradient_fraction"] = float((norms > 0).float().mean())
            # Clip each restart separately, preserving independent searches.
            gradients.mul_(
                (options.gradient_clip / norms.clamp_min(1e-12))
                .clamp_max(1)[:, None]
            )
            optimizer.step()
        history.append(record)
        del weights, amplitude, train_loss, query_loss, mask, scores

    if method != "prior_sampling" and options.final_restarts:
        final_mask = best_mask[:, None].expand(
            -1, options.final_restarts, -1, -1
        ).reshape(-1, 8, 2)
        # Same independent final initializations for both two-stage methods.
        final_rng = torch.Generator(device=device).manual_seed(seed + 90001)
        final_initial = options.initial_scale * torch.randn(
            trials * options.final_restarts, 8, 2,
            generator=final_rng, device=device,
        )
        with torch.no_grad():
            train_loss, query_loss, weights, amplitude = inner_sgd(
                final_mask, final_initial, batches, targets,
                query, query_targets, options,
            )
        population = retain(final_mask, weights, amplitude)
        history.append({
            "round": options.rounds,
            "network_fits": options.fits_per_search,
            "train_mse": float(train_loss.mean()),
            "query_mse": float(query_loss.mean()),
            "candidate_population_mse": float(population.mean()),
            "candidate_recall_at_2": float(
                (final_mask * oracle).sum((-1, -2)).mean() / 2
            ),
            "selected_population_mse": float(best_population.mean()),
            "selected_recall_at_2": float(
                (best_mask * oracle).sum((-1, -2)).mean() / 2
            ),
            "selected_success_rate": float(
                (best_population < 0.1).float().mean()
            ),
        })
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Frozen VAE unexpectedly acquired gradients")
    if not torch.isfinite(best_population).all():
        raise RuntimeError("Non-finite selected loss")
    result = {
        "method": method,
        "searches": trials,
        "success": int((best_population < 0.1).sum()),
        "mean_selected_population_mse": float(best_population.mean()),
        "mean_selected_recall_at_2": float(
            (best_mask * oracle).sum((-1, -2)).mean() / 2
        ),
        "seconds": time.perf_counter() - started,
        "fits_per_search": options.fits_per_search,
        "training_examples_per_search": (
            options.fits_per_search * options.inner_steps * options.batch
        ),
        "history": history,
        "vae_has_gradients": False,
    }
    arrays = {
        "selected_population": best_population.cpu().numpy(),
        "selected_empirical": best_empirical.cpu().numpy(),
        "selected_masks": best_mask.cpu().numpy(),
        "candidate_masks": np.concatenate(all_masks),
        "candidate_empirical": np.concatenate(all_empirical, axis=1),
        "candidate_population": np.concatenate(all_population, axis=1),
        "initial_latent": initial_latent.cpu().numpy(),
        "final_latent": latent.detach().cpu().numpy(),
    }
    return result, arrays


def prepare_run(args, config, options):
    """Freeze configuration, checkpoint hashes and relevant source files."""
    destination = args.run_dir.resolve()
    reference = args.reference_dir.resolve()
    if destination == reference or reference in destination.parents:
        raise ValueError("Use a separate directory outside the frozen reference run")
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        name: digest(DIRECTORY / name)
        for name in ("latent_search.py", "latent_report.py", "core.py", "models.py")
    }
    records = json.loads((reference / "models.json").read_text())
    checkpoints = {
        str(r["seed"]): r["sha256"] for r in records
        if r["variant"] == config["reference_variant"]
        and r["seed"] in config["model_seeds"]
    }
    plan = {
        "config": config,
        "reference": str(reference),
        "checkpoint_sha256": checkpoints,
        "implementation_sha256": source_hashes,
        "device": args.device,
        "torch_version": torch.__version__,
        "methods": list(METHODS),
        "fits_per_search": options.fits_per_search,
        "protocol": {
            "gradient": "Full unrolled SGD; hard top-2 forward, RelaxedTopK backward",
            "latent_optimizer": "Adam; VAE frozen; no target-support gradient",
            "two_stage": (
                f"{options.restarts * options.rounds} exploration fits, then "
                f"{options.final_restarts} fresh fits of selected mask"
            ),
            "paired_control": "random_z uses identical initial weights and batches",
            "sampling_control": "64 independent latent codes and weight initializations",
            "data": "Common IID training batches reused across rounds and methods",
            "selection": "Independent 4096-example sample; population MSE only reported",
            "cost": "Equal forward SGD samples; hypergradient backward cost is additional",
            "holdout": "Fresh functions 301/302; structural families previously studied",
            "recall": "True support used only for diagnostic Recall@2",
            "hyperparameters": "Fixed before evaluation; validation is diagnostic",
        },
    }
    path = destination / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError("Changed plan or implementation; choose a new run directory")
    save_json(path, plan)
    snapshot = destination / "source"
    snapshot.mkdir(exist_ok=True)
    for name in source_hashes:
        shutil.copy2(DIRECTORY / name, snapshot / name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("validate", "evaluate", "report", "run"))
    parser.add_argument("--config", type=Path, default=DIRECTORY / "latent_search.json")
    parser.add_argument("--reference-dir", type=Path, default=DIRECTORY / "runs/v1")
    parser.add_argument("--run-dir", type=Path, default=DIRECTORY / "runs/latent_search_v1")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    options = LatentSearchOptions(**config["search"])
    for name, value in asdict(options).items():
        if not isinstance(value, (int, float)) or value <= 0:
            parser.error(f"search.{name} must be positive")
    if options.fits_per_search % options.rounds:
        parser.error("The total number of fits must be divisible by rounds")
    if not config["model_seeds"] or min(
        config["validation_trials"], config["evaluation_trials"]
    ) <= 0:
        parser.error("Provide model seeds and positive trial counts")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable in this environment")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = prepare_run(args, config, options)
    if args.phase == "report":
        from .latent_report import publish

        publish(directory)
        return
    splits = ("validation", "test") if args.phase == "run" else (
        "validation" if args.phase == "validate" else "test",
    )
    if "test" in splits and args.phase != "run" and not (
        directory / "VALIDATION_COMPLETE.json"
    ).exists():
        parser.error("Run validation before evaluating the fixed protocol")
    for split in splits:
        functions = (
            [config["validation_function"]] if split == "validation"
            else config["evaluation_functions"]
        )
        trials = config[
            "validation_trials" if split == "validation" else "evaluation_trials"
        ]
        output = directory / split
        output.mkdir(exist_ok=True)
        for model_seed in config["model_seeds"]:
            model, record = load_decoder(
                args.reference_dir, config["reference_variant"], model_seed, args.device
            )
            for function in functions:
                for a, b in families(split):
                    spec = task(a, b, function)
                    seed = (
                        480100 + function * 10007 + a * 1009
                        + b * 17 + model_seed * 31
                    )
                    for method in METHODS:
                        name = f"{method}_a{a}_b{b}_f{function}_s{model_seed}"
                        path = output / f"{name}.json"
                        if path.exists():
                            continue
                        result, arrays = run_search(
                            model, spec, seed, trials, method, options, args.device
                        )
                        result.update(
                            a=a, b=b, function=function, model_seed=model_seed,
                            checkpoint_sha256=record["sha256"],
                        )
                        np.savez_compressed(output / f"{name}.npz", **arrays)
                        save_json(path, result)
                        print(
                            split, name, f"{result['success']}/{trials}",
                            f"MSE={result['mean_selected_population_mse']:.5f}",
                            f"seconds={result['seconds']:.1f}", flush=True,
                        )
            if digest(args.reference_dir / record["checkpoint"]) != record["sha256"]:
                raise RuntimeError("Reference checkpoint changed during search")
        records = [json.loads(p.read_text()) for p in output.glob("*.json")]
        expected = len(functions) * len(config["model_seeds"]) * 4 * len(METHODS)
        if len(records) != expected:
            raise RuntimeError(f"Incomplete {split}: {len(records)}/{expected}")
        save_json(directory / f"{split.upper()}_COMPLETE.json", {"contexts": len(records)})
    from .latent_report import publish

    publish(directory)


if __name__ == "__main__":
    main()

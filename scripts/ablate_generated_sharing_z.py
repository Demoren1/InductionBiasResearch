"""Direct global-z ablation for the 469-parameter generated-sharing model.

The original decoder and every training/evaluation hyperparameter are retained.
The controls freeze a bank of 16 codes, repeat one frozen code across all
restart slots, or remove the latent input from the coordinate generator.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch import nn

from pattern.bilevel_mask.generated_sharing import (
    ALL_PATTERNS, SharingConfig, SharingGenerator, _fit_shared,
    _fixed_assignment, _gather_restart, _network_losses, _new_z,
    _regularization, _seed, _temperature, adapt_weights, assignments,
    make_data, sharing_forward, train_generator,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _permutation_buffers(generator: nn.Module, config: SharingConfig,
                         device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    permutations = getattr(generator, "assignment_permutations", None)
    if permutations is None:
        permutations = torch.tensor(list(itertools.permutations(range(config.filter_dim))), device=device)
    one_hot = getattr(generator, "assignment_permutation_one_hot", None)
    if one_hot is None:
        one_hot = F.one_hot(permutations, config.filter_dim).to(torch.float32)
    return permutations, one_hot


class CoordinateGenerator(nn.Module):
    """The original generator with its initial global z folded into the bias."""

    def __init__(self, config: SharingConfig, initial_z: torch.Tensor,
                 initialization_seed: int | None = None) -> None:
        super().__init__()
        # SharingGenerator seeds its own weights from config.seed, independently
        # of torch's global RNG. Change that seed only for initialization.
        generator_config = (config if initialization_seed is None
                            else replace(config, seed=initialization_seed))
        original = SharingGenerator(generator_config).to(initial_z.device)
        first = original.network[0]
        folded = nn.Linear(2, config.generator_width).to(initial_z.device)
        with torch.no_grad():
            folded.weight.copy_(first.weight[:, config.latent_dim:])
            folded.bias.copy_(first.bias + first.weight[:, :config.latent_dim] @ initial_z.reshape(-1))
        self.network = nn.Sequential(folded, *list(original.network.children())[1:])
        self.register_buffer("coordinates", original.coordinates.detach().clone())
        permutations, one_hot = _permutation_buffers(original, config, initial_z.device)
        self.register_buffer("assignment_permutations", permutations.detach().clone(), persistent=False)
        self.register_buffer("assignment_permutation_one_hot", one_hot.detach().clone(), persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        prefix = z.shape[:-1]
        coordinates = self.coordinates.view(*(1 for _ in prefix), *self.coordinates.shape)
        return self.network(coordinates.expand(*prefix, -1, -1, -1))


def train_fixed_z(config: SharingConfig, device: torch.device,
                  policy: str, initialization_seed: int | None = None,
                  ) -> tuple[nn.Module, torch.Tensor, dict]:
    if policy not in ("frozen_bank", "single_fixed", "coordinate_only"):
        raise ValueError(policy)
    torch.manual_seed(config.seed)
    data = make_data(ALL_PATTERNS, config, device)
    initial_bank = _new_z(1, config.train_restarts, config, device,
                          "meta-train-shared").detach()
    z = initial_bank if policy == "frozen_bank" else initial_bank[:, :1].expand_as(initial_bank).clone()
    if initialization_seed is not None and policy != "coordinate_only":
        raise ValueError("initialization_seed is only supported without z")
    generator = (CoordinateGenerator(config, initial_bank[:, :1], initialization_seed)
                 if policy == "coordinate_only"
                 else SharingGenerator(config).to(device))
    optimizer = torch.optim.Adam(generator.parameters(), lr=config.generator_lr)
    best_score = math.inf
    best_step = 0
    best_generator = {name: value.detach().clone()
                      for name, value in generator.state_dict().items()}
    best_z = z[:, :1].detach().clone()
    history = []
    started = time.monotonic()
    for outer in range(1, config.outer_steps + 1):
        temperature = _temperature(config, outer)
        expanded_z = z.expand(len(ALL_PATTERNS), -1, -1)
        structure = assignments(generator, expanded_z, config, temperature, "ste")
        weights = adapt_weights(
            structure, data.support_x, data.support_y, config,
            steps=config.inner_steps, seed=_seed(config.seed, "meta-adapt"),
            create_graph=True,
        )
        validation = _network_losses(
            sharing_forward(data.validation_x, weights, structure), data.validation_y,
        )
        restart = validation.detach().mean(0).argmin()
        selected = restart.expand(len(ALL_PATTERNS))
        chosen_structure = _gather_restart(structure, selected)
        chosen_weights = tuple(_gather_restart(value, selected) for value in weights)
        query = _network_losses(
            sharing_forward(data.query_x, chosen_weights, chosen_structure),
            data.query_y,
        ).mean()
        soft = assignments(generator, expanded_z, config, temperature, "soft")
        penalty, _ = _regularization(soft, config)
        score = float(validation.mean(0).min().detach())
        if score < best_score:
            best_score = score
            best_step = outer
            best_generator = {name: value.detach().clone()
                              for name, value in generator.state_dict().items()}
            best_z = z[:, restart:restart + 1].detach().clone()

        optimizer.zero_grad(set_to_none=True)
        gradients = torch.autograd.grad(query + penalty, tuple(generator.parameters()))
        for parameter, gradient in zip(generator.parameters(), gradients):
            parameter.grad = gradient
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
        optimizer.step()
        history.append({"outer_step": outer, "query_bce": float(query.detach()),
                        "selected_restart": int(restart),
                        "mean_restart_validation_bce": score})
        if outer == 1 or outer % 10 == 0 or outer == config.outer_steps:
            print(f"TRAIN policy={policy} seed={config.seed} step={outer}/{config.outer_steps} "
                  f"query={history[-1]['query_bce']:.6f} best={best_score:.6f}", flush=True)
    generator.load_state_dict(best_generator)
    return generator, best_z, {
        "initialization_seed": config.seed if initialization_seed is None else initialization_seed,
        "selected_outer_step": best_step,
        "selected_training_validation_bce": best_score,
        "training_seconds": time.monotonic() - started,
        "history": history,
    }


def score_generated(generator: nn.Module, z: torch.Tensor,
                    config: SharingConfig, device: torch.device, split: str) -> dict:
    offset = 10_000 if split == "validation" else 20_000
    eval_config = SharingConfig(**{**config.to_dict(), "seed": config.seed + offset})
    data = make_data(ALL_PATTERNS, eval_config, device)
    structure = _fixed_assignment(generator, z, len(ALL_PATTERNS),
                                  config.eval_restarts, config)
    return _fit_shared(structure, data, eval_config, device, "generated sharing")


def fold_global_z(generator: nn.Module, z: torch.Tensor,
                  config: SharingConfig) -> dict:
    """Verify that a single global z is absorbed by the first-layer bias."""
    if z.shape != (1, 1, config.latent_dim):
        raise ValueError(f"expected one global z, got {tuple(z.shape)}")
    if generator.network[0].in_features == 2:
        return {"original_generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
                "folded_coordinate_generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
                "max_abs_logit_difference": 0.0, "hard_assignment_mismatched_entries": 0}
    first = generator.network[0]
    coordinates = generator.coordinates.reshape(-1, 2)
    folded_bias = first.bias + first.weight[:, :config.latent_dim] @ z.reshape(-1)
    hidden = torch.tanh(F.linear(coordinates, first.weight[:, config.latent_dim:], folded_bias))
    hidden = torch.tanh(generator.network[2](hidden))
    folded_logits = generator.network[4](hidden).reshape(
        1, 1, config.seq_len, config.hidden, 1 + config.filter_dim,
    )
    with torch.no_grad():
        original_logits = generator(z)
        permutations, one_hot = _permutation_buffers(generator, config, z.device)

        class FoldedOutput:
            assignment_permutations = permutations
            assignment_permutation_one_hot = one_hot

            def __call__(self, unused_z: torch.Tensor) -> torch.Tensor:
                return folded_logits

        original_u = assignments(generator, z, config, config.temperature_end, "hard")
        folded_u = assignments(FoldedOutput(), z, config, config.temperature_end, "hard")
    return {
        "original_generator_parameters": sum(parameter.numel() for parameter in generator.parameters()),
        "folded_coordinate_generator_parameters": sum(parameter.numel() for parameter in generator.parameters())
                                                - config.latent_dim * config.generator_width,
        "max_abs_logit_difference": float((original_logits - folded_logits).abs().max()),
        "hard_assignment_mismatched_entries": int((original_u != folded_u).sum()),
    }


def run(output: Path, reference: Path, policy: str, device: str,
        outer_steps: int | None = None,
        initialization_seed: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    reference_state = torch.load(reference, map_location="cpu", weights_only=True)
    config = SharingConfig(**reference_state["config"])
    if outer_steps is not None:
        config = replace(config, outer_steps=outer_steps)
    actual_device = torch.device(device)
    if policy == "historical":
        generator = SharingGenerator(config).to(actual_device)
        generator.load_state_dict(reference_state["generator"])
        z = reference_state["shared_z"].to(actual_device)
        training = {"selected_outer_step": reference_state["history"][-1]["selected_outer_step"],
                    "selected_training_validation_bce": reference_state["history"][-1]["selected_validation_bce"]}
    elif policy == "learned":
        torch.manual_seed(config.seed)
        started = time.monotonic()
        generator, z, history = train_generator(config, actual_device, ALL_PATTERNS)
        training = {"selected_outer_step": history[-1]["selected_outer_step"],
                    "selected_training_validation_bce": history[-1]["selected_validation_bce"],
                    "training_seconds": time.monotonic() - started, "history": history}
    else:
        generator, z, training = train_fixed_z(config, actual_device, policy, initialization_seed)
    fold = fold_global_z(generator, z, config)
    if fold["hard_assignment_mismatched_entries"]:
        raise ValueError("folding the global latent changed hard U")
    output.mkdir(parents=True)
    torch.save({"config": config.to_dict(), "generator": generator.state_dict(),
                "shared_z": z.detach().cpu(), "policy": policy,
                "reference": str(reference.resolve()), "training": training},
               output / "training.pt")
    validation = score_generated(generator, z, config, actual_device, "validation")
    test = score_generated(generator, z, config, actual_device, "test")
    result = {"policy": policy, "seed": config.seed, "config": config.to_dict(),
              "reference_checkpoint": str(reference.resolve()),
              "reference_checkpoint_sha256": _sha256(reference),
              "train_patterns": list(ALL_PATTERNS), "training": training,
              "fold_global_z": fold, "validation": validation, "test": test}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE policy={policy} seed={config.seed} test_bce={test['mean_query_bce']:.6f} "
          f"test_accuracy={test['mean_query_accuracy']:.6f}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--policy", choices=("historical", "learned", "frozen_bank", "single_fixed",
                                             "coordinate_only"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--outer-steps", type=int, default=None)
    parser.add_argument("--initialization-seed", type=int, default=None)
    args = parser.parse_args()
    run(args.output, args.reference, args.policy, args.device,
        args.outer_steps, args.initialization_seed)


if __name__ == "__main__":
    main()

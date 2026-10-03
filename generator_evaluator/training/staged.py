"""Two-stage generator objectives with one resumable shared latent.

The runner owns real measurements and evaluator fitting.  This module owns
only generator policy updates, direct Hungarian-aligned agreement, measured
target feedback, latent optimizer state, and device-resident bank caching.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from generator_evaluator.training.objectives import joint_mask_agreement, reconstruct_bank_masks
from generator_evaluator.training.updates import generator_update
from generator_evaluator.search.policy import align_elite_to_logits
from generator_evaluator.search.quality import validate_quality_objective


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None or not parameter.is_floating_point():
        raise ValueError("module must have floating-point parameters")
    return parameter.device, parameter.dtype


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class StagedGeneratorTrainer:
    """Train task-specific generators through quality, then joint cooperation.

    ``quality_update`` performs an own-task score-function update only.  The
    caller can run it during the quality stage without accidentally enabling
    agreement or measured-target feedback.  ``cooperation_update`` performs
    configured all-task quality updates, direct agreement and measured-target
    distillation before one joint optimizer step at the same latent.

    Banks are cached in each generator's device and dtype.  Replacing a bank
    with :meth:`set_bank` invalidates its prepared copy after real feedback.
    """

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        banks: Mapping[str, Any],
        optimizers: Mapping[str, torch.optim.Optimizer] | None,
        ensemble: nn.Module,
        contexts: Tensor,
        dense_quality: Tensor,
        rngs: Mapping[str, torch.Generator] | None = None,
        shared_noise_rng: torch.Generator | None = None,
        shared_latent_dim: int | None = None,
        generator_lr: float | None = None,
        latent_lr: float = 1e-3,
        *,
        quality_sample_count: int = 2,
        permutation_weight: float = 1.0,
        uncertainty_weight: float = 0.0,
        quality_objective: str = "worst",
        agreement_sample_count: int = 2,
        seed: int = 0,
        latent_device: str | torch.device | None = None,
        elite_limit: int = 8,
        reconstruction_weight: float = 0.0,
        reconstruction_batch_size: int = 8,
    ) -> None:
        if not isinstance(models, Mapping) or len(models) < 2:
            raise ValueError("models must be a mapping with at least two generators")
        if set(models) != set(banks):
            raise ValueError("models and banks must have identical task keys")
        names = tuple(models)
        if optimizers is not None and set(optimizers) != set(models):
            raise ValueError("models and optimizers must have identical task keys")
        if optimizers is None and (generator_lr is None or generator_lr <= 0):
            raise ValueError("generator_lr must be positive when optimizers are omitted")
        if generator_lr is not None and (not torch.isfinite(torch.tensor(generator_lr)) or generator_lr <= 0):
            raise ValueError("generator_lr must be a positive finite number")
        if not torch.isfinite(torch.tensor(latent_lr)) or latent_lr <= 0:
            raise ValueError("latent_lr must be a positive finite number")
        self.quality_sample_count = _positive_int(quality_sample_count, "quality_sample_count")
        if self.quality_sample_count < 2:
            raise ValueError("quality_sample_count must be at least two")
        self.agreement_sample_count = _positive_int(agreement_sample_count, "agreement_sample_count")
        if self.agreement_sample_count < 1:
            raise ValueError("agreement_sample_count must be positive")
        if permutation_weight < 0 or uncertainty_weight < 0:
            raise ValueError("quality objective weights must be nonnegative")
        self.quality_objective = validate_quality_objective(quality_objective)
        if (isinstance(reconstruction_weight, bool) or
                not isinstance(reconstruction_weight, (int, float)) or
                not torch.isfinite(torch.tensor(float(reconstruction_weight))) or
                reconstruction_weight < 0):
            raise ValueError("reconstruction_weight must be finite and nonnegative")
        self.permutation_weight = float(permutation_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.elite_limit = _positive_int(elite_limit, "elite_limit")
        self.reconstruction_weight = float(reconstruction_weight)
        self.reconstruction_batch_size = _positive_int(
            reconstruction_batch_size, "reconstruction_batch_size"
        )

        self.models = dict(models)
        self.banks = dict(banks)
        if optimizers is None:
            self.optimizers = {
                name: torch.optim.Adam(model.parameters(), lr=float(generator_lr))
                for name, model in self.models.items()
            }
        else:
            self.optimizers = dict(optimizers)
        self.ensemble = ensemble
        if contexts.ndim != 2 or not contexts.is_floating_point() or not bool(torch.isfinite(contexts).all()):
            raise ValueError("contexts must be a finite floating [tasks, context_dim] tensor")
        if (dense_quality.ndim != 1 or not dense_quality.is_floating_point() or
                not bool(torch.isfinite(dense_quality).all())):
            raise ValueError("dense_quality must be a finite floating [tasks] tensor")
        if len(contexts) != len(names) or len(dense_quality) != len(names):
            raise ValueError("contexts and dense_quality must have one row per generator")
        self.contexts = contexts.detach().clone()
        self.dense_quality = dense_quality.detach().clone()

        noise_dims = {int(getattr(model, "noise_dim", 0)) for model in self.models.values()}
        if len(noise_dims) != 1 or 0 in noise_dims:
            raise ValueError("all generators must have the same positive noise_dim")
        inferred_dim = next(iter(noise_dims))
        latent_dim = (inferred_dim if shared_latent_dim is None else
                      _positive_int(shared_latent_dim, "shared_latent_dim"))
        if latent_dim != inferred_dim:
            raise ValueError("shared_latent_dim must match every generator noise_dim")

        if shared_noise_rng is None:
            shared_noise_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 71_117)
        self.shared_noise_rng = shared_noise_rng
        if latent_device is None:
            latent_device = _module_device_dtype(ensemble)[0]
        self.latent_device = torch.device(latent_device)
        latent_rng_device = torch.device(self.shared_noise_rng.device)
        initial = torch.randn((latent_dim,), generator=self.shared_noise_rng, device=latent_rng_device,
                              dtype=torch.float32).to(device=self.latent_device) * 0.1
        self.shared_latent = nn.Parameter(initial)
        self.latent_optimizer = torch.optim.Adam([self.shared_latent], lr=float(latent_lr))

        if rngs is not None and set(rngs) - set(self.models):
            raise ValueError("rngs may only contain generator task keys")
        self.rngs: dict[str, torch.Generator] = {}
        for index, name in enumerate(names):
            supplied = None if rngs is None else rngs.get(name)
            self.rngs[name] = supplied or torch.Generator(device="cpu").manual_seed(
                int(seed) + 100_003 * (index + 1)
            )

        self._prepared_banks: dict[str, dict[tuple[Any, ...], SimpleNamespace]] = {}

    def _bank_signature(self, name: str, device: torch.device, dtype: torch.dtype) -> tuple[Any, ...]:
        bank = self.banks[name]
        tokens, quality, masks, baseline = (
            bank.tokens, bank.quality, bank.masks, bank.baseline_mask
        )
        return (id(bank), id(tokens), tokens.data_ptr(), tokens._version, id(quality),
                None if quality is None else (quality.data_ptr(), quality._version),
                id(masks), masks.data_ptr(), masks._version,
                id(baseline), baseline.data_ptr(), baseline._version,
                device.type, device.index, dtype)

    def _prepared_bank(self, name: str) -> SimpleNamespace:
        if name not in self.models:
            raise KeyError(f"unknown generator task {name!r}")
        device, dtype = _module_device_dtype(self.models[name])
        signature = self._bank_signature(name, device, dtype)
        cached_by_signature = self._prepared_banks.setdefault(name, {})
        if signature in cached_by_signature:
            return cached_by_signature[signature]
        bank = self.banks[name]
        prepared = SimpleNamespace(
            tokens=bank.tokens.to(device=device, dtype=dtype),
            quality=None if bank.quality is None else bank.quality.to(device=device, dtype=dtype),
            masks=bank.masks.to(device=device, dtype=dtype),
            baseline_mask=bank.baseline_mask.to(device=device, dtype=dtype),
        )
        cached_by_signature[signature] = prepared
        # Mixed and single-density views partition the same source bank. Keep
        # a bounded working set so toggling views does not retransmit it.
        while len(cached_by_signature) > 32:
            cached_by_signature.pop(next(iter(cached_by_signature)))
        return prepared

    def set_bank(self, name: str, bank: Any) -> None:
        """Replace a bank after feedback and discard its device copy."""
        if name not in self.models:
            raise KeyError(f"unknown generator task {name!r}")
        if self.banks[name] is bank:
            return
        self.banks[name] = bank

    def invalidate_bank_cache(self, name: str) -> None:
        """Drop all placed views after a source-bank feedback mutation."""
        if name not in self.models:
            raise KeyError(f"unknown generator task {name!r}")
        self._prepared_banks.pop(name, None)

    def _validate_budget(self, output_k: int, auxiliary_budgets: Sequence[int] | None) -> None:
        if isinstance(output_k, bool) or not isinstance(output_k, int):
            raise TypeError("output_k must be an integer")
        first = next(iter(self.models.values()))
        edge_count = int(first.features) * int(first.hidden)
        if not 1 <= output_k <= edge_count:
            raise ValueError("output_k must be within the generator mask size")
        if auxiliary_budgets is not None:
            if not isinstance(auxiliary_budgets, Sequence):
                raise TypeError("auxiliary_budgets must be a sequence of integers")
            for budget in auxiliary_budgets:
                if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= edge_count:
                    raise ValueError("each auxiliary budget must be within the generator mask size")

    def quality_update(
        self,
        name: str,
        output_k: int,
        auxiliary_budgets: Sequence[int] | None = None,
        ordinal: int = 0,
        *,
        train_shared_latent: bool = False,
        accumulate: bool = False,
        all_tasks: bool = False,
        loss_scale: float = 1.0,
    ) -> dict[str, float | str]:
        """Run one own-task PL/LOO quality update and permutation consistency."""
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        self._validate_budget(output_k, auxiliary_budgets)
        names = tuple(self.models)
        if name not in self.models:
            raise KeyError(f"unknown generator task {name!r}")
        index = names.index(name)
        model = self.models[name]
        if hasattr(model, "set_budget"):
            model.set_budget(output_k)  # type: ignore[attr-defined]
        bank = self._prepared_bank(name)
        if train_shared_latent and not accumulate:
            self.latent_optimizer.zero_grad(set_to_none=True)
        regularize_reconstruction = self.reconstruction_weight > 0
        if regularize_reconstruction and not accumulate:
            self.optimizers[name].zero_grad(set_to_none=True)
        logs = generator_update(
            model, self.ensemble, bank.tokens, bank.quality,
            self.contexts if all_tasks else self.contexts[index:index + 1],
            self.dense_quality if all_tasks else self.dense_quality[index:index + 1],
            self.optimizers[name], output_k, self.rngs[name],
            permutation_weight=self.permutation_weight,
            uncertainty_weight=self.uncertainty_weight,
            quality_objective=self.quality_objective,
            sample_count=self.quality_sample_count,
            shared_noise=(self.shared_latent if train_shared_latent else None),
            accumulate=(accumulate or regularize_reconstruction), loss_scale=loss_scale,
        )
        if regularize_reconstruction:
            # Own-task quality always predicts the requested output budget,
            # even from a single-density source view. Restore the same bank
            # target on every view instead of switching to teacher densities.
            # Cooperation keeps teacher reconstruction alongside the shared
            # measured-target distillation objective.
            bank_consensus = not all_tasks
            reconstruction = reconstruct_bank_masks(
                model, bank, self.optimizers[name], rng=self.rngs[name],
                batch_size=self.reconstruction_batch_size,
                weight=self.reconstruction_weight * loss_scale,
                accumulate=True, bank_consensus=bank_consensus,
            )
            logs.update(reconstruction)
            if not accumulate:
                parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
                if not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError("non-finite combined generator gradient")
                self.optimizers[name].step()
        else:
            logs.update(reconstruction_loss=0.0, reconstruction_overlap=0.0,
                        reconstruction_scope="disabled")
        if train_shared_latent and not accumulate:
            if self.shared_latent.grad is None:
                raise RuntimeError("shared latent received no quality-policy gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_([self.shared_latent], max_norm=10.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite shared-latent quality gradient")
            self.latent_optimizer.step()
            logs["shared_latent_grad_norm"] = float(grad_norm.detach().cpu())
        else:
            logs["shared_latent_grad_norm"] = 0.0
        logs["shared_latent_norm"] = float(self.shared_latent.detach().norm().cpu())
        logs["output_k"] = float(output_k)
        logs["ordinal"] = float(ordinal)
        return logs

    def _distill_targets(
        self, name: str, targets: Tensor | Mapping[str, Tensor], output_k: int, weight: float,
        *, accumulate: bool = False,
    ) -> dict[str, float]:
        if isinstance(targets, Mapping):
            source = targets.get(name)
            if source is None:
                return {"measured_distillation_loss": 0.0, "measured_target_count": 0.0}
        else:
            source = targets
        if not isinstance(source, Tensor) or source.ndim != 3:
            raise ValueError("targets must be [N, features, hidden] masks or a task mapping")
        model = self.models[name]
        device, dtype = _module_device_dtype(model)
        source = source.to(device=device, dtype=dtype)
        if source.shape[1:] != (model.features, model.hidden):
            raise ValueError("measured target dimensions must match each generator")
        if not bool(((source == 0) | (source == 1)).all()):
            raise ValueError("measured targets must be binary masks")
        source = source[source.sum((1, 2)) == output_k][:self.elite_limit]
        if len(source) == 0 or weight == 0:
            return {"measured_distillation_loss": 0.0, "measured_target_count": float(len(source))}

        bank = self._prepared_bank(name)
        draws = len(source)
        noise = self.shared_latent.detach().to(device=device, dtype=dtype).reshape(1, -1).expand(draws, -1)
        tokens = bank.tokens.expand(draws, *bank.tokens.shape[1:])
        quality = None if bank.quality is None else bank.quality.expand(draws, *bank.quality.shape[1:])
        logits = model(tokens, noise, quality)
        aligned = torch.stack([
            align_elite_to_logits(source[index], logits[index]) for index in range(draws)
        ])
        loss = F.binary_cross_entropy_with_logits(logits, aligned)
        optimizer = self.optimizers[name]
        if not accumulate:
            optimizer.zero_grad(set_to_none=True)
        (weight * loss).backward()
        if not accumulate:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite generator measured-target gradient")
            optimizer.step()
        return {"measured_distillation_loss": float(loss.detach().cpu()),
                "measured_target_count": float(len(source))}

    def cooperation_update(
        self,
        output_k: int,
        auxiliary_budgets: Sequence[int] | None = None,
        ordinal: int = 0,
        agreement_weight: float | None = None,
        targets: Tensor | Mapping[str, Tensor] | None = None,
        *,
        elite_weight: float = 0.0,
    ) -> dict[str, dict[str, float | str]]:
        """Combine configured quality, agreement and feedback in one Adam step.

        Measured targets are optional and are only consumed here.  The critic
        is frozen by each quality-policy step and is never part of agreement.
        """
        self._validate_budget(output_k, auxiliary_budgets)
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise ValueError("ordinal must be a nonnegative integer")
        selected_agreement_weight = 0.1 if agreement_weight is None else float(agreement_weight)
        if not torch.isfinite(torch.tensor(selected_agreement_weight)) or selected_agreement_weight < 0:
            raise ValueError("agreement_weight must be finite and nonnegative")
        if not torch.isfinite(torch.tensor(elite_weight)) or elite_weight < 0:
            raise ValueError("elite_weight must be finite and nonnegative")

        # Combine objectives before stepping Adam. Sequential own-task steps
        # moved the same latent repeatedly before agreement could respond.
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self.latent_optimizer.zero_grad(set_to_none=True)
        task_scale = 1.0 / len(self.models)
        rows: dict[str, dict[str, float]] = {
            name: self.quality_update(name, output_k, auxiliary_budgets, ordinal,
                                      train_shared_latent=True, accumulate=True,
                                      all_tasks=True, loss_scale=task_scale)
            for name in self.models
        }
        names = tuple(self.models)
        agreement = joint_mask_agreement(
            [self.models[name] for name in names],
            [self._prepared_bank(name) for name in names],
            [self.optimizers[name] for name in names],
            output_k,
            rng=self.shared_noise_rng,
            weight=selected_agreement_weight,
            sample_count=self.agreement_sample_count,
            accumulate=True,
            shared_noise=self.shared_latent,
            latent_optimizer=self.latent_optimizer,
        )
        for name in names:
            rows[name].update(agreement)
            if targets is None:
                rows[name].update(measured_distillation_loss=0.0, measured_target_count=0.0)
            else:
                rows[name].update(self._distill_targets(name, targets, output_k,
                                                      elite_weight * task_scale, accumulate=True))
        for name in names:
            parameters = [p for p in self.models[name].parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=10.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite joint generator gradient")
            rows[name]["joint_generator_grad_norm"] = float(grad_norm.detach().cpu())
            self.optimizers[name].step()
        if self.shared_latent.grad is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_([self.shared_latent], max_norm=10.0)
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("non-finite joint latent gradient")
            self.latent_optimizer.step()
            for name in names:
                rows[name]["shared_latent_grad_norm"] = float(grad_norm.detach().cpu())
        for name in names:
            rows[name]["shared_latent_norm"] = float(self.shared_latent.detach().norm().cpu())
        return rows

    @torch.no_grad()
    def proposal_noise(self, count: int, *, rng: torch.Generator | None = None,
                       perturbation: float = 0.1) -> Tensor:
        """Sample proposal latents around the learned shared z for acquisition."""
        _positive_int(count, "count")
        if not torch.isfinite(torch.tensor(perturbation)) or perturbation < 0:
            raise ValueError("perturbation must be finite and nonnegative")
        generator = self.shared_noise_rng if rng is None else rng
        random_device = torch.device(generator.device)
        noise = torch.randn((count, self.shared_latent.numel()), generator=generator,
                            device=random_device, dtype=torch.float32)
        center = self.shared_latent.detach().to(device=random_device, dtype=noise.dtype)
        proposals = center.unsqueeze(0) + float(perturbation) * noise
        # Retain one exact z-conditioned proposal, then explore a shared
        # neighbourhood before the policy's independent stochastic draws.
        proposals[0] = center
        return proposals

    def state_dict(self) -> dict[str, Any]:
        """Return deterministic-resume state for z, Adam, and every RNG."""
        return {
            "version": 1,
            "shared_latent": self.shared_latent.detach().cpu().clone(),
            "latent_optimizer": self.latent_optimizer.state_dict(),
            "shared_noise_rng_state": self.shared_noise_rng.get_state().cpu(),
            "rng_states": {name: rng.get_state().cpu() for name, rng in self.rngs.items()},
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise ValueError("unsupported staged trainer state")
        saved_latent = state.get("shared_latent")
        if not isinstance(saved_latent, Tensor) or saved_latent.shape != self.shared_latent.shape:
            raise ValueError("saved shared latent has incompatible dimensions")
        with torch.no_grad():
            self.shared_latent.copy_(saved_latent.to(self.shared_latent))
        self.latent_optimizer.load_state_dict(state["latent_optimizer"])
        for optimizer_state in self.latent_optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, Tensor):
                    optimizer_state[key] = value.to(self.shared_latent.device)
        self.shared_noise_rng.set_state(state["shared_noise_rng_state"].cpu())
        saved_rngs = state.get("rng_states", {})
        if set(saved_rngs) != set(self.rngs):
            raise ValueError("saved generator RNG keys must match the current trainer")
        for name, rng in self.rngs.items():
            rng.set_state(saved_rngs[name].cpu())

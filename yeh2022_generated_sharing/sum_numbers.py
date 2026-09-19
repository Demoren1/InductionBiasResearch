"""Sum-of-numbers parameter-sharing benchmark from Yeh et al. (2022).

This is the non-linear benchmark from Sec. 5.2.  Its first position-wise
fully connected layer has a categorical sharing matrix ``A``::

    W_effective[i] = sum_j A[i, j] W_base[j].

The four methods differ only in how that matrix is obtained: independent
rows (``no_sharing``), the known oracle, free relaxed logits (``direct``), or
a coordinate-conditioned generator ``A = G_psi(z)`` (``generated``).

The released implementation uses an implicit Neumann hypergradient rather
than backpropagating through all lower optimiser updates.  We reproduce that
choice here: model parameters are approximately optimized on the train split,
then a 20-term Neumann inverse-Hessian-vector approximation updates A on the
validation split.  In particular, this is *not* truncated unrolling.

One unavoidable protocol deviation is explicit in the result files: the
paper's released code keeps a separate H=250 set for learning-rate tuning and
early stopping.  ``tasks.make_sum_numbers`` deliberately exposes only the
published T/V/test split, so this module uses a fixed optimisation budget and
selects a soft-validation checkpoint; it never reads test data during model
or assignment selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
from torch import nn
import torch.nn.functional as F

from .core import (
    CoordinateAssignmentGenerator,
    DirectAssignment,
    release_assignment_regularizers,
    partition_distance,
)
from .tasks import Benchmark, Split, SumNumbersSpec, make_sum_numbers


Method = Literal["direct", "generated"]


@dataclass(frozen=True, slots=True)
class SumNumbersConfig:
    """Training configuration for the permutation-invariance benchmark.

    The architecture, T/V/test split (from :class:`SumNumbersSpec`), 250
    lower steps, 20 Neumann terms, and the 1e-3/1e-2 lower/upper learning
    rates match Appendix F.2 of Yeh et al.  The source release trains for at
    most 500 outer updates and uses a separate H split to stop/refit models.
    Here ``refit_steps`` is fixed because H is intentionally absent from the
    public task generator; use :meth:`quick` only for smoke tests.
    """

    seed: int = 0
    sequence_length: int = 10
    outer_steps: int = 500
    inner_steps: int = 250
    lower_lr: float = 1e-3
    outer_lr: float = 1e-2
    lower_weight_decay: float = 1e-3
    outer_weight_decay: float = 0.0
    neumann_iterations: int = 20
    neumann_alpha: float = 1e-2
    embedding_dim: int = 500
    hidden_dim: int = 50
    latent_dim: int = 8
    generator_width: int = 64
    temperature: float = 1.0
    assignment_regularizer_weight: float = 0.05
    entropy_regularizer_weight: float = 0.5
    nuclear_regularizer_weight: float = 1.0
    outer_checkpoint_every: int = 1
    outer_scheduler_warmup: int = 30
    refit_steps: int = 100_000
    test_batch_size: int = 8_192

    def __post_init__(self) -> None:
        positive_ints = (
            self.sequence_length,
            self.outer_steps,
            self.inner_steps,
            self.neumann_iterations,
            self.embedding_dim,
            self.hidden_dim,
            self.latent_dim,
            self.generator_width,
            self.outer_checkpoint_every,
            self.refit_steps,
            self.test_batch_size,
        )
        if any(value <= 0 for value in positive_ints):
            raise ValueError("all dimensions, steps, and batch sizes must be positive")
        if self.outer_scheduler_warmup < 0:
            raise ValueError("outer_scheduler_warmup must be non-negative")
        positive_scalars = (self.lower_lr, self.outer_lr, self.neumann_alpha, self.temperature)
        if any(value <= 0 for value in positive_scalars):
            raise ValueError("learning rates, Neumann alpha, and temperature must be positive")
        non_negative = (
            self.lower_weight_decay,
            self.outer_weight_decay,
            self.assignment_regularizer_weight,
            self.entropy_regularizer_weight,
            self.nuclear_regularizer_weight,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("weight decay and regularization weights must be non-negative")

    @classmethod
    def quick(cls, **overrides: Any) -> "SumNumbersConfig":
        """Small deterministic CPU configuration for a complete smoke run."""
        values: dict[str, Any] = {
            "sequence_length": 4,
            "outer_steps": 3,
            "inner_steps": 3,
            "neumann_iterations": 2,
            "embedding_dim": 24,
            "hidden_dim": 8,
            "latent_dim": 4,
            "generator_width": 16,
            "outer_checkpoint_every": 1,
            "outer_scheduler_warmup": 0,
            "refit_steps": 20,
            "test_batch_size": 128,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class LabelStandardization:
    """Mean/std fitted on T union V, as prescribed in Appendix F.2."""

    mean: float
    std: float


class PositionSharedSumModel(nn.Module):
    """Paper architecture with an externally supplied position assignment."""

    def __init__(self, config: SumNumbersConfig, *, seed: int) -> None:
        super().__init__()
        self.sequence_length = config.sequence_length
        self.embedding = nn.Embedding(11, config.embedding_dim, padding_idx=0)
        # These are the unshared basis parameters psi.  A mixes their first
        # (position) axis on every forward pass.
        self.position_weight = nn.Parameter(
            torch.empty(config.sequence_length, config.hidden_dim, config.embedding_dim)
        )
        self.position_bias = nn.Parameter(torch.empty(config.sequence_length, config.hidden_dim))
        self.output = nn.Linear(config.hidden_dim, 1)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, generator=generator)
            self.embedding.weight[0].zero_()
            # This follows the three-dimensional Xavier call in the released
            # StructFCModel rather than silently treating positions as batch.
            nn.init.xavier_uniform_(self.position_weight, generator=generator)
            nn.init.uniform_(self.position_bias, -0.1, 0.1, generator=generator)
            nn.init.xavier_uniform_(self.output.weight, generator=generator)
            nn.init.zeros_(self.output.bias)

    def forward(self, tokens: torch.Tensor, assignment: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[-1] != self.sequence_length:
            raise ValueError("tokens must have shape [batch, sequence_length]")
        if assignment.shape != (self.sequence_length, self.sequence_length):
            raise ValueError("assignment must have shape [sequence_length, sequence_length]")
        features = self.embedding(tokens.to(torch.long))
        effective_weight = torch.einsum("ij,jhe->ihe", assignment, self.position_weight)
        effective_bias = assignment @ self.position_bias
        hidden = torch.einsum("nke,khe->nkh", features, effective_weight) + effective_bias
        return self.output(F.relu(hidden).sum(dim=1)).squeeze(-1)


def _normalization(benchmark: Benchmark) -> LabelStandardization:
    labels = torch.cat((benchmark.splits.train.y, benchmark.splits.validation.y)).to(torch.float64)
    std = float(labels.std(unbiased=False))
    if not math.isfinite(std) or std <= 0:
        raise ValueError("T union V labels must have positive finite standard deviation")
    return LabelStandardization(mean=float(labels.mean()), std=std)


def _targets(split: Split, statistics: LabelStandardization) -> torch.Tensor:
    return (split.y.reshape(-1) - statistics.mean) / statistics.std


def _l1_loss(model: PositionSharedSumModel, split: Split, assignment: torch.Tensor, statistics: LabelStandardization) -> torch.Tensor:
    """Source-release lower/upper loss on standardized labels (not test MSE)."""
    prediction = model(split.x.to(torch.long), assignment)
    return F.l1_loss(prediction, _targets(split, statistics))


def _assignment_penalty(assignment: torch.Tensor, config: SumNumbersConfig) -> torch.Tensor:
    """The released `total_val_loss` regularizer, including its 0.05 scale."""
    entropy, nuclear = release_assignment_regularizers(assignment)
    return config.assignment_regularizer_weight * (
        config.nuclear_regularizer_weight * nuclear / assignment.shape[0]
        + config.entropy_regularizer_weight * entropy
    )


def _hard_assignment(soft: torch.Tensor) -> torch.Tensor:
    return F.one_hot(soft.argmax(dim=-1), num_classes=soft.shape[-1]).to(dtype=soft.dtype)


def _combine(first: Split, second: Split) -> Split:
    return Split(x=torch.cat((first.x, second.x)), y=torch.cat((first.y, second.y)))


def _clone_parameters(parameters: Iterable[nn.Parameter]) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in parameters]


def _restore_parameters(parameters: Iterable[nn.Parameter], state: Iterable[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(parameters, state, strict=True):
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))


def _finite_gradients(
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
    inputs: list[nn.Parameter],
    *,
    grad_outputs: tuple[torch.Tensor, ...] | None = None,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> tuple[torch.Tensor, ...]:
    values = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=grad_outputs,
        allow_unused=True,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return tuple(torch.zeros_like(parameter) if value is None else value for parameter, value in zip(inputs, values, strict=True))


def _neumann_inverse_hvp(
    train_gradient: tuple[torch.Tensor, ...],
    lower_parameters: list[nn.Parameter],
    validation_gradient: tuple[torch.Tensor, ...],
    *,
    alpha: float,
    iterations: int,
) -> tuple[torch.Tensor, ...]:
    """20-term Neumann inverse-Hessian-vector approximation from the paper.

    The Hessian-vector products intentionally do not create a third-order
    graph; only the final mixed VJP needs the graph of ``train_gradient``.
    """
    vector = tuple(value.detach().clone() for value in validation_gradient)
    inverse_product = tuple(value.detach().clone() for value in validation_gradient)
    for _ in range(iterations):
        hessian_vector = _finite_gradients(
            train_gradient,
            lower_parameters,
            grad_outputs=vector,
            retain_graph=True,
        )
        vector = tuple(current - alpha * hessian for current, hessian in zip(vector, hessian_vector, strict=True))
        inverse_product = tuple(total + current for total, current in zip(inverse_product, vector, strict=True))
    # (H^{-1}v) ≈ alpha * sum_j (I - alpha H)^j v.
    return tuple(alpha * value for value in inverse_product)


def _assignment_factory(
    method: Method,
    config: SumNumbersConfig,
    device: torch.device,
) -> tuple[nn.Module, nn.Parameter | None, list[nn.Parameter]]:
    if method == "direct":
        module: nn.Module = DirectAssignment(
            config.sequence_length,
            seed=config.seed + 10_001,
        ).to(device)
        return module, None, list(module.parameters())
    if method == "generated":
        module = CoordinateAssignmentGenerator(
            config.sequence_length,
            latent_dim=config.latent_dim,
            width=config.generator_width,
            seed=config.seed + 10_002,
        ).to(device)
        latent_generator = torch.Generator(device="cpu").manual_seed(config.seed + 10_003)
        latent = nn.Parameter(
            (0.1 * torch.randn(config.latent_dim, generator=latent_generator)).to(device)
        )
        return module, latent, list(module.parameters()) + [latent]
    raise ValueError(f"unknown assignment method: {method}")


def _current_assignment(
    method: Method,
    module: nn.Module,
    latent: nn.Parameter | None,
    config: SumNumbersConfig,
    *,
    hard: bool = False,
) -> torch.Tensor:
    mode: Literal["soft", "hard"] = "hard" if hard else "soft"
    if method == "direct":
        return module.assignment(temperature=config.temperature, mode=mode).squeeze(0)  # type: ignore[attr-defined]
    assert latent is not None
    return module.assignment(latent, temperature=config.temperature, mode=mode)  # type: ignore[attr-defined]


def _fit_fixed_assignment(
    benchmark: Benchmark,
    assignment: torch.Tensor,
    statistics: LabelStandardization,
    config: SumNumbersConfig,
    *,
    seed_offset: int,
) -> PositionSharedSumModel:
    """Freshly fit psi on T union V after hardening A; test remains untouched."""
    device = benchmark.splits.train.x.device
    model = PositionSharedSumModel(config, seed=config.seed + seed_offset).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lower_lr)
    all_data = _combine(benchmark.splits.train, benchmark.splits.validation)
    fixed_assignment = assignment.detach()
    for _ in range(config.refit_steps):
        loss = _l1_loss(model, all_data, fixed_assignment, statistics)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model


def _test_metrics(
    model: PositionSharedSumModel,
    benchmark: Benchmark,
    assignment: torch.Tensor,
    statistics: LabelStandardization,
    config: SumNumbersConfig,
) -> dict[str, float | int]:
    """Evaluate original-scale losses in bounded GPU batches and compute PD."""
    test = benchmark.splits.test
    total_squared = torch.zeros((), device=test.x.device, dtype=torch.float64)
    total_absolute = torch.zeros((), device=test.x.device, dtype=torch.float64)
    with torch.no_grad():
        for begin in range(0, test.size, config.test_batch_size):
            end = min(begin + config.test_batch_size, test.size)
            normalized = model(test.x[begin:end].to(torch.long), assignment)
            prediction = normalized * statistics.std + statistics.mean
            residual = prediction - test.y[begin:end].reshape(-1)
            total_squared += residual.to(torch.float64).square().sum()
            total_absolute += residual.to(torch.float64).abs().sum()
    distance = partition_distance(assignment.detach(), benchmark.oracle_categories.detach())
    labels = assignment.argmax(dim=-1)
    return {
        "test_mse": float((total_squared / test.size).cpu()),
        "test_mae": float((total_absolute / test.size).cpu()),
        "partition_distance": int(distance),
        "normalized_partition_distance": float(distance / assignment.shape[0]),
        "assignment_groups": int(torch.unique(labels).numel()),
    }


def _serialize_model(model: PositionSharedSumModel) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def _fit_learned_assignment(
    method: Method,
    benchmark: Benchmark,
    statistics: LabelStandardization,
    config: SumNumbersConfig,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    """Run the paper-style lower solve plus implicit Neumann outer updates."""
    device = benchmark.splits.train.x.device
    learner = PositionSharedSumModel(config, seed=config.seed + 20_001).to(device)
    module, latent, hyper_parameters = _assignment_factory(method, config, device)
    lower_optimizer = torch.optim.AdamW(
        learner.parameters(), lr=config.lower_lr, weight_decay=config.lower_weight_decay
    )
    outer_optimizer = torch.optim.Adam(
        hyper_parameters, lr=config.outer_lr, weight_decay=config.outer_weight_decay
    )
    outer_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        outer_optimizer, mode="min", factor=0.5, patience=50, cooldown=0
    )

    best_objective = float("inf")
    best_step = 0
    best_hyper: list[torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for step in range(config.outer_steps):
        # The lower optimiser follows the current A but deliberately does not
        # retain a graph through its 250 updates; IFT handles its dependence.
        fixed_assignment = _current_assignment(method, module, latent, config).detach()
        for _ in range(config.inner_steps):
            train_loss = _l1_loss(learner, benchmark.splits.train, fixed_assignment, statistics)
            lower_optimizer.zero_grad(set_to_none=True)
            train_loss.backward()
            lower_optimizer.step()

        soft_assignment = _current_assignment(method, module, latent, config)
        train_loss = _l1_loss(learner, benchmark.splits.train, soft_assignment, statistics)
        validation_loss = _l1_loss(learner, benchmark.splits.validation, soft_assignment, statistics)
        validation_objective = validation_loss + _assignment_penalty(soft_assignment, config)
        lower_parameters = list(learner.parameters())
        train_gradient = _finite_gradients(
            train_loss, lower_parameters, create_graph=True, retain_graph=True
        )
        validation_gradient = _finite_gradients(validation_loss, lower_parameters, retain_graph=True)
        inverse_product = _neumann_inverse_hvp(
            train_gradient,
            lower_parameters,
            validation_gradient,
            alpha=config.neumann_alpha,
            iterations=config.neumann_iterations,
        )
        direct = _finite_gradients(validation_objective, hyper_parameters, retain_graph=True)
        mixed = _finite_gradients(
            train_gradient,
            hyper_parameters,
            grad_outputs=inverse_product,
        )
        outer_optimizer.zero_grad(set_to_none=True)
        for parameter, direct_gradient, mixed_gradient in zip(hyper_parameters, direct, mixed, strict=True):
            parameter.grad = (direct_gradient - mixed_gradient).detach()
        torch.nn.utils.clip_grad_norm_(hyper_parameters, max_norm=25.0)
        outer_optimizer.step()
        # The source starts ReduceLROnPlateau after a 30-step warm-up.
        if step >= config.outer_scheduler_warmup:
            outer_scheduler.step(float(validation_objective.detach()))

        if step % config.outer_checkpoint_every == 0 or step + 1 == config.outer_steps:
            with torch.no_grad():
                # Selection is explicitly on the relaxed validation objective,
                # just like the source's `best_model` checkpoint.  The selected
                # A is hardened only after the outer optimisation is complete.
                current = float(validation_objective.detach())
                hard = _hard_assignment(soft_assignment)
                hard_pd = partition_distance(hard, benchmark.oracle_categories)
                entropy, nuclear = release_assignment_regularizers(soft_assignment)
            history.append(
                {
                    "step": step,
                    "soft_validation_l1": float(validation_loss.detach()),
                    "soft_validation_objective": current,
                    "entropy": float(entropy.detach()),
                    "nuclear": float(nuclear.detach()),
                    "hard_partition_distance": int(hard_pd),
                }
            )
            if current < best_objective:
                best_objective = current
                best_step = step
                best_hyper = _clone_parameters(hyper_parameters)

    if best_hyper is None:  # pragma: no cover - config validates nonzero steps
        raise RuntimeError("no outer checkpoint was recorded")
    _restore_parameters(hyper_parameters, best_hyper)
    with torch.no_grad():
        final_soft = _current_assignment(method, module, latent, config).detach()
        final_hard = _hard_assignment(final_soft)
    artifact: dict[str, Any] = {
        "assignment_module": {key: value.detach().cpu() for key, value in module.state_dict().items()},
        "latent": None if latent is None else latent.detach().cpu(),
        "soft_assignment": final_soft.cpu(),
        "hard_assignment": final_hard.cpu(),
        "lower_model_before_refit": _serialize_model(learner),
    }
    selection = {
        "selection_step": best_step,
        "selection_soft_validation_l1": best_objective,
        "outer_history": history,
    }
    return final_hard, selection, artifact


def _fixed_controls(benchmark: Benchmark, config: SumNumbersConfig) -> dict[str, torch.Tensor]:
    device = benchmark.splits.train.x.device
    return {
        "no_sharing": torch.eye(config.sequence_length, device=device),
        "oracle": F.one_hot(
            benchmark.oracle_categories.to(torch.long), num_classes=config.sequence_length
        ).to(dtype=torch.float32, device=device),
    }


def run_sum_numbers_benchmark(
    config: SumNumbersConfig,
    device: torch.device | str,
    *,
    alternating: bool,
    train_size: int = 100,
    validation_size: int = 150,
    test_size: int = 100_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one standard/alternating target with four comparable methods.

    `no_sharing` and `oracle` each receive a fresh model and the same T union
    V refit budget as the hard assignments selected by `direct`/`generated`.
    Thus the test MSE comparison isolates the discovered sharing topology.
    """
    device = torch.device(device)
    benchmark = make_sum_numbers(
        SumNumbersSpec(
            sequence_length=config.sequence_length,
            train_size=train_size,
            validation_size=validation_size,
            test_size=test_size,
            alternating=alternating,
        ),
        seed=config.seed,
        device=device,
    )
    statistics = _normalization(benchmark)
    methods: dict[str, dict[str, Any]] = {}
    artifact_methods: dict[str, Any] = {}
    for index, (name, assignment) in enumerate(_fixed_controls(benchmark, config).items()):
        model = _fit_fixed_assignment(
            benchmark, assignment, statistics, config, seed_offset=30_000 + index
        )
        methods[name] = _test_metrics(model, benchmark, assignment, statistics, config)
        artifact_methods[name] = {
            "hard_assignment": assignment.detach().cpu(),
            "refit_model": _serialize_model(model),
        }

    for index, name in enumerate(("direct", "generated")):
        assignment, selection, artifact = _fit_learned_assignment(name, benchmark, statistics, config)
        model = _fit_fixed_assignment(
            benchmark, assignment, statistics, config, seed_offset=40_000 + index
        )
        methods[name] = _test_metrics(model, benchmark, assignment, statistics, config) | selection
        artifact_methods[name] = artifact | {"refit_model": _serialize_model(model)}

    protocol = {
        "source": "Yeh et al. (2022), Appendix F.2; released PermutationSharing experiment",
        "architecture": "token embedding -> position-wise FC + ReLU -> sum positions -> scalar",
        "sharing": "A mixes only the position axis of the first fully connected layer",
        "label_standardization": "mean/std fitted on train+validation; test predictions transformed back before MSE",
        "bilevel_update": "implicit Neumann inverse-Hessian-vector hypergradient; no unrolling through lower optimizer",
        "selection": "lowest relaxed validation objective; row-argmax hard A; fresh refit on train+validation; test only after refit",
        "deviation_from_released_code": (
            "tasks.make_sum_numbers has T/V/test only, whereas the release has a separate H=250 set "
            "for early stopping and learning-rate tuning; this implementation uses fixed budgets and no H/tuning data; "
            "the mathematically required leading alpha is included in the Neumann inverse-Hessian approximation, "
            "while the released helper omits it"
        ),
        "lower_upper_loss": "standardized L1 plus released assignment regularizer; original-scale test MSE/MAE reported",
    }
    summary: dict[str, Any] = {
        "benchmark": benchmark.name,
        "protocol": protocol,
        "config": asdict(config),
        "split_sizes": {"train": train_size, "validation": validation_size, "test": test_size},
        "label_standardization": asdict(statistics),
        "methods": methods,
    }
    artifacts: dict[str, Any] = {
        "config": asdict(config),
        "oracle_categories": benchmark.oracle_categories.detach().cpu(),
        "label_standardization": asdict(statistics),
        "methods": artifact_methods,
    }
    return summary, artifacts


def markdown_report(summary: dict[str, Any]) -> str:
    """Render one compact Markdown table per target result."""
    lines = [
        f"# Yeh et al. (2022): {summary['benchmark']}",
        "",
        "| method | test MSE | test MAE | PD | normalized PD | groups |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("no_sharing", "oracle", "direct", "generated"):
        row = summary["methods"][name]
        lines.append(
            f"| {name} | {row['test_mse']:.6g} | {row['test_mae']:.6g} | "
            f"{row['partition_distance']} | {row['normalized_partition_distance']:.4f} | {row['assignment_groups']} |"
        )
    lines.extend(
        [
            "",
            f"Label standardization on T∪V: mean={summary['label_standardization']['mean']:.6g}, "
            f"std={summary['label_standardization']['std']:.6g}.",
            "",
            "`direct` is the Yeh-style free relaxed assignment; `generated` is A=Gψ(z). "
            "Both are selected by relaxed validation objective, row-hardened, and freshly refit on T∪V.",
        ]
    )
    return "\n".join(lines) + "\n"


def save_sum_numbers_outputs(
    output: Path,
    summaries: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> None:
    """Persist the stable JSON/PT/Markdown artifact trio used by the CLI."""
    output.mkdir(parents=True, exist_ok=True)
    combined = {
        "protocol": {
            "paper": "Yeh et al. (2022), Equivariance Discovery by Learned Parameter-Sharing",
            "targets": ["standard", "alternating"],
        },
        "targets": summaries,
    }
    (output / "sum_numbers_results.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    torch.save({"targets": artifacts}, output / "sum_numbers_artifacts.pt")
    report = "\n".join(markdown_report(summaries[target]) for target in summaries)
    (output / "SUM_NUMBERS_RESULTS.md").write_text(report, encoding="utf-8")


__all__ = [
    "LabelStandardization",
    "PositionSharedSumModel",
    "SumNumbersConfig",
    "markdown_report",
    "run_sum_numbers_benchmark",
    "save_sum_numbers_outputs",
]

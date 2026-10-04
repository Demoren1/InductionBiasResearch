"""Run settings for cooperative exact-K generator search."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

from generator_evaluator.data.pattern import _validate_roles
from generator_evaluator.search.quality import validate_quality_objective


@dataclass(frozen=True)
class CooperativeConfig:
    domain: str = "pattern"
    features: int = 11
    hidden: int = 8
    phase: str = "search"
    training_mode: str = "joint"
    seed: int = 4100
    train_patterns: tuple[str, ...] = ("0001", "0011")
    test_pattern: str = "0101"
    test_patterns: tuple[str, ...] = ()
    test_task_count: int = 2
    k: int = 32
    generator_epochs: int = 10
    updates_per_epoch: int = 10
    cooperation_rounds: int = 5
    cooperation_updates: int = 20
    latent_lr: float = .001
    refresh_every: int = 5
    minimum_refresh_every: int = 1
    acquisition_budget: int = 6
    auxiliary_budget: int = 0
    candidates: int = 24
    initial_random: int = 8
    bootstrap_generators: bool = False
    evaluator_epochs: int = 50
    evaluator_batch_size: int = 32
    generator_lr: float = .001
    evaluator_lr: float = .001
    width: int = 64
    heads: int = 4
    layers: int = 2
    noise_dim: int = 16
    ensemble_members: int = 3
    feedback_masks: int = 3
    bank_capacity: int = 100
    elite_limit: int = 8
    elite_margin: float = 0.
    agreement_weight: float = .1
    agreement_ramp_epochs: int = 5
    quality_objective: str = "average"
    elite_distillation_weight: float = .1
    reconstruction_weight: float = .1
    reconstruction_batch_size: int = 8
    generator_pretrain_epochs: int = 0
    pretrain_updates_per_epoch: int = 20
    permutation_weight: float = 1.
    gap_threshold: float = .1
    smoke: bool = False
    preset: str = "full"
    batch_children: bool = False
    tune_dense: bool = True
    initial_global_density: bool = False
    output_budgets: tuple[int, ...] = ()

    def __post_init__(self):
        if self.domain == "pattern":
            tests = tuple(self.test_patterns) if self.test_patterns else (self.test_pattern,)
            if not tests:
                raise ValueError("pattern requires at least one held-out test pattern")
            if self.test_patterns:
                object.__setattr__(self, "test_patterns", tests)
                object.__setattr__(self, "test_pattern", tests[0])
            _validate_roles(tuple(self.train_patterns), tests[0] if len(tests) == 1 else tests)
            if (self.features, self.hidden) != (11, 8):
                raise ValueError("pattern requires 11 input features and 8 hidden neurons")
        elif self.domain == "deepsets":
            expected = tuple(str(index) for index in range(len(self.train_patterns)))
            if (self.features != 784 or self.hidden < 1 or len(expected) < 2 or
                    tuple(self.train_patterns) != expected or self.test_task_count < 1):
                raise ValueError("DeepSets requires ordered task generators 0..N-1, N >= 2, and heldout tasks")
        else:
            raise ValueError("unknown cooperative domain")
        if self.phase not in ("bootstrap", "search"):
            raise ValueError("phase must be bootstrap or search")
        if self.training_mode not in ("joint", "staged"):
            raise ValueError("training_mode must be joint or staged")
        validate_quality_objective(self.quality_objective)
        positive = (self.k, self.generator_epochs, self.updates_per_epoch,
                    self.cooperation_rounds, self.cooperation_updates,
                    self.refresh_every, self.acquisition_budget,
                    self.candidates, self.initial_random, self.evaluator_epochs,
                    self.evaluator_batch_size, self.feedback_masks, self.bank_capacity,
                    self.elite_limit, self.ensemble_members)
        edges = self.features * self.hidden
        if min(positive) < 1 or self.bank_capacity < 7 or not 0 < self.k < edges:
            raise ValueError("invalid cooperative budgets")
        if not 1 <= self.minimum_refresh_every <= self.refresh_every:
            raise ValueError("minimum refresh interval must be between 1 and refresh_every")
        weights = (self.agreement_weight, self.elite_distillation_weight,
                   self.reconstruction_weight)
        if (not math.isfinite(self.latent_lr) or self.latent_lr <= 0 or
                not all(math.isfinite(value) and value >= 0 for value in weights) or
                not math.isfinite(self.elite_margin) or self.elite_margin < 0 or
                not math.isfinite(self.gap_threshold) or self.gap_threshold <= 0):
            raise ValueError("invalid agreement or calibration settings")
        if (self.generator_pretrain_epochs < 0 or
                min(self.pretrain_updates_per_epoch, self.reconstruction_batch_size,
                    self.agreement_ramp_epochs) < 1):
            raise ValueError("invalid generator reconstruction or agreement schedule")
        if self.auxiliary_budget < 0 or any(not 0 < k < edges for k in self.output_budgets):
            raise ValueError("invalid auxiliary budgets")
        if self.auxiliary_budget and not any(k != self.k for k in self.output_budgets):
            raise ValueError("auxiliary acquisition requires non-target output budgets")

    @property
    def effective_test_patterns(self) -> tuple[str, ...]:
        """Ordered pattern holdouts, with the scalar field for legacy configs."""
        if self.domain != "pattern":
            return ()
        return tuple(self.test_patterns) if self.test_patterns else (self.test_pattern,)

def pattern_small_config(**overrides):
    settings = dict(preset="pattern-small", batch_children=True, tune_dense=False,
        initial_global_density=True, output_budgets=(), width=16, heads=2, layers=1,
        noise_dim=4, ensemble_members=2, generator_epochs=2, updates_per_epoch=4,
        refresh_every=1, evaluator_epochs=10, candidates=8, acquisition_budget=2,
        auxiliary_budget=0, initial_random=2, feedback_masks=1, bank_capacity=100, elite_limit=4,
        generator_pretrain_epochs=0, pretrain_updates_per_epoch=10)
    settings.update(overrides)
    return CooperativeConfig(**settings)


def _config_metadata(config):
    """Serialize the effective pattern roles without inventing DeepSets roles."""
    metadata = asdict(config)
    if config.domain == "pattern":
        metadata["test_pattern"] = config.effective_test_patterns[0]
        metadata["test_patterns"] = list(config.effective_test_patterns)
    else:
        metadata["test_patterns"] = []
    return metadata


def deepsets_config(**overrides):
    settings = dict(domain="deepsets", features=784, hidden=32, preset="deepsets",
        train_patterns=("0", "1"), test_pattern="heldout", k=7526,
        batch_children=True, width=64, heads=4, layers=2, noise_dim=8,
        ensemble_members=2, generator_epochs=20, updates_per_epoch=10,
        refresh_every=2, evaluator_epochs=100, evaluator_batch_size=64,
        evaluator_lr=.0003, candidates=24, acquisition_budget=6,
        auxiliary_budget=0, output_budgets=(),
        initial_random=8, feedback_masks=2, bank_capacity=1024, elite_limit=8)
    settings.update(overrides)
    return CooperativeConfig(**settings)

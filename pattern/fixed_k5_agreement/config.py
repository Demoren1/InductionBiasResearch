from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Config:
    """Immutable protocol for the overnight seq-32, pattern-length-5 run."""

    seq_len: int = 32
    hidden: int = 32
    pattern_len: int = 5
    experiment_seed: int = 20260912
    task_split_seed: int = 42
    input_split_seed: int = 1729

    # A larger bank than the historical pattern-8 experiment.  The best 10%
    # from every meta-train task become VAE examples.
    bank_mlps: int = 4000
    bank_steps: int = 2000
    bank_batch: int = 128
    bank_lr: float = 1e-3
    bank_val_size: int = 8192
    bank_support_size: int = 65_536
    top_fraction: float = 0.1

    vae_pairs: int = 64
    vae_seed_start: int = 1000
    vae_epochs: int = 160
    vae_batch: int = 256
    vae_lr: float = 1e-3
    vae_beta: float = 0.1
    vae_hidden: int = 256
    latent_dim: int = 32
    vae_val_fraction: float = 0.15
    map_split_seed: int = 42

    n_starts: int = 64
    agreement_steps: int = 2000
    agreement_lr: float = 0.03
    temperature: float = 0.5
    latent_radius: float = 12.0
    random_proposals: int = 2001

    # Established direct-gradient single-z baseline.  This intentionally has
    # target-label access and is not compute-matched to decoder agreement.
    task_outer_steps: int = 30
    task_warmup_steps: int = 300
    task_grad_steps: int = 100
    task_z_lr: float = 0.05
    task_search_val_size: int = 2048

    task_support_size: int = 65_536
    eval_support_size: int = 65_536
    eval_test_size: int = 8192
    eval_steps: int = 2000
    eval_batch: int = 128
    eval_lr: float = 1e-3
    eval_repeats: int = 2

    # Smoke-only caps; zero means all 24/8 train/test tasks.
    train_task_limit: int = 0
    test_task_limit: int = 0

    def __post_init__(self) -> None:
        if (self.seq_len, self.hidden, self.pattern_len) != (32, 32, 5):
            raise ValueError("This protocol is fixed to seq_len=32, hidden=32, pattern_len=5")
        positive = (
            "bank_mlps", "bank_steps", "bank_batch", "bank_val_size", "bank_support_size",
            "vae_pairs", "vae_epochs", "vae_batch", "vae_hidden", "latent_dim", "n_starts",
            "agreement_steps", "random_proposals", "task_outer_steps", "task_search_val_size",
            "task_support_size", "eval_support_size", "eval_test_size", "eval_steps",
            "eval_batch", "eval_repeats",
        )
        if any(getattr(self, name) < 1 for name in positive):
            raise ValueError("all count and step parameters must be positive")
        nonnegative = ("task_warmup_steps", "task_grad_steps", "train_task_limit", "test_task_limit")
        if any(getattr(self, name) < 0 for name in nonnegative):
            raise ValueError("caps and optional step counts must be nonnegative")
        rates = (self.bank_lr, self.vae_lr, self.agreement_lr, self.task_z_lr, self.eval_lr)
        if min(rates) <= 0 or self.temperature <= 0 or self.latent_radius <= 0:
            raise ValueError("learning rates, temperature and radius must be positive")
        if not 0 < self.top_fraction <= 1 or not 0 < self.vae_val_fraction < 1:
            raise ValueError("fractions must lie strictly inside their valid range")
        if self.bank_mlps % 10:
            raise ValueError("bank_mlps must be divisible by 10 for an exact top-10% selection")
        if self.random_proposals != self.agreement_steps + 1:
            raise ValueError("random search must inspect the same number of states as Adam")

    @property
    def mask_dim(self) -> int:
        return self.seq_len * self.hidden

    @property
    def k_active(self) -> int:
        return self.pattern_len * self.hidden

    # Compatibility aliases used by the audited length_interp bank builder.
    @property
    def seed(self) -> int:
        return self.experiment_seed

    @property
    def support_pool_size(self) -> int:
        return self.bank_support_size

    @property
    def vae_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.vae_seed_start, self.vae_seed_start + 2 * self.vae_pairs))

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        seeds = self.vae_seeds
        return tuple((seeds[index], seeds[index + 1]) for index in range(0, len(seeds), 2))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def smoke(cls) -> "Config":
        return replace(
            cls(), bank_mlps=20, bank_steps=2, bank_val_size=64, bank_support_size=128,
            vae_pairs=1, vae_epochs=2, vae_batch=8, vae_hidden=32, latent_dim=4,
            n_starts=2, agreement_steps=2, random_proposals=3,
            task_outer_steps=1, task_warmup_steps=1, task_grad_steps=1,
            task_search_val_size=32, task_support_size=64, eval_support_size=64,
            eval_test_size=64, eval_steps=2, eval_batch=16, eval_repeats=1,
            train_task_limit=3, test_task_limit=2,
        )

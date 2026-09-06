from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    seq_len: int = 32
    hidden: int = 32
    train_lengths: tuple = (3, 4, 6, 8)
    heldout_lengths: tuple = (5, 7)
    seed: int = 20260906
    task_split_seed: int = 42
    input_split_seed: int = 1729
    bank_mlps: int = 2000
    bank_steps: int = 2000
    bank_batch: int = 128
    bank_lr: float = 0.001
    bank_val_size: int = 2048
    support_pool_size: int = 8192
    top_fraction: float = 0.1
    cvae_epochs: int = 80
    cvae_batch: int = 256
    cvae_lr: float = 0.001
    cvae_beta: float = 0.1
    cvae_hidden: int = 256
    latent_dim: int = 32
    cvae_seeds: tuple = (42, 43)
    eval_masks: int = 32
    eval_repeats: int = 2
    eval_steps: int = 2000
    eval_batch: int = 128
    eval_lr: float = 0.001
    eval_test_size: int = 2048
    # A nonzero value is a smoke-only cap, sampled reproducibly within length.
    task_limit: int = 0
    eval_task_limit: int = 0

    def __post_init__(self):
        for field in ("train_lengths", "heldout_lengths", "cvae_seeds"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.seq_len != 32 or self.hidden != 32:
            raise ValueError("This experiment uses a 32-input, 32-hidden-unit MLP")
        if not self.train_lengths or not self.heldout_lengths:
            raise ValueError("Both training and interpolation lengths are required")
        if set(self.train_lengths) & set(self.heldout_lengths):
            raise ValueError("Training and interpolation lengths overlap")
        if any(k not in range(3, 9) for k in self.train_lengths + self.heldout_lengths):
            raise ValueError("Pattern lengths must be in 3..8")
        if any(not min(self.train_lengths) < k < max(self.train_lengths) for k in self.heldout_lengths):
            raise ValueError("Only interpolation, not extrapolation, is permitted")
        for field in ("train_lengths", "heldout_lengths", "cvae_seeds"):
            if len(set(getattr(self, field))) != len(getattr(self, field)):
                raise ValueError(f"Duplicate values in {field}")
        for field in ("bank_mlps", "bank_steps", "bank_batch", "bank_val_size", "support_pool_size",
                      "cvae_epochs", "cvae_batch", "cvae_hidden", "latent_dim", "eval_masks",
                      "eval_repeats", "eval_steps", "eval_batch", "eval_test_size"):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive")
        if self.top_fraction != 0.1:
            raise ValueError("The requested protocol selects exactly the top 10%")
        if self.bank_mlps % 10:
            raise ValueError("bank_mlps must be divisible by 10 for an exact 10% selection")
        if self.task_limit < 0 or self.eval_task_limit < 0:
            raise ValueError("Task caps must be nonnegative")
        if min(self.bank_lr, self.cvae_lr, self.eval_lr) <= 0 or self.cvae_beta < 0:
            raise ValueError("Invalid learning rates or beta")

    def to_dict(self):
        return asdict(self)

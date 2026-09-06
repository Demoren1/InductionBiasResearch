"""Independent settings; existing pattern and motif_pair runs are unaffected."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    method: str = "generator"
    seq_len: int = 32
    hidden: int = 32
    lengths: tuple = (3, 4, 5, 6, 7, 8)
    train_lengths: tuple = (3, 4, 5, 6, 7, 8)
    rank1: int = 16
    rank2: int = 4
    width: int = 64
    generator_depth: int = 2
    condition_length: bool = True
    inner_optimizer: str = "sgd"
    init_scale: float = 0.1
    data_seed: int | None = None
    all_unseen_patterns: bool = False
    seed: int = 42
    task_split_seed: int = 42
    input_split_seed: int = 1729
    outer_steps: int = 1000
    tasks_per_step: int = 6
    inner_steps: int = 20
    inner_lr: float = 0.1
    outer_lr: float = 0.001
    support_size: int = 256
    query_size: int = 256
    batch_size: int = 128
    validate_every: int = 50
    val_tasks_per_length: int = 2
    grad_clip: float = 5.0

    def __post_init__(self):
        object.__setattr__(self, "lengths", tuple(self.lengths))
        object.__setattr__(self, "train_lengths", tuple(self.train_lengths))
        if self.method not in {"generator", "table", "random", "ideal"}:
            raise ValueError("Unknown method")
        if self.seq_len != 32 or not self.lengths or any(k < 3 or k > 8 for k in self.lengths):
            raise ValueError("This protocol uses length-32 inputs and pattern lengths 3..8")
        if not self.train_lengths or not set(self.train_lengths) <= set(self.lengths):
            raise ValueError("train_lengths must be a nonempty subset of lengths")
        if len(set(self.lengths)) != len(self.lengths) or len(set(self.train_lengths)) != len(self.train_lengths):
            raise ValueError("Lengths must be unique")
        if self.method == "ideal" and (self.rank1 < max(self.lengths) + 2 or self.rank2 < 2
                                       or self.hidden < self.seq_len - min(self.lengths) + 1):
            raise ValueError("Ideal requires rank1 >= max(lengths)+2, rank2 >= 2, and enough hidden units for every window")
        for key in ("hidden", "rank1", "rank2", "width", "generator_depth", "tasks_per_step", "support_size",
                    "query_size", "batch_size", "validate_every"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.inner_steps < 1 or self.outer_steps < 0 or min(self.inner_lr, self.outer_lr, self.grad_clip) <= 0:
            raise ValueError("Invalid training budget or learning rate")
        if self.val_tasks_per_length < 0 or self.init_scale <= 0 or self.inner_optimizer not in {"sgd", "adam"}:
            raise ValueError("Invalid validation cap or inner optimizer/initialization")

    def to_dict(self):
        return asdict(self)

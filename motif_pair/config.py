"""Shared configuration and task representation for the motif-pair experiment.

A task asks whether a circular length-16 sequence contains motif ``A`` followed
by motif ``B`` exactly ``gap`` positions later.  The task identifier is stable
and shell-safe, so it is also used in filenames and split manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SEQ_LEN = 16
MOTIF_LEN = 3
H = 16
GAPS = tuple(range(3, 11))
MOTIFS = tuple(format(i, f"0{MOTIF_LEN}b") for i in range(2 ** MOTIF_LEN))

# The intended first-layer support has two three-bit receptive fields for each
# of the 16 hidden units: exactly 6 * 16 = 96 active entries.
MASK_DIM = SEQ_LEN * H
K_ACTIVE = 6 * H

TASKS_PER_GAP = 8
TRAIN_TASKS_PER_GAP = 6
TEST_TASKS_PER_GAP = 2

N_VAL_SAMPLES = 2_048
POS_FRACTION = 0.5

# Downstream-search and generator defaults.  They live here so all runners
# share one reproducible experiment scale.
P = K_ACTIVE / MASK_DIM
N_MLPS_PER_TASK = 512
TRAIN_STEPS = 1_000
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 256
LR = 1e-3
EVAL_EVERY = 200
TOP_FRACTION = 0.1

CVAE_HIDDEN = 256
LATENT_DIM = 32
CVAE_EPOCHS = 80
CVAE_BATCH_SIZE = 256
CVAE_LR = 1e-3
CVAE_BETA = 0.1
EVAL_STEPS = 1_000

# The structural oracle depends only on separation, not motif identities.
# ``one_hot`` is deliberately the default: it is the representation used by
# existing artifacts and checkpoints.  New gap-OOD runs select ``scalar``
# explicitly, so a checkpoint records a representation rather than inheriting
# a mutable process-wide default.
CONDITION_ENCODING_ONE_HOT = "one_hot"
CONDITION_ENCODING_SCALAR = "scalar"
CONDITION_ENCODING_NONE = "none"
DEFAULT_CONDITION_ENCODING = CONDITION_ENCODING_ONE_HOT
COND_DIM = len(GAPS)  # Legacy one-hot dimension; prefer ``condition_dim`` in new code.

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA_DIR = OUTPUTS / "data"
CKPT_DIR = OUTPUTS / "checkpoints"
CVAE_DIR = OUTPUTS / "cvae"
EVAL_DIR = OUTPUTS / "eval"
SPLIT_DIR = OUTPUTS / "splits"

_TASK_RE = re.compile(r"^A([01]{3})_B([01]{3})_G(0[3-9]|10)$")


@dataclass(frozen=True, order=True)
class Task:
    """A validated motif-pair classification task."""

    a: str
    b: str
    gap: int

    def __post_init__(self) -> None:
        if self.a not in MOTIFS or self.b not in MOTIFS:
            raise ValueError(f"motifs must be {MOTIF_LEN}-bit strings")
        if self.a == self.b:
            raise ValueError("A and B must be distinct motifs")
        if self.gap not in GAPS:
            raise ValueError(f"gap must be one of {GAPS}")

    @property
    def id(self) -> str:
        return task_id(self)


def task_id(task: Task | str, b: str | None = None, gap: int | None = None) -> str:
    """Return canonical ``Axxx_Byyy_Gzz`` encoding for a task.

    Both ``task_id(Task(...))`` and ``task_id("000", "111", 5)`` are
    accepted to keep call sites compact.
    """
    if isinstance(task, Task):
        if b is not None or gap is not None:
            raise TypeError("b/gap are only valid when the first argument is motif A")
        return f"A{task.a}_B{task.b}_G{task.gap:02d}"
    if b is None or gap is None:
        return parse_task(task).id
    return Task(task, b, gap).id


def parse_task(task: Task | str) -> Task:
    """Parse a :class:`Task` or its canonical identifier."""
    if isinstance(task, Task):
        return task
    if not isinstance(task, str):
        raise TypeError("task must be a Task or canonical task-id string")
    match = _TASK_RE.fullmatch(task)
    if match is None:
        raise ValueError(f"invalid task id {task!r}; expected Axxx_Byyy_Gzz")
    return Task(match.group(1), match.group(2), int(match.group(3)))


def task_components(task: Task | str) -> tuple[str, str, int]:
    """Return ``(A, B, gap)`` for a task id or :class:`Task`."""
    parsed = parse_task(task)
    return parsed.a, parsed.b, parsed.gap


def all_tasks() -> tuple[Task, ...]:
    """Return the complete 8 * 7 * 8 task universe in stable order."""
    return tuple(Task(a, b, gap) for gap in GAPS for a in MOTIFS for b in MOTIFS if a != b)


ALL_TASKS = all_tasks()


def normalize_condition_encoding(encoding: str | None = None) -> str:
    """Return a validated, canonical condition-encoding name.

    ``None`` intentionally means the legacy one-hot encoding.  New run
    configuration should pass a concrete encoding and persist
    :func:`condition_metadata` in its checkpoint.
    """
    value = DEFAULT_CONDITION_ENCODING if encoding is None else encoding
    if not isinstance(value, str):
        raise TypeError("condition encoding must be a string or None")
    value = value.lower()
    aliases = {
        "onehot": CONDITION_ENCODING_ONE_HOT,
        "scalar_normalized_gap": CONDITION_ENCODING_SCALAR,
    }
    value = aliases.get(value, value)
    if value not in {
        CONDITION_ENCODING_ONE_HOT,
        CONDITION_ENCODING_SCALAR,
        CONDITION_ENCODING_NONE,
    }:
        raise ValueError(
            f"unknown condition encoding {encoding!r}; expected one of "
            f"{(CONDITION_ENCODING_ONE_HOT, CONDITION_ENCODING_SCALAR, CONDITION_ENCODING_NONE)}"
        )
    return value


def condition_dim(encoding: str | None = None) -> int:
    """Dimension of a canonical gap-condition representation."""
    encoding = normalize_condition_encoding(encoding)
    if encoding == CONDITION_ENCODING_ONE_HOT:
        return len(GAPS)
    if encoding == CONDITION_ENCODING_SCALAR:
        return 1
    return 0


def condition_metadata(encoding: str | None = None) -> dict[str, object]:
    """Stable checkpoint metadata describing a condition representation."""
    encoding = normalize_condition_encoding(encoding)
    metadata: dict[str, object] = {"condition_encoding": encoding}
    if encoding == CONDITION_ENCODING_SCALAR:
        metadata.update({
            "condition_scalar_normalization": "(gap - min_gap) / (max_gap - min_gap)",
            "condition_gap_min": min(GAPS),
            "condition_gap_max": max(GAPS),
        })
    return metadata


def task_to_condition(task: Task | str, *, encoding: str | None = None):
    """Encode the task's gap using the requested structural representation.

    ``one_hot`` is retained as the default for old callers.  ``scalar`` is a
    single normalized value in ``[0, 1]`` based on the full declared gap
    universe, never just the gaps visible in a meta-training split.
    """
    import torch

    parsed = parse_task(task)
    encoding = normalize_condition_encoding(encoding)
    if encoding == CONDITION_ENCODING_ONE_HOT:
        condition = torch.zeros(condition_dim(encoding), dtype=torch.float32)
        condition[GAPS.index(parsed.gap)] = 1.0
        return condition
    if encoding == CONDITION_ENCODING_SCALAR:
        span = max(GAPS) - min(GAPS)
        if span <= 0:  # Defensive: scalar normalization requires a gap range.
            raise ValueError("scalar condition encoding requires at least two distinct gaps")
        return torch.tensor([(parsed.gap - min(GAPS)) / span], dtype=torch.float32)
    return torch.zeros(0, dtype=torch.float32)


def task_dir(task: Task | str) -> Path:
    return CKPT_DIR / f"task_{parse_task(task).id}"


def val_path(task: Task | str) -> Path:
    return DATA_DIR / f"val_{parse_task(task).id}.pt"


def bank_path(task: Task | str) -> Path:
    return DATA_DIR / f"bank_{parse_task(task).id}.pt"


def ensure_dirs() -> None:
    for directory in (OUTPUTS, DATA_DIR, CKPT_DIR, CVAE_DIR, EVAL_DIR, SPLIT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

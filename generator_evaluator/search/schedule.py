"""Shared stage schedule for quality search followed by cooperation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from generator_evaluator.storage.progress import progress


STAGE_QUALITY = "quality"
STAGE_COOPERATION = "cooperation"
STAGE_BOOTSTRAP_QUALITY = "bootstrap_quality"
STAGE_JOINT = "joint"
STAGE_TRAINING_COMPLETE = "training_complete"


@dataclass(frozen=True)
class StageEpoch:
    stage: str
    epoch: int
    updates: int


class StagedSearch:
    """Run named stages through one loop and expose its restart position.

    A checkpoint stores the last completed ``stage`` and ``stage_epoch``. The
    same loop then resumes at the following epoch, or moves to the next stage
    after a completed stage. The update and epoch callbacks keep the search
    policy and real measurement cadence owned by the runner.
    """

    def __init__(self, stages: Iterable[tuple[str, int, int]]):
        rows = tuple((str(name), int(epochs), int(updates)) for name, epochs, updates in stages)
        if not rows or any(not name or epochs < 1 or updates < 1 for name, epochs, updates in rows):
            raise ValueError("each stage needs a name, positive epoch count, and positive update count")
        if len({name for name, _, _ in rows}) != len(rows):
            raise ValueError("stage names must be unique")
        self.stages = rows

    def run(self, *, start_stage: str | None, start_epoch: int,
            update: Callable[[str, int, int], None],
            finish_epoch: Callable[[str, int], None],
            checkpoint: Callable[[str, int], None]) -> str:
        names = [name for name, _, _ in self.stages]
        if start_stage is None:
            stage_index, start_at = 0, 1
        elif start_stage == STAGE_TRAINING_COMPLETE:
            return STAGE_TRAINING_COMPLETE
        else:
            if start_stage not in names:
                raise ValueError(f"checkpoint stage {start_stage!r} is not in the active schedule")
            stage_index = names.index(start_stage)
            start_at = start_epoch + 1

        for index in range(stage_index, len(self.stages)):
            name, epochs, updates = self.stages[index]
            first_epoch = start_at if index == stage_index else 1
            stage_epochs = progress(range(first_epoch, epochs + 1),
                                    desc=f"{name.replace('_', ' ').title()} stage", unit="epoch")
            for epoch in stage_epochs:
                stage_updates = progress(range(updates),
                    desc=f"{name.replace('_', ' ').title()} {epoch}: generator updates", unit="update")
                for ordinal in stage_updates:
                    update(name, epoch, ordinal)
                finish_epoch(name, epoch)
                checkpoint(name, epoch)
            # A checkpoint may have been taken at this stage's final epoch.
            # Moving the next stage marker into the checkpoint is the caller's
            # responsibility after this method returns from one stage.
        return STAGE_TRAINING_COMPLETE


def search_stages(*, phase: str, quality_epochs: int, quality_updates: int,
                  cooperation_rounds: int, cooperation_updates: int):
    """Describe the configured helper or main search stages."""
    if phase == "bootstrap":
        return StagedSearch(((STAGE_BOOTSTRAP_QUALITY, quality_epochs, quality_updates),))
    if phase != "search":
        raise ValueError("phase must be bootstrap or search")
    return StagedSearch(((STAGE_QUALITY, quality_epochs, quality_updates),
                         (STAGE_COOPERATION, cooperation_rounds, cooperation_updates)))


def joint_search_stages(*, epochs: int, updates: int) -> StagedSearch:
    """Describe the default interleaved joint training schedule."""
    return StagedSearch(((STAGE_JOINT, epochs, updates),))

"""Terminal progress bars; cosmetic output never changes experiment state."""
import os
import sys

from tqdm.auto import tqdm


def progress(iterable=None, *, desc, total=None, position=0, leave=True, unit="it"):
    setting = os.environ.get("GENERATOR_EVALUATOR_PROGRESS", "auto")
    disabled = setting == "0" or (setting == "auto" and not sys.stderr.isatty())
    return tqdm(iterable, total=total, desc=desc, position=position, leave=leave,
                unit=unit, dynamic_ncols=True, mininterval=1.0, disable=disabled)

"""Re-evaluate a generated-sharing checkpoint with validation checkpointing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .generated_sharing import reevaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pattern-coverage", choices=("train10", "all16"), default="all16")
    args = parser.parse_args()
    reevaluate(args.checkpoint, args.output, args.device, args.pattern_coverage)


if __name__ == "__main__":
    main()

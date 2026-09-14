"""Re-evaluate a trained length-32 generator with longer inner searches."""

from __future__ import annotations

import argparse
from pathlib import Path

from .length32_joint import reevaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-inner-max-steps", type=int, default=1000)
    args = parser.parse_args()
    reevaluate(args.checkpoint, args.output, args.device, args.eval_inner_max_steps)


if __name__ == "__main__":
    main()

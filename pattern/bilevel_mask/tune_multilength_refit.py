"""Evaluate weight-training hyperparameters on the fixed analytic structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .multilength_sharing import (
    MultiLengthConfig,
    _fit_shared,
    _oracle_fixed,
    all_tasks,
    make_data,
)


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--support-per-class", type=int, required=True)
    parser.add_argument("--validation-per-class", type=int, default=256)
    parser.add_argument("--query-per-class", type=int, default=512)
    parser.add_argument("--restarts", type=int, default=16)
    args = parser.parse_args()
    config = MultiLengthConfig(
        seed=args.seed, final_refit_steps=args.steps, final_refit_lr=args.lr,
        support_per_class=args.support_per_class,
        validation_per_class=args.validation_per_class,
        query_per_class=args.query_per_class, eval_restarts=args.restarts,
    )
    device = torch.device(args.device)
    tasks = all_tasks(config.pattern_lengths)
    data = make_data(tasks, config, device, "refit-tune")
    oracle = _oracle_fixed(tasks, config, device)
    result = _fit_shared(oracle, data, config, device, "analytic cyclic sharing", "refit-tune")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"config": config.to_dict(), "result": result}, indent=2))


if __name__ == "__main__":
    main()

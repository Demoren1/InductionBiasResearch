"""Single-stage CLI for source-bank and meta-generator work.

Output layout is controlled by ``--out``.  The production launcher can keep
GPU allocation and seed scheduling outside this module and invoke one stage at
a time for clean resumption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .bank import BankConfig, build_bank
from .meta import MetaConfig, train_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)

    bank_parser = stages.add_parser("bank", help="train source teachers and save bank.pt")
    bank_parser.add_argument("--seed", type=int, required=True)
    bank_parser.add_argument("--method", choices=("transformer_mask", "free_mask"),
                             default="transformer_mask")
    bank_parser.add_argument("--out", type=Path, required=True,
                             help="seed-specific bank output directory")
    bank_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    bank_parser.add_argument("--min-steps", type=int, default=BankConfig.min_steps)
    bank_parser.add_argument("--max-steps", type=int, default=BankConfig.max_steps)
    bank_parser.add_argument("--extend", action="store_true",
                             help="extend a prior capped bank run with all other settings fixed")
    bank_parser.add_argument("--no-resume", action="store_true")

    meta_parser = stages.add_parser("meta", help="train one mask generator")
    meta_parser.add_argument("--seed", type=int, required=True)
    meta_parser.add_argument("--method", choices=("transformer_mask", "free_mask"), required=True)
    meta_parser.add_argument("--out", type=Path, required=True,
                             help="seed/method-specific output directory")
    meta_parser.add_argument("--bank", type=Path,
                             help="source bank path; defaults to sibling bank/bank.pt")
    meta_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    meta_parser.add_argument("--min-steps", type=int, default=MetaConfig.min_steps)
    meta_parser.add_argument("--max-steps", type=int, default=MetaConfig.max_steps)
    meta_parser.add_argument("--replicas", type=int, default=MetaConfig.replicas)
    meta_parser.add_argument("--extend", action="store_true",
                             help="extend a prior capped meta run with all other settings fixed")
    meta_parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    if args.stage == "bank":
        config = BankConfig(min_steps=args.min_steps, max_steps=args.max_steps)
        path = build_bank(args.out, args.seed, args.device, config,
                          resume=not args.no_resume, extend=args.extend)
    else:
        bank_path = args.bank or (args.out.parent / "bank" / "bank.pt")
        config = MetaConfig(method=args.method, min_steps=args.min_steps,
                            max_steps=args.max_steps, replicas=args.replicas)
        path = train_meta(bank_path, args.out, args.seed, args.method, args.device,
                          config, resume=not args.no_resume, extend=args.extend)
    print(path)


if __name__ == "__main__":
    main()

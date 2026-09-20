"""Shuffle image-to-expert routes while preserving their episode distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .meta_u import load_pool
from .meta_u_first_v_moe import LENGTHS, VExpertMoE, evaluate
from .v_moe_sweep import SPECS, stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first_v_moe"))
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/mnist8m"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--router", choices=("conv", "mlp"), default="conv")
    parser.add_argument("--counts", help="Comma-separated expert counts")
    parser.add_argument("--ortho-weight", type=float, default=0.2)
    args = parser.parse_args()
    if args.counts is None:
        specs = SPECS
    else:
        try:
            specs = tuple((int(part.strip()), args.ortho_weight)
                          for part in args.counts.split(","))
        except ValueError:
            parser.error("--counts must be comma-separated integers")
        if not specs:
            parser.error("--counts must not be empty")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    configs = {}
    for spec in specs:
        path = args.results / f"{stem(*spec, 42, args.router)}.json"
        configs[spec] = json.loads(path.read_text())["config"]
    reference = configs[specs[0]]
    pool = load_pool(args.data_dir, seed=42, split="test",
                     per_digit=reference["eval_images_per_digit"],
                     device=device)
    for spec in specs:
        path = args.results / f"{stem(*spec, 42, args.router)}.json"
        data = json.loads(path.read_text())
        if "test_shuffled_route" in data:
            print(f"SKIP {stem(*spec, 42, args.router)}", flush=True)
            continue
        config = data["config"]
        model = VExpertMoE(spec[0], 42, args.router).to(device)
        best_path = args.results / f"{stem(*spec, 42, args.router)}_best.pt"
        model.load_state_dict(torch.load(
            best_path, map_location=device, weights_only=True))
        data["test_shuffled_route"] = evaluate(
            model, pool, seed=42 + 19000, tasks=config["test_tasks"],
            support=config["support"], query=config["query"],
            steps=config["inner_steps"], lr_coeff=config["lr_coeff"],
            lr_readout=config["lr_readout"], lengths=LENGTHS,
            shuffle_route=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(path)
        print(f"{stem(*spec, 42, args.router)} MAE5={data['test']['5']['mae']:.4f} "
              f"shuffled={data['test_shuffled_route']['5']['mae']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()

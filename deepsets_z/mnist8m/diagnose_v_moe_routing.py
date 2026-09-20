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
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    configs = {}
    for spec in SPECS:
        path = args.results / f"{stem(*spec, 42)}.json"
        configs[spec] = json.loads(path.read_text())["config"]
    reference = configs[SPECS[0]]
    pool = load_pool(args.data_dir, seed=42, split="test",
                     per_digit=reference["eval_images_per_digit"],
                     device=device)
    for spec in SPECS:
        path = args.results / f"{stem(*spec, 42)}.json"
        data = json.loads(path.read_text())
        if "test_shuffled_route" in data:
            print(f"SKIP {stem(*spec, 42)}", flush=True)
            continue
        config = data["config"]
        model = VExpertMoE(spec[0], 42).to(device)
        best_path = args.results / f"{stem(*spec, 42)}_best.pt"
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
        print(f"{stem(*spec, 42)} MAE5={data['test']['5']['mae']:.4f} "
              f"shuffled={data['test_shuffled_route']['5']['mae']:.4f}",
              flush=True)


if __name__ == "__main__":
    main()

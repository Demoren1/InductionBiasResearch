"""Post-training adaptation and translation checks for the first-layer U run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .meta_u import load_pool
from .meta_u_first import FirstLayerU, evaluate, translate


@torch.no_grad()
def shift_error(model: FirstLayerU, images: torch.Tensor) -> float:
    """Normalized first-layer equivariance error for a three-pixel downshift.

    The output grid is fixed to the three 10x10 feature maps specified by the
    experiment.  A hidden-unit permutation can hide an equivalent structure.
    """
    u = model.u()
    weight = (model.dense_weight if model.arm == "dense" else
              (u @ model.initial_v1).reshape(785, 300))

    def features(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.pad(x, (0, 1), value=1.0) @ weight).reshape(
            -1, 3, 10, 10)

    # Exclude the last comparison: its shifted receptive field crosses the
    # bottom image boundary, while its unshifted counterpart does not.
    original = features(images)[:, :, :8, :]
    shifted = features(translate(images, 3, 0))[:, :, 1:9, :]
    scale = (original.square().mean() + shifted.square().mean()) / 2
    return float(((original - shifted).square().mean() / scale).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(
        "deepsets_z/mnist8m/outputs/meta_u_first/full_v1"))
    parser.add_argument("--out", type=Path, default=Path(
        "deepsets_z/mnist8m/meta_u_first_diagnostics.json"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("datasets/mnist8m"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device)
    output = {"no_v_adaptation": {}, "first_layer_shift_error": {},
              "shift_error_definition":
              "mean((f(shift_down_3(x))[:,1:9,:] - f(x)[:,:8,:])^2) "
              "/ mean((f(shift_down_3(x))[:,1:9,:]^2 + "
              "f(x)[:,:8,:]^2)/2); f is first-layer tanh reshaped "
              "to 3x10x10"}
    for seed in (42, 43, 44):
        pool = load_pool(args.data_dir, seed=seed, split="test",
                         per_digit=1000, device=device)
        images = pool[0][:128].float().div(255.0)
        for arm in ("learned", "random", "dense") + (
                ("convolution",) if seed == 42 else ()):
            key = f"{arm}_seed{seed}"
            metadata = json.loads((args.root / f"{key}.json").read_text())
            model = FirstLayerU(arm, seed).to(device)
            model.load_state_dict(torch.load(args.root / f"{key}_best.pt",
                                             map_location=device,
                                             weights_only=True))
            model.eval()
            output["first_layer_shift_error"][key] = shift_error(model, images)
            if arm == "learned":
                config = metadata["config"]
                output["no_v_adaptation"][key] = evaluate(
                    model, pool, seed=seed + 19000,
                    tasks=int(config["test_tasks"]),
                    support=int(config["support"]),
                    query=int(config["query"]),
                    steps=int(config["inner_steps"]), lr_v1=0.0,
                    lr_readout=float(config["lr_readout"]),
                    lengths=(5, 10, 20))
            print(key, output["first_layer_shift_error"][key], flush=True)
            del model
    output["no_v_adaptation_mean_mae"] = {
        str(length): float(np.mean([
            output["no_v_adaptation"][f"learned_seed{seed}"][str(length)][
                "mae"] for seed in (42, 43, 44)]))
        for length in (5, 10, 20)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"no_v_adaptation_mean_mae":
                      output["no_v_adaptation_mean_mae"],
                      "first_layer_shift_error":
                      output["first_layer_shift_error"]}, indent=2))


if __name__ == "__main__":
    main()

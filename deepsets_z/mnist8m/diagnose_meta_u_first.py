"""Post-training adaptation and translation checks for the first-layer U run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .meta_u import load_pool, normalized_mse
from .meta_u_first import FirstLayerU, evaluate, shifted_episode, translate


def first_weight(model: FirstLayerU, v1: torch.Tensor | None = None,
                 u: torch.Tensor | None = None) -> torch.Tensor:
    if model.arm == "dense":
        return model.dense_weight
    if u is None:
        u = model.u()
    if v1 is None:
        v1 = model.initial_v1
    return (u @ v1).reshape(785, 300)


@torch.no_grad()
def shift_error(model: FirstLayerU, images: torch.Tensor,
                v1: torch.Tensor | None = None,
                u: torch.Tensor | None = None) -> float:
    """Normalized first-layer equivariance error for a three-pixel downshift.

    The output grid is fixed to the three 10x10 feature maps specified by the
    experiment.  A hidden-unit permutation can hide an equivalent structure.
    """
    weight = first_weight(model, v1, u)

    def features(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.pad(x, (0, 1), value=1.0) @ weight).reshape(
            -1, 3, 10, 10)

    # Exclude the last comparison: its shifted receptive field crosses the
    # bottom image boundary, while its unshifted counterpart does not.
    original = features(images)[:, :, :8, :]
    shifted = features(translate(images, 3, 0))[:, :, 1:9, :]
    scale = (original.square().mean() + shifted.square().mean()) / 2
    return float(((original - shifted).square().mean() / scale).item())


@torch.no_grad()
def image_sensitivity(model: FirstLayerU, images: torch.Tensor,
                      v1: torch.Tensor | None = None,
                      u: torch.Tensor | None = None) -> dict:
    """Check whether apparent equivariance just comes from constant features."""
    weight = first_weight(model, v1, u)
    features = torch.tanh(F.pad(images, (0, 1), value=1.0) @ weight)
    variance = features.var(dim=0, unbiased=False).mean()
    energy = features.square().mean()
    return {"variance": float(variance.item()),
            "variance_to_energy": float((variance / energy).item())}


@torch.no_grad()
def effective_rank(model: FirstLayerU) -> float | None:
    u = model.u()
    if u is None:
        return None
    singular = torch.linalg.eigvalsh(u.T @ u).clamp_min(0).sqrt()
    probabilities = singular / singular.sum()
    positive = probabilities[probabilities > 0]
    return float(torch.exp(-(positive * positive.log()).sum()).item())


def adapted_structure(model: FirstLayerU, pool: tuple,
                      images: torch.Tensor, *, seed: int,
                      tasks: int = 16) -> dict:
    """Measure first-layer behavior after task-specific support adaptation."""
    rng = torch.Generator(device=pool[0].device).manual_seed(seed + 29000)
    with torch.no_grad():
        u = model.u()
    shifts, sensitivities = [], []
    for _ in range(tasks):
        support, _, _ = shifted_episode(pool, support=32, query=32,
                                        rng=rng, query_length=5)
        v1 = model.initial_v1.detach().clone().requires_grad_(
            model.arm != "dense")
        readout = model.initial_readout.detach().clone().requires_grad_(True)
        for _ in range(5):
            prediction = model.predict(support[0], support[1],
                                       v1, readout, u)
            loss = normalized_mse(prediction, support[2], support[3])
            if model.arm == "dense":
                gr, = torch.autograd.grad(loss, (readout,))
            else:
                gv, gr = torch.autograd.grad(loss, (v1, readout))
                v1 = (v1 - 0.1 * gv).detach().requires_grad_(True)
            readout = (readout - 0.03 * gr).detach().requires_grad_(True)
        shifts.append(shift_error(model, images, v1, u))
        sensitivities.append(image_sensitivity(
            model, images, v1, u)["variance_to_energy"])
    return {"tasks": tasks, "shift_error_mean": float(np.mean(shifts)),
            "shift_error_median": float(np.median(shifts)),
            "image_sensitivity_mean": float(np.mean(sensitivities)),
            "image_sensitivity_median": float(np.median(sensitivities))}


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
              "first_layer_image_sensitivity": {},
              "adapted_first_layer": {},
              "basis_effective_rank": {},
              "shift_error_definition":
              "mean((f(shift_down_3(x))[:,1:9,:] - f(x)[:,:8,:])^2) "
              "/ mean((f(shift_down_3(x))[:,1:9,:]^2 + "
              "f(x)[:,:8,:]^2)/2); f is first-layer tanh reshaped "
              "to 3x10x10"}
    for seed in (42, 43, 44):
        pool = load_pool(args.data_dir, seed=seed, split="test",
                         per_digit=1000, device=device)
        images = pool[0][:128].float().div(255.0)
        for arm in ("learned", "random", "dense", "generated",
                    "generated_shuffled", "generated_ortho",
                    "generated_shuffled_ortho") + (
                ("convolution",) if seed == 42 else ()):
            key = f"{arm}_seed{seed}"
            metadata = json.loads((args.root / f"{key}.json").read_text())
            model = FirstLayerU(arm, seed).to(device)
            model.load_state_dict(torch.load(args.root / f"{key}_best.pt",
                                             map_location=device,
                                             weights_only=True))
            model.eval()
            output["first_layer_shift_error"][key] = shift_error(model, images)
            output["first_layer_image_sensitivity"][key] = image_sensitivity(
                model, images)
            output["basis_effective_rank"][key] = effective_rank(model)
            output["adapted_first_layer"][key] = adapted_structure(
                model, pool, images, seed=seed)
            if arm in ("learned", "generated", "generated_shuffled",
                       "generated_ortho", "generated_shuffled_ortho"):
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
        arm: {str(length): float(np.mean([
            output["no_v_adaptation"][f"{arm}_seed{seed}"][str(length)][
                "mae"] for seed in (42, 43, 44)]))
              for length in (5, 10, 20)}
        for arm in ("learned", "generated", "generated_shuffled",
                    "generated_ortho", "generated_shuffled_ortho")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"no_v_adaptation_mean_mae":
                      output["no_v_adaptation_mean_mae"],
                      "first_layer_shift_error":
                      output["first_layer_shift_error"]}, indent=2))


if __name__ == "__main__":
    main()

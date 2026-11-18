"""Paper-described augmented few-shot tasks; full higher-order training."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .images_data import Episodes
from .linear import save_json


def initialize(dataset, shots, ways, method, device):
    """MAML Conv4, applying the paper's changes to its first three layers."""
    omni = dataset == "omniglot"
    channels = [64] * 4 if omni else [32] * 4
    kernels = [3] * 4
    if not omni and shots == 5:
        channels[:3] = [128] * 3
        if method == "proto":
            channels = [128] * 4
        if method != "proto":
            kernels[:3] = [5] * 3
            channels.append(64)
            kernels.append(1)
    if method == "big":
        channels[:3] = [150] * 3 if omni else ([64] * 3 if shots == 1 else [128] * 3)
    parameters = {}
    previous = 1 if omni else 3
    for layer, (outputs, size) in enumerate(zip(channels, kernels)):
        prefix = f"conv{layer}"
        weight = torch.empty(outputs, previous, size, size, device=device)
        torch.nn.init.xavier_uniform_(weight)
        parameters[prefix + ".weight"] = weight
        parameters[prefix + ".bias"] = torch.zeros(outputs, device=device)
        parameters[prefix + ".gamma"] = torch.ones(outputs, device=device)
        parameters[prefix + ".beta"] = torch.zeros(outputs, device=device)
        if method == "msr":
            parameters[prefix + ".warp_out"] = torch.eye(outputs, device=device)
            parameters[prefix + ".warp_in"] = torch.eye(previous, device=device)
            parameters[prefix + ".warp_space"] = torch.eye(size * size, device=device)
        previous = outputs
    features = channels[-1] if omni else channels[-1] * 25
    if method == "big" and not omni:
        extra = 1920 if shots == 1 else 3840
        weight = torch.empty(extra, features, device=device)
        torch.nn.init.xavier_uniform_(weight)
        parameters["extra.weight"] = weight
        parameters["extra.bias"] = torch.zeros(extra, device=device)
        features = extra
    if method != "proto":
        weight = torch.empty(ways, features, device=device)
        if omni:
            torch.nn.init.normal_(weight)
        else:
            torch.nn.init.xavier_uniform_(weight)
        parameters["head.weight"] = weight
        parameters["head.bias"] = torch.zeros(ways, device=device)
        if method == "msr":
            parameters["head.warp_in"] = torch.eye(features, device=device)
            parameters["head.warp_out"] = torch.eye(ways, device=device)
    parameters = {name: tensor.requires_grad_() for name, tensor in parameters.items()}
    adaptable = [name for name in parameters if "warp" not in name
                 and (method != "anil" or name.startswith("head."))]
    rates = {name: torch.tensor(0.4 if omni else 0.05, device=device, requires_grad=True)
             for name in adaptable}
    return parameters, rates, len(channels)


def features(parameters, images, layers, dataset, method):
    if method == "d8":
        from .images_d8 import features as d8_features

        return d8_features(parameters, images, layers, dataset)
    x = images
    for layer in range(layers):
        prefix = f"conv{layer}"
        if prefix + ".index" in parameters:
            index = parameters[prefix + ".index"].long()
            codes = parameters[prefix + ".codes"]
            dictionary = torch.cat((codes.new_zeros(1), codes))
            weight = dictionary[index.abs()] * index.sign()
        else:
            weight = parameters[prefix + ".weight"]
        outputs, inputs, size, _ = weight.shape
        if method == "msr":
            weight = torch.einsum("ij,jkl,mk,nl->imn",
                                  parameters[prefix + ".warp_out"],
                                  weight.reshape(outputs, inputs, -1),
                                  parameters[prefix + ".warp_in"],
                                  parameters[prefix + ".warp_space"]).reshape_as(weight)
        stride = 2 if dataset == "omniglot" else 1
        x = F.conv2d(x, weight, parameters[prefix + ".bias"], stride=stride,
                     padding=size // 2)
        mean = x.mean((0, 2, 3), keepdim=True)
        variance = (x - mean).square().mean((0, 2, 3), keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + 1e-5)
        x = x * parameters[prefix + ".gamma"][None, :, None, None]
        x = F.relu(x + parameters[prefix + ".beta"][None, :, None, None])
        if dataset != "omniglot" and layer < 4:
            x = F.max_pool2d(x, 2)
    x = x.mean((-2, -1)) if dataset == "omniglot" else x.flatten(1)
    if "extra.weight" in parameters:
        x = F.relu(F.linear(x, parameters["extra.weight"], parameters["extra.bias"]))
    return x


def forward(parameters, images, layers, dataset, method):
    x = features(parameters, images, layers, dataset, method)
    if method == "msr":
        x = F.linear(x, parameters["head.warp_in"])
    x = F.linear(x, parameters["head.weight"], parameters["head.bias"])
    if method == "msr":
        x = F.linear(x, parameters["head.warp_out"])
    return x


def episode_logits(parameters, rates, sx, sy, qx, layers, dataset, method, steps, ways, shots):
    if method == "proto":
        support = features(parameters, sx, layers, dataset, method).reshape(ways, shots, -1).mean(1)
        query = features(parameters, qx, layers, dataset, method)
        return -(query[:, None] - support[None]).square().sum(-1)

    adapted = parameters
    for _ in range(steps):
        def support_loss(local):
            combined = {**adapted, **local}
            return F.cross_entropy(forward(combined, sx, layers, dataset, method), sy)

        gradients = torch.func.grad(support_loss)({name: adapted[name] for name in rates})
        adapted = {name: value - rates[name] * gradients[name] if name in rates else value
                   for name, value in adapted.items()}
    return forward(adapted, qx, layers, dataset, method)


def validation(parameters, rates, episodes, layers, dataset, method, ways, shots, count=100):
    accuracy, cross_entropy = [], []
    for _ in range(count):
        sx, sy, qx, qy = [item[0] for item in episodes.next(1)]
        logits = episode_logits(parameters, rates, sx, sy, qx, layers, dataset, method, 3, ways, shots)
        accuracy.append(float((logits.argmax(-1) == qy).float().mean()))
        cross_entropy.append(float(F.cross_entropy(logits, qy).detach()))
    return {"accuracy": float(np.mean(accuracy)), "cross_entropy": float(np.mean(cross_entropy))}


def checkpoint(path, parameters, rates, step, **extra):
    state = {"parameters": {key: value.detach().cpu() for key, value in parameters.items()},
             "rates": {key: value.detach().cpu() for key, value in rates.items()},
             "step": step, **extra}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def run(root, dataset, ways, shots, method, seed, steps, chunk, device):
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(141100 + seed)
    folder = root / "image_fits" / f"{dataset}_{ways}way_{shots}shot" / f"{method}_s{seed}"
    if (folder / "complete.json").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    if method == "d8":
        from .images_d8 import configuration as configuration_d8
        from .images_d8 import initialize as initialize_d8

        parameters, rates, layers = initialize_d8(root, dataset, shots, ways, device)
    else:
        parameters, rates, layers = initialize(dataset, shots, ways, method, device)
    batch = 32 if dataset == "omniglot" else 10
    outer_lr = 0.0005 if dataset == "miniimagenet" and shots == 5 else 0.001
    optimizer = torch.optim.Adam([*parameters.values(), *rates.values()], lr=outer_lr)
    start, best = 0, -math.inf
    if (folder / "latest.pt").exists():
        state = torch.load(folder / "latest.pt", map_location=device, weights_only=True)
        with torch.no_grad():
            for key, value in parameters.items():
                value.copy_(state["parameters"][key])
            for key, value in rates.items():
                value.copy_(state["rates"][key])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu(), device)
        start, best = state["step"], state["best"]
    training = Episodes(root / "images" / dataset, "train", ways, shots, device, 421100 + seed)
    if start:
        training.generator.set_state(state["episode_rng"].cpu())
    save_json(folder / "execution_start.json", {
        "unix_time": time.time(), "resume_from_step": start,
        "optimizer_and_rng_restored": bool(start), "total_steps": steps,
    })
    save_json(folder / "registered.json", {
        "dataset": dataset, "ways": ways, "shots": shots, "method": method, "seed": seed,
        "outer_steps": steps, "outer_lr": outer_lr, "meta_batch": batch,
        "train_inner_steps": 1, "test_inner_steps": 3, "query_per_class": 15,
        "initial_inner_lr": 0.4 if dataset == "omniglot" else 0.05,
        "trainable_parameters": sum(item.numel() for item in parameters.values()),
        "adapted_parameters": 0 if method == "proto" else sum(parameters[name].numel() for name in rates),
        "layers": layers, "implementation": "Reconstruction from paper and cited MAML; image author code is unavailable",
        "query_transforms": "Tensor bilinear equivalent geometry, disclosed PIL rounding differences",
        "normalization": "Task-local batch statistics, separate support and query; transductive query statistics",
        "d8_layout": configuration_d8(root, dataset, shots) if method == "d8" else None,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_code_sha256": hashlib.sha256(Path(__file__).with_name("images_data.py").read_bytes()).hexdigest(),
    })
    for source_name in ("images.py", "images_data.py", "images_d8.py"):
        source_path = Path(__file__).with_name(source_name)
        (folder / ("executed_" + source_name)).write_bytes(source_path.read_bytes())

    def query_loss(params, learning_rates, sx, sy, qx, qy):
        logits = episode_logits(params, learning_rates, sx, sy, qx, layers, dataset, method, 1, ways, shots)
        return F.cross_entropy(logits, qy)

    batched_loss = torch.vmap(query_loss, in_dims=(None, None, 0, 0, 0, 0))
    started = time.perf_counter()
    log = (folder / "history.jsonl").open("a")
    for step in range(start, steps):
        optimizer.zero_grad()
        episode = training.next(batch)
        objective = 0.0
        for begin in range(0, batch, chunk):
            current = [item[begin:begin + chunk] for item in episode]
            values = batched_loss(parameters, rates, *current)
            loss = values.sum() / batch
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite objective at step {step}")
            loss.backward()
            objective += loss.detach().item()
        optimizer.step()
        if (step + 1) % 100 == 0:
            record = {"step": step + 1, "train_cross_entropy": objective,
                      "seconds": time.perf_counter() - started,
                      "steps_per_second": (step + 1 - start) / (time.perf_counter() - started)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            save_json(folder / "progress.json", record)
        if (step + 1) % 2000 == 0 or step + 1 == steps:
            selection = Episodes(root / "images" / dataset, "val", ways, shots, device, 723311)
            metrics = validation(parameters, rates, selection, layers, dataset, method, ways, shots)
            del selection
            improved = metrics["accuracy"] > best
            if improved:
                best = metrics["accuracy"]
                checkpoint(folder / "best.pt", parameters, rates, step + 1, validation=metrics)
            save_json(folder / "validation.json", {"step": step + 1, "metrics": metrics,
                                                   "best_accuracy": best, "improved": improved})
        if (step + 1) % 100 == 0 or step + 1 == steps:
            checkpoint(
                folder / "latest.pt", parameters, rates, step + 1,
                optimizer=optimizer.state_dict(), best=best,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(device),
                episode_rng=training.generator.get_state(),
            )
    log.close()
    save_json(folder / "complete.json", {"status": "complete", "outer_steps": steps,
                                        "best_validation_accuracy": best, "test_targets_read": False,
                                        "seconds_this_invocation": time.perf_counter() - started})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("omniglot", "miniimagenet"), required=True)
    parser.add_argument("--ways", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--method", choices=("maml", "msr", "anil", "proto", "big", "d8"), default="msr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=60000)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args.root.resolve(), args.dataset, args.ways, args.shots, args.method, args.seed,
        args.steps, args.chunk, torch.device(args.device))


if __name__ == "__main__":
    main()

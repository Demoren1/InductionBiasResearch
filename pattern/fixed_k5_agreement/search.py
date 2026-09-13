from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from pattern.evaluation.decoder_agreement import (
    align_columns, hard_topk, optimize_agreement, soft_topk,
)
from pattern.length_interp.bank import random_exact_k_masks
from pattern.length_interp.mlp import BatchedMaskedMLP

from .common import (atomic_save, ideal_mask, load_protocol, pair_dir, seed_for,
                     sha256_file, write_json)
from .model import FixedVAE, load_model


def _check_pair(config, pair: tuple[int, int]) -> None:
    if pair not in config.pairs:
        raise ValueError(f"pair {pair} is absent from the saved protocol")


def load_pair(out: Path, pair: tuple[int, int], device: torch.device) -> tuple[FixedVAE, FixedVAE, list[dict]]:
    config, _ = load_protocol(out)
    _check_pair(config, pair)
    models, provenance = [], []
    root = pair_dir(out, pair)
    for seed in pair:
        checkpoint = root / f"vae_{seed}" / "best.pt"
        metadata_path = root / f"vae_{seed}" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        digest = sha256_file(checkpoint)
        if metadata["checkpoint_sha256"] != digest:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        models.append(load_model(str(checkpoint), device))
        provenance.append({"seed": seed, "checkpoint": str(checkpoint), "sha256": digest})
    return models[0], models[1], provenance


def _decode(model: FixedVAE, latent: torch.Tensor, config) -> tuple[torch.Tensor, torch.Tensor]:
    logits = model.decode(latent, latent.new_empty(len(latent), 0))
    soft = soft_topk(logits, config.k_active, config.temperature).reshape(
        -1, config.seq_len, config.hidden)
    return soft, logits


def _project(latent: torch.Tensor, radius: float) -> torch.Tensor:
    norm = latent.norm(dim=1, keepdim=True).clamp_min(torch.finfo(latent.dtype).tiny)
    return latent * (radius / norm).clamp(max=1.0)


@torch.no_grad()
def random_pair_search(models: tuple[FixedVAE, FixedVAE], config, device: torch.device) -> dict:
    """Equal-state-count random latent search, selected only by agreement."""
    n, proposals = config.n_starts, config.random_proposals
    generator = torch.Generator(device=device).manual_seed(seed_for("random-pair", config.task_split_seed))
    best_loss = torch.full((n,), float("inf"), device=device)
    best_z = [torch.empty(n, config.latent_dim, device=device) for _ in models]
    batch_proposals = 8 if config.mask_dim >= 1024 else 16
    for offset in range(0, proposals, batch_proposals):
        count = min(batch_proposals, proposals - offset)
        latents = [_project(torch.randn(count * n, config.latent_dim, generator=generator, device=device),
                            config.latent_radius) for _ in models]
        masks = [_decode(model, latent, config)[0] for model, latent in zip(models, latents)]
        loss = (masks[0] - align_columns(masks[0], masks[1])).square().mean((1, 2))
        values, indices = loss.reshape(count, n).min(dim=0)
        improved = values < best_loss
        best_loss[improved] = values[improved]
        ordinal = torch.arange(n, device=device)
        for target, latent in zip(best_z, latents):
            candidate = latent.reshape(count, n, -1)[indices, ordinal]
            target[improved] = candidate[improved]
        if offset % 160 == 0:
            print(f"[k5-random-pair] {offset + count}/{proposals} proposals/start", flush=True)
    result: dict[str, Any] = {"loss": best_loss.cpu(), "proposals_per_start": proposals}
    for index, (model, latent) in enumerate(zip(models, best_z), 1):
        soft, logits = _decode(model, latent, config)
        result[f"z{index}"] = latent.cpu()
        result[f"soft{index}"] = soft.cpu()
        result[f"masks{index}"] = hard_topk(logits, config.k_active).reshape(
            n, config.seq_len, config.hidden).cpu()
    return result


def run_agreement(out: Path, pair: tuple[int, int], device_name: str) -> None:
    config, _ = load_protocol(out)
    _check_pair(config, pair)
    root = pair_dir(out, pair)
    agreement_path, random_path, provenance_path = (
        root / "agreement.pt", root / "random_pair.pt", root / "search_provenance.json")
    if all(path.exists() for path in (agreement_path, random_path, provenance_path)):
        provenance = json.loads(provenance_path.read_text())
        for path, key in ((agreement_path, "agreement_sha256"), (random_path, "random_pair_sha256")):
            if provenance[key] != sha256_file(path):
                raise ValueError(f"search artifact hash mismatch: {path}")
        print(f"[k5-search] pair={pair}: verified existing artifacts", flush=True)
        return
    if any(path.exists() for path in (agreement_path, random_path, provenance_path)):
        raise FileExistsError(f"partial search output in {root}")
    device = torch.device(device_name)
    model1, model2, model_provenance = load_pair(out, pair, device)
    result = optimize_agreement(
        model1, model2, n_starts=config.n_starts, steps=config.agreement_steps,
        lr=config.agreement_lr, seed=seed_for("agreement-starts", config.task_split_seed),
        temperature=config.temperature, radius=config.latent_radius, device=device,
        k=config.k_active,
    )
    atomic_save(agreement_path, result)
    random_result = random_pair_search((model1, model2), config, device)
    atomic_save(random_path, random_result)
    write_json(provenance_path, {
        "pair": list(pair), "models": model_provenance,
        "agreement_sha256": sha256_file(agreement_path),
        "random_pair_sha256": sha256_file(random_path),
        "agreement_start_seed": seed_for("agreement-starts", config.task_split_seed),
        "selection": "agreement-only; no task data, labels, or ideal mask accessed",
    })


def _fresh_parameters(n: int, config, seed: int, device: torch.device) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return [
        torch.nn.Parameter(torch.randn(n, config.seq_len, config.hidden,
                                       generator=generator, device=device) * 0.1),
        torch.nn.Parameter(torch.zeros(n, config.hidden, device=device)),
        torch.nn.Parameter(torch.randn(n, config.hidden, generator=generator, device=device) * 0.1),
        torch.nn.Parameter(torch.zeros(n, device=device)),
    ]


def _forward(x: torch.Tensor, params: list[torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    w1, b1, w2, b2 = params
    hidden = F.relu(torch.einsum("bl,nlh->bnh", x, w1 * mask) + b1)
    return torch.einsum("bnh,nh->bn", hidden, w2) + b2


def _network_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, target[:, None].expand_as(logits), reduction="none").mean(0)


def _project_(latent: torch.Tensor, radius: float) -> None:
    with torch.no_grad():
        latent.copy_(_project(latent, radius))


def optimize_task_z(model: FixedVAE, initial_z: torch.Tensor, data: dict, pattern: str,
                    config, device: torch.device) -> dict:
    """Direct-gradient target-aware single-z baseline for one held-out task."""
    model.eval().requires_grad_(False)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    latent = torch.nn.Parameter(initial_z.to(device).clone())
    _project_(latent, config.latent_radius)
    initial = latent.detach().cpu().clone()
    optimizer_z = torch.optim.Adam([latent], lr=config.task_z_lr)
    support_x = data["task_support"]["x"].to(device)
    support_y = data["task_support"]["y"].to(device)
    query_x = data["task_query"]["x"].to(device)
    query_y = data["task_query"]["y"].to(device)
    n = len(latent)

    with torch.no_grad():
        _, initial_logits = _decode(model, latent, config)
        initial_masks = hard_topk(initial_logits, config.k_active).reshape(
            n, config.seq_len, config.hidden).cpu()
    best_loss = torch.full((n,), float("inf"), device=device)
    best_z = latent.detach().clone()
    best_outer = torch.full((n,), -1, dtype=torch.long, device=device)
    history = []

    for outer in range(config.task_outer_steps):
        optimizer_z.zero_grad(set_to_none=True)
        _, detached_logits = _decode(model, latent, config)
        detached_mask = soft_topk(detached_logits, config.k_active, config.temperature).reshape(
            n, config.seq_len, config.hidden).detach()
        params = _fresh_parameters(
            n, config, seed_for("task-mlp", config.task_split_seed, pattern, outer), device)
        inner = torch.optim.Adam(params, lr=config.eval_lr)
        batch_generator = torch.Generator().manual_seed(
            seed_for("task-batches", config.input_split_seed, pattern, outer))
        train_sum, train_count = 0.0, 0

        for step in range(config.task_warmup_steps):
            index = torch.randint(len(support_x), (config.eval_batch,), generator=batch_generator).to(device)
            loss = _network_bce(_forward(support_x[index], params, detached_mask), support_y[index])
            inner.zero_grad(set_to_none=True)
            loss.sum().backward()
            inner.step()
            train_sum += float(loss.detach().mean())
            train_count += 1

        for step in range(config.task_grad_steps):
            index = torch.randint(len(support_x), (config.eval_batch,), generator=batch_generator).to(device)
            live_mask, _ = _decode(model, latent, config)
            loss = _network_bce(_forward(support_x[index], params, live_mask), support_y[index])
            inner.zero_grad(set_to_none=True)
            loss.sum().backward()
            inner.step()
            train_sum += float(loss.detach().mean())
            train_count += 1

        live_mask, _ = _decode(model, latent, config)
        validation = _network_bce(_forward(query_x, params, live_mask), query_y)
        with torch.no_grad():
            improved = validation.detach() < best_loss
            best_loss = torch.where(improved, validation.detach(), best_loss)
            best_z[improved] = latent.detach()[improved]
            best_outer[improved] = outer
        validation.sum().backward()
        if latent.grad is None or not torch.isfinite(latent.grad).all():
            raise RuntimeError("individual latent search produced an invalid gradient")
        raw_norm = latent.grad.detach().norm(dim=1)
        with torch.no_grad():
            scale = (10.0 / raw_norm.clamp_min(torch.finfo(latent.dtype).tiny)).clamp(max=1.0)
            latent.grad.mul_(scale[:, None])
        optimizer_z.step()
        _project_(latent, config.latent_radius)
        history.append({
            "outer": outer, "train_bce": train_sum / max(1, train_count),
            "query_bce": float(validation.detach().mean()),
            "raw_gradient_norm": float(raw_norm.mean()), "z_norm": float(latent.detach().norm(dim=1).mean()),
        })
        print(f"[k5-task-z] pattern={pattern} outer={outer + 1}/{config.task_outer_steps} "
              f"query={history[-1]['query_bce']:.5f} z_norm={history[-1]['z_norm']:.3f}", flush=True)

    def pack(codes: torch.Tensor) -> dict:
        with torch.no_grad():
            soft, logits = _decode(model, codes, config)
            masks = hard_topk(logits, config.k_active).reshape(n, config.seq_len, config.hidden)
        return {"z": codes.detach().cpu(), "soft": soft.cpu(), "masks": masks.cpu()}

    for name, value in model.state_dict().items():
        if not torch.equal(before[name], value):
            raise AssertionError(f"frozen decoder changed: {name}")
    return {
        "initial_z": initial, "initial_masks": initial_masks,
        "final": pack(latent), "best_query": pack(best_z),
        "best_query_loss": best_loss.cpu(), "best_outer": best_outer.cpu(),
        "history": history, "uses_target_labels": True, "uses_gold": False,
        "uses_final_test": False, "decoder_unchanged": True,
    }


def run_task_z(out: Path, pair: tuple[int, int], pattern: str, device_name: str) -> None:
    config, protocol = load_protocol(out)
    _check_pair(config, pair)
    if pattern not in protocol["task_split"]["test_patterns"]:
        raise ValueError("individual optimization is restricted to held-out tasks")
    root = pair_dir(out, pair)
    destination = root / "task_z" / f"pattern_{pattern}.pt"
    if destination.exists():
        record = torch.load(destination, map_location="cpu", weights_only=True)
        if record.get("pair") != list(pair) or record.get("pattern") != pattern:
            raise ValueError(f"incompatible task-z artifact: {destination}")
        print(f"[k5-task-z] pair={pair} pattern={pattern}: existing artifact", flush=True)
        return
    device = torch.device(device_name)
    model, _, provenance = load_pair(out, pair, device)
    agreement_path = root / "agreement.pt"
    agreement = torch.load(agreement_path, map_location="cpu", weights_only=True)
    data_path = out / "task_data" / f"pattern_{pattern}.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    result = optimize_task_z(model, agreement["initial_z1"], data, pattern, config, device)
    if not torch.equal(result["initial_z"], agreement["initial_z1"]):
        raise AssertionError("individual and agreement latent starts differ")
    result.update({
        "pair": list(pair), "pattern": pattern, "checkpoint": provenance[0],
        "agreement_sha256": sha256_file(agreement_path), "task_data_sha256": sha256_file(data_path),
    })
    atomic_save(destination, result)


def _structure(masks: torch.Tensor, config) -> dict:
    target = ideal_mask(config).to(masks).unsqueeze(0).expand(len(masks), -1, -1)
    aligned = align_columns(target, masks.float())
    intersection = (target * aligned).sum((1, 2))
    union = 2 * config.k_active - intersection
    iou = intersection / union
    hamming = (target - aligned).abs().sum((1, 2))
    return {
        "gold_iou": iou.tolist(), "gold_iou_mean": float(iou.mean()),
        "gold_hamming": hamming.tolist(), "gold_hamming_mean": float(hamming.mean()),
        "gold_hamming_normalized_mean": float((hamming / config.mask_dim).mean()),
        "unique_masks": int(torch.unique(masks.flatten(1), dim=0).shape[0]),
    }


def _pair_metrics(first: torch.Tensor, second: torch.Tensor, config) -> dict:
    aligned = align_columns(first.float(), second.float())
    intersection = (first * aligned).sum((1, 2))
    iou = intersection / (2 * config.k_active - intersection)
    hamming = (first - aligned).abs().sum((1, 2))
    return {
        "iou": iou.tolist(), "iou_mean": float(iou.mean()),
        "hamming": hamming.tolist(), "hamming_mean": float(hamming.mean()),
        "hamming_normalized_mean": float((hamming / config.mask_dim).mean()),
        "exact_count": int((hamming == 0).sum()), "count": len(first),
    }


def _eval_masks(masks: dict[str, torch.Tensor], data: dict, pattern: str, config,
                device: torch.device) -> dict:
    names = list(masks)
    count = config.n_starts
    stacked = torch.cat([masks[name] for name in names]).to(device)
    results = {name: {"accuracy": [], "bce": []} for name in names}
    train_x, train_y = data["eval_support"]["x"].to(device), data["eval_support"]["y"].to(device)
    test_x, test_y = data["eval_test"]["x"].to(device), data["eval_test"]["y"].to(device)
    for repeat in range(config.eval_repeats):
        base = BatchedMaskedMLP(
            torch.ones(count, config.seq_len, config.hidden),
            seed=seed_for("eval-init", config.task_split_seed, pattern, repeat)).to(device)
        model = BatchedMaskedMLP(stacked.cpu(), seed=0).to(device)
        with torch.no_grad():
            for name in ("w1", "b1", "w2", "b2"):
                template = getattr(base, name)
                target = getattr(model, name)
                target.copy_(template.repeat((len(names),) + (1,) * (target.ndim - 1)))
        optimizer = torch.optim.Adam(model.parameters(), lr=config.eval_lr)
        generator = torch.Generator().manual_seed(
            seed_for("eval-batches", config.input_split_seed, pattern, repeat))
        for step in range(config.eval_steps):
            index = torch.randint(len(train_x), (config.eval_batch,), generator=generator).to(device)
            logits = model(train_x[index])
            per_model = F.binary_cross_entropy_with_logits(
                logits, train_y[index, None].expand_as(logits), reduction="none").mean(0)
            optimizer.zero_grad(set_to_none=True)
            per_model.sum().backward()
            optimizer.step()
            if (step + 1) % 500 == 0 or step + 1 == config.eval_steps:
                print(f"[k5-eval] pattern={pattern} repeat={repeat + 1}/{config.eval_repeats} "
                      f"step={step + 1}/{config.eval_steps}", flush=True)
        metrics = model.validation(test_x, test_y, batch_size=512)
        accuracy = metrics["accuracy"].cpu().reshape(len(names), count)
        bce = metrics["bce"].cpu().reshape(len(names), count)
        for index, name in enumerate(names):
            results[name]["accuracy"].append(accuracy[index].tolist())
            results[name]["bce"].append(bce[index].tolist())
    for method in results.values():
        for metric in ("accuracy", "bce"):
            values = torch.tensor(method[metric])
            method[f"{metric}_mean"] = float(values.mean())
            method[f"{metric}_std"] = float(values.std(unbiased=False))
    return results


def run_evaluation(out: Path, pair: tuple[int, int], pattern: str, device_name: str) -> None:
    config, protocol = load_protocol(out)
    _check_pair(config, pair)
    if pattern not in protocol["task_split"]["test_patterns"]:
        raise ValueError("evaluation is restricted to held-out tasks")
    root = pair_dir(out, pair)
    destination = root / "evaluation" / f"pattern_{pattern}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("pair") != list(pair) or saved.get("pattern") != pattern:
            raise ValueError(f"incompatible evaluation artifact: {destination}")
        print(f"[k5-eval] pair={pair} pattern={pattern}: existing artifact", flush=True)
        return
    agreement = torch.load(root / "agreement.pt", map_location="cpu", weights_only=True)
    random_pair = torch.load(root / "random_pair.pt", map_location="cpu", weights_only=True)
    task = torch.load(root / "task_z" / f"pattern_{pattern}.pt", map_location="cpu", weights_only=True)
    random_masks = random_exact_k_masks(
        config.n_starts, seq_len=config.seq_len, hidden=config.hidden, k_active=config.k_active,
        seed=seed_for("random-exact-k", config.task_split_seed, pair[0], pair[1], pattern),
    )
    target = ideal_mask(config).unsqueeze(0).repeat(config.n_starts, 1, 1)
    masks = {
        "prior": agreement["initial_masks1"],
        "agreement": agreement["final_masks1"],
        "random_latent_pair": random_pair["masks1"],
        "individual_final": task["final"]["masks"],
        "individual_best_query": task["best_query"]["masks"],
        "random_exact_k": random_masks,
        "ideal": target,
    }
    for name, value in masks.items():
        if value.shape != (config.n_starts, config.seq_len, config.hidden):
            raise ValueError(f"{name} has wrong shape {tuple(value.shape)}")
        if not ((value == 0) | (value == 1)).all() or not (value.sum((1, 2)) == config.k_active).all():
            raise ValueError(f"{name} is not an exact-{config.k_active} binary mask")
    data_path = out / "task_data" / f"pattern_{pattern}.pt"
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    metrics = _eval_masks(masks, data, pattern, config, torch.device(device_name))
    structure = {name: _structure(value, config) for name, value in masks.items()}
    write_json(destination, {
        "pair": list(pair), "pattern": pattern, "methods": metrics, "structure": structure,
        "agreement_artifact_sha256": sha256_file(root / "agreement.pt"),
        "random_pair_artifact_sha256": sha256_file(root / "random_pair.pt"),
        "task_z_artifact_sha256": sha256_file(root / "task_z" / f"pattern_{pattern}.pt"),
        "task_data_sha256": sha256_file(data_path),
    })


def pair_summary(out: Path, pair: tuple[int, int]) -> None:
    config, protocol = load_protocol(out)
    root = pair_dir(out, pair)
    agreement = torch.load(root / "agreement.pt", map_location="cpu", weights_only=True)
    random_pair = torch.load(root / "random_pair.pt", map_location="cpu", weights_only=True)
    summary = {
        "pair": list(pair),
        "soft_loss": {
            "prior": float(agreement["initial_loss"].mean()),
            "agreement": float(agreement["final_loss"].mean()),
            "random_latent_pair": float(random_pair["loss"].mean()),
        },
        "pair_agreement": {
            "prior": _pair_metrics(agreement["initial_masks1"], agreement["initial_masks2"], config),
            "agreement": _pair_metrics(agreement["final_masks1"], agreement["final_masks2"], config),
            "random_latent_pair": _pair_metrics(random_pair["masks1"], random_pair["masks2"], config),
        },
        "tasks": {},
    }
    for pattern in protocol["task_split"]["test_patterns"]:
        summary["tasks"][pattern] = json.loads(
            (root / "evaluation" / f"pattern_{pattern}.json").read_text())
    write_json(root / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pattern")
    parser.add_argument("command", choices=("agreement", "task-z", "evaluate", "summarize"))
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    pair = tuple(args.pair)
    if args.command != "summarize" and args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.command == "agreement":
        run_agreement(args.out, pair, args.device)
    elif args.command == "task-z":
        if args.pattern is None:
            parser.error("task-z requires --pattern")
        run_task_z(args.out, pair, args.pattern, args.device)
    elif args.command == "evaluate":
        if args.pattern is None:
            parser.error("evaluate requires --pattern")
        run_evaluation(args.out, pair, args.pattern, args.device)
    else:
        pair_summary(args.out, pair)


if __name__ == "__main__":
    main()

"""Multi-task search with per-ordinal gradient-normalized agreement.

The two frozen CVAE decoders and the paired initial latents are the same as
the completed combined-task-agreement experiment.  Only the five new task
search trajectories are computed here; agreement controls are provenance-
checked copies of the old artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import configure_compute_device, ideal_mask
from evaluation.combined_task_agreement import (PARENT, TaskBank, _freeze,
    _validate_checkpoints, _verify_frozen, agreement, dataset, decode,
    evaluate_bank, per_network_bce, project, train_bank)

OLD = ROOT / "outputs/combined_task_agreement/20260906"
DEFAULT_OUT = ROOT / "outputs/normalized_task_agreement/20260906"
VARIANTS = ("task_only", "normalized_0p1", "normalized_0p3", "normalized_1", "fixed_0p01")
METHODS = VARIANTS + ("agreement_30", "agreement_1000", "prior", "ideal")
SOFT_METHODS = VARIANTS + ("prior",)
ALPHAS = (0., .1, .3, 1., None)
EPS = 1e-12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def save_torch(path, value):
    tmp = Path(str(path) + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def combine_gradients(task_grad, agreement_grad, eps=EPS):
    """Return the five manually composed gradients and per-ordinal diagnostics."""
    if task_grad.shape != agreement_grad.shape or task_grad.ndim != 4:
        raise ValueError("expected matching [variant, decoder, ordinal, latent] gradients")
    if task_grad.shape[0] != len(ALPHAS):
        raise ValueError(f"expected {len(ALPHAS)} variants")
    # These quantities deliberately have no gradient path: Adam receives the
    # explicitly assigned vector field, not a derivative through its scale.
    with torch.no_grad():
        task_grad, agreement_grad = task_grad.detach(), agreement_grad.detach()
        task_norm = task_grad.norm(dim=(1, 3))
        agreement_norm = agreement_grad.norm(dim=(1, 3))
        dot = (task_grad * agreement_grad).sum(dim=(1, 3))
        valid = (task_norm > eps) & (agreement_norm > eps)
        cosine = torch.where(valid, dot / (task_norm.clamp_min(eps) * agreement_norm.clamp_min(eps)),
                             torch.zeros_like(dot))
        coefficient = torch.zeros_like(task_norm)
        for index, alpha in enumerate(ALPHAS):
            if alpha is None:
                coefficient[index].fill_(.01)
            elif alpha:
                coefficient[index] = torch.where(valid[index], alpha * task_norm[index] /
                                                   (agreement_norm[index] + eps), coefficient[index])
        weighted = coefficient[:, None, :, None] * agreement_grad
        direction = task_grad + weighted
        weighted_ratio = weighted.norm(dim=(1, 3)) / task_norm.clamp_min(eps)
    return direction, {"task_grad_norm": task_norm, "agreement_grad_norm": agreement_norm,
                       "cosine": cosine, "negative_cosine": cosine < 0,
                       "effective_coefficient": coefficient,
                       "weighted_agreement_ratio": weighted_ratio, "valid_norms": valid}


def _history_value(value):
    return value.detach().cpu().tolist()


def prepare(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    old_protocol_path = OLD / "protocol.json"
    old_protocol = json.loads(old_protocol_path.read_text())
    for path, digest in old_protocol["source_hashes"].items():
        assert sha(path) == digest, f"combined source changed: {path}"
    profiles, control_sources = {}, {}
    for name, spec in old_protocol["profiles"].items():
        split = json.loads(Path(spec["split"]).read_text())
        tasks = {str(g): [t for t in split["test_tasks"] if config.parse_task(t).gap == g]
                 for g in spec["heldout_gaps"]}
        assert all(len(value) == 8 for value in tasks.values())
        profiles[name] = {**spec, "tasks": tasks}
        for gap in spec["heldout_gaps"]:
            for shard in range(2):
                path = OLD / name / f"gap{gap}_shard{shard}" / "search.pt"
                control_sources[str(path)] = sha(path)
    source_files = [Path(__file__), ROOT / "scripts/run_normalized_task_agreement.py",
                    ROOT / "evaluation/combined_task_agreement.py", ROOT / "data/generate.py",
                    ROOT / "config.py", ROOT / "evaluation/oracle_ideal.py", ROOT / "models/cvae.py",
                    ROOT / "evaluation/decoder_pair_search.py", ROOT / "evaluation/eval_generated_masks.py"]
    settings = {"n_starts": 64, "shards_per_gap": 2, "variants": list(VARIANTS),
                "methods": list(METHODS), "soft_methods": list(SOFT_METHODS),
                "outer_steps": 30, "inner_steps": 400, "batch_size": 128, "mlp_lr": .001,
                "z_lr": .05, "temperature": .5, "radius": 8., "search_val_samples": 1024,
                "evaluation_seeds": [0, 1, 2], "evaluation_steps": 1000,
                "evaluation_samples": 2048, "evaluation_batch": 256,
                "agreement_controls": {"agreement_30": [30, .05], "agreement_1000": [1000, .03]},
                "normalization": {"alphas": [0., .1, .3, 1.], "fixed_lambda": .01,
                                  "norm_dims": [1, 3], "eps": EPS,
                                  "zero_norm": "zero normalized agreement addition"}}
    protocol = {"experiment": "multi_task_gradient_normalized_decoder_agreement", "profiles": profiles,
                "parent_protocol_sha256": sha(PARENT / "protocol.json"),
                "combined_protocol_sha256": sha(old_protocol_path),
                "source_hashes": {str(path): sha(path) for path in source_files},
                "control_source_hashes": control_sources, "settings": settings,
                "objective": "task BCE gradient plus per-ordinal normalized agreement gradient; coefficients detached",
                "search_selection": "last iterate for every method; no gold or final-eval selection",
                "gradient": "fresh MLP weights frozen for each z update; no inner optimizer differentiation",
                "initialization": "old combined artifact initial z, checked against paired decoder-agreement prior",
                "sampling": "same seeds/data and paired MLP initialization as combined task agreement",
                "seed_formulas": old_protocol["seed_formulas"],
                "scope": old_protocol["scope"],
                "interpretation": "report every predeclared variant; do not select alpha/lambda using final test or ideal. Primary comparisons are each new variant versus task_only and prior, with agreement_30 secondary; report conditional paired CIs and per-gap directions. An isolated positive gap result is exploratory and does not support a broad claim.",
                "control_reuse": "agreement_30, agreement_1000, and prior copied byte-for-tensor from hashed completed combined search"}
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("existing protocol differs; use a new output directory")
    else:
        write_json(path, protocol)
    print(f"Protocol ready: {path}", flush=True)


def search(models, initial, condition, tasks, start, stop, settings, device):
    z = torch.nn.Parameter(initial[None].expand(len(VARIANTS), -1, -1, -1).clone())
    project(z)
    optimizer = torch.optim.Adam([z], lr=settings["z_lr"])
    validation = dataset(tasks, settings["search_val_samples"], 330000003, device)
    history = []
    for outer in range(settings["outer_steps"]):
        tick = time.monotonic()
        with torch.no_grad():
            soft_before, hard_before = decode(models, z, condition)
        bank = TaskBank(tasks, len(VARIANTS), 2, start, stop, 310000003 + outer, device)
        train_x, train_y = dataset(tasks, settings["inner_steps"] * settings["batch_size"],
                                  320000003 + outer, device)
        inner = train_bank(bank, soft_before, train_x, train_y, settings["inner_steps"], settings["batch_size"])
        del train_x, train_y
        for parameter in bank.parameters():
            parameter.requires_grad_(False); parameter.grad = None
        frozen_weights = [parameter.detach().clone() for parameter in bank.parameters()]
        with torch.no_grad():
            soft_bce_before = per_network_bce(bank(validation[0], soft_before), validation[1]).mean(2)
            hard_bce_before = per_network_bce(bank(validation[0], hard_before), validation[1]).mean(2)
        optimizer.zero_grad(set_to_none=True)
        soft, _ = decode(models, z, condition)
        task_loss = per_network_bce(bank(validation[0], soft), validation[1]).mean((1, 2))
        pair_loss = agreement(soft)
        task_grad = torch.autograd.grad(task_loss.sum(), z, retain_graph=True)[0]
        agreement_grad = torch.autograd.grad(pair_loss.sum(), z)[0]
        direction, diagnostics = combine_gradients(task_grad, agreement_grad)
        if not bool(torch.isfinite(direction).all()):
            raise RuntimeError("non-finite manual z gradient")
        z_before = z.detach().clone()
        z.grad = direction
        optimizer.step(); project(z)
        step = z.detach() - z_before
        with torch.no_grad():
            soft_after, hard_after = decode(models, z, condition)
            soft_bce_after = per_network_bce(bank(validation[0], soft_after), validation[1]).mean(2)
            hard_bce_after = per_network_bce(bank(validation[0], hard_after), validation[1]).mean(2)
        assert all(torch.equal(p, old) and p.grad is None for p, old in zip(bank.parameters(), frozen_weights))
        row = {"outer": outer, "task_bce": _history_value(task_loss), "pair_mse": _history_value(pair_loss),
               "task_grad_norm": _history_value(diagnostics["task_grad_norm"]),
               "pair_grad_norm": _history_value(diagnostics["agreement_grad_norm"]),
               "gradient_cosine": _history_value(diagnostics["cosine"]),
               "negative_cosine": _history_value(diagnostics["negative_cosine"]),
               "effective_coefficient": _history_value(diagnostics["effective_coefficient"]),
               "weighted_agreement_ratio": _history_value(diagnostics["weighted_agreement_ratio"]),
               "valid_gradient_norms": _history_value(diagnostics["valid_norms"]),
               "projected_step_dot_task_grad": _history_value((step * task_grad).sum((1, 3))),
               "projected_step_dot_agreement_grad": _history_value((step * agreement_grad).sum((1, 3))),
               "validation_soft_bce_before": _history_value(soft_bce_before),
               "validation_hard_bce_before": _history_value(hard_bce_before),
               "validation_soft_bce_after": _history_value(soft_bce_after),
               "validation_hard_bce_after": _history_value(hard_bce_after),
               "inner_bce_mean": float(inner.mean()), "seconds": time.monotonic() - tick}
        history.append(row)
        print(f"[search] outer={outer+1}/{settings['outer_steps']} seconds={row['seconds']:.1f} "
              f"BCE={task_loss.detach().mean(1).tolist()} pair={pair_loss.detach().mean(1).tolist()}", flush=True)
        del bank, frozen_weights, soft, soft_before, hard_before, soft_after, hard_after
        del task_loss, pair_loss, task_grad, agreement_grad, direction, step
    with torch.no_grad():
        soft, hard = decode(models, z, condition)
    return {"z": z.detach().cpu(), "soft": soft.cpu(), "hard": hard.cpu(), "history": history}


def evaluate(masks, tasks, start, stop, settings, device, destination, metadata, suffix=""):
    masks = masks.to(device)
    x_test, y_test = dataset(tasks, settings["evaluation_samples"], 430000003, device)
    records = []
    for seed in settings["evaluation_seeds"]:
        path = destination / f"evaluation{suffix}_seed{seed}.pt"
        if path.exists():
            result = torch.load(path, weights_only=True, map_location="cpu")
            assert result["metadata"] == metadata
            records.append(result); continue
        tick = time.monotonic()
        bank = TaskBank(tasks, masks.shape[0], 2, start, stop, 410000003 + seed, device)
        train_x, train_y = dataset(tasks, settings["evaluation_steps"] * settings["batch_size"],
                                  420000003 + seed, device)
        train_bank(bank, masks, train_x, train_y, settings["evaluation_steps"], settings["batch_size"])
        bce, acc = evaluate_bank(bank, masks, x_test, y_test, settings["evaluation_batch"])
        result = {"metadata": metadata, "seed": seed, "bce": bce.cpu(), "acc": acc.cpu()}
        save_torch(path, result); records.append(result)
        print(f"[evaluation{suffix} seed={seed}] seconds={time.monotonic()-tick:.1f} accuracy={acc.mean((1,2,3)).tolist()}", flush=True)
        del bank, train_x, train_y
    return records


def _load_old(profile, gap, shard, initial, tasks, metadata, protocol):
    path = OLD / profile / f"gap{gap}_shard{shard}" / "search.pt"
    assert sha(path) == protocol["control_source_hashes"][str(path)]
    old = torch.load(path, weights_only=True, map_location="cpu")
    assert old["metadata"]["protocol_sha256"] == protocol["combined_protocol_sha256"]
    assert old["metadata"]["profile"] == profile and old["metadata"]["gap"] == gap
    assert old["metadata"]["start"] == metadata["start"] and old["metadata"]["stop"] == metadata["stop"]
    assert old["metadata"]["tasks"] == tasks and torch.equal(old["initial_z"], initial.cpu())
    assert tensor_sha(old["initial_z"]) == tensor_sha(initial.cpu())
    assert old["metadata"]["methods"] == ["task_only", "combined_0p1", "combined_1", "combined_10",
                                               "agreement_30", "agreement_1000", "prior", "ideal"]
    assert set(old["controls"]) == {"agreement_30", "agreement_1000"}
    return old


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for production experiment")
    device = torch.device("cuda")
    configure_compute_device("cuda")
    protocol_path = args.out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    for path, digest in protocol["source_hashes"].items():
        assert sha(path) == digest, f"source changed: {path}"
    spec, settings = protocol["profiles"][args.profile], protocol["settings"]
    assert args.gap in spec["heldout_gaps"] and args.shard in (0, 1)
    tasks, start, stop = spec["tasks"][str(args.gap)], args.shard * 32, (args.shard + 1) * 32
    for key in ("checkpoint", "checkpoint2", "prior", "split"):
        assert sha(spec[key]) == spec[key + "_sha256"]
    models = _validate_checkpoints(Path(spec["checkpoint"]), Path(spec["checkpoint2"]), Path(spec["split"]), device, .1, "importance.pt")[:2]
    frozen = [_freeze(model) for model in models]
    parent = torch.load(spec["prior"], weights_only=True, map_location="cpu")
    gap_index = parent["gaps"].tolist().index(args.gap)
    reconstructed = torch.stack([parent["latents"]["prior"][key].reshape(8, 64, 32)[gap_index, start:stop]
                                 for key in ("z1", "z2")]).to(device)
    condition = models[0].condition([tasks[0]], device=device)
    assert torch.equal(condition, models[1].condition([tasks[0]], device=device))
    destination = args.out / args.profile / f"gap{args.gap}_shard{args.shard}"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol_sha256": sha(protocol_path), "profile": args.profile, "gap": args.gap,
                "start": start, "stop": stop, "tasks": tasks, "methods": list(METHODS),
                "torch_version": str(torch.__version__)}
    old = _load_old(args.profile, args.gap, args.shard, reconstructed, tasks, metadata, protocol)
    initial = old["initial_z"].to(device)
    search_path = destination / "search.pt"
    if search_path.exists():
        saved = torch.load(search_path, weights_only=True, map_location="cpu")
        assert saved["metadata"] == metadata and torch.equal(saved["initial_z"], old["initial_z"])
    else:
        print(f"Starting {args.profile} gap={args.gap} ordinals={start}:{stop} GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} {torch.cuda.get_device_name()}", flush=True)
        combined = search(models, initial, condition, tasks, start, stop, settings, device)
        saved = {"metadata": metadata, "initial_z": initial.cpu(), "combined": combined,
                 "controls": old["controls"], "prior": old["prior"]}
        save_torch(search_path, saved)
    gold = ideal_mask(tasks[0]).float()[None, None, None].expand(1, 2, stop-start, 16, 16)
    hard = torch.cat([saved["combined"]["hard"], saved["controls"]["agreement_30"]["hard"],
                      saved["controls"]["agreement_1000"]["hard"], saved["prior"]["hard"], gold])
    soft = torch.cat([saved["combined"]["soft"], saved["prior"]["soft"]])
    assert hard.shape == (len(METHODS), 2, stop-start, 16, 16)
    assert soft.shape == (len(SOFT_METHODS), 2, stop-start, 16, 16)
    assert bool(((hard == 0) | (hard == 1)).all()) and bool((hard.sum((-1, -2)) == 96).all())
    base_metadata = {**metadata, "search_sha256": sha(search_path)}
    hard_metadata = {**base_metadata, "hard_masks_sha256": tensor_sha(hard)}
    soft_metadata = {**base_metadata, "methods": list(SOFT_METHODS), "soft_masks_sha256": tensor_sha(soft)}
    evaluate(hard, tasks, start, stop, settings, device, destination, hard_metadata)
    evaluate(soft, tasks, start, stop, settings, device, destination, soft_metadata, "_soft")
    for index, model in enumerate(models):
        _verify_frozen(model, frozen[index], f"decoder{index}")
    for key in ("checkpoint", "checkpoint2", "prior", "split"):
        assert sha(spec[key]) == spec[key + "_sha256"]
    write_json(destination / "done.json", {**hard_metadata, "soft_evaluation_metadata": soft_metadata,
               "decoder_unchanged": True, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "max_memory_gb": torch.cuda.max_memory_allocated() / 1e9})
    print(f"DONE {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile", choices=("interp", "extrap"))
    parser.add_argument("--gap", type=int)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if args.prepare:
        prepare(args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()

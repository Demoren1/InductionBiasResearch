"""Multi-task latent search: validation BCE plus frozen-decoder agreement.

Each gap has one latent pair per restart, shared by all eight motif tasks.
MLPs are independent per task, decoder and restart; their initialization and
training examples are paired across objective variants. This is a direct
mask gradient through freshly trained MLPs, not an unrolled hypergradient.
Gold and final evaluation are used only after all search masks are saved.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import configure_compute_device, ideal_mask, make_dataset
from evaluation.decoder_pair_search import _freeze, _verify_frozen, _validate_checkpoints
from evaluation.eval_generated_masks import _seed
from evaluation.oracle_ideal import align_columns, hard_topk, soft_topk

PARENT = ROOT / "outputs/decoder_agreement/20260906"
DEFAULT_OUT = ROOT / "outputs/combined_task_agreement/20260906"
LAMBDAS = [0., .1, 1., 10.]
METHODS = ["task_only", "combined_0p1", "combined_1", "combined_10",
           "agreement_30", "agreement_1000", "prior", "ideal"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def prepare(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    parent = json.loads((PARENT / "protocol.json").read_text())
    profiles = {}
    for name, spec in parent["profiles"].items():
        split = json.loads(Path(spec["split"]).read_text())
        checkpoint2 = PARENT / name / "vae_43/best.pt"
        prior = PARENT / name / "search/masks.pt"
        assert sha(spec["checkpoint"]) == spec["checkpoint_sha256"]
        assert sha(spec["split"]) == spec["split_sha256"]
        profiles[name] = {**spec, "checkpoint2": str(checkpoint2),
                          "checkpoint2_sha256": sha(checkpoint2),
                          "prior": str(prior), "prior_sha256": sha(prior),
                          "tasks": {str(g): [t for t in split["test_tasks"]
                                             if config.parse_task(t).gap == g]
                                    for g in spec["heldout_gaps"]}}
        assert all(len(ts) == 8 for ts in profiles[name]["tasks"].values())
    source_files = [Path(__file__), ROOT / "data/generate.py", ROOT / "config.py",
                    ROOT / "evaluation/oracle_ideal.py", ROOT / "models/cvae.py",
                    ROOT / "evaluation/decoder_pair_search.py",
                    ROOT / "evaluation/eval_generated_masks.py"]
    protocol = {"experiment": "multi_task_BCE_plus_decoder_agreement", "profiles": profiles,
                "parent_protocol_sha256": sha(PARENT / "protocol.json"),
                "source_hashes": {str(p): sha(p) for p in source_files},
                "settings": {"n_starts": 64, "shards_per_gap": 2, "lambdas": LAMBDAS,
                             "methods": METHODS, "outer_steps": 30, "inner_steps": 400,
                             "batch_size": 128, "mlp_lr": .001, "z_lr": .05,
                             "temperature": .5, "radius": 8., "search_val_samples": 1024,
                             "evaluation_seeds": [0, 1, 2], "evaluation_steps": 1000,
                             "evaluation_samples": 2048, "evaluation_batch": 256,
                             "agreement_controls": {"agreement_30": [30, .05],
                                                    "agreement_1000": [1000, .03]}},
                "objective": "mean validation BCE over 2 decoders and 8 motif tasks + lambda * aligned soft mask MSE; sum independent restart/variant losses for backward",
                "search_selection": "last iterate for every method; no gold or final-eval selection",
                "gradient": "partial derivative through live mask; trained MLP weights frozen for each z update; no inner optimizer differentiation",
                "initialization": "full 64-ordinal CPU bank per task and outer/eval seed, then slice shard; identical MLP weights across methods and decoders",
                "sampling": "one balanced matched-hard-negative dataset per task/outer with inner_steps*batch_size examples; sequential minibatches; common to all methods/shards",
                "seed_formulas": {"search_mlp": "_seed(task, 310000003 + outer)",
                                  "search_train": "_seed(task, 320000003 + outer)",
                                  "search_validation": "_seed(task, 330000003)",
                                  "eval_mlp": "_seed(task, 410000003 + eval_seed)",
                                  "eval_train": "_seed(task, 420000003 + eval_seed)",
                                  "eval_test": "_seed(task, 430000003)"},
                "scope": "supervised adaptation to eight target motif tasks jointly at a CVAE-heldout gap; evaluation uses independent draws from the same finite populations, not disjoint input support or unseen motif tasks",
                "lambda_selection": "report all predeclared values; do not select a winning lambda on test or ideal IoU"}
    path = out / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("existing protocol differs; use a new output directory")
    else:
        write_json(path, protocol)
    print(f"Protocol ready: {path}", flush=True)


class TaskBank(torch.nn.Module):
    """Independent weights [variant, decoder, task, ordinal, ...]."""
    def __init__(self, tasks, variants, decoders, start, stop, seed_base, device):
        super().__init__()
        self.shape = (variants, decoders, len(tasks), stop - start)
        rows = {name: [] for name in ("w1", "b1", "w2", "b2")}
        for task in tasks:
            generator = torch.Generator().manual_seed(_seed(task, seed_base))
            rows["w1"].append((torch.randn(64, 16, 16, generator=generator) * .1)[start:stop])
            rows["b1"].append(torch.zeros(stop-start, 16))
            rows["w2"].append((torch.randn(64, 16, generator=generator) * .1)[start:stop])
            rows["b2"].append(torch.zeros(stop-start))
        for name, values in rows.items():
            base = torch.stack(values).to(device)
            value = base[None, None].expand(variants, decoders, *base.shape).clone()
            self.register_parameter(name, torch.nn.Parameter(value))

    def forward(self, x, masks):
        # x [task,batch,input], masks [variant,decoder,ordinal,input,hidden].
        weight = self.w1 * masks[:, :, None]
        hidden = torch.matmul(x[None, None, :, None], weight)
        hidden = F.relu(hidden + self.b1[..., None, :])
        return (hidden * self.w2[..., None, :]).sum(-1) + self.b2[..., None]


def per_network_bce(logits, labels):
    target = labels[None, None, :, None, :].expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)


def dataset(tasks, samples, seed_base, device):
    rows = [make_dataset(task, samples, _seed(task, seed_base)) for task in tasks]
    return (torch.stack([r["x"] for r in rows]).to(device),
            torch.stack([r["y"] for r in rows]).to(device))


def train_bank(bank, masks, x, y, steps, batch_size):
    optimizer = torch.optim.Adam(bank.parameters(), lr=.001)
    assert x.shape[1] == steps * batch_size
    fixed = masks.detach()
    for step in range(steps):
        lo, hi = step * batch_size, (step + 1) * batch_size
        loss = per_network_bce(bank(x[:, lo:hi], fixed), y[:, lo:hi])
        optimizer.zero_grad(set_to_none=True)
        loss.sum().backward()
        optimizer.step()
    if not bool(torch.isfinite(loss).all()):
        raise RuntimeError("non-finite inner loss")
    return loss.detach()


def decode(models, z, condition):
    a, _, n, latent = z.shape
    c = condition.expand(a * n, -1)
    logits = torch.stack([m.decode(z[:, d].reshape(a*n, latent), c).reshape(a, n, 256)
                          for d, m in enumerate(models)], dim=1)
    soft = soft_topk(logits, 96, .5).reshape(a, 2, n, 16, 16)
    hard = hard_topk(logits, 96).reshape(a, 2, n, 16, 16)
    return soft, hard


def agreement(soft):
    a, _, n, _, _ = soft.shape
    left, right = soft[:, 0].reshape(a*n, 16, 16), soft[:, 1].reshape(a*n, 16, 16)
    return (left - align_columns(left, right)).square().mean((1, 2)).reshape(a, n)


def project(z):
    with torch.no_grad():
        z.mul_((8. / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.))


def search(models, initial, condition, tasks, start, stop, settings, device):
    z = torch.nn.Parameter(initial[None].expand(len(LAMBDAS), -1, -1, -1).clone())
    project(z)
    optimizer = torch.optim.Adam([z], lr=settings["z_lr"])
    validation = dataset(tasks, settings["search_val_samples"], 330000003, device)
    weights = torch.tensor(LAMBDAS, device=device)[:, None]
    history = []
    for outer in range(settings["outer_steps"]):
        tick = time.monotonic()
        with torch.no_grad():
            soft, _ = decode(models, z, condition)
        bank = TaskBank(tasks, len(LAMBDAS), 2, start, stop, 310000003 + outer, device)
        train_x, train_y = dataset(tasks, settings["inner_steps"]*settings["batch_size"],
                                  320000003 + outer, device)
        inner = train_bank(bank, soft, train_x, train_y, settings["inner_steps"], settings["batch_size"])
        del train_x, train_y
        for p in bank.parameters():
            p.requires_grad_(False); p.grad = None
        before = [p.detach().clone() for p in bank.parameters()]
        optimizer.zero_grad(set_to_none=True)
        soft, _ = decode(models, z, condition)
        task_loss = per_network_bce(bank(validation[0], soft), validation[1]).mean((1, 2))
        pair_loss = agreement(soft)
        objective = task_loss + weights * pair_loss
        task_grad = torch.autograd.grad(task_loss.sum(), z, retain_graph=True)[0]
        pair_grad = torch.autograd.grad(pair_loss.sum(), z, retain_graph=True)[0]
        objective.sum().backward()
        if z.grad is None or not bool(torch.isfinite(z.grad).all()):
            raise RuntimeError("missing/non-finite z gradient")
        optimizer.step(); project(z)
        assert all(torch.equal(p, b) and p.grad is None for p, b in zip(bank.parameters(), before))
        row = {"outer": outer, "task_bce": task_loss.detach().cpu().tolist(),
               "pair_mse": pair_loss.detach().cpu().tolist(),
               "objective": objective.detach().cpu().tolist(),
               "task_grad_norm": task_grad.flatten(1).norm(dim=1).cpu().tolist(),
               "pair_grad_norm": pair_grad.flatten(1).norm(dim=1).cpu().tolist(),
               "inner_bce_mean": float(inner.mean()), "seconds": time.monotonic()-tick}
        history.append(row)
        print(f"[search] outer={outer+1}/{settings['outer_steps']} seconds={row['seconds']:.1f} "
              f"BCE={task_loss.detach().mean(1).tolist()} pair={pair_loss.detach().mean(1).tolist()}", flush=True)
        del bank, before, soft, objective, task_loss, pair_loss, task_grad, pair_grad
    with torch.no_grad():
        soft, hard = decode(models, z, condition)
    return {"z": z.detach().cpu(), "soft": soft.cpu(), "hard": hard.cpu(), "history": history}


def pair_control(models, initial, condition, steps, lr):
    z = torch.nn.Parameter(initial[None].clone())
    project(z)
    optimizer = torch.optim.Adam([z], lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        soft, _ = decode(models, z, condition)
        loss = agreement(soft)
        loss.sum().backward(); optimizer.step(); project(z)
        if step % 100 == 0 or step + 1 == steps:
            history.append({"step": step, "mse": float(loss.mean().detach())})
            print(f"[pair {steps}] {step+1}/{steps} MSE={history[-1]['mse']:.6f}", flush=True)
    with torch.no_grad():
        soft, hard = decode(models, z, condition)
    return {"z": z.detach().cpu(), "soft": soft.cpu(), "hard": hard.cpu(), "history": history}


@torch.no_grad()
def evaluate_bank(bank, masks, x, y, batch_size):
    count = x.shape[1]
    bce = torch.zeros(bank.shape, device=x.device)
    accuracy = torch.zeros_like(bce)
    for lo in range(0, count, batch_size):
        hi = min(count, lo + batch_size)
        logits = bank(x[:, lo:hi], masks)
        bce += per_network_bce(logits, y[:, lo:hi]) * (hi-lo)
        target = y[None, None, :, None, lo:hi]
        accuracy += ((logits > 0) == target.bool()).sum(-1)
    return bce / count, accuracy / count


def evaluate(hard, tasks, start, stop, settings, device, destination, metadata):
    masks = hard.to(device)
    x_test, y_test = dataset(tasks, settings["evaluation_samples"], 430000003, device)
    records = []
    for seed in settings["evaluation_seeds"]:
        path = destination / f"evaluation_seed{seed}.pt"
        if path.exists():
            result = torch.load(path, weights_only=True)
            assert result["metadata"] == metadata
            records.append(result); continue
        tick = time.monotonic()
        bank = TaskBank(tasks, len(METHODS), 2, start, stop, 410000003 + seed, device)
        train_x, train_y = dataset(tasks, settings["evaluation_steps"]*settings["batch_size"],
                                  420000003 + seed, device)
        train_bank(bank, masks, train_x, train_y, settings["evaluation_steps"], settings["batch_size"])
        bce, acc = evaluate_bank(bank, masks, x_test, y_test, settings["evaluation_batch"])
        result = {"metadata": metadata, "seed": seed, "bce": bce.cpu(), "acc": acc.cpu()}
        torch.save(result, path); records.append(result)
        print(f"[evaluation seed={seed}] seconds={time.monotonic()-tick:.1f} "
              f"accuracy={acc.mean((1,2,3)).tolist()}", flush=True)
        del bank, train_x, train_y
    return records


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
    tasks = spec["tasks"][str(args.gap)]
    start, stop = args.shard * 32, (args.shard + 1)*32
    for key in ("checkpoint", "checkpoint2", "prior", "split"):
        assert sha(spec[key]) == spec[key + "_sha256"]
    models = _validate_checkpoints(Path(spec["checkpoint"]), Path(spec["checkpoint2"]),
                                   Path(spec["split"]), device, .1, "importance.pt")[:2]
    frozen = [_freeze(m) for m in models]
    parent = torch.load(spec["prior"], weights_only=True, map_location="cpu")
    gap_index = parent["gaps"].tolist().index(args.gap)
    initial = torch.stack([parent["latents"]["prior"][key].reshape(8,64,32)[gap_index, start:stop]
                           for key in ("z1", "z2")]).to(device)
    condition = models[0].condition([tasks[0]], device=device)
    assert torch.equal(condition, models[1].condition([tasks[0]], device=device))
    destination = args.out / args.profile / f"gap{args.gap}_shard{args.shard}"
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {"protocol_sha256": sha(protocol_path), "profile": args.profile, "gap": args.gap,
                "start": start, "stop": stop, "tasks": tasks, "methods": METHODS,
                "torch_version": str(torch.__version__)}
    search_path = destination / "search.pt"
    if search_path.exists():
        saved = torch.load(search_path, weights_only=True, map_location="cpu")
        assert saved["metadata"] == metadata
    else:
        print(f"Starting {args.profile} gap={args.gap} ordinals={start}:{stop} "
              f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} {torch.cuda.get_device_name()}", flush=True)
        combined = search(models, initial, condition, tasks, start, stop, settings, device)
        controls = {name: pair_control(models, initial, condition, steps, lr)
                    for name, (steps, lr) in settings["agreement_controls"].items()}
        with torch.no_grad():
            prior_soft, prior_hard = decode(models, initial[None], condition)
        saved = {"metadata": metadata, "initial_z": initial.cpu(), "combined": combined,
                 "controls": controls, "prior": {"soft": prior_soft.cpu(), "hard": prior_hard.cpu()}}
        torch.save(saved, search_path)
    # Only after all search results are fixed, materialize ideal for diagnostics.
    gold = ideal_mask(tasks[0]).float()[None,None,None].expand(1,2,stop-start,16,16)
    hard = torch.cat([saved["combined"]["hard"], saved["controls"]["agreement_30"]["hard"],
                      saved["controls"]["agreement_1000"]["hard"], saved["prior"]["hard"], gold])
    assert hard.shape == (len(METHODS), 2, stop-start, 16, 16)
    assert bool(((hard == 0) | (hard == 1)).all()) and bool((hard.sum((-1,-2)) == 96).all())
    eval_metadata = {**metadata, "search_sha256": sha(search_path),
                     "hard_masks_sha256": hashlib.sha256(hard.contiguous().numpy().tobytes()).hexdigest()}
    evaluate(hard, tasks, start, stop, settings, device, destination, eval_metadata)
    for d, model in enumerate(models):
        _verify_frozen(model, frozen[d], f"decoder{d}")
    for key in ("checkpoint", "checkpoint2", "prior", "split"):
        assert sha(spec[key]) == spec[key + "_sha256"]
    write_json(destination / "done.json", {**eval_metadata, "decoder_unchanged": True,
                                           "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                           "max_memory_gb": torch.cuda.max_memory_allocated()/1e9})
    print(f"DONE {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--profile", choices=("interp", "extrap"))
    parser.add_argument("--gap", type=int)
    parser.add_argument("--shard", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if args.prepare:
        prepare(args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()

"""Paired few-shot refits with public query inputs and separately scored answers."""

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np
import torch

from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .optimization import Layout, initialization, losses, predict
from .trajectory_data import LOCAL, PLAN, ROOT, episode_indices


def public_panels(dataset, phase):
    data_dir = LOCAL / dataset
    public = data_dir / phase
    public.mkdir(exist_ok=True)
    archive = public / "answers.npz"
    if archive.exists():
        return public
    episode_indices(dataset, phase)
    with (
        np.load(data_dir / f"{phase}_indices.npz") as f,
        np.load(data_dir / "records.npz") as data,
    ):
        x, y = data["x"], data["y"]
        support, query, units = f["support"], f["query"], f["units"]
        for i, start in enumerate(range(0, len(units), 32)):
            s, q = support[start : start + 32], query[start : start + 32]
            np.savez(public / f"public_{i}.npz", sx=x[s], sy=y[s], qx=x[q])
        np.savez_compressed(archive, qy=y[query], units=units)
    return public


def refit(sx, sy, qx, width, index, pretrained, steps, lr, seed):
    layout = Layout(1, sx.shape[-1], width, 10)
    if index is None:
        size = layout.connections
    else:
        labels = torch.as_tensor(index.ravel(), dtype=torch.long, device=sx.device)
        size = int(abs(index).max())
        position, sign = labels.abs() - 1, labels.sign()
    if pretrained is None:
        parameter = initialization(layout, size, 1, len(sx), sx.device, seed)
    else:
        initial = torch.as_tensor(pretrained, device=sx.device)[None, None]
        parameter = initial.expand(len(sx), -1, -1).clone().requires_grad_()
    optimizer = torch.optim.Adam([parameter], lr=lr)
    requested = set(steps)
    last = max(requested)
    result = {}
    for i in range(last + 1):
        w, tail = parameter[..., :size], parameter[..., size:]
        if index is not None:
            w = w[..., position] * sign
        if i in requested:
            with torch.no_grad():
                result[i] = predict(qx, w, tail, layout)[:, 0].softmax(-1).cpu().numpy()
        if i == last:
            break
        loss = losses(predict(sx, w, tail, layout), sy).mean(-1).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return result


def predict_batches(gpu, batches, dataset, phase, study, shots, manifest):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    public = LOCAL / dataset / phase
    cache = LOCAL / dataset / study
    with np.load(ROOT / dataset / manifest["bank"]) as models:
        indices = {k: models[k] for k in manifest["methods"] if k in models}
    with np.load(ROOT / dataset / "pooled.npz") as f:
        warm = f["parameter"]
    for b in batches:
        output = cache / f"pred_{b}.npz"
        if output.exists():
            continue
        with np.load(public / f"public_{b}.npz") as f:
            x = f["sx"][:, :, :shots]
            sx = torch.as_tensor(x.reshape(len(x), -1, 1, x.shape[-1]), device=device)
            sy = torch.as_tensor(
                f["sy"][:, :, :shots].reshape(len(x), -1), device=device
            )
            qx = torch.as_tensor(f["qx"][:, :, None, :], device=device)
        predictions = {}
        for name, spec in manifest["methods"].items():
            if spec["kind"] == "index" and name not in indices:
                raise ValueError(f"Missing indexator {name}")
            values = refit(
                sx,
                sy,
                qx,
                spec["width"],
                indices.get(name),
                warm if spec["kind"] == "pooled" else None,
                spec["steps"],
                manifest["lr"],
                manifest["numeric_seed"] + b,
            )
            predictions.update(
                {f"{name}__{step}": prob for step, prob in values.items()}
            )
        tmp = output.with_suffix(".partial.npz")
        np.savez(tmp, **predictions)
        tmp.replace(output)


def evaluate(dataset, study, phase, shots, methods, bank, gpus, decision):
    plan = json.loads(PLAN.read_text())
    folder = ROOT / dataset
    public = public_panels(dataset, phase)
    cache = LOCAL / dataset / study
    cache.mkdir(exist_ok=True)
    manifest = {
        "plan_sha256": digest(PLAN),
        "source_sha256": digest(folder / "source.json"),
        "bank": bank,
        "bank_sha256": digest(folder / bank),
        "decision_sha256": digest(folder / decision) if decision else None,
        "indices_sha256": digest(LOCAL / dataset / f"{phase}_indices.npz"),
        "answers_sha256": digest(public / "answers.npz"),
        "refit_sha256": digest(Path(__file__)),
        "numeric_seed": plan["seeds"]["numeric"][phase],
        "shots": shots,
        "lr": plan["network"]["target_lr"],
        "methods": methods,
    }
    receipt = folder / f"{study}_manifest.json"
    if receipt.exists() and json.loads(receipt.read_text()) != manifest:
        raise ValueError(f"Inputs or choices changed for {study}")
    save_json(receipt, manifest)
    scores_path = folder / f"{study}_scores.npz"
    if scores_path.exists():
        return
    batches = len(list(public.glob("public_*.npz")))
    with ProcessPoolExecutor(
        max_workers=len(gpus), mp_context=mp.get_context("spawn")
    ) as pool:
        list(
            pool.map(
                predict_batches,
                gpus,
                [list(range(i, batches, len(gpus))) for i in range(len(gpus))],
                repeat(dataset),
                repeat(phase),
                repeat(study),
                repeat(shots),
                repeat(manifest),
            )
        )
    save_json(
        folder / f"{study}_predictions.json",
        {str(b): digest(cache / f"pred_{b}.npz") for b in range(batches)},
    )
    with np.load(public / "answers.npz") as f:
        units, query = f["units"], f["qy"]
    scores = {}
    for b in range(batches):
        y = query[b * 32 : (b + 1) * 32]
        with np.load(cache / f"pred_{b}.npz") as f:
            for name in f.files:
                probability = f[name]
                p = np.take_along_axis(probability, y[..., None], axis=-1)[..., 0]
                acc = (probability.argmax(-1) == y).mean(-1)
                brier = ((probability**2).sum(-1) - 2 * p + 1).mean(-1) / 2
                ce = -np.log(np.maximum(p, 1e-30)).mean(-1)
                scores.setdefault(name, []).append(np.stack([acc, brier, ce], axis=1))
    np.savez_compressed(
        scores_path, units=units, **{k: np.concatenate(v) for k, v in scores.items()}
    )
    print(
        dataset, study, len(units), "episodes,", len(scores), "evaluations", flush=True
    )

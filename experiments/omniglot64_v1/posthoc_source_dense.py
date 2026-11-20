"""Equal-budget Dense comparison with supervised access to the same source tasks."""

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import posthoc_model as model
from .posthoc_budget import BATCH, CURVE, Run
from .posthoc_scores import interval, metrics
from .posthoc_source import PLAN, ROOT, WIDTHS, C, D, H, sha, write_json

SPEC = Path(__file__).with_name("posthoc_source_dense_plan.json")
CFG = json.loads(SPEC.read_text())
OUT = ROOT / "source_dense"


def source_fit(job):
    gpu, width, restart = job
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    gen = torch.Generator(device=device).manual_seed(680183 + 2017 * restart + width)
    folder = ROOT / "source"
    shots = PLAN["source"]["fit_shots"]
    sx, sy, qx = (
        torch.tensor(np.load(folder / str(shots) / f"{k}.npy"), device=device)
        for k in ("sx", "sy", "qx")
    )
    qy = torch.tensor(np.load(folder / f"answers_{shots}.npy"), device=device).long()
    sx, sy, qx = sx.float() / 255, sy.long(), qx.float() / 255
    w = torch.randn(D, width, generator=gen, device=device) * math.sqrt(2 / D)
    b = torch.randn(width, generator=gen, device=device) * 0.1
    v = torch.randn(len(sx), width, C, generator=gen, device=device)
    a = torch.zeros(len(sx), 1, C, device=device)
    opt = torch.optim.Adam(
        [x.requires_grad_() for x in (w, b, v, a)], lr=CFG["source_lr"], fused=True
    )

    def forward(x, heads, bias):
        return ((x @ w + b).relu() @ heads) / math.sqrt(width) + bias

    best, curve = math.inf, []
    for i in range(CFG["source_updates"] + 1):
        if i % CFG["source_stopping_every"] == 0:
            with torch.no_grad():
                q = forward(qx, v, a)
                p = forward(sx, v, a)
                loss = F.cross_entropy(q.flatten(0, 1), qy.flatten()).item()
                curve.append(
                    [
                        i,
                        F.cross_entropy(p.flatten(0, 1), sy.flatten()).item(),
                        (p.argmax(-1) == sy).float().mean().item(),
                        loss,
                        (q.argmax(-1) == qy).float().mean().item(),
                    ]
                )
                if loss < best:
                    best, step = loss, i
                    saved = (
                        w.detach().cpu().numpy().copy(),
                        b.detach().cpu().numpy().copy(),
                    )
        if i == CFG["source_updates"]:
            break
        ids = torch.randperm(len(sx), generator=gen, device=device)[
            : CFG["tasks_per_update"]
        ]
        loss = F.cross_entropy(
            forward(sx[ids], v[ids], a[ids]).flatten(0, 1), sy[ids].flatten()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    np.savez(
        OUT / f"source_{width}_{restart}.npz",
        w=saved[0],
        b=saved[1],
        ce=best,
        step=step,
        curve=np.asarray(curve),
    )


def bank(gpus):
    OUT.mkdir(exist_ok=True)
    jobs = [(width, r) for width in WIDTHS for r in range(CFG["starts"])]
    for i in range(0, len(jobs), len(gpus)):
        wave = jobs[i : i + len(gpus)]
        with mp.get_context("spawn").Pool(len(wave)) as pool:
            pool.map(source_fit, [(gpu, *job) for gpu, job in zip(gpus, wave)])
    chosen = {}
    for width in WIDTHS:
        trials = []
        for r in range(CFG["starts"]):
            with np.load(OUT / f"source_{width}_{r}.npz") as data:
                trials.append((float(data["ce"]), r, int(data["step"])))
        ce, r, step = min(trials)
        chosen[str(width)] = {"restart": r, "step": step, "ce": ce}
    write_json(
        OUT / "source_frozen.json",
        {
            "plan_sha256": sha(SPEC),
            "source_draws_sha256": sha(ROOT / "source" / "draws.npz"),
            "choices": chosen,
        },
    )
    print("Source-trained Dense", chosen, flush=True)


def family(target, ridge, rate):
    return f"{target}_{round(ridge * 1000)}_{round(rate * 1000)}"


def fit_batch(stage, shot, offset, gpu, source, selected, base):
    run = Run(stage, shot, offset, gpu, {}, base[str(shot)])
    sx, qx = run.dense_x
    teacher_start = run.scratch(H, 1, D, H)
    teacher = run.fixed(
        "teacher", base[str(shot)]["teacher"], teacher_start, run.dense_x
    )
    run.out.clear()
    teacher_y = model.predict(sx, teacher)
    hard_y = F.one_hot(run.sy, C).float() * 4
    for width in WIDTHS:
        w, b = source[width]
        state = (
            w.view(1, 1, D, width).expand(len(sx), -1, -1, -1),
            b.view(1, 1, width).expand(len(sx), -1, -1),
            torch.zeros(len(sx), width, C, device=run.device),
            torch.zeros(len(sx), 1, C, device=run.device),
        )
        for target, y in (("labels", hard_y), ("teacher", teacher_y)):
            soft = None if target == "labels" else teacher_y
            for ridge in CFG["ridge"]:
                for rate in CFG["lr"]:
                    prefix = f"w{width}_" + family(target, ridge, rate)
                    if stage == "test":
                        selection = selected[str(shot)][str(width)]
                        if prefix != selection.rsplit(".", 1)[0]:
                            continue
                    init = model.repair(state, sx, y, coefficient=ridge)
                    if stage == "test":
                        steps = int(selection.rsplit(".", 1)[-1])
                        fitted = model.fit(
                            init, sx, run.sy, rate, [steps], lambda *_: None, soft
                        )
                        run.record(
                            f"source_dense_{width}", fitted, (sx, qx), deploy=True
                        )
                    else:

                        def record(step, fitted, prefix=prefix):
                            run.record(f"{prefix}.{step}", fitted, (sx, qx))

                        model.fit(init, sx, run.sy, rate, CURVE, record, soft)
    return run.out


def worker(job):
    gpu, stage, jobs, selected, base, source_cfg = job
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    source = {}
    for width in WIDTHS:
        r = source_cfg[str(width)]["restart"]
        with np.load(OUT / f"source_{width}_{r}.npz") as data:
            source[width] = [
                torch.tensor(data[k], device=f"cuda:{gpu}") for k in ("w", "b")
            ]
    for shot, offset in jobs:
        result = fit_batch(stage, shot, offset, gpu, source, selected, base)
        path = OUT / stage / str(shot) / f"pred_{offset:04d}.npz"
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez(stream, **result)
        path.with_suffix(".tmp").replace(path)


def predict(stage, gpus):
    base = json.loads((ROOT / "teacher_frozen.json").read_text())["choices"]
    source_cfg = json.loads((OUT / "source_frozen.json").read_text())["choices"]
    selected = (
        json.loads((OUT / "policy_frozen.json").read_text())["choices"]
        if stage == "test"
        else {}
    )
    jobs = []
    for shot in PLAN["target_shots"]:
        folder = OUT / stage / str(shot)
        folder.mkdir(parents=True, exist_ok=True)
        count = len(np.load(ROOT / stage / str(shot) / "sy.npy", mmap_mode="r"))
        jobs += [
            (shot, i)
            for i in range(0, count, BATCH)
            if not (folder / f"pred_{i:04d}.npz").exists()
        ]
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        pool.map(
            worker,
            [
                (gpu, stage, jobs[i :: len(gpus)], selected, base, source_cfg)
                for i, gpu in enumerate(gpus)
            ],
        )
    files = sorted((OUT / stage).glob("*/pred_*.npz"))
    write_json(
        OUT / stage / "seal.json",
        {
            "source_sha256": sha(OUT / "source_frozen.json"),
            "choices_sha256": sha(OUT / "policy_frozen.json")
            if stage == "test"
            else None,
            "files": {str(p.relative_to(OUT / stage)): sha(p) for p in files},
        },
    )


def score(stage):
    seal = json.loads((OUT / stage / "seal.json").read_text())
    curves, choices, totals = {}, {}, {}
    for shot in PLAN["target_shots"]:
        sy = np.load(ROOT / stage / f"answers_{shot}.npy")
        stats = {}
        for i in range(0, len(sy), BATCH):
            path = OUT / stage / str(shot) / f"pred_{i:04d}.npz"
            if sha(path) != seal["files"][f"{shot}/{path.name}"]:
                raise RuntimeError(f"Changed predictions at {path}")
            with np.load(path) as data:
                for k in data.files:
                    if k.endswith("_support"):
                        continue
                    value = metrics(data[k], sy[i : i + BATCH])
                    if k + "_support" in data:
                        value["train_ce"], value["train_recall"] = data[
                            k + "_support"
                        ].T
                    for m, v in value.items():
                        stats.setdefault(k, {}).setdefault(m, []).append(v)
        rows = {
            k: {m: np.concatenate(v) for m, v in item.items()}
            for k, item in stats.items()
        }
        means = {
            k: {m: float(v.mean()) for m, v in item.items()} for k, item in rows.items()
        }
        if stage == "policy_dev":
            curves[str(shot)] = means
            choices[str(shot)] = {
                str(width): min(
                    (
                        k
                        for k in means
                        if k.startswith(f"w{width}_")
                        and int(k.rsplit(".", 1)[-1]) in CFG["steps"]
                    ),
                    key=lambda k: (
                        -means[k]["accuracy"],
                        means[k]["ce"],
                        int(k.rsplit(".", 1)[-1]),
                        k,
                    ),
                )
                for width in WIDTHS
            }
        else:
            totals[str(shot)] = {}
            with np.load(ROOT / "episode_metrics.npz") as main:
                for width in WIDTHS:
                    key = f"source_dense_{width}"
                    diff = (
                        main[f"s{shot}_asrr_{width}_accuracy"] - rows[key]["accuracy"]
                    ) * 100
                    totals[str(shot)][str(width)] = {
                        **means[key],
                        "asrr_minus_source_dense_pp": interval(diff, 0.95),
                        "cohort_pp": [float(v.mean()) for v in np.split(diff, 2)],
                    }
    if stage == "policy_dev":
        write_json(OUT / "curves.json", curves)
        write_json(
            OUT / "policy_frozen.json",
            {"choices": choices, "dev_seal_sha256": sha(OUT / stage / "seal.json")},
        )
        print("Policy choices", choices, flush=True)
    else:
        write_json(
            OUT / "results.json",
            {"shots": totals, "test_seal_sha256": sha(OUT / stage / "seal.json")},
        )
        print("Supplement results", totals, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("source", "policy_dev", "test", "score_test"))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    args = parser.parse_args()
    if args.stage == "source":
        bank(args.gpus)
    elif args.stage == "score_test":
        score("test")
    else:
        predict(args.stage, args.gpus)
        if args.stage == "policy_dev":
            score("policy_dev")

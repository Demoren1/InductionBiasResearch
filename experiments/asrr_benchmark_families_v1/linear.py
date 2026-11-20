"""Exact and partial translations with observed output coordinates."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
import torch
from scipy.stats import t

from experiments.msr_reproduction_v1 import linear as msr
from experiments.sharing_structure_research_v1.blind_contract import digest, save_json

from .groups import LOCAL, PLAN, PLAN_PATH, ROOT, shuffled, signed_bank, untied

CFG = PLAN["linear"]
WORK = LOCAL / "linear"
OUT = ROOT / "linear"
D = CFG["inputs"]
WIDTH = CFG["filter_width"]
H = D - WIDTH + 1
CASES = ("shift", "partial_shift")
YEH_LATENTS = (3, 16, 54)


def atomic_npz(path, **arrays):
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def tolerance_interval(values):
    half = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [
        float(v) for v in (values.mean(), values.mean() - half, values.mean() + half)
    ]


def panel(index, count, samples, seed, sigma):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((count, samples, D))
    codes = rng.standard_normal((count, index.max()))
    w = np.column_stack((np.zeros(count), codes))[:, index]
    y = x @ w
    if sigma:
        y += sigma * rng.standard_normal(y.shape)
    return x, y


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for case_no, case in enumerate(CASES):
        folder = WORK / case
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "registration.json").exists():
            continue
        rng = np.random.default_rng(CFG["seed"] + case_no)
        index = np.zeros((D, H), dtype=np.int32)
        code = WIDTH + 1
        for offset in range(WIDTH):
            index[np.arange(H) + offset, np.arange(H)] = offset + 1
            if case == "partial_shift":
                cols = rng.choice(H, CFG["local_locations_per_tap"], replace=False)
                index[cols + offset, cols] = np.arange(code, code + len(cols))
                code += len(cols)
        seed = CFG["seed"] + 77777 * case_no
        x, y = panel(
            index,
            CFG["source_tasks"],
            CFG["source_samples"],
            seed + 53,
            CFG["source_sigma"],
        )
        atomic_npz(folder / "train.npz", x=x.astype(np.float32), y=y.astype(np.float32))
        for name, offset in (("dev", 107), ("test", 163)):
            x, y = panel(
                index,
                CFG[f"{name}_tasks"],
                max(CFG["shots"]) + CFG["query"],
                seed + offset,
                0,
            )
            if name == "dev":
                atomic_npz(
                    folder / "val.npz", x=x.astype(np.float32), y=y.astype(np.float32)
                )
            else:
                support = max(CFG["shots"])
                atomic_npz(folder / "test.npz", x=x, y_support=y[:, :support])
                atomic_npz(folder / "answers.npz", y=y[:, support:], index=index)
        # The teacher is available as a reference model in the separate final scoring phase.
        np.save(folder / "oracle.npy", index)
        save_json(
            folder / "registration.json",
            {
                "examples_per_train_task": CFG["source_samples"],
                "protocol_sha256": digest(PLAN_PATH),
                "data_sha256": {
                    f.name: digest(f) for f in sorted(folder.glob("*.npz"))
                },
            },
        )


def msr_job(args):
    case, latent, gpu = args
    msr.run(WORK / case, "msr", latent, 0.0005, 0, f"cuda:{gpu}")


def fit_msr(gpus):
    jobs = list(product(CASES, CFG["msr_latents"]))
    for i in range(0, len(jobs), len(gpus)):
        wave = jobs[i : i + len(gpus)]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            list(pool.map(msr_job, [(case, k, g) for (case, k), g in zip(wave, gpus)]))


def source_weights(folder):
    with np.load(folder / "train.npz") as data:
        x, y = data["x"].astype(float), data["y"].astype(float)
    return np.linalg.solve(x.transpose(0, 2, 1) @ x, x.transpose(0, 2, 1) @ y)


def magnitude_mask(weights, threshold):
    rms = np.sqrt(np.square(weights).mean(0))
    return rms >= threshold * np.median(rms.max(axis=0))


def yeh_job(args):
    case, m, gpu = args
    folder = WORK / case
    dest = folder / f"yeh_{m}.npz"
    if dest.exists():
        return
    torch.set_num_threads(1)
    device = f"cuda:{gpu}"
    with np.load(folder / "train.npz") as data:
        sx, sy = (torch.tensor(data[k], device=device) for k in ("x", "y"))
    i, h = np.where(magnitude_mask(source_weights(folder), 0.25))
    out = torch.tensor(h, device=device)
    gen = torch.Generator(device=device).manual_seed(91579921 + m)
    logits = (
        0.4 * torch.randn(len(i), m, generator=gen, device=device)
    ).requires_grad_()
    identity = torch.eye(m, device=device)
    opt = torch.optim.Adam([logits], lr=0.03, fused=True)
    for _ in range(800):
        tasks = torch.randperm(len(sx), generator=gen, device=device)[:12]
        take = torch.randperm(sx.shape[1], generator=gen, device=device)[:17]
        target = sy[tasks[:, None], take]
        values = sx[tasks[:, None], take][:, :, i]
        a = logits.softmax(-1)
        design = torch.zeros(len(tasks), len(take), H, m, device=device).index_add_(
            2, out, values[..., None] * a
        )
        inner = 1 if _ % 2 == 0 else 5
        x = design[:, :inner].reshape(len(tasks), -1, m)
        y = target[:, :inner].reshape(len(tasks), -1, 1)
        xt = x.transpose(1, 2)
        ridge = 0.01 * x.square().sum((1, 2)).clamp_min(1e-8) / m
        coef = torch.linalg.solve(xt @ x + ridge[:, None, None] * identity, xt @ y)
        validation = (design[:, inner:] * coef[:, None, None, :, 0]).sum(-1)
        entropy = -(a * logits.log_softmax(-1)).sum(-1).mean()
        loss = (validation - target[:, inner:]).square().mean() + 0.002 * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    a = logits.detach().softmax(-1).cpu().numpy()
    basis = np.zeros((D, H, m))
    basis[i, h] = a
    hard = np.zeros((D, H), dtype=np.int32)
    hard[i, h] = np.unique(a.argmax(-1), return_inverse=True)[1] + 1
    atomic_npz(dest, soft=basis, hard=hard)


def fit_yeh(gpus):
    save_json(
        OUT / "yeh_protocol.json",
        {
            "source_data": "same 96 source tasks; mask from source OLS RMS at threshold 0.25",
            "assignment": "unsigned row softmax; validation gradient through analytic ridge",
            "latent_grid": YEH_LATENTS,
            "learning_rate": 0.03,
            "updates": 800,
            "selection": "MAP and soft with fresh numerical coefficients; choose on development",
        },
    )
    jobs = list(product(CASES, YEH_LATENTS))
    for i in range(0, len(jobs), len(gpus)):
        wave = jobs[i : i + len(gpus)]
        with ProcessPoolExecutor(max_workers=len(wave)) as pool:
            list(pool.map(yeh_job, [(case, k, g) for (case, k), g in zip(wave, gpus)]))


def ridge(a, b, strength):
    energy = np.square(a).sum() / max(a.shape[1], 1)
    if a.shape[0] < a.shape[1]:
        gram = a @ a.T + np.eye(len(a)) * max(energy * strength, 1e-12)
        return a.T @ np.linalg.solve(gram, b)
    gram = a.T @ a + np.eye(a.shape[1]) * max(energy * strength, 1e-12)
    return np.linalg.solve(gram, a.T @ b)


def predict(task_x, support_y, shot, representation, strength):
    sx = task_x[:shot]
    if representation is None:
        weight = ridge(sx, support_y, strength)
    elif representation.ndim == 3:
        design = np.einsum("nd,dhk->nhk", sx, representation, optimize=True)
        codes = ridge(design.reshape(shot * H, -1), support_y.ravel(), strength)
        weight = np.einsum("dhk,k->dh", representation, codes)
    else:
        inp, out = np.nonzero(representation)
        symbols = representation[inp, out]
        ids = np.abs(symbols) - 1
        a = np.zeros((shot, H, int(ids.max()) + 1))
        np.add.at(
            a, (np.arange(shot)[:, None], out, ids), sx[:, inp] * np.sign(symbols)
        )
        codes = ridge(a.reshape(shot * H, -1), support_y.ravel(), strength)
        weight = np.r_[0, codes][np.abs(representation)] * np.sign(representation)
    return task_x[max(CFG["shots"]) :] @ weight


def predictions(x, y, shot, rep, strength):
    return np.stack(
        [
            predict(task_x, yy[:shot], shot, rep, strength)
            for task_x, yy in zip(x, y, strict=True)
        ]
    )


def best(x, y, shot, options):
    scores = {}
    for name, reps in options.items():
        for strength in CFG["ridge_grid"]:
            mse = [
                np.square(
                    predictions(x, y, shot, rep, strength) - y[:, max(CFG["shots"]) :]
                ).mean()
                for rep in reps
            ]
            scores[name, strength] = float(np.mean(mse))
    winner = min(scores, key=lambda key: (scores[key], key))
    return {"name": winner[0], "ridge": winner[1], "mse_dev": scores[winner]}, scores


def banks(folder):
    weights = source_weights(folder)
    partial, ward = {}, {}
    for cut in CFG["relative_support_thresholds"]:
        mask = magnitude_mask(weights, cut)
        partial.update(
            {
                f"partial_{cut:g}_{v:g}": (index,)
                for v, index in signed_bank(
                    weights, mask, CFG["thresholds"], partial=True
                ).items()
            }
        )
        ward.update(
            {
                f"ward_{cut:g}_{v:g}": (index,)
                for v, index in signed_bank(
                    weights, mask, (3, 12, 30, 54, 102, 153), partial=False
                ).items()
            }
        )
    return deduplicate(partial), deduplicate(ward)


def deduplicate(bank):
    unique = {}
    for name, reps in bank.items():
        unique.setdefault(reps[0].tobytes(), (name, reps))
    return dict(unique.values())


def select(folder, output, case):
    with np.load(folder / "val.npz") as data:
        x, y = data["x"].astype(float), data["y"].astype(float)
    partial, ward = banks(folder)
    msr_bank = {}
    for k in CFG["msr_latents"]:
        path = folder / "fits" / f"msr_m{k}_lr0.0005_s0" / "model.npz"
        with np.load(path) as data:
            basis = data["basis"].astype(float).reshape(H, D, k).transpose(1, 0, 2)
        # An orthonormal basis makes fresh ridge invariant to the coordinates of U.
        orthogonal = np.linalg.svd(basis.reshape(D * H, k), full_matrices=False)[0]
        msr_bank[f"msr_{k}"] = (orthogonal.reshape(D, H, k),)
    yeh_bank = {}
    for k in YEH_LATENTS:
        with np.load(folder / f"yeh_{k}.npz") as data:
            for form in ("soft", "hard"):
                yeh_bank[f"yeh_{form}_{k}"] = (data[form],)
    oracle = np.load(folder / "oracle.npy")
    frozen, decisions, dev = {}, {}, {}
    for shot in CFG["shots"]:
        picks, table = {}, {}
        for name, bank in (
            ("partial", partial),
            ("ward", ward),
            ("msr", msr_bank),
            ("yeh", yeh_bank),
        ):
            choice, tr = best(x, y, shot, bank)
            picks[name] = choice
            table[name] = {f"{key}_{r}": score for (key, r), score in tr.items()}
            frozen[f"{shot}_{name}"] = bank[choice["name"]][0]
        selected = frozen[f"{shot}_partial"]
        rng = np.random.default_rng(CFG["seed"] + len(case) + shot)
        controls = {
            "dense": (None,),
            "untied": (untied(selected != 0),),
            "shuffle": tuple(shuffled(selected, rng) for _ in range(4)),
            "oracle": (oracle,),
        }
        for name, reps in controls.items():
            picks[name], _ = best(x, y, shot, {name: reps})
            for i, rep in enumerate(reps):
                if rep is not None:
                    frozen[f"{shot}_{name}_{i}"] = rep
        decisions[str(shot)], dev[str(shot)] = picks, table
    atomic_npz(output / "indices_and_bases.npz", **frozen)
    save_json(output / "dev.json", dev)
    save_json(output / "choices.json", decisions)


def evaluate(folder, output):
    choices = json.loads((output / "choices.json").read_text())
    with np.load(folder / "test.npz") as data:
        x, y = data["x"], data["y_support"]
    with np.load(output / "indices_and_bases.npz") as data:
        frozen = dict(data)
    result = {}
    for shot, methods in choices.items():
        for method, choice in methods.items():
            base = f"{shot}_{method}"
            if method == "dense":
                reps = (None,)
            elif base in frozen:
                reps = (frozen[base],)
            else:
                reps = tuple(v for k, v in frozen.items() if k.startswith(base + "_"))
            result[base] = np.stack(
                [predictions(x, y, int(shot), rep, choice["ridge"]) for rep in reps]
            )
    atomic_npz(output / "predictions.npz", **result)
    save_json(
        output / "seal.json",
        {
            key: digest(output / key)
            for key in ("choices.json", "indices_and_bases.npz", "predictions.npz")
        },
    )


def structural(index, truth):
    keep = (truth != 0) | (index != 0)
    ix, iy = np.triu_indices(keep.sum(), k=1)

    def links(array):
        values = array[keep]
        a, b = values[ix], values[iy]
        return np.where((a != 0) & (np.abs(a) == np.abs(b)), np.sign(a * b), 0)

    a, b = links(index), links(truth)
    correct = np.count_nonzero((a == b) & (a != 0))
    return {
        "mask_errors": int(np.count_nonzero((index != 0) != (truth != 0))),
        "codes": int(np.abs(index).max()),
        "correct_signed_ties": int(correct),
        "proposed_ties": int(np.count_nonzero(a)),
        "teacher_ties": int(np.count_nonzero(b)),
    }


def score(folder, output):
    for name, sha in json.loads((output / "seal.json").read_text()).items():
        if digest(output / name) != sha:
            raise ValueError(f"Frozen linear result changed: {name}")
    with np.load(folder / "answers.npz") as data:
        truth, y = data["index"], data["y"]
    with np.load(output / "predictions.npz") as data:
        errors = {k: np.square(data[k] - y).mean(axis=(0, 2, 3)) for k in data.files}
    atomic_npz(output / "per_task_mse.npz", **errors)
    summary = {}
    with np.load(output / "indices_and_bases.npz") as data:
        for shot in CFG["shots"]:
            head = f"{shot}_"
            rows = {
                key[len(head) :]: v for key, v in errors.items() if key.startswith(head)
            }
            summary[str(shot)] = {
                "mse": {k: float(v.mean()) for k, v in rows.items()},
                "gains_95_ci": {
                    k: tolerance_interval(v - rows["partial"])
                    for k, v in rows.items()
                    if k != "partial"
                },
                "structure": {
                    method: structural(data[head + method], truth)
                    for method in ("partial", "ward")
                },
            }
    save_json(output / "results.json", summary)
    return summary


def run():
    for name in CASES:
        source, output = WORK / name, OUT / name
        output.mkdir(exist_ok=True)
        select(source, output, name)
        evaluate(source, output)
    for name in CASES:
        print(name, json.dumps(score(WORK / name, OUT / name)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "msr", "yeh", "run"))
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(6)))
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "msr":
        fit_msr(args.gpus)
    elif args.phase == "yeh":
        fit_yeh(args.gpus)
    else:
        run()

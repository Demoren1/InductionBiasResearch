"""Measure fresh numerical adaptation with frozen learned DeepSets relations."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

from experiments.paper_structure_benchmark_v1.fitting.adaptation import (
    feature,
    fit_adam,
    signed_basis,
    values,
)
from experiments.paper_structure_benchmark_v1.fitting.numerical import fit_features
from experiments.paper_structure_benchmark_v1.protocol.prepare import (
    numerical_tasks,
    outputs_for,
)
from experiments.paper_structure_benchmark_v1.reporting.evaluate import align_index
from experiments.sharing_structure_research_v1.blind_contract import (
    canonical_index,
    digest,
    save_json,
)

ARTIFACTS = Path("artifacts")
ARCHIVE = ARTIFACTS / "theorem_pair_recovery_weighted_2026-11-12"
ANCHOR = ARTIFACTS / "anchor_start_2026-11-10" / "selected_structures.npz"
OUTPUT = ARTIFACTS / "asrr_deepsets_structure_2026-11-19"
SPEC = Path(__file__).with_name("deepsets_structure_plan.json")
PLAN = json.loads(SPEC.read_text())
CASES = (3, 4, 5)
CONDITIONS = (0, 1)
D, H, K, O = 16, 16, 4, 2
SHUFFLES = tuple(f"shuffle_{i}" for i in range(4))


def learned(number, condition, frozen):
    name = f"standard/case_{number:02d}_c{condition}/selection.npz"
    path = ARCHIVE / name
    if digest(path) != frozen[name]:
        raise ValueError(f"The archived selection changed: {name}")
    with np.load(path) as data:
        return data["signature_only_index"], data["baseline_full_index"]


def shuffled(index, rng):
    result = index.copy()
    for original, destination in zip(index.T, result.T, strict=True):
        active = original != 0
        destination[active] = np.sign(original[active]) * rng.permutation(
            np.abs(original[active])
        )
    return result


def untied(index):
    mask = index != 0
    mask = mask[:, np.lexsort(mask[::-1])]
    result = np.zeros(mask.shape, dtype=np.int16)
    result[mask] = np.arange(1, mask.sum() + 1)
    return result


def teacher(number):
    rng = np.random.default_rng(27111054013 + 100003 * number)
    pixels = rng.permutation(D).reshape(-1, 2)
    index = np.zeros((D, H), dtype=np.int16)
    for position, pair in enumerate(pixels):
        index[np.ix_(pair, [2 * position, 2 * position + 1])] = [[1, 2], [3, 4]]
    columns = rng.permutation(H)
    return index[:, columns], columns


def prepare_case(number, folder, truth_archive, frozen):
    index, columns = teacher(number)
    if not np.array_equal(
        index, truth_archive[f"standard_case_{number:02d}_c0_full_truth"]
    ):
        raise ValueError("Teacher geometry differs from the historical benchmark")
    cfg = PLAN["target"]
    rng = np.random.default_rng(cfg["seed"] + 104729 * number)
    codes, bias, head, offset = numerical_tasks(
        rng, "deepsets", cfg["tasks_per_geometry"], H, O, K
    )
    truth = {}
    for name, count in (
        ("support", max(cfg["support"])),
        ("selection", max(cfg["support"]) // 4),
        ("query", cfg["query"]),
    ):
        x = rng.normal(size=(len(codes), count, D))
        y = np.stack(
            [
                outputs_for(row, index, z, b, v, a)
                for row, z, b, v, a in zip(
                    x, codes, bias[:, columns], head[:, columns], offset, strict=True
                )
            ]
        )
        if name == "query":
            np.save(folder / "query_x.npy", x)
            truth["query_y"] = y
        else:
            np.savez(folder / f"{name}.npz", x=x, y=y)
    np.savez(folder / "answers.npz", **truth, index=index)

    models = {"oracle": canonical_index(index, K)}
    for condition in CONDITIONS:
        estimate, g34 = learned(number, condition, frozen)
        rng = np.random.default_rng(73119319 + number * 101 + condition)
        proposed = {"signature": estimate, "g34": g34}
        proposed.update({name: shuffled(estimate, rng) for name in SHUFFLES})
        for name, array in proposed.items():
            models[f"c{condition}_{name}"] = canonical_index(array, K)
        models[f"c{condition}_untied"] = untied(estimate)

    identity, names = {}, {}
    for name, index in models.items():
        key = identity.setdefault(index.tobytes(), f"m{len(identity):02d}")
        names[name] = key
    if any(len({names[f"c{c}_{k}"] for k in SHUFFLES}) < 2 for c in CONDITIONS):
        raise ValueError("Matched permutations do not change the geometry sufficiently")
    np.savez(folder / "indices.npz", **{names[k]: v for k, v in models.items()})
    save_json(folder / "designs.json", names)


def prepare():
    OUTPUT.mkdir(parents=True)
    frozen = json.loads((ARCHIVE / "selection_freeze.json").read_text())["files"]
    with np.load(ANCHOR) as truth_archive:
        for number in CASES:
            folder = OUTPUT / f"case_{number:02d}"
            folder.mkdir()
            prepare_case(number, folder, truth_archive, frozen)
    inputs = {
        str(p.relative_to(OUTPUT)): digest(p) for p in sorted(OUTPUT.glob("case_*/*"))
    }
    save_json(
        OUTPUT / "registration.json",
        {
            "protocol_sha256": digest(SPEC),
            "inputs_sha256": inputs,
            "historical_selects": [
                p
                for p in frozen
                if p.startswith("standard/case_0")
                and "/selection.npz" in p
                and p.split("/")[1][6] in "345"
            ],
        },
    )


def read_panel(folder, name, n, device):
    with np.load(folder / f"{name}.npz") as panel:
        return tuple(
            torch.tensor(panel[k][:, :n], device=device, dtype=torch.float32)
            for k in ("x", "y")
        )


def fit_one(folder, budget, model, device):
    config = PLAN["fit"]
    if model == "dense":
        codes = D * H
        basis = torch.eye(codes, dtype=torch.float64, device=device).reshape(
            D, H, codes
        )
        adam_basis = None
    else:
        with np.load(folder / "indices.npz") as data:
            index = data[model]
        codes = int(np.abs(index).max())
        basis = signed_basis(index, codes, device)
        adam_basis = basis.float()
    x, y = read_panel(folder, "support", budget, device)
    vx, vy = read_panel(folder, "selection", budget // 4, device)
    qx = torch.tensor(
        np.load(folder / "query_x.npy"), device=device, dtype=torch.float64
    )
    center = y.mean(1, keepdim=True)
    scale = (y - center).square().mean((1, 2), keepdim=True).sqrt().clamp_min(1e-8)
    y, vy = (y - center) / scale, (vy - center) / scale
    initial, trace = fit_adam(
        x,
        y,
        vx,
        vy,
        adam_basis,
        H,
        O,
        steps=config["adam_updates"],
        seed=config["seed"] + budget,
        keep=config["gn_starts"],
    )
    restarts, tasks = initial.shape[:2]
    feat, vfeat = (
        feature(a.double(), basis).repeat(restarts, 1, 1, 1) for a in (x, vx)
    )
    candidate, gn = fit_features(
        feat,
        vfeat,
        initial.flatten(0, 1).double(),
        y.double().repeat(restarts, 1, 1),
        vy.double().repeat(restarts, 1, 1),
        codes,
        steps=config["gn_steps"],
    )
    ranks = gn["numerical_selection_mse"].reshape(restarts, tasks).argmin(axis=0)
    state = candidate.reshape(restarts, tasks, -1)[ranks, np.arange(tasks)]
    query_basis = None if model == "dense" else basis
    logits = values(qx, query_basis, state[None], H, O)[0] * scale + center
    arrays = {"query": logits.cpu().numpy(), "adam_steps": trace["steps"]}
    if model == "dense":
        adam_prediction = values(qx, None, initial[:1].double(), H, O)[0]
        arrays["adam_query"] = (adam_prediction * scale + center).cpu().numpy()
    for stage, log in (("adam", trace), ("gn", gn)):
        for split in ("train", "validation"):
            metric = log[f"{split}_history"].reshape(
                len(log[f"{split}_history"]), -1, tasks
            )
            arrays[f"{stage}_{split}"] = metric
    destination = folder / f"{budget}_{model}.npz"
    temp = destination.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(destination)


def worker(job):
    gpu, tasks = job
    torch.set_num_threads(1)
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    for folder, budget, model in tasks:
        fit_one(folder, budget, model, device)
    print(f"GPU {gpu}: {len(tasks)} completed", flush=True)


def fit(gpus, *, dense=False):
    jobs = []
    if dense:
        registration = json.loads((OUTPUT / "registration.json").read_text())
        if digest(SPEC) != registration["protocol_sha256"]:
            raise ValueError("Fitting plan differs from the original experiment")
    for number in CASES:
        folder = OUTPUT / f"case_{number:02d}"
        if dense:
            for name in ("support.npz", "selection.npz", "query_x.npy"):
                path = folder / name
                key = str(path.relative_to(OUTPUT))
                if digest(path) != registration["inputs_sha256"][key]:
                    raise ValueError(f"Original observations changed: {path}")
            models = ["dense"]
        else:
            names = json.loads((folder / "designs.json").read_text())
            models = sorted(set(names.values()))
        for budget in PLAN["target"]["support"]:
            jobs.extend((folder, budget, name) for name in models)
    pending = [(f, b, n) for f, b, n in jobs if not (f / f"{b}_{n}.npz").exists()]
    with mp.get_context("spawn").Pool(len(gpus)) as pool:
        pool.map(worker, [(g, pending[i :: len(gpus)]) for i, g in enumerate(gpus)])
    save_json(
        OUTPUT
        / ("dense_predictions_sealed.json" if dense else "predictions_sealed.json"),
        {
            str((folder / f"{b}_{n}.npz").relative_to(OUTPUT)): digest(
                folder / f"{b}_{n}.npz"
            )
            for folder, b, n in jobs
        },
    )


def paired(x):
    spread = x.std(ddof=1) * t.ppf(0.975, len(x) - 1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - spread), float(x.mean() + spread)]


def learning_curve(data):
    scores = {}
    steps = data["adam_steps"]
    gn_steps = len(data["gn_validation"])
    for split in ("train", "validation"):
        parts = []
        for stage in ("adam", "gn"):
            validation = data[f"{stage}_validation"]
            best = validation.argmin(axis=1, keepdims=True)
            metric = np.take_along_axis(data[f"{stage}_{split}"], best, axis=1)
            parts.append(metric.mean(axis=(1, 2)))
        scores[split] = np.concatenate(parts).tolist()
    return {
        **scores,
        "steps": np.r_[steps, steps[-1] + np.arange(1, gn_steps + 1)].tolist(),
    }


def plot_learning(curves, *, dense=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = (
        ("signature", "Найденные группы", "#057355"),
        ("untied", "Та же маска, свободные веса", "#265992"),
        ("shuffle_0", "Одна перестановка групп", "#b06a26"),
        ("g34", "G34", "#6750a4"),
    )
    if dense:
        methods += (("dense", "Dense", "#36393e"),)
    for budget in PLAN["target"]["support"]:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6.0), layout="constrained")
        for c, row in zip(CONDITIONS, axes, strict=True):
            for method, label, color in methods:
                cells = [a[str(budget)][f"c{c}_{method}"] for a in curves.values()]
                steps = cells[0]["steps"]
                for split, ax in zip(("train", "validation"), row, strict=True):
                    ys = np.asarray([a[split] for a in cells]).mean(0)
                    ax.plot(steps, np.maximum(ys, 1e-10), color=color, label=label)
            for split, ax in zip(("Обучение", "Численная валидация"), row, strict=True):
                ax.set_title(f"{split} · исходный шум {0.1 * c:g}")
                ax.set_yscale("log")
                ax.set_ylabel("Нормированная MSE")
                ax.axvline(PLAN["fit"]["adam_updates"], color="#888", ls="--", lw=0.7)
                ax.grid(alpha=0.18)
        for ax in axes[-1]:
            ax.set_xlabel("Обновления Adam, затем шаги Гаусса — Ньютона")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        fig.suptitle(
            f"DeepSets · {budget} обучающих примеров на новую функцию", fontsize=13
        )
        tag = "learning_with_dense" if dense else "learning"
        fig.savefig(OUTPUT / f"{tag}_{budget}.png", dpi=155)
        plt.close(fig)


def score():
    seal = json.loads((OUTPUT / "predictions_sealed.json").read_text())
    all_cases, structure, fit_curves = {}, {}, {}
    for number in CASES:
        key = f"case_{number:02d}"
        folder = OUTPUT / key
        names = json.loads((folder / "designs.json").read_text())
        with np.load(folder / "answers.npz") as data:
            answer, truth = data["query_y"], data["index"]
        with np.load(folder / "indices.npz") as data:
            structure[key] = {
                name: align_index(data[names[name]], truth, K)[1]
                for c in CONDITIONS
                for name in (f"c{c}_signature", f"c{c}_g34")
            }
        all_cases[key], fit_curves[key] = {}, {}
        for budget in PLAN["target"]["support"]:
            case = all_cases[key][str(budget)] = {}
            curves = fit_curves[key][str(budget)] = {}
            for model in set(names.values()):
                path = folder / f"{budget}_{model}.npz"
                if digest(path) != seal[f"{key}/{path.name}"]:
                    raise ValueError(f"Prediction has changed: {path}")
                with np.load(path) as data:
                    case[model] = np.mean((data["query"] - answer) ** 2, axis=(1, 2))
                    curves[model] = learning_curve(data)
            all_cases[key][str(budget)] = {
                name: case[model] for name, model in names.items()
            }
            for c in CONDITIONS:
                names_shuffle = [f"c{c}_{name}" for name in SHUFFLES]
                rows = all_cases[key][str(budget)]
                rows[f"c{c}_shuffle"] = np.mean(
                    [rows[name] for name in names_shuffle], axis=0
                )
            fit_curves[key][str(budget)] = {
                name: curves[model]
                for name, model in names.items()
                if name.endswith(("signature", "untied", "g34", "shuffle_0"))
            }
    pooled = {}
    for budget in PLAN["target"]["support"]:
        pooled[str(budget)] = {}
        rows = [case[str(budget)] for case in all_cases.values()]
        for c in CONDITIONS:
            pair = {
                name: np.concatenate([r[f"c{c}_{name}"] for r in rows])
                for name in ("signature", "g34", "shuffle", "untied")
            }
            pair["oracle"] = np.concatenate([r["oracle"] for r in rows])
            pooled[str(budget)][str(c)] = {
                "mean_mse": {m: float(x.mean()) for m, x in pair.items()},
                "gains_95_ci": {
                    m: paired(x - pair["signature"])
                    for m, x in pair.items()
                    if m != "signature"
                },
            }
    save_json(
        OUTPUT / "results.json",
        {
            "pooled": pooled,
            "structure": structure,
            "per_geometry": {
                case: {
                    b: {m: float(x.mean()) for m, x in z.items()} for b, z in y.items()
                }
                for case, y in all_cases.items()
            },
        },
    )
    save_json(OUTPUT / "curves.json", fit_curves)
    plot_learning(fit_curves)
    np.savez_compressed(
        OUTPUT / "function_mse.npz",
        **{
            f"{case}_n{b}_{m}": x
            for case, y in all_cases.items()
            for b, z in y.items()
            for m, x in z.items()
        },
    )
    print(json.dumps(pooled, indent=2), flush=True)


def score_dense():
    seal = json.loads((OUTPUT / "dense_predictions_sealed.json").read_text())
    registration = json.loads((OUTPUT / "registration.json").read_text())
    curves = json.loads((OUTPUT / "curves.json").read_text())
    pooled, by_case, errors = {}, {}, {}
    with np.load(OUTPUT / "function_mse.npz") as original:
        for budget in PLAN["target"]["support"]:
            dense, adam, learned = [], [], {c: [] for c in CONDITIONS}
            for number in CASES:
                key = f"case_{number:02d}"
                folder = OUTPUT / key
                path = folder / f"{budget}_dense.npz"
                if digest(path) != seal[f"{key}/{path.name}"]:
                    raise ValueError(f"Dense prediction changed: {path}")
                answers = folder / "answers.npz"
                if (
                    digest(answers)
                    != registration["inputs_sha256"][f"{key}/answers.npz"]
                ):
                    raise ValueError(f"Original answers changed: {answers}")
                with np.load(answers) as data:
                    target = data["query_y"]
                with np.load(path) as data:
                    gn_mse = np.square(data["query"] - target).mean(axis=(1, 2))
                    adam_mse = np.square(data["adam_query"] - target).mean(axis=(1, 2))
                    curve = learning_curve(data)
                dense.append(gn_mse)
                adam.append(adam_mse)
                errors[f"{key}_n{budget}_dense"] = gn_mse
                errors[f"{key}_n{budget}_adam"] = adam_mse
                by_case.setdefault(key, {})[str(budget)] = {
                    "dense": float(gn_mse.mean()),
                    "adam": float(adam_mse.mean()),
                }
                for c in CONDITIONS:
                    learned[c].append(original[f"{key}_n{budget}_c{c}_signature"])
                    curves[key][str(budget)][f"c{c}_dense"] = curve
            dense, adam = np.concatenate(dense), np.concatenate(adam)
            pooled[str(budget)] = {
                "dense": float(dense.mean()),
                "dense_adam": float(adam.mean()),
                "signature": {
                    c: float(np.concatenate(v).mean()) for c, v in learned.items()
                },
                "dense_minus_signature_95_ci": {
                    c: paired(dense - np.concatenate(v)) for c, v in learned.items()
                },
                "signature_wins": {
                    c: int((dense > np.concatenate(v)).sum())
                    for c, v in learned.items()
                },
            }
    save_json(
        OUTPUT / "dense_results.json",
        {
            "plan_sha256": digest(SPEC),
            "prior_metrics_sha256": digest(OUTPUT / "function_mse.npz"),
            "prediction_seal_sha256": digest(OUTPUT / "dense_predictions_sealed.json"),
            "pooled": pooled,
            "per_geometry": by_case,
        },
    )
    np.savez_compressed(OUTPUT / "dense_function_mse.npz", **errors)
    plot_learning(curves, dense=True)
    print(json.dumps(pooled, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "fit", "score", "fit-dense", "score-dense")
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    options = parser.parse_args()
    if options.phase == "prepare":
        prepare()
    elif options.phase == "score":
        score()
    elif options.phase == "score-dense":
        score_dense()
    else:
        fit(options.gpus, dense=options.phase == "fit-dense")

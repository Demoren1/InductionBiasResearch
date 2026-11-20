"""Freeze development selections, then score sealed held-out predictions."""

import argparse
import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from .posthoc_budget import BATCH, unpack
from .posthoc_source import PLAN, ROOT, WIDTHS, C, D, H, sha, write_json


def metrics(logits, targets, teacher=None):
    logp = logits.astype(float) - logsumexp(
        logits.astype(float), axis=-1, keepdims=True
    )
    pred = logits.argmax(-1)
    result = {
        "accuracy": (pred == targets).mean(-1),
        "ce": -np.take_along_axis(logp, targets[..., None], axis=-1)[:, :, 0].mean(-1),
    }
    if teacher is not None:
        logq = teacher.astype(float) - logsumexp(
            teacher.astype(float), axis=-1, keepdims=True
        )
        result["teacher_kl"] = (np.exp(logq) * (logq - logp)).sum(-1).mean(-1)
        result["agreement"] = (teacher.argmax(-1) == pred).mean(-1)
    return result


def read_stage(stage):
    sealed = json.loads((ROOT / stage / "predictions_sealed.json").read_text())
    for shot in PLAN["target_shots"]:
        folder = ROOT / stage / str(shot)
        targets = np.load(ROOT / stage / f"answers_{shot}.npy")
        rows = {}
        for i in range(0, len(targets), BATCH):
            path = folder / f"pred_{i:04d}.npz"
            if sealed["files"][f"{shot}/{path.name}"] != sha(path):
                raise RuntimeError(f"Changed predictions at {path}")
            with np.load(path) as data:
                teacher = data["teacher"] if stage == "test" else None
                for name in data.files:
                    if name.endswith("_support"):
                        continue
                    result = metrics(data[name], targets[i : i + BATCH], teacher)
                    if name + "_support" in data:
                        trace = data[name + "_support"]
                        result["train_ce"], result["train_recall"] = trace.T
                    group = rows.setdefault(name, {})
                    for metric, value in result.items():
                        group.setdefault(metric, []).append(value)
        yield (
            shot,
            {k: {m: np.concatenate(x) for m, x in v.items()} for k, v in rows.items()},
        )


def rank(item):
    name, values = item
    return (-float(values["accuracy"]), float(values["ce"]), unpack(name)[-1], name)


def winner(rows, family, width, steps):
    options = [
        (key, values)
        for key, values in rows.items()
        if key.startswith(f"{family}.{width}.") and unpack(key)[-1] in steps
    ]
    return min(options, key=rank)[0]


def freeze(stage):
    teacher = stage == "teacher_dev"
    name = "teacher" if teacher else "policy"
    choices = (
        {}
        if teacher
        else json.loads((ROOT / "teacher_frozen.json").read_text())["choices"]
    )
    curves = {}
    for shot, rows in read_stage(stage):
        curves[str(shot)] = avg = {
            key: {m: float(x.mean()) for m, x in vals.items()}
            for key, vals in rows.items()
        }
        if teacher:
            choices[str(shot)] = {
                "teacher": winner(avg, "teacher", H, PLAN["teacher_steps"])
            }
        else:
            for width in WIDTHS:
                picks = {
                    family: winner(avg, family, width, PLAN[steps])
                    for family, steps in (
                        ("narrow", "scratch_steps"),
                        ("prune", "warm_steps"),
                        ("asrr", "warm_steps"),
                        ("fresh", "scratch_steps"),
                    )
                }
                blocks = unpack(picks["asrr"])[2]
                same_index = {
                    k: v
                    for k, v in avg.items()
                    if k.startswith(f"fresh.{width}.{blocks}.")
                }
                picks["fresh_same_index"] = winner(
                    same_index, "fresh", width, PLAN["scratch_steps"]
                )
                picked = min(
                    ((picks[k], avg[picks[k]]) for k in ("narrow", "prune")), key=rank
                )[0]
                picks["best_dense"] = unpack(picked)[0]
                choices[str(shot)][str(width)] = picks
    write_json(ROOT / f"{name}_curves.json", curves)
    write_json(
        ROOT / f"{name}_frozen.json",
        {
            "choices": choices,
            "dev_seal_sha256": sha(ROOT / stage / "predictions_sealed.json"),
            "source_bank_sha256": None if teacher else sha(ROOT / "bank.npz"),
        },
    )
    print(json.dumps(choices, indent=1), flush=True)


def interval(x, level):
    x = np.asarray(x)
    half = t.ppf((1 + level) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean()), float(x.mean() - half), float(x.mean() + half)]


def deployment(shot, expected):
    folder = ROOT / "models"
    public = ROOT / "test" / str(shot)
    query = np.load(public / "qx.npy", mmap_mode="r")[0].astype(np.float32) / 255
    models = {}
    keys = {"teacher": H}
    keys.update(
        {
            f"{method}_{width}": width
            for width in WIDTHS
            for method in ("asrr", "narrow", "prune")
        }
    )
    for key, width in keys.items():
        with np.load(folder / f"{shot}shot_{key}.npz") as data:
            w, b, v, a = (data[k] for k in ("w", "b", "v", "a"))
            labels = data.get("groups")
        expanded = (
            w[0]
            if labels is None
            else np.concatenate([mat[ids] for mat, ids in zip(w, labels)], axis=1)
        )
        numeric = int(sum(x.size for x in (w, b, v, a)))
        intended = width * (D + C + 1) + C
        if numeric != intended:
            raise RuntimeError(
                f"{key} has {numeric} numerical parameters, expected {intended}"
            )
        logits = (
            np.maximum(query @ expanded + b, 0) @ v / np.sqrt(np.float32(v.shape[0]))
            + a
        )
        err = float(np.abs(logits - expected[key][0]).max())
        if not np.allclose(logits, expected[key][0], atol=1e-4, rtol=2e-5):
            raise RuntimeError(
                f"{key}: deployment disagrees with saved logits, max error {err}"
            )
        models[key] = {
            "trainable_scalars": numeric,
            "numeric_bytes": numeric * 4,
            "index_bytes": int(labels.nbytes) if labels is not None else 0,
            "max_logit_error": err,
        }
    return models


def neighbor_agreement():
    pixels = np.arange(D).reshape(28, 28)
    left = np.r_[pixels[:, :-1].ravel(), pixels[:-1].ravel()]
    right = np.r_[pixels[:, 1:].ravel(), pixels[1:].ravel()]
    result = {}
    with np.load(ROOT / "bank.npz") as bank:
        for width in WIDTHS:
            for blocks in (1, 4):
                key = f"w{width}_b{blocks}"
                labels = bank[key]
                chance = []
                for row in labels:
                    sizes = np.bincount(row)
                    chance.append(np.sum(sizes * (sizes - 1)) / (D * (D - 1)))
                result[key] = {
                    "neighbors_share_group": float(
                        np.mean(labels[:, left] == labels[:, right])
                    ),
                    "expected_after_shuffle": float(np.mean(chance)),
                }
    return result


def evaluate():
    decisions = json.loads((ROOT / "policy_frozen.json").read_text())["choices"]
    report = {
        "test_seal_sha256": sha(ROOT / "test" / "predictions_sealed.json"),
        "pixel_neighbors": neighbor_agreement(),
        "shots": {},
    }
    arrays = {}
    for shot, rows in read_stage("test"):
        result = {"methods": {}, "contrasts": {}, "deployments": {}}
        with np.load(ROOT / "test" / str(shot) / "pred_0000.npz") as first:
            result["deployments"] = deployment(shot, first)
        for width in WIDTHS:
            rows[f"shuffle_{width}"] = {
                m: np.stack([rows[f"shuffle_{width}_{i}"][m] for i in range(4)]).mean(0)
                for m in rows[f"shuffle_{width}_0"]
            }
            best = decisions[str(shot)][str(width)]["best_dense"]
            names = {
                "asrr_vs_narrow": (f"asrr_{width}", f"narrow_{width}"),
                "asrr_vs_prune": (f"asrr_{width}", f"prune_{width}"),
                "asrr_vs_best_dense": (f"asrr_{width}", f"{best}_{width}"),
                "asrr_vs_shuffle": (f"asrr_{width}", f"shuffle_{width}"),
                "asrr_vs_fresh": (f"asrr_{width}", f"fresh_{width}"),
                "asrr_vs_fresh_same": (f"asrr_{width}", f"fresh_same_{width}"),
                "asrr_vs_raw": (f"asrr_{width}", f"raw_{width}"),
                "asrr_vs_teacher": (f"asrr_{width}", "teacher"),
                "raw_vs_teacher": (f"raw_{width}", "teacher"),
            }
            for k, (left, right) in names.items():
                diff = (rows[left]["accuracy"] - rows[right]["accuracy"]) * 100
                level = 0.975 if shot == 5 and k == "asrr_vs_narrow" else 0.95
                result["contrasts"][f"{k}_{width}"] = {
                    "level": level,
                    "diff_pp": interval(diff, level),
                    "cohort_pp": [float(x.mean()) for x in np.split(diff, 2)],
                }
        for key, vals in rows.items():
            result["methods"][key] = {m: float(x.mean()) for m, x in vals.items()}
            arrays.update({f"s{shot}_{key}_{m}": x for m, x in vals.items()})
        report["shots"][str(shot)] = result
    write_json(ROOT / "results.json", report)
    np.savez_compressed(ROOT / "episode_metrics.npz", **arrays)
    print("Results saved to", ROOT / "results.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("teacher_dev", "policy_dev", "test"))
    stage = parser.parse_args().stage
    if stage == "test":
        evaluate()
    else:
        freeze(stage)

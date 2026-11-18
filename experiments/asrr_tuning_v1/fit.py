"""Fresh numerical fitting for direct development search and frozen targets."""

import argparse
import itertools
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from experiments.omniglot_prior_v1.model import numerical_fit
from experiments.paper_recovery_extension_v1.adapter import (
    cache_input_groups,
    fit_adam,
    values,
)
from experiments.paper_recovery_extension_v1.regression_target import (
    TargetBoundary,
)
from experiments.sharing_structure_research_v1.blind_contract import (
    digest,
    save_json,
)

from .data import SETTINGS
from .replicate import OUTPUT, ROOT


def statistics(probability, truth):
    accuracy = (probability.argmax(-1) == truth.argmax(-1)).mean(-1)
    ce = -(np.log(np.maximum(probability, 1e-30)) * truth).sum(-1).mean(-1)
    return {"accuracy": float(accuracy.mean()), "ce": float(ce.mean())}


def ensemble_point(x, y, representation, parameter, hidden, outputs):
    logits = values(x, representation, parameter, hidden, outputs)
    probabilities = logits.softmax(-1).mean(0)
    ce = -(probabilities.clamp_min(1e-30).log() * y).sum(-1).mean(-1)
    recall = (probabilities.argmax(-1) == y.argmax(-1)).float().mean(-1)
    return torch.stack((ce, recall), dim=-1).cpu().numpy()


def run(job, device, *, root=ROOT):
    torch.set_num_threads(2)
    setting, name, phase = job["setting"], job["name"], job["phase"]
    dataset, _shots, validation, outputs = SETTINGS[setting]
    hidden = job.get("hidden", 16)
    folder = root / "study" / setting
    boundary = None
    if not phase.startswith("d"):
        boundary = TargetBoundary(root.resolve(), folder.resolve())
        boundary.install()
    result_folder = folder / f"fits/{phase}"
    result_folder.mkdir(parents=True, exist_ok=True)
    key = name + "__" + job.get("variant", "screen")
    path = result_folder / f"{key}.npz"
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    panel_path = folder / f"panels/{phase}.npz"
    if job.get("panel_sha256") and digest(panel_path) != job["panel_sha256"]:
        raise RuntimeError("Changed frozen target observations")
    with np.load(panel_path) as data:
        panels = {k: data[k] for k in data.files if k != "query_y"}
    index, representation = None, None
    if name != "dense":
        model_path = root / "banks" / dataset / f"models/{name}.npz"
        if (
            job.get("model_sha256")
            and digest(model_path) != job["model_sha256"]
        ):
            raise RuntimeError("Changed frozen indexator")
        with np.load(model_path) as model:
            if "index" in model:
                index = model["index"]
                representation = {
                    "index": torch.as_tensor(
                        index, device=device, dtype=torch.long
                    ),
                    "codes": int(model["codes"])
                    if "codes" in model
                    else int(np.abs(index).max()),
                    "scale": float(model["scale"])
                    if "scale" in model
                    else 1.0,
                }
            else:
                representation = torch.as_tensor(
                    model["basis"], device=device, dtype=torch.float32
                )
    if "input_blocks" in job:
        cache_input_groups(representation, job["input_blocks"])
    x, y, qx = [
        torch.as_tensor(panels[k], device=device, dtype=torch.float32)
        for k in ("train_x", "train_y", "query_x")
    ]
    seed = job.get("seed", 7611172201 if phase.startswith("d") else 7611173301)
    started, predictions, rules = time.perf_counter(), {}, {}
    measure = job.get("measure_fit", False)
    fit_seconds = 0.0
    peak_allocated, peak_reserved = 0, 0
    if validation:
        variant = job.get("variant", "legacy6_200")
        if (
            dataset == "omniglot"
            and variant == "legacy6_200"
            and hidden == 16
            and not torch.is_tensor(representation)
        ):
            fitted = numerical_fit(
                panels, index=index, precompute=False, device=device, seed=seed
            )
            probability = (
                torch.as_tensor(fitted["prediction"]).softmax(-1).numpy()
            )
            predictions[variant] = probability
            rules[variant] = {
                "variant": variant,
                "numeric_starts": 6,
                "steps": 200,
            }
        else:
            wide = variant.startswith("wide")
            starts = 8 if variant.startswith("legacy24") else 2
            rates = (0.001, 0.003, 0.01, 0.03) if wide else (0.003, 0.01, 0.03)
            quantiles = (0.1, 0.5, 0.9) if wide else (0.5,)
            settings = [
                {"lr": rate, "quantile": quantile, "restart": r}
                for rate, quantile, r in itertools.product(
                    rates, quantiles, range(starts)
                )
            ]
            steps = int(variant.rsplit("_", 1)[1])
            vx, vy = [
                torch.as_tensor(panels[k], device=device, dtype=torch.float32)
                for k in ("valid_x", "valid_y")
            ]
            shard_size = job.get(
                "shard_size",
                128 if torch.is_tensor(representation) else len(x),
            )
            ensembles = {1: [], 4: []}
            learning_curves = []
            for begin in range(0, len(x), shard_size):
                shard = slice(begin, begin + shard_size)
                history = []

                def observe(step, selected):
                    train = ensemble_point(
                        x[shard], y[shard], representation, selected,
                        hidden, outputs,
                    )
                    valid = ensemble_point(
                        vx[shard], vy[shard], representation, selected,
                        hidden, outputs,
                    )
                    history.append(np.column_stack((train, valid)))

                if measure:
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    started_fit = time.perf_counter()
                parameter, trace = fit_adam(
                    x[shard],
                    y[shard],
                    vx[shard],
                    vy[shard],
                    representation,
                    hidden,
                    outputs,
                    steps=steps,
                    loss="cross_entropy",
                    seed=seed + begin,
                    settings=settings,
                    keep=4,
                    on_checkpoint=observe if job.get("record_curves") else None,
                )
                if measure:
                    torch.cuda.synchronize(device)
                    fit_seconds += time.perf_counter() - started_fit
                    peak_allocated = max(
                        peak_allocated, torch.cuda.max_memory_allocated(device)
                    )
                    peak_reserved = max(
                        peak_reserved, torch.cuda.max_memory_reserved(device)
                    )
                with torch.no_grad():
                    probability = values(
                        qx[shard], representation, parameter, hidden, outputs
                    ).softmax(-1)
                    for count, batches in ensembles.items():
                        batches.append(
                            probability[:count].mean(0).cpu().numpy()
                        )
                del parameter, probability
                if history:
                    learning_curves.append(np.stack(history))
            if learning_curves:
                predictions["curve_steps"] = trace["steps"]
                predictions["ensemble_curves"] = np.concatenate(
                    learning_curves, axis=1
                )
            for count, batches in ensembles.items():
                label = f"{variant}_top{count}"
                predictions[label] = (
                    batches[0]
                    if len(batches) == 1
                    else np.concatenate(batches)
                )
                rules[label] = {
                    "variant": variant,
                    "ensemble": count,
                    "numeric_starts": len(settings),
                    "steps": steps,
                }
    elif phase.startswith("d"):
        with np.load(folder / f"panels/{phase}.npz") as data:
            query_y = data["query_y"]
        target = torch.as_tensor(query_y, device=device, dtype=torch.float32)
        wide = phase == "d2"
        settings = [
            {"lr": rate, "quantile": quantile, "restart": r}
            for rate, quantile, r in itertools.product(
                (0.001, 0.003, 0.01, 0.03) if wide else (0.003, 0.01, 0.03),
                (0.1, 0.5, 0.9),
                range(2),
            )
        ]
        _, trace = fit_adam(
            x,
            y,
            qx,
            target,
            representation,
            hidden,
            outputs,
            steps=1000 if wide else 200,
            loss="cross_entropy",
            seed=seed,
            settings=settings,
        )
        score = (
            trace["validation_history"]
            .mean(-1)
            .reshape(len(trace["steps"]), -1, 2)
            .mean(-1)
        )
        configurations = []
        for column in range(score.shape[1]):
            step = int(score[:, column].argmin())
            configurations.append((float(score[step, column]), step, column))
        configurations.sort()
        for rank, (_, step, column) in enumerate(
            configurations[: 3 if wide else 1]
        ):
            config = settings[2 * column]
            count = 16 if wide else 2
            parameter, _ = fit_adam(
                x,
                y,
                None,
                None,
                representation,
                hidden,
                outputs,
                fixed_step=int(trace["steps"][step]),
                seed=seed,
                settings=[{**config, "restart": r} for r in range(count)],
                keep=count,
                preserve_setting_order=True,
                loss="cross_entropy",
            )
            with torch.no_grad():
                probability = values(
                    qx, representation, parameter, hidden, outputs
                ).softmax(-1)
                for ensemble in (1, 4, 16) if wide else (2,):
                    label = f"rule{rank}_ensemble{ensemble}"
                    predictions[label] = (
                        probability[:ensemble].mean(0).cpu().numpy()
                    )
                    rules[label] = {
                        "variant": "support",
                        "global_setting": config,
                        "steps": int(trace["steps"][step]),
                        "ensemble": ensemble,
                        "numeric_starts": count,
                    }
    else:
        rule = job["rule"]
        count = rule["numeric_starts"]
        parameter, _ = fit_adam(
            x,
            y,
            None,
            None,
            representation,
            hidden,
            outputs,
            fixed_step=rule["steps"],
            seed=seed,
            settings=[
                {**rule["global_setting"], "restart": r} for r in range(count)
            ],
            keep=count,
            preserve_setting_order=True,
            loss="cross_entropy",
        )
        with torch.no_grad():
            probability = values(
                qx, representation, parameter, hidden, outputs
            ).softmax(-1)
            predictions["frozen"] = (
                probability[: rule["ensemble"]].mean(0).cpu().numpy()
            )
            rules["frozen"] = rule
    np.savez_compressed(path, **predictions)
    record = {
        "job": job,
        "rules": rules,
        "prediction_sha256": digest(path),
        "seconds": time.perf_counter() - started,
        "query_answers_used_for_target_fitting": False,
    }
    if measure:
        record["fit_timing"] = {
            "seconds": fit_seconds,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
            "cpu_peak_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss
            * 1024,
            "device": torch.cuda.get_device_name(device),
            "device_index": device.index,
            "episodes": len(x),
        }
    if boundary is not None:
        record["access_violations"] = boundary.violations
        record["opened_workspace_files"] = sorted(boundary.opened)
    if phase.startswith("d"):
        with np.load(folder / f"panels/{phase}.npz") as data:
            truth = data["query_y"]
        record["metrics"] = {
            key: statistics(p, truth) for key, p in predictions.items()
        }
    save_json(path.with_suffix(".json"), record)
    print(
        json.dumps(
            {
                "setting": setting,
                "name": name,
                "seconds": record["seconds"],
                "metrics": record.get("metrics"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", required=True)
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jobs = json.loads((args.output / f"{args.queue}_jobs.json").read_text())
    run(jobs[args.job], torch.device(args.device), root=args.root)

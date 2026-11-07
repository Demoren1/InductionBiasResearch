"""Calibrate, freeze and evaluate the adaptive tanh mechanism pilot."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from experiments.adaptive_tanh_v1.fit import evaluate_selected, fit
from experiments.adaptive_tanh_v1.source import fit_source, make_tasks


def validate_protocol(plan):
    positive = [
        "inputs",
        "hidden",
        "groups",
        "calibration_banks",
        "evaluation_banks",
        "source_tasks",
        "source_starts",
        "source_support",
        "source_query",
        "source_steps",
        "source_batch",
        "calibration_tasks",
        "evaluation_tasks",
        "support",
        "query",
        "test",
        "starts",
        "steps",
        "log_every",
    ]
    for name in positive:
        if type(plan[name]) is not int or plan[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    positions = plan["inputs"] * plan["hidden"]
    if positions % plan["groups"]:
        raise ValueError("Initial teacher groups must have equal occupancy")
    previous_step, previous_groups = 0, plan["groups"]
    for step, groups in plan["split_schedule"]:
        if not previous_step < step < plan["steps"]:
            raise ValueError(
                "Split steps must increase within the training budget"
            )
        if not previous_groups < groups <= positions:
            raise ValueError(
                "Split group counts must increase up to weight count"
            )
        if step % plan["log_every"]:
            raise ValueError(
                "Split events must coincide with logged checkpoints"
            )
        previous_step, previous_groups = step, groups
    for name in [
        "raw_rates",
        "normalized_rates",
        "dense_sgd_rates",
        "dense_adam_rates",
    ]:
        rates = plan[name]
        if len(rates) != 5 or any(not np.isfinite(r) or r <= 0 for r in rates):
            raise ValueError(f"{name} must contain five finite positive rates")
    if plan["residual_ratios"][0] != 0 or any(
        not np.isfinite(r) or r < 0 for r in plan["residual_ratios"]
    ):
        raise ValueError(
            "Residual ratios must be nonnegative and start with zero"
        )
    if not 0 <= plan["momentum"] < 1:
        raise ValueError("Momentum must belong to [0, 1)")
    if plan["dtype"] != "float32" or not plan["device"].startswith("cuda"):
        raise ValueError("This protocol implements CUDA float32 only")
    expected = [
        "fixed_raw",
        "fixed_normalized",
        "adaptive",
        "random_matched",
        "dense_sgd",
        "dense_adam",
    ]
    if plan["methods"] != expected:
        raise ValueError(
            "Methods/order must include adaptive before matched random"
        )


def save_fit(folder, method, result, history, events, timing):
    np.savez_compressed(folder / f"{method}.npz", **result)
    (folder / f"{method}_history.json").write_text(json.dumps(history))
    (folder / f"{method}_splits.json").write_text(json.dumps(events))
    (folder / f"{method}_timing.json").write_text(json.dumps(timing, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", default="experiments/adaptive_tanh_v1/protocol.json"
    )
    parser.add_argument("--resume-calibration", action="store_true")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    plan = json.loads(protocol_path.read_text())
    validate_protocol(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; CPU fallback is forbidden")
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    device = torch.device(plan["device"])
    torch.cuda.set_device(device)
    output = Path(plan["output"])
    if args.resume_calibration:
        if (output / "frozen_selection.json").exists():
            raise ValueError(
                "Cannot resume calibration after freezing selection"
            )
        if (
            output / "protocol.json"
        ).read_bytes() != protocol_path.read_bytes():
            raise ValueError("Resume requires the unchanged protocol")
    else:
        output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    code = Path(__file__).resolve().parent
    snapshots = output / "code"
    snapshots.mkdir(exist_ok=args.resume_calibration)
    paths = [code / name for name in ["source.py", "fit.py", "run.py"]]
    paths.append(code.parent / "shared_codebook_mlp_v1/model.py")
    hashes = {}
    for path in paths:
        shutil.copy2(path, snapshots / path.name)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    shutil.copy2(protocol_path, output / "protocol.json")
    environment = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "code_sha256": hashes,
        "protocol_sha256": hashlib.sha256(
            protocol_path.read_bytes()
        ).hexdigest(),
        "started_unix": time.time(),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2))
    grids = {
        "fixed_raw": plan["raw_rates"],
        "fixed_normalized": plan["normalized_rates"],
        "adaptive": plan["normalized_rates"],
        "random_matched": plan["normalized_rates"],
        "dense_sgd": plan["dense_sgd_rates"],
        "dense_adam": plan["dense_adam_rates"],
    }
    calibration_scores = {method: [] for method in plan["methods"]}
    calibration_seconds = {}
    for bank in range(plan["calibration_banks"]):
        folder = output / f"calibration_bank_{bank}"
        seed = plan["seed"] + bank * 1000
        if (
            args.resume_calibration
            and (folder / "source_diagnostic.json").exists()
        ):
            consensus = np.load(folder / "bank.npz")["consensus"]
            teacher_i = torch.tensor(
                np.load(folder / "source_data.npz")["teacher_i"], device=device
            )
        else:
            consensus, teacher_i = fit_source(plan, seed, folder, device)
        data = make_tasks(
            plan,
            teacher_i,
            plan["calibration_tasks"],
            seed + 100,
            False,
            device,
        )
        np.savez_compressed(
            folder / "calibration_data.npz",
            **{key: value.cpu().numpy() for key, value in data.items()},
        )
        mirrored = None
        for method in plan["methods"]:
            result, history, events, timing = fit(
                plan,
                data["train_x"],
                data["train_y"],
                data["query_x"],
                data["query_y"],
                consensus,
                method,
                grids[method],
                seed + 200,
                mirrored_events=mirrored,
            )
            if method == "adaptive":
                mirrored = events
            save_fit(folder, method, result, history, events, timing)
            scores = (
                result["query_nmse"]
                .min(0)
                .reshape(-1, len(grids[method]), plan["starts"])
                .min(2)
                .mean(0)
            )
            calibration_scores[method].append(scores.tolist())
            calibration_seconds[f"{bank}/{method}"] = timing
            print(
                json.dumps(
                    {
                        "stage": "calibration",
                        "bank": bank,
                        "method": method,
                        "rates": grids[method],
                        "scores": scores.tolist(),
                        **timing,
                    }
                ),
                flush=True,
            )
        del data
    mean_scores = {
        method: np.mean(scores, axis=0)
        for method, scores in calibration_scores.items()
    }
    normalized_methods = ["fixed_normalized", "adaptive", "random_matched"]
    normalized_score = np.mean(
        [mean_scores[method] for method in normalized_methods], axis=0
    )
    common_id = int(normalized_score.argmin())
    rates = {}
    for method in plan["methods"]:
        choice = (
            common_id
            if method in normalized_methods
            else int(mean_scores[method].argmin())
        )
        rates[method] = grids[method][choice]
    frozen = {
        "selected_rates": rates,
        "calibration_scores_by_bank": calibration_scores,
        "mean_scores": {
            key: value.tolist() for key, value in mean_scores.items()
        },
        "common_normalized_scores": normalized_score.tolist(),
        "frozen_unix": time.time(),
        "protocol_sha256": environment["protocol_sha256"],
        "code_sha256": hashes,
        "calibration_bank_sha256": {
            str(path.relative_to(output)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(output.glob("calibration_bank_*/bank.npz"))
        },
    }
    frozen_path = output / "frozen_selection.json"
    frozen_path.write_text(json.dumps(frozen, indent=2))
    frozen_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print(json.dumps({"stage": "frozen", "rates": rates}), flush=True)

    metrics = []
    evaluation_seconds = {}
    for bank in range(plan["evaluation_banks"]):
        folder = output / f"evaluation_bank_{bank}"
        seed = plan["seed"] + 100000 + bank * 1000
        consensus, teacher_i = fit_source(plan, seed, folder, device)
        bank_frozen = {
            "global_selection_sha256": frozen_hash,
            "source_bank_sha256": hashlib.sha256(
                (folder / "bank.npz").read_bytes()
            ).hexdigest(),
            "before_target_generation_unix": time.time(),
            "target_seed": seed + 100,
        }
        (folder / "frozen_before_target.json").write_text(
            json.dumps(bank_frozen, indent=2)
        )
        data = make_tasks(
            plan, teacher_i, plan["evaluation_tasks"], seed + 100, True, device
        )
        np.savez_compressed(
            folder / "target_data.npz",
            **{key: value.cpu().numpy() for key, value in data.items()},
        )
        mirrored = None
        bank_results = {}
        for method in plan["methods"]:
            result, history, events, timing = fit(
                plan,
                data["train_x"],
                data["train_y"],
                data["query_x"],
                data["query_y"],
                consensus,
                method,
                [rates[method]],
                seed + 200,
                mirrored_events=mirrored,
            )
            if method == "adaptive":
                mirrored = events
            save_fit(folder, method, result, history, events, timing)
            bank_results[method] = result
            evaluation_seconds[f"{bank}/{method}"] = timing
            print(
                json.dumps(
                    {
                        "stage": "target_fit",
                        "bank": bank,
                        "method": method,
                        **timing,
                    }
                ),
                flush=True,
            )
        # All six methods are fitted and selected before held-out evaluation.
        for method, result in bank_results.items():
            rows = evaluate_selected(
                plan, result, data["test_x"], data["test_y"]
            )
            for row in rows:
                task = row["task_row"]
                row.update(
                    {
                        "bank": bank,
                        "method": method,
                        "residual_ratio": float(data["ratios"][task]),
                        "task": int(data["task_ids"][task]),
                    }
                )
            metrics.extend(rows)
            print(
                json.dumps(
                    {
                        "stage": "target_evaluation",
                        "bank": bank,
                        "method": method,
                        "selected_mean_nmse": float(
                            np.mean(
                                [
                                    row["test_nmse"]
                                    for row in rows
                                    if row["selection"] == "selected"
                                ]
                            )
                        ),
                    }
                ),
                flush=True,
            )
        with (output / "metrics.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        del data, bank_results
    completion = {
        "complete": True,
        "seconds": time.perf_counter() - started,
        "global_selection_sha256": frozen_hash,
        "calibration_timings": calibration_seconds,
        "evaluation_timings": evaluation_seconds,
        "completed_unix": time.time(),
    }
    (output / "complete.json").write_text(json.dumps(completion, indent=2))
    print(
        json.dumps({"stage": "complete", "seconds": completion["seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()

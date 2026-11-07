"""Check whether complete mixed-logic tasks yield a usable MLP bank."""

import argparse
import hashlib
import json
import math
import shutil
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

HERE = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def logic(values, kinds):
    """Kinds: 0 is parity, 2/3 are thresholds on four binary inputs."""
    total = values.sum(-1)
    return np.where(kinds == 0, total % 2, total >= kinds).astype(np.float32)


def numpy_labels(x, task):
    first = logic(x[:, task["parents"]], task["kinds_first"])
    second = logic(first.reshape(-1, 4, 4), task["kinds_second"])
    return logic(second[:, None, :], task["kind_output"])[..., 0]


def make_tasks(plan, groups, count, rng):
    tasks, attempts = [], []
    spec = plan["family"]
    while len(tasks) < count:
        parents = np.array(
            [
                rng.choice(group, 4, replace=False)
                for group in groups
                for _ in range(2)
            ]
        )
        kinds = np.where(
            rng.random(21) < spec["xor_probability"],
            0,
            rng.integers(2, 4, 21),
        )
        task = {
            "parents": parents,
            "kinds_first": kinds[:16],
            "kinds_second": kinds[16:20],
            "kind_output": kinds[20:21],
        }
        x = rng.integers(0, 2, (spec["balance_examples"], 64))
        rate = float(numpy_labels(x, task).mean())
        accepted = (
            spec["positive_rate_range"][0]
            <= rate
            <= spec["positive_rate_range"][1]
            and np.any(kinds == 0)
            and np.any(kinds != 0)
        )
        attempts.append({"positive_rate": rate, "accepted": bool(accepted)})
        if accepted:
            tasks.append(task)
    return tasks, attempts


def teacher_tensors(tasks, device):
    return {
        key: torch.from_numpy(np.stack([task[key] for task in tasks])).to(
            device
        )
        for key in tasks[0]
    }


def torch_labels(x, teachers):
    flat = teachers["parents"].flatten(1)
    first_sum = (
        x.gather(2, flat[:, None, :].expand(-1, x.shape[1], -1))
        .reshape(x.shape[0], x.shape[1], 16, 4)
        .sum(-1)
    )
    kinds = teachers["kinds_first"][:, None, :]
    first = torch.where(kinds == 0, first_sum.remainder(2), first_sum >= kinds)
    second_sum = first.reshape(x.shape[0], x.shape[1], 4, 4).sum(-1)
    kinds = teachers["kinds_second"][:, None, :]
    second = torch.where(
        kinds == 0, second_sum.remainder(2), second_sum >= kinds
    )
    final_sum = second.sum(-1)
    kind = teachers["kind_output"]
    return torch.where(kind == 0, final_sum.remainder(2), final_sum >= kind)


def forward(x, parameters):
    for index in range(0, len(parameters), 2):
        x = x.bmm(parameters[index]) + parameters[index + 1]
        if index + 2 < len(parameters):
            x = x.relu()
    return x[..., 0]


def initialize(model_count, hidden, gains, device, masks=None):
    dimensions = [64, *hidden, 1]
    parameters = []
    for index, (inputs, outputs) in enumerate(
        pairwise(dimensions)
    ):
        weights = torch.randn(model_count, inputs, outputs, device=device)
        weights *= gains[:, None, None] * math.sqrt(2 / inputs)
        if masks is not None:
            mask = masks[index].to(device)
            degree = mask.sum(0).clamp_min(1)
            weights *= mask[None] * (inputs / degree).sqrt()[None, None, :]
        parameters.extend(
            [
                weights.requires_grad_(),
                torch.zeros(
                    model_count, 1, outputs, device=device, requires_grad=True
                ),
            ]
        )
    return parameters


@torch.no_grad()
def evaluate(parameters, x, y, models_per_task, chunk=256):
    count = parameters[0].shape[0]
    loss = torch.zeros(count, device=x.device)
    correct = torch.zeros_like(loss)
    true_positive = torch.zeros_like(loss)
    true_negative = torch.zeros_like(loss)
    positives = torch.zeros_like(loss)
    negatives = torch.zeros_like(loss)
    for start in range(0, x.shape[1], chunk):
        batch_x = x[:, start : start + chunk].repeat_interleave(
            models_per_task, 0
        )
        batch_y = y[:, start : start + chunk].repeat_interleave(
            models_per_task, 0
        )
        logits = forward(batch_x, parameters)
        loss += functional.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        ).sum(1)
        pred = logits >= 0
        label = batch_y.bool()
        correct += (pred == label).sum(1)
        true_positive += (pred & label).sum(1)
        true_negative += (~pred & ~label).sum(1)
        positives += label.sum(1)
        negatives += (~label).sum(1)
    return {
        "loss": loss / x.shape[1],
        "accuracy": correct / x.shape[1],
        "recall_positive": true_positive / positives.clamp_min(1),
        "recall_negative": true_negative / negatives.clamp_min(1),
        "positive_rate": positives / x.shape[1],
    }


def settings_grid(spec, optimizer):
    if optimizer == "adam":
        return [
            {"optimizer": optimizer, "learning_rate": rate, "gain": gain}
            for rate in spec["adam_learning_rates"]
            for gain in spec["initial_gains"]
        ]
    return [
        {"optimizer": optimizer, "learning_rate": rate, "gain": 1.0}
        for rate in spec["sgd_learning_rates"]
    ]


def fit(plan, tasks, settings, output, seed, device, masks=None):
    spec = plan["calibration"]
    task_count = len(tasks)
    per_task = len(settings) * spec["restarts"]
    count = task_count * per_task
    torch.manual_seed(seed)
    teachers = teacher_tensors(tasks, device)
    repeated_settings = settings * task_count
    rates = torch.tensor(
        [s["learning_rate"] for s in repeated_settings], device=device
    ).repeat_interleave(spec["restarts"])
    gains = torch.tensor(
        [s["gain"] for s in repeated_settings], device=device
    ).repeat_interleave(spec["restarts"])
    parameters = initialize(
        count, plan["model"]["hidden"], gains, device, masks
    )
    first_moment = [torch.zeros_like(p) for p in parameters]
    second_moment = [torch.zeros_like(p) for p in parameters]
    generator = torch.Generator(device=device).manual_seed(plan["seed"] + 50)
    selection_x = torch.randint(
        0,
        2,
        (task_count, spec["selection_examples"], 64),
        device=device,
        generator=generator,
    ).float()
    selection_y = torch_labels(selection_x, teachers)
    evaluation_x = torch.randint(
        0,
        2,
        (task_count, spec["evaluation_examples"], 64),
        device=device,
        generator=generator,
    ).float()
    evaluation_y = torch_labels(evaluation_x, teachers)
    generator.manual_seed(seed + 1)
    best_loss = torch.full((count,), float("inf"), device=device)
    best_parameters = [p.detach().clone() for p in parameters]
    best_step = torch.zeros(count, dtype=torch.int64, device=device)
    curves = []
    accumulated_loss = torch.zeros(count, device=device)
    accumulated_steps = 0
    optimizer = settings[0]["optimizer"]
    start = time.perf_counter()
    for step in range(spec["steps"] + 1):
        if step % spec["log_every"] == 0 or step == spec["steps"]:
            metrics = evaluate(parameters, selection_x, selection_y, per_task)
            with torch.no_grad():
                improved = metrics["loss"] < best_loss
                best_loss[improved] = metrics["loss"][improved]
                best_step[improved] = step
                for best, current in zip(best_parameters, parameters):
                    best[improved] = current[improved]
            row = {
                "step": step,
                **{
                    key: value.cpu().tolist() for key, value in metrics.items()
                },
            }
            if step:
                row["train_loss"] = (
                    (accumulated_loss / accumulated_steps).cpu().tolist()
                )
                accumulated_loss.zero_()
                accumulated_steps = 0
            curves.append(row)
            print(
                json.dumps(
                    {
                        "optimizer": optimizer,
                        "step": step,
                        "seconds": round(time.perf_counter() - start, 1),
                        "mean_validation_accuracy": float(
                            metrics["accuracy"].mean()
                        ),
                        "best_validation_accuracy_per_task": metrics[
                            "accuracy"
                        ]
                        .reshape(task_count, per_task)
                        .max(1)
                        .values.tolist(),
                    }
                ),
                flush=True,
            )
            save_json(output / f"{optimizer}_curves.json", curves)
        if step == spec["steps"]:
            break
        x = torch.randint(
            0,
            2,
            (task_count, spec["batch_size"], 64),
            generator=generator,
            device=device,
        ).float()
        y = torch_labels(x, teachers).repeat_interleave(per_task, 0)
        logits = forward(x.repeat_interleave(per_task, 0), parameters)
        train_losses = functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none"
        ).mean(1)
        accumulated_loss += train_losses.detach()
        accumulated_steps += 1
        train_losses.sum().backward()
        with torch.no_grad():
            for index, parameter in enumerate(parameters):
                grad = parameter.grad
                if masks is not None and index % 2 == 0:
                    grad *= masks[index // 2].to(device)[None]
                if optimizer == "adam":
                    first_moment[index].mul_(0.9).add_(grad, alpha=0.1)
                    second_moment[index].mul_(0.999).addcmul_(
                        grad, grad, value=0.001
                    )
                    denominator = (
                        second_moment[index].sqrt()
                        / math.sqrt(1 - 0.999 ** (step + 1))
                        + 1e-8
                    )
                    update = first_moment[index] / denominator
                    update *= (rates / (1 - 0.9 ** (step + 1)))[:, None, None]
                else:
                    first_moment[index].mul_(spec["sgd_momentum"]).add_(grad)
                    update = first_moment[index] * rates[:, None, None]
                parameter.sub_(update)
                parameter.grad = None
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    selection = evaluate(best_parameters, selection_x, selection_y, per_task)
    evaluation = evaluate(
        best_parameters, evaluation_x, evaluation_y, per_task
    )
    records = []
    for index in range(count):
        task_id = index // per_task
        configuration = (index % per_task) // spec["restarts"]
        records.append(
            {
                "task": task_id,
                "configuration": settings[configuration],
                "restart": index % spec["restarts"],
                "model_index": index,
                "selected_step": int(best_step[index]),
                "selection": {
                    key: float(value[index])
                    for key, value in selection.items()
                },
                "evaluation": {
                    key: float(value[index])
                    for key, value in evaluation.items()
                },
            }
        )
    arrays = {
        f"parameter_{index}": value.cpu().numpy()
        for index, value in enumerate(best_parameters)
    }
    np.savez_compressed(output / f"{optimizer}_models.npz", **arrays)
    save_json(output / f"{optimizer}_records.json", records)
    save_json(
        output / f"{optimizer}_runtime.json",
        {
            "seconds": elapsed,
            "fits": count,
            "steps_per_fit": spec["steps"],
            "gradient_examples": count * spec["steps"] * spec["batch_size"],
        },
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocol.json"
    )
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    if plan["family"]["dimension"] != 64:
        raise ValueError("This pilot implements the frozen 64-bit family")
    if not torch.cuda.is_available():
        raise RuntimeError("Run outside sandbox on an available CUDA GPU")
    torch.set_num_threads(2)
    output = HERE / args.output
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "run_snapshot.py")
    shutil.copyfile(args.protocol, output / "protocol.json")
    save_json(
        output / "seal.json",
        {
            "protocol_sha256": hashlib.sha256(
                args.protocol.read_bytes()
            ).hexdigest(),
            "script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0),
            "seed": plan["seed"],
            "stage": "source_feasibility",
        },
    )
    rng = np.random.default_rng(plan["seed"])
    groups = rng.permutation(64).reshape(8, 8)
    tasks, attempts = make_tasks(
        plan, groups, plan["calibration"]["tasks"], rng
    )
    np.savez_compressed(
        output / "source_tasks.npz",
        groups_diagnostic=groups,
        **{key: np.stack([task[key] for task in tasks]) for key in tasks[0]},
    )
    save_json(output / "generation_attempts.json", attempts)
    records = []
    for index, optimizer in enumerate(["adam", "sgd"]):
        records.extend(
            fit(
                plan,
                tasks,
                settings_grid(plan["calibration"], optimizer),
                output,
                plan["seed"] + 100 + index,
                "cuda",
            )
        )
    selected = []
    spec = plan["calibration"]
    for task_id in range(spec["tasks"]):
        best = min(
            [row for row in records if row["task"] == task_id],
            key=lambda row: row["selection"]["loss"],
        )
        metrics = best["evaluation"]
        best["success"] = (
            metrics["accuracy"] >= spec["success_accuracy"]
            and min(metrics["recall_positive"], metrics["recall_negative"])
            >= spec["success_minimum_class_recall"]
        )
        selected.append(best)
    successes = sum(row["success"] for row in selected)
    passed = successes >= spec["minimum_successful_tasks"]
    summary = {
        "stage": "source_feasibility",
        "successes": successes,
        "tasks": spec["tasks"],
        "fits": len(records),
        "passed": passed,
        "next_stage": "source_bank" if passed else "stopped_source_gate",
        "selected_by": "minimum independent selection BCE, not evaluation accuracy",
        "selected_models": selected,
        "target_tasks_opened": False,
        "bank_collected": False,
    }
    save_json(output / "summary.json", summary)
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

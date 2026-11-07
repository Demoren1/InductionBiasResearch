"""Compare random/evolutionary structures and index/numeric SGD."""

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from run import forward, save_json, teacher_tensors, torch_labels
from torch.nn import functional

HERE = Path(__file__).resolve().parent


def validate(plan):
    model, search = plan["model"], plan["search"]
    dimensions = model["dimensions"]
    if dimensions != [64, 64, 16, 4, 1]:
        raise ValueError("This source pilot fixes the teacher-sized MLP")
    if len(model["fan_in_options"]) != len(dimensions) - 1:
        raise ValueError("Specify fan-in choices for every layer")
    for inputs, choices in zip(dimensions[:-1], model["fan_in_options"]):
        if not choices or any(not 1 <= degree <= inputs for degree in choices):
            raise ValueError("Fan-in must fit its input layer")
    if 0 not in model["codebook_base"]:
        raise ValueError("Zero is needed for masked and quantized weights")
    if search["elite"] + search["immigrants"] >= search["population"]:
        raise ValueError("Leave space for evolutionary offspring")
    for key in ["population", "generations", "candidate_steps", "batch_size"]:
        if search[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 < search["mutated_rows_fraction"] <= 1:
        raise ValueError("Mutation fraction must be in (0, 1]")
    if (
        not 0
        < model["codebook_scale_range"][0]
        < model["codebook_scale_range"][1]
    ):
        raise ValueError("Codebook scales need positive ordered bounds")
    for method in plan["methods"]:
        if method["outer"] not in {"random", "evolution"}:
            raise ValueError("Unknown outer search")
        if method["inner"] not in {"weights", "indices"}:
            raise ValueError("Unknown inner learner")


def sample_structure(plan, rng, dense=False):
    model = plan["model"]
    masks = []
    for layer, (inputs, outputs) in enumerate(
        zip(model["dimensions"][:-1], model["dimensions"][1:])
    ):
        degree = (
            inputs if dense else rng.choice(model["fan_in_options"][layer])
        )
        mask = np.zeros((inputs, outputs), dtype=bool)
        for column in range(outputs):
            mask[rng.choice(inputs, degree, replace=False), column] = True
        masks.append(mask)
    low, high = np.log(model["codebook_scale_range"])
    return {"masks": masks, "scales": np.exp(rng.uniform(low, high, (4, 2)))}


def offspring(plan, left, right, rng):
    masks = []
    spec = plan["search"]
    for layer, (mask_left, mask_right) in enumerate(
        zip(left["masks"], right["masks"])
    ):
        inputs, outputs = mask_left.shape
        choose_left = rng.random(outputs) < 0.5
        mask = np.where(choose_left[None], mask_left, mask_right).copy()
        count = max(1, math.ceil(outputs * spec["mutated_rows_fraction"]))
        for column in rng.choice(outputs, count, replace=False):
            active = np.flatnonzero(mask[:, column])
            inactive = np.flatnonzero(~mask[:, column])
            if inactive.size and rng.random() < 0.5:
                mask[rng.choice(active), column] = False
                mask[rng.choice(inactive), column] = True
            else:
                degree = rng.choice(plan["model"]["fan_in_options"][layer])
                mask[:, column] = False
                mask[rng.choice(inputs, degree, replace=False), column] = True
        masks.append(mask)
    scales = np.where(
        rng.random((4, 2)) < 0.5, left["scales"], right["scales"]
    )
    scales = scales * np.exp(
        rng.normal(0, spec["codebook_log_scale_mutation"], (4, 2))
    )
    scales = np.clip(scales, *plan["model"]["codebook_scale_range"])
    return {"masks": masks, "scales": scales}


def structure_hash(structure, include_scales=True):
    digest = hashlib.sha256()
    for mask in structure["masks"]:
        digest.update(mask.tobytes())
    if include_scales:
        digest.update(structure["scales"].tobytes())
    return digest.hexdigest()


def population_tensors(plan, structures, device):
    base = torch.tensor(plan["model"]["codebook_base"], device=device).float()
    masks, codebooks = [], []
    for layer in range(4):
        mask = (
            torch.from_numpy(np.stack([s["masks"][layer] for s in structures]))
            .to(device)
            .float()
        )
        scale = (
            torch.from_numpy(
                np.stack([s["scales"][layer] for s in structures])
            )
            .to(device)
            .float()
        )
        masks.append(mask)
        codebooks.extend(
            [
                base[None]
                * scale[:, :1]
                / mask.sum(1).mean(1).sqrt()[:, None],
                base[None]
                * scale[:, 1:]
                * plan["model"]["bias_codebook_step"],
            ]
        )
    return masks, codebooks


def initialize(masks, seed):
    torch.manual_seed(seed)
    parameters = []
    for mask in masks:
        count, inputs, outputs = mask.shape
        weights = torch.randn(count, inputs, outputs, device=mask.device)
        weights *= mask * (2 / mask.sum(1).clamp_min(1)).sqrt()[:, None, :]
        parameters.extend(
            [
                weights.requires_grad_(),
                torch.zeros(
                    count, 1, outputs, device=mask.device
                ).requires_grad_(),
            ]
        )
    return parameters


def effective_parameters(parameters, codebooks, inner):
    if inner == "weights":
        return parameters
    effective = []
    for shadow, codebook in zip(parameters, codebooks):
        with torch.no_grad():
            indices = (
                (shadow[..., None] - codebook[:, None, None]).abs().argmin(-1)
            )
            quantized = codebook.gather(1, indices.flatten(1)).reshape_as(
                shadow
            )
        effective.append(quantized + (shadow - shadow.detach()))
    return effective


@torch.no_grad()
def metrics(parameters, x, y, per_task, chunk=256):
    count = parameters[0].shape[0]
    sums = torch.zeros(6, count, device=x.device)
    for start in range(0, x.shape[1], chunk):
        batch_x = x[:, start : start + chunk].repeat_interleave(per_task, 0)
        batch_y = y[:, start : start + chunk].repeat_interleave(per_task, 0)
        logits = forward(batch_x, parameters)
        predicted, positive = logits >= 0, batch_y.bool()
        sums[0] += functional.binary_cross_entropy_with_logits(
            logits, batch_y, reduction="none"
        ).sum(1)
        sums[1] += (predicted == positive).sum(1)
        sums[2] += (predicted & positive).sum(1)
        sums[3] += (~predicted & ~positive).sum(1)
        sums[4] += positive.sum(1)
        sums[5] += (~positive).sum(1)
    return {
        "loss": sums[0] / x.shape[1],
        "accuracy": sums[1] / x.shape[1],
        "recall_positive": sums[2] / sums[4].clamp_min(1),
        "recall_negative": sums[3] / sums[5].clamp_min(1),
        "positive_rate": sums[4] / x.shape[1],
    }


def train_population(plan, structures, inner, generation, data, device):
    spec = plan["search"]
    count, population = len(structures), spec["population"]
    masks, codebooks = population_tensors(plan, structures, device)
    parameters = initialize(masks, plan["seed"] + 1000 + generation)
    if inner == "indices":
        with torch.no_grad():
            for shadow, codebook in zip(parameters, codebooks):
                shadow.clamp_(
                    codebook.min(1).values[:, None, None],
                    codebook.max(1).values[:, None, None],
                )
    velocity = [torch.zeros_like(p) for p in parameters]
    learning_rates = spec[
        f"{'index' if inner == 'indices' else 'numeric'}_learning_rates"
    ]
    rates = torch.tensor(
        [
            learning_rates[
                (generation * population + i % population)
                % len(learning_rates)
            ]
            for i in range(count)
        ],
        device=device,
    )
    best_loss = torch.full((count,), float("inf"), device=device)
    best_parameters = [
        p.detach().clone()
        for p in effective_parameters(parameters, codebooks, inner)
    ]
    best_step = torch.zeros(count, dtype=torch.int64, device=device)
    curves = []
    generator = torch.Generator(device=device).manual_seed(
        plan["seed"] + 2000 + generation
    )
    train_sum = torch.zeros(count, device=device)
    accumulated_steps = 0
    for step in range(spec["candidate_steps"] + 1):
        if step % spec["log_every"] == 0:
            actual = effective_parameters(parameters, codebooks, inner)
            scores = metrics(
                actual, data["selection_x"], data["selection_y"], population
            )
            with torch.no_grad():
                improved = scores["loss"] < best_loss
                best_loss[improved] = scores["loss"][improved]
                best_step[improved] = step
                for best, current in zip(best_parameters, actual):
                    best[improved] = current[improved]
            row = {
                "step": step,
                **{key: value.cpu().tolist() for key, value in scores.items()},
            }
            if accumulated_steps:
                row["train_loss"] = (
                    (train_sum / accumulated_steps).cpu().tolist()
                )
                train_sum.zero_()
                accumulated_steps = 0
            curves.append(row)
        if step == spec["candidate_steps"]:
            break
        bits = torch.randint(
            0,
            2,
            (plan["tasks"], spec["batch_size"], 64),
            device=device,
            generator=generator,
        ).float()
        labels = torch_labels(bits, data["teachers"]).repeat_interleave(
            population, 0
        )
        inputs = 2 * bits - 1 if plan["model"]["center_inputs"] else bits
        actual = effective_parameters(parameters, codebooks, inner)
        logits = forward(inputs.repeat_interleave(population, 0), actual)
        losses = functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(1)
        losses.sum().backward()
        train_sum += losses.detach()
        accumulated_steps += 1
        with torch.no_grad():
            grad_norm = sum(
                p.grad.square().flatten(1).sum(1) for p in parameters
            ).sqrt()
            factor = (
                spec["gradient_norm_clip"] / grad_norm.clamp_min(1e-12)
            ).clamp_max(1)
            for index, (parameter, momentum) in enumerate(
                zip(parameters, velocity)
            ):
                gradient = parameter.grad * factor[:, None, None]
                if index % 2 == 0:
                    gradient *= masks[index // 2]
                momentum.mul_(spec["momentum"]).add_(gradient)
                parameter.sub_(rates[:, None, None] * momentum)
                if inner == "indices":
                    codebook = codebooks[index]
                    parameter.clamp_(
                        codebook.min(1).values[:, None, None],
                        codebook.max(1).values[:, None, None],
                    )
                parameter.grad = None
    final = metrics(
        best_parameters, data["selection_x"], data["selection_y"], population
    )
    return (
        best_parameters,
        final,
        best_step.cpu().tolist(),
        curves,
        rates.cpu().tolist(),
    )


def successful(scores, spec):
    return (
        scores["accuracy"] >= spec["success_accuracy"]
        and min(scores["recall_positive"], scores["recall_negative"])
        >= spec["success_minimum_class_recall"]
    )


def run_search(plan, method, data, output, device):
    spec = plan["search"]
    population, tasks = spec["population"], plan["tasks"]
    rng = np.random.default_rng(plan["seed"] + 3000)
    current = [
        [sample_structure(plan, rng, dense=i == 0) for i in range(population)]
        for _ in range(tasks)
    ]
    archive, curves_all, progress, winners = [], [], [], [None] * tasks
    structure_arrays = {f"mask_{layer}": [] for layer in range(4)}
    structure_arrays["scales"] = []
    qualified = []
    start = time.perf_counter()
    for generation in range(spec["generations"]):
        if generation and method["outer"] == "random":
            current = [
                [sample_structure(plan, rng) for _ in range(population)]
                for _ in range(tasks)
            ]
        elif generation:
            next_population = []
            for task in range(tasks):
                rows = [
                    r
                    for r in archive
                    if r["task"] == task and r["generation"] == generation - 1
                ]
                elite_ids = np.argsort([r["selection"]["loss"] for r in rows])[
                    : spec["elite"]
                ]
                elite = [current[task][i] for i in elite_ids]
                children = [elite[0]]
                while len(children) < population - spec["immigrants"]:
                    left, right = rng.integers(len(elite), size=2)
                    children.append(
                        offspring(plan, elite[left], elite[right], rng)
                    )
                children.extend(
                    sample_structure(plan, rng)
                    for _ in range(spec["immigrants"])
                )
                next_population.append(children)
            current = next_population
        flattened = [s for task_population in current for s in task_population]
        fitted, scores, best_step, curves, rates = train_population(
            plan, flattened, method["inner"], generation, data, device
        )
        curves_all.append({"generation": generation, "curves": curves})
        for layer in range(4):
            structure_arrays[f"mask_{layer}"].append(
                np.stack([s["masks"][layer] for s in flattened])
            )
        structure_arrays["scales"].append(
            np.stack([s["scales"] for s in flattened])
        )
        for index, structure in enumerate(flattened):
            task = index // population
            selected = {
                key: float(value[index]) for key, value in scores.items()
            }
            row = {
                "task": task,
                "generation": generation,
                "candidate": index % population,
                "learning_rate": rates[index],
                "selected_step": best_step[index],
                "structure_hash": structure_hash(structure),
                "mask_hash": structure_hash(structure, include_scales=False),
                "active_weights": sum(
                    int(m.sum()) for m in structure["masks"]
                ),
                "selection": selected,
            }
            archive.append(row)
            if (
                winners[task] is None
                or selected["loss"]
                < winners[task]["record"]["selection"]["loss"]
            ):
                winners[task] = {
                    "record": row,
                    "structure": structure,
                    "parameters": [
                        p[index : index + 1].cpu().clone() for p in fitted
                    ],
                }
            if successful(selected, spec):
                qualified.append(
                    {
                        "record": row,
                        "parameters": [
                            p[index : index + 1].cpu().clone() for p in fitted
                        ],
                    }
                )
        row = {
            "generation": generation,
            "candidates_per_task": (generation + 1) * population,
            "seconds": time.perf_counter() - start,
            "incumbent_selection": [
                winner["record"]["selection"] for winner in winners
            ],
        }
        progress.append(row)
        save_json(output / "records.json", archive)
        save_json(output / "curves.json", curves_all)
        save_json(output / "progress.json", progress)
        print(
            json.dumps(
                {
                    "method": method["name"],
                    "generation": generation,
                    "seconds": round(row["seconds"], 1),
                    "best_selection_accuracy": [
                        w["record"]["selection"]["accuracy"] for w in winners
                    ],
                    "selection_qualified_candidates": len(qualified),
                }
            ),
            flush=True,
        )
    torch.cuda.synchronize()
    arrays = {key: np.stack(value) for key, value in structure_arrays.items()}
    np.savez_compressed(output / "structures.npz", **arrays)
    winner_parameters = [
        torch.cat([w["parameters"][index] for w in winners]).to(device)
        for index in range(8)
    ]
    np.savez_compressed(
        output / "winner_models.npz",
        **{
            f"parameter_{index}": value.cpu().numpy()
            for index, value in enumerate(winner_parameters)
        },
    )
    if qualified:
        np.savez_compressed(
            output / "selection_qualified_models.npz",
            **{
                f"parameter_{index}": torch.cat(
                    [q["parameters"][index] for q in qualified]
                ).numpy()
                for index in range(8)
            },
        )
        save_json(
            output / "selection_qualified_records.json",
            [q["record"] for q in qualified],
        )
    save_json(
        output / "winners_selection.json", [w["record"] for w in winners]
    )
    save_json(
        output / "runtime.json",
        {
            "seconds": time.perf_counter() - start,
            "fits": len(archive),
            "gradient_examples": len(archive)
            * spec["candidate_steps"]
            * spec["batch_size"],
            "unique_structures": len({r["structure_hash"] for r in archive}),
            "unique_masks": len({r["mask_hash"] for r in archive}),
        },
    )
    return winners, qualified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocol_search.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    validate(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("Use an available GPU outside sandbox")
    torch.set_num_threads(2)
    output = HERE / args.output
    output.mkdir(parents=True, exist_ok=False)
    for source, destination in [
        (Path(__file__), "search_snapshot.py"),
        (HERE / "run.py", "run_snapshot.py"),
        (args.protocol, "protocol.json"),
    ]:
        shutil.copyfile(source, output / destination)
    save_json(
        output / "seal.json",
        {
            "sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in [
                    "search_snapshot.py",
                    "run_snapshot.py",
                    "protocol.json",
                ]
            },
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "stage": "structural_source_search",
        },
    )
    source = np.load(HERE / plan["tasks_from"])
    tasks = [
        {
            key: source[key][i]
            for key in [
                "parents",
                "kinds_first",
                "kinds_second",
                "kind_output",
            ]
        }
        for i in range(plan["tasks"])
    ]
    shutil.copyfile(HERE / plan["tasks_from"], output / "source_tasks.npz")
    teachers = teacher_tensors(tasks, "cuda")
    generator = torch.Generator(device="cuda").manual_seed(plan["seed"] + 4000)
    selection_bits = torch.randint(
        0,
        2,
        (plan["tasks"], plan["search"]["selection_examples"], 64),
        generator=generator,
        device="cuda",
    ).float()
    evaluation_bits = torch.randint(
        0,
        2,
        (plan["tasks"], plan["search"]["evaluation_examples"], 64),
        generator=generator,
        device="cuda",
    ).float()
    data = {
        "teachers": teachers,
        "selection_x": 2 * selection_bits - 1
        if plan["model"]["center_inputs"]
        else selection_bits,
        "selection_y": torch_labels(selection_bits, teachers),
    }
    results = []
    for method in plan["methods"]:
        method_output = output / method["name"]
        method_output.mkdir()
        winners, qualified = run_search(
            plan, method, data, method_output, "cuda"
        )
        results.append((method, winners, qualified))
    # Hold evaluation labels out of all four outer searches and checkpoint choices.
    evaluation_y = torch_labels(evaluation_bits, teachers)
    evaluation_x = (
        2 * evaluation_bits - 1
        if plan["model"]["center_inputs"]
        else evaluation_bits
    )
    summaries = []
    for method, winners, qualified in results:
        parameters = [
            torch.cat([w["parameters"][index] for w in winners]).cuda()
            for index in range(8)
        ]
        evaluation = metrics(parameters, evaluation_x, evaluation_y, 1)
        records = []
        for task, winner in enumerate(winners):
            scores = {
                key: float(value[task]) for key, value in evaluation.items()
            }
            records.append(
                {
                    **winner["record"],
                    "evaluation": scores,
                    "success": successful(scores, plan["search"]),
                }
            )
        save_json(output / method["name"] / "winners_evaluation.json", records)
        qualified_evaluation = []
        for item in qualified:
            task = item["record"]["task"]
            scores = metrics(
                [p.cuda() for p in item["parameters"]],
                evaluation_x[task : task + 1],
                evaluation_y[task : task + 1],
                1,
            )
            scores = {key: float(value[0]) for key, value in scores.items()}
            qualified_evaluation.append(
                {
                    **item["record"],
                    "evaluation": scores,
                    "success": successful(scores, plan["search"]),
                }
            )
        save_json(
            output / method["name"] / "bank_evaluation.json",
            qualified_evaluation,
        )
        summary = {
            "method": method["name"],
            "successes": sum(r["success"] for r in records),
            "tasks": plan["tasks"],
            "mean_accuracy": float(
                np.mean([r["evaluation"]["accuracy"] for r in records])
            ),
            "mean_balanced_accuracy": float(
                np.mean(
                    [
                        (
                            r["evaluation"]["recall_positive"]
                            + r["evaluation"]["recall_negative"]
                        )
                        / 2
                        for r in records
                    ]
                )
            ),
            "selection_qualified_candidates": len(qualified),
            "independently_qualified_candidates": sum(
                q["success"] for q in qualified_evaluation
            ),
        }
        summaries.append(summary)
    save_json(
        output / "summary.json",
        {
            "stage": "structural_source_search",
            "methods": summaries,
            "transfer_evaluated": False,
            "prior_trained": False,
            "successful_bank_obtained": any(
                m["independently_qualified_candidates"] for m in summaries
            ),
            "source_tasks_previously_viewed": True,
        },
    )
    print("SUMMARY " + json.dumps(summaries), flush=True)


if __name__ == "__main__":
    main()

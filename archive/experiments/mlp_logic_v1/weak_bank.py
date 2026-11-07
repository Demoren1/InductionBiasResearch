"""Extract structure from partial solutions and evaluate fresh full tasks."""

import argparse
import hashlib
import json
import math
import shutil
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from run import forward, make_tasks, save_json, teacher_tensors, torch_labels
from search import metrics
from torch.nn import functional

HERE = Path(__file__).resolve().parent


def coassociation(features, importance):
    features = features**2
    features = features / np.maximum(features.sum(0, keepdims=True), 1e-12)
    importance = importance / max(importance.sum(), 1e-12)
    score = (features * importance[None]) @ features.T
    np.fill_diagonal(score, 0)
    return score / max(score.sum(), 1e-12)


def pair_clusters(clusters, score):
    available = set(range(len(clusters)))
    paired, pairs = [], []
    candidates = sorted(
        (
            (-float(score[np.ix_(left, right)].mean()), i, j)
            for i, left in enumerate(clusters)
            for j, right in enumerate(clusters)
            if i < j
        )
    )
    for _, i, j in candidates:
        if i in available and j in available:
            paired.append(clusters[i] + clusters[j])
            pairs.append([i, j])
            available.remove(i)
            available.remove(j)
    return paired, pairs


def extract_hierarchy(first, second):
    groups = [[i] for i in range(64)]
    while len(groups) > 8:
        groups, _ = pair_clusters(groups, first)
    _, pairs = pair_clusters(groups, second)
    return np.asarray(groups), np.asarray(pairs)


def hierarchy_masks(groups, pairs):
    first = np.zeros((64, 64), dtype=np.float32)
    for index, group in enumerate(groups):
        first[np.ix_(group, np.arange(8 * index, 8 * index + 8))] = 1
    second = np.zeros((64, 16), dtype=np.float32)
    for index, pair in enumerate(pairs):
        rows = np.concatenate([np.arange(8 * g, 8 * g + 8) for g in pair])
        second[np.ix_(rows, np.arange(4 * index, 4 * index + 4))] = 1
    return [
        first,
        second,
        np.ones((16, 4), dtype=np.float32),
        np.ones((4, 1), dtype=np.float32),
    ]


def pair_auc(score, truth):
    i, j = np.triu_indices(64, 1)
    labels = truth[i] == truth[j]
    positive, negative = score[i, j][labels], score[i, j][~labels]
    return float(
        (positive[:, None] > negative).mean()
        + 0.5 * (positive[:, None] == negative).mean()
    )


def structural_diagnostics(score, groups, truth_groups, rng, repeats):
    truth = np.empty(64, dtype=int)
    for index, group in enumerate(truth_groups):
        truth[group] = index
    auc = pair_auc(score, truth)
    null = [
        pair_auc(score, truth[rng.permutation(64)]) for _ in range(repeats)
    ]
    pairs = np.array(
        [
            (a, b)
            for group in groups
            for i, a in enumerate(group)
            for b in group[i + 1 :]
        ]
    )
    precision = float((truth[pairs[:, 0]] == truth[pairs[:, 1]]).mean())
    return {
        "pair_auc": auc,
        "permutation_p_upper": float(
            (1 + np.sum(np.asarray(null) >= auc)) / (repeats + 1)
        ),
        "group_pair_precision": precision,
        "chance_group_pair_precision": 7 / 63,
        "exact_groups": sum(
            any(set(group) == set(actual) for actual in truth_groups)
            for group in groups
        ),
        "permutation_auc": null,
    }


def build_prior(plan, output):
    source = HERE / plan["source_refit"]
    refit_plan = json.loads((source / "effective_protocol.json").read_text())
    source_tasks = np.load(HERE / "results/search_v1/source_tasks.npz")
    all_scores, model_rows = [], []
    dimensions = refit_plan["model"]["dimensions"]
    for method in refit_plan["methods"]:
        name = method["name"]
        records = json.loads(
            (source / name / "records_selection.json").read_text()
        )
        models = np.load(source / name / "fitted_models.npz")
        structures = np.load(source / name / "structures.npz")
        value_sum = np.zeros((2, 64, 64), dtype=np.float64)
        mask_sum = np.zeros_like(value_sum)
        mass = 0.0
        for row in records:
            record = row["selection"]
            p = np.clip(record["positive_rate"], 1e-8, 1 - 1e-8)
            entropy = -p * np.log(p) - (1 - p) * np.log(1 - p)
            quality = max(
                0.0,
                entropy - record["loss"] - plan["bank"]["minimum_bce_gain"],
            )
            model_rows.append(
                {
                    "method": name,
                    "task": row["task"],
                    "restart": row["restart"],
                    "validation_bce": record["loss"],
                    "constant_bce": float(entropy),
                    "quality_weight": float(quality),
                    "balanced_accuracy": (
                        record["recall_positive"] + record["recall_negative"]
                    )
                    / 2,
                }
            )
            if quality == 0:
                continue
            weights = [
                np.abs(
                    models[f"parameter_{2 * layer}"][row["model_index"]]
                ).astype(np.float64)
                for layer in range(4)
            ]
            masks = [
                structures[f"mask_{layer}"][row["task"]].astype(np.float64)
                for layer in range(4)
            ]
            value_first = coassociation(
                weights[0], (weights[1] @ weights[2] @ weights[3])[:, 0]
            )
            value_second = coassociation(
                weights[0] @ weights[1], (weights[2] @ weights[3])[:, 0]
            )
            mask_first = coassociation(masks[0], np.ones(dimensions[1]))
            mask_second = coassociation(
                masks[0] @ masks[1], np.ones(dimensions[2])
            )
            value_sum += quality * np.stack([value_first, value_second])
            mask_sum += quality * np.stack([mask_first, mask_second])
            mass += quality
        all_scores.append(
            {
                "method": name,
                "value_sum": value_sum,
                "mask_sum": mask_sum,
                "mass": mass,
            }
        )
    value_sum = sum(item["value_sum"] for item in all_scores)
    mask_sum = sum(item["mask_sum"] for item in all_scores)
    mass = sum(item["mass"] for item in all_scores)
    values = value_sum / max(mass, 1e-12)
    masks = mask_sum / max(mass, 1e-12)
    value_groups, value_pairs = extract_hierarchy(*values)
    mask_groups, mask_pairs = extract_hierarchy(*masks)
    # The proposal is frozen before consulting true groups for diagnostics.
    rng = np.random.default_rng(plan["seed"])
    permutation = rng.permutation(64)
    groups = {
        "weak_values": value_groups,
        "weak_masks": mask_groups,
        "shuffled": permutation[value_groups],
    }
    pairs = {
        "weak_values": value_pairs,
        "weak_masks": mask_pairs,
        "shuffled": value_pairs,
    }
    np.savez_compressed(
        output / "learned_prior.npz",
        value_scores=values,
        mask_scores=masks,
        **{f"{name}_groups": group for name, group in groups.items()},
        **{f"{name}_pairs": pair for name, pair in pairs.items()},
    )
    save_json(output / "source_quality_weights.json", model_rows)
    truth_groups = source_tasks["groups_diagnostic"]
    groups["oracle"] = truth_groups
    pairs["oracle"] = np.arange(8).reshape(4, 2)
    method_masks = {
        name: hierarchy_masks(group, pairs[name])
        for name, group in groups.items()
    }
    dense = [
        np.ones((left, right), dtype=np.float32)
        for left, right in pairwise(dimensions)
    ]
    method_masks["dense_adam"] = dense
    method_masks["dense_sgd"] = dense
    diagnostic = []
    for item in all_scores + [
        {"method": "pooled_values", "value_sum": value_sum, "mass": mass},
        {"method": "pooled_masks", "value_sum": mask_sum, "mass": mass},
    ]:
        score = item["value_sum"] / max(item["mass"], 1e-12)
        recovered, _ = extract_hierarchy(*score)
        diagnostic.append(
            {
                "method": item["method"],
                "quality_mass": item["mass"],
                **structural_diagnostics(
                    score[0],
                    recovered,
                    truth_groups,
                    rng,
                    plan["bank"]["permutation_diagnostics"],
                ),
            }
        )
    save_json(output / "structural_diagnostics.json", diagnostic)
    np.savez_compressed(
        output / "transfer_masks.npz",
        **{
            f"{name}_{layer}": mask
            for name, layers in method_masks.items()
            for layer, mask in enumerate(layers)
        },
    )
    print(
        "PRIOR "
        + json.dumps(
            [
                {
                    key: value
                    for key, value in item.items()
                    if key != "permutation_auc"
                }
                for item in diagnostic
            ]
        ),
        flush=True,
    )
    return method_masks, truth_groups


def fit_transfer(
    plan, tasks, cells, method_masks, steps, restarts, seed, output
):
    spec = plan["transfer"]
    task_count, cell_count = len(tasks), len(cells)
    torch.manual_seed(seed)
    teachers = teacher_tensors(tasks, "cuda")
    masks = [
        torch.from_numpy(
            np.stack([method_masks[cell["name"]][layer] for cell in cells])
        )
        .cuda()
        .repeat_interleave(task_count * restarts, 0)
        for layer in range(4)
    ]
    parameters = []
    for mask in masks:
        _, inputs, outputs = mask.shape
        base = torch.randn(
            task_count * restarts, inputs, outputs, device="cuda"
        ).repeat(cell_count, 1, 1)
        weights = base * mask * (2 / mask.sum(1).clamp_min(1)).sqrt()[:, None]
        parameters.extend(
            [
                weights.requires_grad_(),
                torch.zeros(
                    mask.shape[0], 1, outputs, device="cuda"
                ).requires_grad_(),
            ]
        )
    first = [torch.zeros_like(p) for p in parameters]
    second = [torch.zeros_like(p) for p in parameters]
    rates = torch.tensor(
        [cell["learning_rate"] for cell in cells], device="cuda"
    ).repeat_interleave(task_count * restarts)
    adam = torch.tensor(
        [cell["optimizer"] == "adam" for cell in cells], device="cuda"
    ).repeat_interleave(task_count * restarts)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    bits = torch.randint(
        0,
        2,
        (task_count, spec["selection_examples"], 64),
        device="cuda",
        generator=generator,
    ).float()
    selection_x, selection_y = 2 * bits - 1, torch_labels(bits, teachers)
    expanded_x = selection_x.repeat(cell_count, 1, 1)
    expanded_y = selection_y.repeat(cell_count, 1)
    count = len(cells) * task_count * restarts
    best_loss = torch.full((count,), float("inf"), device="cuda")
    best = [p.detach().clone() for p in parameters]
    best_step = torch.zeros(count, dtype=torch.int64, device="cuda")
    curves, train_sum, since = [], torch.zeros(count, device="cuda"), 0
    for step in range(steps + 1):
        if step % spec["log_every"] == 0:
            scores = metrics(parameters, expanded_x, expanded_y, restarts)
            with torch.no_grad():
                improved = scores["loss"] < best_loss
                best_loss[improved] = scores["loss"][improved]
                best_step[improved] = step
                for saved, current in zip(best, parameters):
                    saved[improved] = current[improved]
            row = {
                "step": step,
                **{key: value.cpu().tolist() for key, value in scores.items()},
            }
            if since:
                row["train_loss"] = (train_sum / since).cpu().tolist()
                train_sum.zero_()
                since = 0
            curves.append(row)
            save_json(output / "curves.json", curves)
            print(
                json.dumps(
                    {
                        "stage": output.name,
                        "step": step,
                        "mean_validation_bce": float(scores["loss"].mean()),
                    }
                ),
                flush=True,
            )
        if step == steps:
            break
        bits = torch.randint(
            0,
            2,
            (task_count, spec["batch_size"], 64),
            device="cuda",
            generator=generator,
        ).float()
        labels = (
            torch_labels(bits, teachers)
            .repeat_interleave(restarts, 0)
            .repeat(cell_count, 1)
        )
        inputs = (
            (2 * bits - 1)
            .repeat_interleave(restarts, 0)
            .repeat(cell_count, 1, 1)
        )
        logits = forward(inputs, parameters)
        losses = functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        ).mean(1)
        losses.sum().backward()
        train_sum += losses.detach()
        since += 1
        with torch.no_grad():
            norm = sum(
                p.grad.square().flatten(1).sum(1) for p in parameters
            ).sqrt()
            clip = (
                spec["gradient_norm_clip"] / norm.clamp_min(1e-12)
            ).clamp_max(1)
            for index, parameter in enumerate(parameters):
                grad = parameter.grad * clip[:, None, None]
                if index % 2 == 0:
                    grad *= masks[index // 2]
                first[index].mul_(0.9).add_(grad, alpha=0.1)
                second[index].mul_(0.999).addcmul_(grad, grad, value=0.001)
                denominator = (
                    second[index].sqrt() / math.sqrt(1 - 0.999 ** (step + 1))
                    + 1e-8
                )
                adam_update = (
                    first[index] / (1 - 0.9 ** (step + 1)) / denominator
                )
                update = torch.where(
                    adam[:, None, None], adam_update, 10 * first[index]
                )
                parameter.sub_(rates[:, None, None] * update)
                parameter.grad = None
    selection = metrics(best, expanded_x, expanded_y, restarts)
    records = []
    for index in range(count):
        cell = index // (task_count * restarts)
        task = (index // restarts) % task_count
        records.append(
            {
                "model_index": index,
                "cell": cell,
                "method": cells[cell]["name"],
                "optimizer": cells[cell]["optimizer"],
                "learning_rate": cells[cell]["learning_rate"],
                "task": task,
                "restart": index % restarts,
                "selected_step": int(best_step[index]),
                "selection": {
                    key: float(value[index])
                    for key, value in selection.items()
                },
            }
        )
    save_json(output / "records_selection.json", records)
    np.savez_compressed(
        output / "models.npz",
        **{
            f"parameter_{index}": value.cpu().numpy()
            for index, value in enumerate(best)
        },
    )
    return records, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=HERE / "protocol_weak_bank.json"
    )
    args = parser.parse_args()
    plan = json.loads(args.protocol.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Use a CUDA GPU outside sandbox")
    torch.set_num_threads(2)
    output = HERE / plan["output"]
    output.mkdir(parents=True, exist_ok=False)
    for original, name in [
        (Path(__file__), "weak_bank_snapshot.py"),
        (HERE / "search.py", "search_snapshot.py"),
        (HERE / "run.py", "run_snapshot.py"),
        (args.protocol, "protocol.json"),
    ]:
        shutil.copyfile(original, output / name)
    save_json(
        output / "seal.json",
        {
            "sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in [
                    "weak_bank_snapshot.py",
                    "search_snapshot.py",
                    "run_snapshot.py",
                    "protocol.json",
                ]
            },
            "source_refit_seal_sha256": hashlib.sha256(
                (HERE / plan["source_refit"] / "seal.json").read_bytes()
            ).hexdigest(),
        },
    )
    method_masks, truth_groups = build_prior(plan, output)
    family = json.loads((HERE / "protocol.json").read_text())
    rng = np.random.default_rng(plan["seed"] + 10000)
    spec = plan["transfer"]
    tasks, attempts = make_tasks(
        family,
        truth_groups,
        spec["tuning_tasks"] + spec["evaluation_tasks"],
        rng,
    )
    np.savez_compressed(
        output / "new_tasks.npz",
        **{key: np.stack([task[key] for task in tasks]) for key in tasks[0]},
    )
    save_json(output / "generation_attempts.json", attempts)
    tuning = output / "tuning"
    tuning.mkdir()
    cells = [
        {**method, "learning_rate": rate}
        for method in spec["methods"]
        for rate in spec[f"{method['optimizer']}_learning_rates"]
    ]
    records, _ = fit_transfer(
        plan,
        tasks[: spec["tuning_tasks"]],
        cells,
        method_masks,
        spec["tuning_steps"],
        spec["tuning_restarts"],
        plan["seed"] + 20000,
        tuning,
    )
    selected = []
    for method in spec["methods"]:
        options = [cell for cell in cells if cell["name"] == method["name"]]
        losses = [
            np.mean(
                [
                    min(
                        row["selection"]["loss"]
                        for row in records
                        if row["method"] == option["name"]
                        and row["learning_rate"] == option["learning_rate"]
                        and row["task"] == task
                    )
                    for task in range(spec["tuning_tasks"])
                ]
            )
            for option in options
        ]
        selected.append(options[int(np.argmin(losses))])
    save_json(output / "selected_settings.json", selected)
    evaluation = output / "evaluation"
    evaluation.mkdir()
    target_tasks = tasks[spec["tuning_tasks"] :]
    records, fitted = fit_transfer(
        plan,
        target_tasks,
        selected,
        method_masks,
        spec["evaluation_steps"],
        spec["evaluation_restarts"],
        plan["seed"] + 30000,
        evaluation,
    )
    generator = torch.Generator(device="cuda").manual_seed(
        plan["seed"] + 40000
    )
    bits = torch.randint(
        0,
        2,
        (spec["evaluation_tasks"], spec["evaluation_examples"], 64),
        device="cuda",
        generator=generator,
    ).float()
    labels = torch_labels(bits, teacher_tensors(target_tasks, "cuda"))
    scores = metrics(
        fitted,
        (2 * bits - 1).repeat(len(selected), 1, 1),
        labels.repeat(len(selected), 1),
        spec["evaluation_restarts"],
    )
    for index, row in enumerate(records):
        row["evaluation"] = {
            key: float(value[index]) for key, value in scores.items()
        }
    winners = [
        min(
            [
                row
                for row in records
                if row["method"] == method["name"] and row["task"] == task
            ],
            key=lambda row: row["selection"]["loss"],
        )
        for method in spec["methods"]
        for task in range(spec["evaluation_tasks"])
    ]
    save_json(evaluation / "records_evaluation.json", records)
    save_json(evaluation / "winners.json", winners)
    summary = []
    for method in spec["methods"]:
        rows = [row for row in winners if row["method"] == method["name"]]
        summary.append(
            {
                "method": method["name"],
                "mean_bce": float(
                    np.mean([row["evaluation"]["loss"] for row in rows])
                ),
                "mean_accuracy": float(
                    np.mean([row["evaluation"]["accuracy"] for row in rows])
                ),
                "mean_balanced_accuracy": float(
                    np.mean(
                        [
                            (
                                row["evaluation"]["recall_positive"]
                                + row["evaluation"]["recall_negative"]
                            )
                            / 2
                            for row in rows
                        ]
                    )
                ),
                "tasks": spec["evaluation_tasks"],
            }
        )
    save_json(
        output / "summary.json",
        {
            "stage": "weak_bank_transfer",
            "source_exact_solutions_required": False,
            "teacher_used_for_prior": False,
            "methods": summary,
        },
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

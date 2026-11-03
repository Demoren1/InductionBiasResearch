"""Prepare sources, calibrate, and evaluate frozen structural transfer."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.adaptive_tanh_v1.source import scalar_kmeans
from experiments.basin_structure_transfer_v1.core import (
    canonicalize,
    fit,
    make_tasks,
)
from pattern.autoregressive.model import IndexMADE

ROOT = Path(__file__).resolve().parent


def save_fit(directory, result, history, timing):
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "models.npz", **result)
    with gzip.open(directory / "history.json.gz", "wt") as stream:
        json.dump(history, stream)
    (directory / "timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps({"fit": str(directory), **timing}), flush=True)


def prepare(plan, output, device):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(ROOT / "protocol.json", output / "protocol.json")
    spec = make_tasks(plan, "source", device)
    source_plan = {**plan, "batch": plan["source_batch"]}
    configs = [
        {
            "optimizer": "adam",
            "lr": plan["source_lr"],
            "radius": plan["source_radius"],
        }
    ]
    result, history, timing = fit(
        source_plan,
        spec,
        configs,
        plan["source_starts"],
        plan["seed"] + 10,
        device,
        steps=plan["source_steps"],
    )
    save_fit(output / "source", result, history, timing)
    rows = np.arange(len(spec["families"]))
    choice = result["query"][:, 0].argmin(1)
    weights = result["weights"][rows, 0, choice]
    torch.manual_seed(plan["seed"] + 11)
    raw = scalar_kmeans(torch.tensor(weights, device=device), plan["groups"])
    inferred = canonicalize(raw.cpu().numpy())
    train = np.tile(
        np.arange(plan["source_tasks_per_family"])
        < plan["source_train_per_family"],
        plan["families"],
    )
    np.savez_compressed(
        output / "structure_bank.npz",
        train=inferred[train],
        validation=inferred[~train],
    )
    np.savez_compressed(
        output / "numeric_baseline_bank.npz", weights=weights[train]
    )
    np.savez_compressed(
        output / "source_diagnostic.npz",
        inferred=inferred,
        teacher=spec["indices"],
        weights=weights,
        families=spec["families"],
        population=result["population"][rows, 0, choice],
    )
    diagnostic = {
        "exact_source_indexators": int(
            np.all(inferred == spec["indices"], axis=1).sum()
        ),
        "source_functions": len(rows),
        "unique_train_indexators": len(np.unique(inferred[train], axis=0)),
        "maximum_selected_source_mse": float(
            result["population"][rows, 0, choice].max()
        ),
        "extraction": "scalar k-means; known K; first-occurrence labels",
    }
    (output / "source_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2)
    )
    print(json.dumps(diagnostic), flush=True)
    bank = np.load(output / "structure_bank.npz")
    training = torch.tensor(bank["train"], device=device)
    validation = torch.tensor(bank["validation"], device=device)
    for index, seed in enumerate(plan["prior_seeds"]):
        torch.manual_seed(seed)
        model = IndexMADE(
            plan["inputs"], plan["groups"], plan["prior_width"]
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=plan["prior_lr"])
        best, history = float("inf"), []
        for step in range(plan["prior_steps"] + 1):
            if step % 100 == 0:
                with torch.no_grad():
                    value = float(
                        F.cross_entropy(
                            model(validation).flatten(0, 1),
                            validation.flatten(),
                        )
                    )
                    history.append({"step": step, "validation_nll": value})
                    if value < best:
                        best = value
                        torch.save(
                            model.state_dict(), output / f"prior_{index}.pt"
                        )
            if step == plan["prior_steps"]:
                break
            loss = F.cross_entropy(
                model(training).flatten(0, 1), training.flatten()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        (output / f"prior_{index}_history.json").write_text(
            json.dumps(history, indent=2)
        )
        print(
            json.dumps({"prior": index, "best_validation_nll": best}),
            flush=True,
        )


def load_prior(plan, output, index, device):
    model = IndexMADE(plan["inputs"], plan["groups"], plan["prior_width"]).to(
        device
    )
    model.load_state_dict(
        torch.load(
            output / f"prior_{index}.pt",
            map_location=device,
            weights_only=True,
        )
    )
    return model.eval().requires_grad_(False)


@torch.no_grad()
def draw(plan, bank, method, tasks, starts, seed, model=None, temperature=1):
    torch.manual_seed(seed)
    count = tasks * starts
    if method == "ar":
        result = model.sample_cached(count, temperature)
    elif method == "retrieval":
        result = bank[torch.randint(len(bank), (count,), device=bank.device)]
    elif method == "frequency":
        probability = F.one_hot(bank, plan["groups"]).float().mean(0)
        result = torch.multinomial(probability, count, replacement=True).T
    elif method == "random":
        result = torch.rand(count, plan["inputs"], device=bank.device).argsort(
            1
        )
        result %= plan["groups"]
    else:
        raise ValueError(method)
    return result.cpu().numpy().reshape(tasks, starts, plan["inputs"])


def choose(result, configs):
    """Calibrate only on empirical query; never on population/oracle values."""
    query = result["query"].min(2).mean(0)
    selected = int(query.argmin())
    return {**configs[selected], "calibration_query": float(query[selected])}


def calibrate(plan, output, device):
    directory = output / "calibration"
    directory.mkdir(exist_ok=False)
    spec = make_tasks(plan, "calibration", device)
    starts, tasks = plan["calibration_starts"], len(spec["families"])
    bank = torch.tensor(
        np.load(output / "structure_bank.npz")["train"], device=device
    )
    pretrained = torch.tensor(
        np.load(output / "numeric_baseline_bank.npz")["weights"], device=device
    )
    proposals = draw(plan, bank, "retrieval", tasks, starts, plan["seed"] + 20)
    chosen, code_choices = {}, []
    for mode in ("code", "dense", "long", "source_transfer"):
        for optimizer in plan["optimizers"]:
            if mode == "source_transfer" and optimizer != "adam":
                continue
            configs = [
                {"optimizer": optimizer, "lr": lr, "radius": radius}
                for radius in plan["radii"]
                for lr in plan["rates"]
            ]
            count = (
                starts // plan["long_multiplier"] if mode == "long" else starts
            )
            steps = plan["steps"] * (
                plan["long_multiplier"] if mode == "long" else 1
            )
            result, history, timing = fit(
                plan,
                spec,
                configs,
                count,
                plan["seed"] + 30,
                device,
                indices=proposals if mode == "code" else None,
                source_weights=pretrained
                if mode == "source_transfer"
                else None,
                steps=steps,
            )
            name = f"{mode}_{optimizer}"
            save_fit(directory / name, result, history, timing)
            chosen[name] = choose(result, configs)
            if mode == "code":
                code_choices.append(chosen[name])
    chosen["code"] = min(
        code_choices, key=lambda item: item["calibration_query"]
    )
    release_choices = []
    for optimizer in plan["optimizers"]:
        configs = [
            {
                "optimizer": optimizer,
                "lr": lr,
                "radius": chosen["code"]["radius"],
            }
            for lr in plan["rates"]
        ]
        result, history, timing = fit(
            plan,
            spec,
            configs,
            starts,
            plan["seed"] + 30,
            device,
            indices=proposals,
            release_initial=True,
        )
        save_fit(directory / f"release_{optimizer}", result, history, timing)
        release_choices.append(choose(result, configs))
    chosen["release"] = min(
        release_choices, key=lambda item: item["calibration_query"]
    )
    temperatures = []
    for temperature in plan["temperatures"]:
        queries = []
        for index in range(len(plan["prior_seeds"])):
            model = load_prior(plan, output, index, device)
            proposals = draw(
                plan,
                bank,
                "ar",
                tasks,
                starts,
                plan["seed"] + 40 + index,
                model,
                temperature,
            )
            result, history, timing = fit(
                plan,
                spec,
                [chosen["code"]],
                starts,
                plan["seed"] + 30,
                device,
                indices=proposals,
            )
            save_fit(
                directory / f"temperature_{temperature}_{index}",
                result,
                history,
                timing,
            )
            queries.append(float(result["query"].min(2).mean()))
        temperatures.append(
            {"temperature": temperature, "query": float(np.mean(queries))}
        )
    chosen["temperature"] = min(temperatures, key=lambda item: item["query"])
    (output / "frozen_selection.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps({"selection": chosen}), flush=True)


def input_hashes(output, plan):
    paths = [
        ROOT / "core.py",
        ROOT / "run.py",
        output / "protocol.json",
        output / "structure_bank.npz",
        output / "numeric_baseline_bank.npz",
        output / "frozen_selection.json",
    ]
    paths += [
        output / f"prior_{index}.pt"
        for index in range(len(plan["prior_seeds"]))
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def evaluate(plan, output, device):
    directory = output / "evaluation"
    directory.mkdir(exist_ok=False)
    hashes = input_hashes(output, plan)
    (directory / "frozen_before_target.json").write_text(
        json.dumps(hashes, indent=2)
    )
    chosen = json.loads((output / "frozen_selection.json").read_text())
    spec = make_tasks(plan, "evaluation", device)
    np.savez_compressed(
        directory / "teacher_diagnostic.npz",
        **{
            key: value.cpu().numpy()
            if isinstance(value, torch.Tensor)
            else value
            for key, value in spec.items()
        },
    )
    tasks, starts = len(spec["families"]), plan["starts"]
    bank = torch.tensor(
        np.load(output / "structure_bank.npz")["train"], device=device
    )
    pretrained = torch.tensor(
        np.load(output / "numeric_baseline_bank.npz")["weights"], device=device
    )
    methods = [f"ar_{index}" for index in range(len(plan["prior_seeds"]))]
    methods += ["retrieval", "frequency", "random", "oracle"]
    methods += [f"dense_{optimizer}" for optimizer in plan["optimizers"]]
    methods += [f"long_{optimizer}" for optimizer in plan["optimizers"]]
    methods += ["source_transfer_adam", "release_ar_0"]
    ar_initial = None
    for method in methods:
        indices = None
        release = method == "release_ar_0"
        count, steps = starts, plan["steps"]
        if method.startswith("ar_"):
            index = int(method.split("_")[-1])
            model = load_prior(plan, output, index, device)
            indices = draw(
                plan,
                bank,
                "ar",
                tasks,
                starts,
                plan["seed"] + 30040 + index,
                model,
                chosen["temperature"]["temperature"],
            )
            if index == 0:
                ar_initial = indices.copy()
        elif method in ("retrieval", "frequency", "random"):
            indices = draw(
                plan, bank, method, tasks, starts, plan["seed"] + 30040
            )
        elif method == "oracle":
            indices = np.repeat(spec["indices"][:, None], starts, axis=1)
        elif release:
            indices = ar_initial
        elif method.startswith("long_"):
            count //= plan["long_multiplier"]
            steps *= plan["long_multiplier"]
        config = chosen[
            "release" if release else "code" if indices is not None else method
        ]
        result, history, timing = fit(
            plan,
            spec,
            [config],
            count,
            plan["seed"] + 30030,
            device,
            indices=indices,
            release_initial=release,
            steps=steps,
            source_weights=pretrained
            if method == "source_transfer_adam"
            else None,
        )
        if indices is not None:
            result["indices"] = indices
        save_fit(directory / method, result, history, timing)
    (directory / "complete.json").write_text(
        json.dumps(
            {
                "frozen_inputs_unchanged": hashes
                == input_hashes(output, plan),
                "known_functions": int(
                    (spec["families"] < plan["families"]).sum()
                ),
                "foreign_functions": plan["foreign_tasks"],
                "steps_per_search": starts * plan["steps"],
                "main_method_generates": "indexator only",
                "main_numeric_initialization": (
                    "fresh; exact chosen radius; common radius grid with dense"
                ),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "calibrate", "evaluate"))
    args = parser.parse_args()
    plan = json.loads((ROOT / "protocol.json").read_text())
    if plan["inputs"] % plan["groups"]:
        raise ValueError("Teacher groups must have equal sizes")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    output = Path(plan["output"])
    if args.stage != "prepare":
        if plan != json.loads((output / "protocol.json").read_text()):
            raise ValueError("Protocol changed after source preparation")
    {"prepare": prepare, "calibrate": calibrate, "evaluate": evaluate}[
        args.stage
    ](plan, output, device)


if __name__ == "__main__":
    main()

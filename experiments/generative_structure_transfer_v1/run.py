"""Learn structures, freeze the generator, and fit fresh target codebooks."""

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from experiments.adaptive_tanh_v1.source import align_indices
from experiments.generative_structure_transfer_v1.fit import (
    evaluate_attempts,
    fit_numeric,
)
from pattern.autoregressive.model import IndexMADE

ROOT = Path(__file__).resolve().parent


def prepare_bank(plan, output):
    matrices, qualities, roles, numeric_weights = [], [], [], []
    for family in range(plan["families"]):
        bank = np.load(output / f"source_{family}" / "bank.npz")
        quality = bank["quality"].reshape(
            plan["source_tasks"], plan["source_starts"]
        )
        chosen = np.arange(plan["source_tasks"]) * plan["source_starts"]
        chosen += quality.argmin(1)
        training = np.arange(plan["source_tasks"]) < plan["prior_train_tasks"]
        raw = bank["raw_indices"][chosen]
        quality = bank["quality"][chosen]
        reference = raw[np.flatnonzero(training)[quality[training].argmin()]]
        matrices.extend(
            align_indices(matrix, reference, plan["groups"])[0]
            for matrix in raw
        )
        qualities.extend(quality)
        roles.extend(training)
        numeric_weights.extend(bank["weights"][chosen[training]])
    matrices = np.stack(matrices).reshape(len(matrices), -1)
    roles = np.array(roles, dtype=bool)
    np.savez_compressed(
        output / "structure_bank.npz",
        train=matrices[roles],
        validation=matrices[~roles],
        quality=np.array(qualities),
    )
    np.savez_compressed(
        output / "numeric_baseline_bank.npz", weights=np.stack(numeric_weights)
    )


def train_prior(plan, output, device):
    bank = np.load(output / "structure_bank.npz")
    train = torch.as_tensor(bank["train"], device=device)
    validation = torch.as_tensor(bank["validation"], device=device)
    torch.manual_seed(plan["seed"] + 10)
    model = IndexMADE(
        plan["inputs"] * plan["hidden"],
        plan["groups"],
        plan["prior_width"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=plan["prior_lr"])
    history, best = [], float("inf")
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(plan["prior_steps"] + 1):
        if step % plan["prior_log_every"] == 0:
            with torch.no_grad():
                train_nll = F.cross_entropy(
                    model(train).flatten(0, 1), train.flatten()
                )
                validation_nll = F.cross_entropy(
                    model(validation).flatten(0, 1), validation.flatten()
                )
                item = {
                    "step": step,
                    "train_nll": float(train_nll),
                    "validation_nll": float(validation_nll),
                }
                history.append(item)
                if item["validation_nll"] < best:
                    best = item["validation_nll"]
                    torch.save(model.state_dict(), output / "prior.pt")
                print(json.dumps({"stage": "prior", **item}), flush=True)
        if step == plan["prior_steps"]:
            break
        batch = train[torch.randint(
            len(train), (plan["prior_batch"],), device=device
        )]
        loss = F.cross_entropy(model(batch).flatten(0, 1), batch.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize(device)
    (output / "prior_history.json").write_text(json.dumps(history, indent=2))
    (output / "prior_summary.json").write_text(json.dumps({
        "seconds": time.perf_counter() - started,
        "best_validation_nll": best,
        "training_structures": len(train),
        "validation_structures": len(validation),
        "unique_training_structures": len(torch.unique(train, dim=0)),
        "generator_inputs": ["inferred source indexators"],
        "generator_outputs": ["categorical indexators"],
        "numeric_codebook_outputs": False,
    }, indent=2))


def load_prior(plan, output, device):
    model = IndexMADE(
        plan["inputs"] * plan["hidden"],
        plan["groups"],
        plan["prior_width"],
    ).to(device)
    model.load_state_dict(torch.load(
        output / "prior.pt", map_location=device, weights_only=True
    ))
    model.eval().requires_grad_(False)
    bank = np.load(output / "structure_bank.npz")
    train = torch.as_tensor(bank["train"], device=device)
    frequency = F.one_hot(train, plan["groups"]).float().sum(0) + 0.01
    frequency /= frequency.sum(1, keepdim=True)
    return model, train, frequency


@torch.no_grad()
def propose(plan, model, bank, frequency, method, tasks, temperature, seed):
    torch.manual_seed(seed)
    device = bank.device
    positions = plan["inputs"] * plan["hidden"]
    count = tasks * plan["starts"]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if method == "ar":
        indices = model.sample_cached(count, temperature)
    elif method == "frequency":
        probabilities = (frequency.log() / temperature).softmax(1)
        indices = torch.multinomial(probabilities, count, replacement=True).T
    elif method == "retrieval":
        indices = bank[torch.randint(len(bank), (count,), device=device)]
    elif method == "random":
        indices = torch.rand(count, positions, device=device).argsort(1)
        indices %= plan["groups"]
    else:
        raise ValueError(f"Unknown proposal method: {method}")
    torch.cuda.synchronize(device)
    return (
        indices.reshape(tasks, plan["starts"], positions),
        time.perf_counter() - started,
    )


def benchmark_tasks(plan, output, evaluation, device):
    """Create benchmark functions; no teacher values enter the solver/prior."""
    torch.manual_seed(plan["seed"] + (200000 if evaluation else 100000))
    per_family = plan[
        "evaluation_tasks_per_family" if evaluation
        else "calibration_tasks_per_family"
    ]
    dimensions, hidden = plan["inputs"], plan["hidden"]
    structures, families = [], []
    for family in range(plan["families"]):
        source = np.load(output / f"source_{family}" / "source_data.npz")
        structures.extend([source["teacher_i"]] * per_family)
        families.extend([family] * per_family)
    if evaluation:
        foreign = torch.randperm(dimensions * hidden, device=device)
        foreign = (foreign % plan["groups"]).reshape(dimensions, hidden)
        structures.extend([foreign.cpu().numpy()] * plan["foreign_tasks"])
        families.extend([plan["families"]] * plan["foreign_tasks"])
    teacher_i = torch.as_tensor(np.stack(structures), device=device)
    tasks = len(teacher_i)
    codes = torch.rand(tasks, plan["groups"], device=device) * 2 - 1
    weights = codes.gather(1, teacher_i.flatten(1))
    weights = weights.reshape(tasks, dimensions, hidden)
    maximum_support = max(plan["support_sizes"])
    samples = maximum_support + plan["query"]
    if evaluation:
        samples += plan["test"]
    x = torch.randn(tasks, samples, dimensions, device=device)
    y = torch.bmm(x, weights).tanh().sum(2) / hidden**0.5
    support = plan["support"]
    query_end = maximum_support + plan["query"]
    data = {
        "train_x": x[:, :support],
        "train_y": y[:, :support],
        "query_x": x[:, maximum_support:query_end],
        "query_y": y[:, maximum_support:query_end],
    }
    if evaluation:
        data.update(test_x=x[:, query_end:], test_y=y[:, query_end:])
    truth = {"indexators": teacher_i, "values": codes, "families": families}
    return data, truth


def save_fit(folder, result, history, timing):
    folder.mkdir()
    np.savez_compressed(folder / "models.npz", **result)
    with gzip.open(folder / "history.json.gz", "wt") as handle:
        json.dump(history, handle)
    (folder / "timing.json").write_text(json.dumps(timing, indent=2))


def frozen_inputs(output):
    files = [
        ROOT / "protocol.json", ROOT / "run.py", ROOT / "fit.py",
        Path("pattern/autoregressive/model.py"),
        Path("experiments/adaptive_tanh_v1/source.py"),
        output / "prior.pt", output / "structure_bank.npz",
        output / "numeric_baseline_bank.npz",
    ]
    files.extend(sorted(output.glob("source_*/bank.npz")))
    files.extend(sorted(output.glob("source_*/source_data.npz")))
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def calibrate(plan, output, device):
    folder = output / f"calibration_{plan['support']}"
    folder.mkdir()
    model, bank, frequency = load_prior(plan, output, device)
    data, _ = benchmark_tasks(plan, output, False, device)
    tasks = len(data["train_x"])
    choices = {}
    records = []
    seed = plan["seed"] + 300000
    retrieval, _ = propose(
        plan, model, bank, frequency, "retrieval", tasks, 1.0, seed
    )
    for scale in plan["initial_stds"]:
        result, history, timing = fit_numeric(
            plan, data, retrieval, plan["code_rates"], "sgdm", scale,
            seed + 1, plan["steps"],
        )
        save_fit(folder / f"code_scale_{scale}", result, history, timing)
        scores = result["query_nmse"].min(2).mean(0)
        for rate, score in zip(plan["code_rates"], scores):
            records.append({
                "method": "code", "rate": rate,
                "initial_std": scale, "score": float(score),
            })
    choices["code"] = min(records, key=lambda row: row["score"])
    print(json.dumps({"stage": "calibrate_code", **choices["code"]}),
          flush=True)

    for method in ["ar", "frequency"]:
        candidates = []
        for temperature in plan["temperatures"]:
            indices, sampling_seconds = propose(
                plan, model, bank, frequency, method, tasks, temperature,
                seed + 2,
            )
            result, history, timing = fit_numeric(
                plan, data, indices, [choices["code"]["rate"]], "sgdm",
                choices["code"]["initial_std"], seed + 1, plan["steps"],
            )
            timing["sampling_seconds"] = sampling_seconds
            save_fit(
                folder / f"{method}_temperature_{temperature}",
                result, history, timing,
            )
            candidates.append({
                "temperature": temperature,
                "score": float(result["query_nmse"].min(2).mean()),
            })
        choices[method] = min(candidates, key=lambda row: row["score"])
        records.extend({"method": method, **row} for row in candidates)
        print(json.dumps({"stage": f"calibrate_{method}", **choices[method]}),
              flush=True)

    for method, optimizer in [("dense_sgd", "sgdm"), ("dense_adam", "adam")]:
        for scale in plan["initial_stds"]:
            rates = plan[f"{method}_rates"]
            result, history, timing = fit_numeric(
                plan, data, None, rates, optimizer, scale, seed + 1,
                plan["steps"] * plan["long_dense_multiplier"],
            )
            save_fit(
                folder / f"{method}_scale_{scale}", result, history, timing
            )
            short_query = np.stack([
                row["query_nmse"] for row in history
                if row["step"] <= plan["steps"]
            ])
            short_query = np.nan_to_num(short_query, nan=np.inf)
            short_query = short_query.min(0).reshape(
                result["query_nmse"].shape
            )
            for label, query in [
                (method, short_query), (f"{method}_long", result["query_nmse"])
            ]:
                scores = query.min(2).mean(0)
                for rate, score in zip(rates, scores):
                    records.append({
                        "method": label, "rate": rate,
                        "initial_std": scale, "score": float(score),
                        "optimizer": optimizer,
                    })
        for label in [method, f"{method}_long"]:
            choices[label] = min(
                (row for row in records if row["method"] == label),
                key=lambda row: row["score"],
            )
            print(json.dumps({"stage": "calibrate_dense", **choices[label]}),
                  flush=True)

    pretrained = torch.as_tensor(
        np.load(output / "numeric_baseline_bank.npz")["weights"], device=device
    )
    for optimizer in ["sgdm", "adam"]:
        rate_key = (
            "dense_sgd_rates" if optimizer == "sgdm" else "dense_adam_rates"
        )
        rates = plan[rate_key]
        result, history, timing = fit_numeric(
            plan, data, None, rates, optimizer, plan["initial_std"],
            seed + 1, plan["steps"], pretrained,
        )
        save_fit(
            folder / f"source_transfer_{optimizer}", result, history, timing
        )
        for rate, score in zip(rates, result["query_nmse"].min(2).mean(0)):
            records.append({
                "method": "source_transfer", "optimizer": optimizer,
                "rate": rate, "score": float(score),
                "initial_std": plan["initial_std"],
            })
    choices["source_transfer"] = min(
        (row for row in records if row["method"] == "source_transfer"),
        key=lambda row: row["score"],
    )
    choices["dense_reference"] = min(
        ["dense_sgd", "dense_adam"], key=lambda key: choices[key]["score"]
    )
    (folder / "scores.json").write_text(json.dumps(records, indent=2))
    frozen = {
        "support": plan["support"], "selection": choices,
        "hashes": frozen_inputs(output),
        "target_data_created": False,
    }
    (output / f"frozen_selection_{plan['support']}.json").write_text(
        json.dumps(frozen, indent=2)
    )
    print(json.dumps({"stage": "frozen", "support": plan["support"],
                      "selection": choices}), flush=True)


def evaluate(plan, output, device):
    frozen_path = output / f"frozen_selection_{plan['support']}.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["hashes"] != frozen_inputs(output):
        raise RuntimeError("Frozen inputs changed before evaluation")
    if frozen["support"] != plan["support"]:
        raise ValueError("Frozen support size does not match this arm")
    folder = output / f"evaluation_{plan['support']}"
    folder.mkdir()
    (folder / "frozen_before_target.json").write_text(frozen_path.read_text())
    choices = frozen["selection"]
    model, bank, frequency = load_prior(plan, output, device)
    data, truth = benchmark_tasks(plan, output, True, device)
    numeric_data = {
        key: value for key, value in data.items() if not key.startswith("test")
    }
    np.savez_compressed(folder / "data.npz", **{
        key: value.cpu().numpy() for key, value in data.items()
    })
    np.savez_compressed(
        folder / "teacher.npz",
        indexators=truth["indexators"].cpu().numpy(),
        values=truth["values"].cpu().numpy(),
        families=np.array(truth["families"]),
    )
    tasks = len(data["train_x"])
    pretrained = torch.as_tensor(
        np.load(output / "numeric_baseline_bank.npz")["weights"], device=device
    )
    methods = [
        "ar", "frequency", "retrieval", "random", "oracle",
        "dense_sgd", "dense_adam", "dense_sgd_long", "dense_adam_long",
        "source_transfer",
    ]
    final = {}
    for method_id, method in enumerate(methods):
        sampling_seconds = 0.0
        indices = None
        optimizer = "sgdm"
        steps = plan["steps"]
        numeric_source = None
        if method in {"ar", "frequency", "retrieval", "random"}:
            temperature = choices.get(method, {}).get("temperature", 1.0)
            indices, sampling_seconds = propose(
                plan, model, bank, frequency, method, tasks, temperature,
                plan["seed"] + 400000 + method_id,
            )
            settings = choices["code"]
        elif method == "oracle":
            indices = truth["indexators"].flatten(1)[:, None]
            indices = indices.expand(-1, plan["starts"], -1).clone()
            settings = choices["code"]
        else:
            settings = choices[method]
            optimizer = settings["optimizer"]
            if method.endswith("_long"):
                steps *= plan["long_dense_multiplier"]
            if method == "source_transfer":
                numeric_source = pretrained
        result, history, timing = fit_numeric(
            plan, numeric_data, indices, [settings["rate"]], optimizer,
            settings["initial_std"], plan["seed"] + 500000, steps,
            numeric_source,
        )
        timing["sampling_seconds"] = sampling_seconds
        method_folder = folder / method
        save_fit(method_folder, result, history, timing)
        errors = evaluate_attempts(plan, result, data)
        np.savez_compressed(method_folder / "test_metrics.npz", **errors)
        query = result["query_nmse"][:, 0]
        selected = query.argmin(1)
        rows = np.arange(tasks)
        test = errors["test_nmse"][rows, selected]
        final[method] = {
            "query_nmse": query[rows, selected].tolist(),
            "test_nmse": test.tolist(),
            "start": selected.tolist(),
            "seconds": timing["seconds"],
            "sampling_seconds": sampling_seconds,
        }
        known = np.array(truth["families"]) < plan["families"]
        print(json.dumps({
            "stage": "evaluation", "support": plan["support"],
            "method": method, "known_nmse": float(test[known].mean()),
            "known_successes": int((test[known] < plan["success_nmse"]).sum()),
            "foreign_nmse": float(test[~known].mean()), **timing,
        }), flush=True)
    reference = choices["dense_reference"]
    choose_ar = np.array(final["ar"]["query_nmse"]) < np.array(
        final[reference]["query_nmse"]
    )
    final["portfolio"] = {
        "dense_reference": reference,
        "choose_ar": choose_ar.tolist(),
        "query_nmse": np.minimum(
            final["ar"]["query_nmse"], final[reference]["query_nmse"]
        ).tolist(),
        "test_nmse": np.where(
            choose_ar, final["ar"]["test_nmse"], final[reference]["test_nmse"]
        ).tolist(),
        "seconds": final["ar"]["seconds"] + final[reference]["seconds"],
        "sampling_seconds": final["ar"]["sampling_seconds"],
    }
    (folder / "selected_summary.json").write_text(json.dumps({
        "families": truth["families"], "support": plan["support"],
        "methods": final,
    }, indent=2))
    (folder / "complete.json").write_text(json.dumps({
        "frozen_inputs_unchanged": frozen["hashes"] == frozen_inputs(output),
        "methods": methods, "support": plan["support"],
        "target_codebooks_learned_fresh": True,
        "generator_outputs": "indexators_only",
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prior", "calibrate", "evaluate"],
                        required=True)
    parser.add_argument("--support", type=int, default=128)
    args = parser.parse_args()
    plan = json.loads((ROOT / "protocol.json").read_text())
    for name, value in plan.items():
        if isinstance(value, (int, float)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Protocol {name} must be finite and positive")
        elif isinstance(value, list):
            if not value or not all(np.isfinite(x) and x > 0 for x in value):
                raise ValueError(f"Protocol {name} must contain positive values")
    integer_fields = [
        "seed", "inputs", "hidden", "groups", "families", "source_tasks",
        "source_starts", "source_support", "source_query", "source_steps",
        "source_batch", "prior_train_tasks", "prior_width", "prior_steps",
        "prior_batch", "prior_log_every", "calibration_tasks_per_family",
        "evaluation_tasks_per_family", "foreign_tasks", "support", "query",
        "test", "starts", "steps", "long_dense_multiplier", "log_every",
    ]
    if any(type(plan[name]) is not int for name in integer_fields):
        raise ValueError("Protocol dimensions and counts must be integers")
    if any(type(size) is not int or size < 2 for size in plan["support_sizes"]):
        raise ValueError("Support sizes must be integers of at least two")
    if plan["momentum"] >= 1 or plan["steps"] % plan["log_every"]:
        raise ValueError("Invalid momentum or short-horizon logging interval")
    if args.support not in plan["support_sizes"]:
        parser.error("support must be one of the predeclared sample sizes")
    if not 0 < plan["prior_train_tasks"] < plan["source_tasks"]:
        raise ValueError("Separate source train/validation tasks are required")
    positions = plan["inputs"] * plan["hidden"]
    if positions % plan["groups"]:
        raise ValueError("Balanced teacher needs an integer group size")
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")
    plan["support"] = args.support
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    output = Path(plan["output"])
    if args.stage == "prior":
        if (output / "prior.pt").exists():
            raise FileExistsError("A prior already exists in this directory")
        prepare_bank(plan, output)
        train_prior(plan, output, device)
    elif args.stage == "calibrate":
        calibrate(plan, output, device)
    else:
        evaluate(plan, output, device)


if __name__ == "__main__":
    main()

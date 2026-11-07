"""Single-pattern search with exactly zero and two learnable codebook entries."""

import argparse
import itertools
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from pattern.datasets import input_table, labels_for

from .data import BASE, digest, load_archive, observations, save_json
from .model import BatchedAdam, balanced_loss, clipped, initial_numeric, metrics
from .search import roots


def canonical(indices, numeric=None):
    best = None
    for mapping in (np.array([0, 1, 2]), np.array([0, 2, 1])):
        renamed = mapping[indices]
        columns = sorted(range(8), key=lambda column: tuple(renamed[:, column]))
        result = renamed[:, columns].astype(np.int16)
        key = tuple(result.ravel())
        if best is None or key < best[0]:
            converted = None if numeric is None else numeric.copy()
            if converted is not None:
                converted[..., :2] = numeric[..., np.argsort(mapping)[1:] - 1]
            best = (key, result, converted, mapping, columns)
    return best[1:]


def prepare(protocol):
    plan = json.loads(protocol.read_text())
    if plan["pattern"] != "0110" or plan["codes"] != 2 or plan["classes"] != 3:
        raise ValueError(
            "This experiment fixes 0110 and exactly three assignment classes"
        )
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "protocol.json", plan)
    data = observations([plan["pattern"]], plan["seed"], plan)
    heldout = {key: value for key, value in data.items() if key.startswith("test_")}
    observed = {
        key: value for key, value in data.items() if not key.startswith("test_")
    }
    np.savez_compressed(output / "source.npz", **observed)
    np.savez_compressed(output / "heldout.npz", **heldout)
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    for path in BASE.iterdir():
        if path.suffix in (".py", ".json", ".md"):
            shutil.copy2(path, snapshot / path.name)
    save_json(
        output / "started.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": digest(output / "protocol.json"),
            "source_sha256": digest(output / "source.npz"),
            "heldout_sha256": digest(output / "heldout.npz"),
            "heldout_evaluated": False,
            "software_tests_run": False,
        },
    )
    print("PREPARED", str(output), flush=True)


def predict(inputs, indices, parameters):
    codes = parameters.shape[-1] - 3
    codebook = torch.cat(
        [torch.zeros_like(parameters[:, :1]), parameters[:, :codes]], 1
    )
    weights = codebook.gather(1, indices.abs().flatten(1)).reshape(-1, 11, 8)
    weights = weights * indices.sign()
    hidden = (
        torch.einsum("ni,fih->fnh", inputs, weights) + parameters[:, codes, None, None]
    ).relu()
    return (
        hidden.sum(-1) * parameters[:, codes + 1, None] + parameters[:, codes + 2, None]
    )


@torch.no_grad()
def score_split(data, split, indices, parameters):
    inputs = torch.as_tensor(data[split + "_x"], device=parameters.device)
    target = torch.as_tensor(data[split + "_y"][0], device=parameters.device)
    return metrics(predict(inputs, indices, parameters), target[None])


def fit_fixed(data, indices, initial, rates, specification, seed, device):
    started = time.perf_counter()
    paired_steps = specification.get("paired_initialization_steps", 0)
    if paired_steps < 0 or 2 * paired_steps >= specification["steps"]:
        raise ValueError("Paired initialization must leave a positive fitting budget")
    index = torch.tensor(indices, dtype=torch.long, device=device)
    parameters = torch.tensor(
        initial, dtype=torch.float32, device=device
    ).requires_grad_()
    optimizer = BatchedAdam(parameters, rates)
    support_x = torch.tensor(data["support_x"], device=device)
    support_y = torch.tensor(data["support_y"][0], device=device)
    rng = torch.Generator(device=device).manual_seed(seed + 701)

    def numerical_step(current_index, current_parameters, current_optimizer):
        ids = torch.randint(
            len(support_x), (specification["batch"],), generator=rng, device=device
        )
        loss = balanced_loss(
            predict(support_x[ids], current_index, current_parameters),
            support_y[None, ids],
        )
        loss = (
            loss + specification["numeric_l2"] * current_parameters.square().sum(-1) / 2
        )
        gradient = torch.autograd.grad(loss.sum(), current_parameters)[0]
        current_optimizer.step(clipped(gradient, specification["gradient_clip"], (-1,)))

    paired_state = {}
    if paired_steps:
        mirrored = np.asarray(initial).copy()
        mirrored[:, : initial.shape[-1] - 3] *= -1
        candidates = np.concatenate([initial, mirrored])
        paired_index = index.repeat(2, 1, 1)
        paired_parameters = torch.tensor(
            candidates, dtype=torch.float32, device=device
        ).requires_grad_()
        paired_optimizer = BatchedAdam(paired_parameters, list(rates) * 2)
        for _ in range(paired_steps):
            numerical_step(paired_index, paired_parameters, paired_optimizer)
        with torch.no_grad():
            warm_query = score_split(data, "query", paired_index, paired_parameters)
            branch = warm_query[len(initial) :, 0] < warm_query[: len(initial), 0]
            chosen = torch.arange(len(initial), device=device) + branch * len(initial)
        parameters = paired_parameters[chosen].detach().clone().requires_grad_()
        optimizer = BatchedAdam(parameters, rates)
        optimizer.first.copy_(paired_optimizer.first[chosen])
        optimizer.second.copy_(paired_optimizer.second[chosen])
        optimizer.steps = paired_steps
        paired_state = {
            "polarity_branch": branch.cpu().numpy(),
            "selected_initial_parameters": candidates[chosen.cpu().numpy()],
            "warmup_query_metrics": warm_query.cpu()
            .numpy()
            .reshape(2, len(initial), 4)
            .transpose(1, 0, 2),
            "paired_initialization_steps": np.full(len(initial), paired_steps),
            "gradient_evaluations_per_fit": np.full(
                len(initial), specification["steps"]
            ),
        }
    best_loss = torch.full((len(initial),), torch.inf, device=device)
    best_parameters = parameters.detach().clone()
    best_step = torch.zeros(len(initial), dtype=torch.long, device=device)
    history_steps, training, query, recall = [], [], [], []
    remaining = specification["steps"] - 2 * paired_steps
    for step in range(remaining + 1):
        if step % specification["checkpoint_every"] == 0 or step == remaining:
            with torch.no_grad():
                current_query = score_split(data, "query", index, parameters)
                current_train = score_split(data, "support", index, parameters)
                improved = current_query[:, 0] < best_loss
                best_loss[improved] = current_query[improved, 0]
                best_parameters[improved] = parameters[improved]
                best_step[improved] = step + paired_steps
                history_steps.append(step + paired_steps)
                training.append(current_train[:, 0].cpu().numpy())
                query.append(current_query[:, 0].cpu().numpy())
                recall.append(current_query[:, 2].cpu().numpy())
                if not torch.isfinite(current_query).all():
                    raise FloatingPointError("Non-finite single-pattern numeric fit")
        if step == remaining:
            break
        numerical_step(index, parameters, optimizer)
    result = {
        "indices": np.asarray(indices, dtype=np.int16),
        "initial_parameters": np.asarray(initial),
        "parameters": best_parameters.cpu().numpy(),
        "best_step": best_step.cpu().numpy(),
        "history_steps": np.array(history_steps),
        "train_history": np.stack(training),
        "query_history": np.stack(query),
        "recall_history": np.stack(recall),
        "seconds": np.array(time.perf_counter() - started),
        **paired_state,
    }
    for split in ("support", "query", "selection"):
        result[split + "_metrics"] = (
            score_split(data, split, index, best_parameters).cpu().numpy()
        )
    return result


def collect(plan, output, phase):
    entries, matrices, inherited = [], [], []
    count = 0
    for folder in sorted((output / phase / "search").glob("shard_*")):
        json.loads((folder / "complete.json").read_text())
        specifications = json.loads((folder / "roots.json").read_text())
        snapshots = load_archive(folder / "snapshots.npz")
        count += len(specifications)
        for snapshot, step in enumerate(snapshots["steps"]):
            for root, specification in enumerate(specifications):
                index, numeric, mapping, columns = canonical(
                    snapshots["alpha"][snapshot, root].argmax(-1),
                    snapshots["numeric"][snapshot, root, 0],
                )
                matrices.append(index)
                inherited.append(numeric)
                entries.append(
                    {
                        **specification,
                        "entry": len(entries),
                        "step": int(step),
                        "folder": str(folder.relative_to(output)),
                        "snapshot": snapshot,
                        "root": root,
                        "mapping": mapping.tolist(),
                        "columns": columns,
                        "soft_query": snapshots["soft_query"][snapshot, root, 0]
                        .mean(0)
                        .tolist(),
                        "hard_query": snapshots["inherited_hard_query"][
                            snapshot, root, 0
                        ]
                        .mean(0)
                        .tolist(),
                    }
                )
    if count != len(roots(plan, output, phase)):
        raise ValueError("All registered roots must finish before collecting")
    folder = output / phase
    np.savez_compressed(
        folder / "hard_candidates.npz",
        indices=np.stack(matrices),
        inherited_numeric=np.stack(inherited),
    )
    save_json(folder / "hard_candidates.json", entries)
    print("COLLECTED", phase, len(entries), "checkpoints", flush=True)


def fine_tune(plan, output, phase, device, shard, shards):
    folder = output / phase
    archive = load_archive(folder / "hard_candidates.npz")
    entries = json.loads((folder / "hard_candidates.json").read_text())
    ids = np.array(
        [number for number in range(len(entries)) if number % shards == shard]
    )
    restarts = plan["search"]["numeric_restarts"]
    indices = np.repeat(archive["indices"][ids], restarts, axis=0)
    initial = archive["inherited_numeric"][ids].reshape(-1, 5)
    rates = [entries[number]["numeric_lr"] for number in ids for _ in range(restarts)]
    result = fit_fixed(
        load_archive(output / "source.npz"),
        indices,
        initial,
        rates,
        plan["fit"],
        plan["seed"] + 10000001,
        device,
    )
    np.savez_compressed(
        folder / f"hard_fits_{shard}.npz", entry_ids=np.repeat(ids, restarts), **result
    )
    print(
        "HARD FITTED", phase, shard, len(initial), float(result["seconds"]), flush=True
    )


def combined(folder):
    pieces = [load_archive(path) for path in sorted(folder.glob("hard_fits_*.npz"))]
    result = {}
    for key in pieces[0]:
        if key in ("history_steps", "seconds"):
            result[key] = pieces[0][key]
        else:
            axis = 1 if key.endswith("history") else 0
            result[key] = np.concatenate([piece[key] for piece in pieces], axis=axis)
    order = np.argsort(result["entry_ids"], kind="stable")
    for key, value in result.items():
        if key not in ("history_steps", "seconds"):
            result[key] = value[:, order] if key.endswith("history") else value[order]
    return result


def select(plan, output, phase):
    folder = output / phase
    entries = json.loads((folder / "hard_candidates.json").read_text())
    fits = combined(folder)
    expected = np.repeat(np.arange(len(entries)), plan["search"]["numeric_restarts"])
    if not np.array_equal(fits["entry_ids"], expected):
        raise ValueError("Missing or duplicate inherited-codebook fits")
    ranks = []
    for number, origin in enumerate(entries):
        ids = np.flatnonzero(fits["entry_ids"] == number)
        ranks.append(
            {
                "entry": number,
                "origin": origin,
                "selection_bce": float(fits["selection_metrics"][ids, 0].mean()),
                "selection_balanced_accuracy": float(
                    fits["selection_metrics"][ids, 1].mean()
                ),
                "query_bce": float(fits["query_metrics"][ids, 0].mean()),
            }
        )
    ranks.sort(key=lambda row: (row["selection_bce"], row["entry"]))
    per_root = {}
    for row in ranks:
        origin = row["origin"]
        per_root.setdefault((origin["id"], origin["order"], origin["search_seed"]), row)
    groups = []
    for configuration, order in sorted({key[:2] for key in per_root}):
        seeds = [
            row for key, row in per_root.items() if key[:2] == (configuration, order)
        ]
        groups.append(
            {
                "configuration": configuration,
                "order": order,
                "seeds": seeds,
                "mean_selection_bce": float(
                    np.mean([row["selection_bce"] for row in seeds])
                ),
            }
        )
    groups.sort(
        key=lambda row: (row["mean_selection_bce"], row["configuration"], row["order"])
    )
    selected = {
        "groups": groups,
        "selected_group": groups[0],
        "primary": min(groups[0]["seeds"], key=lambda row: row["selection_bce"]),
        "heldout_evaluated": False,
    }
    if phase == "pilot":
        selected["top_configurations"] = [
            row["configuration"]
            for row in groups[: plan["search"]["confirmation_top_configs"]]
        ]
    save_json(output / (phase + "_selection.json"), selected)
    save_json(folder / "ranking.json", ranks)
    print(
        json.dumps({"best_group": groups[0], "primary": selected["primary"]}, indent=2),
        flush=True,
    )


def freeze(plan, output):
    selection = json.loads((output / "confirmation_selection.json").read_text())
    candidates = load_archive(output / "confirmation/hard_candidates.npz")
    fits = combined(output / "confirmation")
    primary = selection["primary"]
    rows = np.flatnonzero(fits["entry_ids"] == primary["entry"])
    row = rows[np.argmin(fits["query_metrics"][rows, 0])]
    np.savez_compressed(
        output / "frozen.npz",
        indices=candidates["indices"][primary["entry"]],
        parameters=fits["parameters"][row],
        fit_row=np.array(row),
        by_seed=np.stack(
            [
                candidates["indices"][seed["entry"]]
                for seed in selection["selected_group"]["seeds"]
            ]
        ),
    )
    save_json(
        output / "frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "selected_group": selection["selected_group"],
            "primary": primary,
            "fit_row": int(row),
            "state_sha256": digest(output / "frozen.npz"),
            "heldout_evaluated": False,
        },
    )
    print("FROZEN", primary, flush=True)


def fresh_calibrate(plan, output, device):
    frozen = load_archive(output / "frozen.npz")
    settings = [
        {"lr": rate, "norm": norm}
        for rate, norm in itertools.product(
            plan["fit"]["fresh_learning_rates"], plan["fit"]["fresh_initial_norms"]
        )
    ]
    restarts = plan["fit"]["calibration_restarts"]
    indices, initial, rates = [], [], []
    for setting in settings:
        values = fresh_parameters(restarts, setting["norm"], plan["seed"] + 12000001)
        initial.append(values)
        indices.append(np.broadcast_to(frozen["indices"], (restarts, 11, 8)))
        rates += [setting["lr"]] * restarts
    result = fit_fixed(
        load_archive(output / "source.npz"),
        np.concatenate(indices),
        np.concatenate(initial),
        rates,
        plan["fit"],
        plan["seed"] + 12000001,
        device,
    )
    np.savez_compressed(output / "fresh_calibration.npz", **result)
    scores = result["selection_metrics"][:, 0].reshape(len(settings), restarts).mean(1)
    chosen = int(scores.argmin())
    save_json(
        output / "fresh_setting.json",
        {
            **settings[chosen],
            "selection_bce": float(scores[chosen]),
            "all_settings": [
                {**setting, "selection_bce": float(score)}
                for setting, score in zip(settings, scores)
            ],
            "heldout_evaluated": False,
        },
    )
    print("FRESH SETTING", settings[chosen], flush=True)


def fresh_parameters(restarts, norm, seed):
    values = (
        initial_numeric(
            [seed], 1, restarts, norm, torch.device("cpu"), codes=2, ordered_signed=True
        )
        .numpy()
        .reshape(-1, 5)
    )
    values[1::2, :2] = values[1::2, :2][:, ::-1]
    return values


def evaluate(plan, output, device):
    frozen = json.loads((output / "frozen.json").read_text())
    state = load_archive(output / "frozen.npz")
    setting = json.loads((output / "fresh_setting.json").read_text())
    pilot = json.loads((output / "pilot_selection.json").read_text())
    pilot_indices = load_archive(output / "pilot/hard_candidates.npz")["indices"][
        pilot["primary"]["entry"]
    ]
    methods = [("primary", state["indices"]), ("pilot_reference", pilot_indices)]
    individual_path = output / "individual_reference.npz"
    if individual_path.exists():
        methods.append(
            ("best_individual_source", load_archive(individual_path)["indices"])
        )
    for row, indices in zip(frozen["selected_group"]["seeds"], state["by_seed"]):
        seed = row["origin"]["search_seed"]
        if seed != frozen["primary"]["origin"]["search_seed"]:
            methods.append((f"search_seed_{seed}", indices))
    truth = np.zeros((11, 8), dtype=np.int16)
    pattern_codes = np.array([1 if bit == "0" else 2 for bit in plan["pattern"]])
    for hidden in range(8):
        truth[hidden : hidden + 4, hidden] = pattern_codes
    oracle = canonical(truth)[0]
    rng = np.random.default_rng(plan["seed"] + 13000001)
    random_indices = canonical(
        rng.permutation(state["indices"].ravel()).reshape(11, 8)
    )[0]
    origin = frozen["primary"]["origin"]
    initial = load_archive(output / origin["folder"] / "snapshots.npz")["alpha"][
        0, origin["root"]
    ].argmax(-1)
    methods.extend(
        [
            ("oracle", oracle),
            ("random_same_counts", random_indices),
            ("initial_indexator", canonical(initial)[0]),
        ]
    )
    save_json(
        output / "evaluation_frozen.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "architecture_sha256": digest(output / "frozen.npz"),
            "fresh_setting_sha256": digest(output / "fresh_setting.json"),
            "methods": [name for name, _ in methods],
            "heldout_evaluated": False,
        },
    )
    restarts = plan["fit"]["evaluation_restarts"]
    start = fresh_parameters(restarts, setting["norm"], plan["seed"] + 14000001)
    indices = np.repeat(np.stack([index for _, index in methods]), restarts, axis=0)
    initial = np.tile(start, (len(methods), 1))
    observed = load_archive(output / "source.npz")
    result = fit_fixed(
        observed,
        indices,
        initial,
        [setting["lr"]] * len(initial),
        plan["fit"],
        plan["seed"] + 14000001,
        device,
    )
    heldout = load_archive(output / "heldout.npz")
    index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    parameters = torch.tensor(result["parameters"], device=device)
    x = torch.tensor(heldout["test_x"], device=device)
    y = torch.tensor(heldout["test_y"][0], device=device)
    _, all_x = input_table()
    all_x = all_x.to(device)
    all_y = labels_for(all_x, plan["pattern"])
    with torch.no_grad():
        result["test_metrics"] = (
            metrics(predict(x, index_tensor, parameters), y[None]).cpu().numpy()
        )
        result["population_metrics"] = (
            metrics(predict(all_x, index_tensor, parameters), all_y[None]).cpu().numpy()
        )
        inherited_parameters = torch.tensor(state["parameters"][None], device=device)
        inherited_index = torch.tensor(
            state["indices"][None], dtype=torch.long, device=device
        )
        inherited_test = (
            metrics(predict(x, inherited_index, inherited_parameters), y[None])[0]
            .cpu()
            .tolist()
        )
        inherited_population = (
            metrics(predict(all_x, inherited_index, inherited_parameters), all_y[None])[
                0
            ]
            .cpu()
            .tolist()
        )
        analytic_index = torch.tensor(truth[None], dtype=torch.long, device=device)
        analytic_parameters = torch.tensor(
            [[-1.0, 1.0, -3.0, 20.0, -10.0]], device=device
        )
        analytic = (
            metrics(predict(all_x, analytic_index, analytic_parameters), all_y[None])[0]
            .cpu()
            .tolist()
        )
    np.savez_compressed(
        output / "evaluation_fits.npz",
        methods=np.array([name for name, _ in methods]),
        **result,
    )
    summary = {
        "inherited_primary_test": inherited_test,
        "inherited_primary_population": inherited_population,
        "analytic_oracle_population": analytic,
        "methods": {},
    }
    for method, (name, index) in enumerate(methods):
        ids = np.arange(method * restarts, (method + 1) * restarts)
        chosen = ids[np.argmin(result["query_metrics"][ids, 0])]
        summary["methods"][name] = {
            "indices": index.tolist(),
            "active_edges": int((index > 0).sum()),
            "mean_test_metrics": result["test_metrics"][ids].mean(0).tolist(),
            "selected_test_metrics": result["test_metrics"][chosen].tolist(),
            "selected_population_metrics": result["population_metrics"][
                chosen
            ].tolist(),
            "selected_row": int(chosen),
            "selected_restart": int(chosen - ids[0]),
            "selected_parameters": result["parameters"][chosen].tolist(),
        }
    save_json(output / "evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "collect",
            "fine_tune",
            "select",
            "freeze",
            "fresh_calibrate",
            "evaluate",
        ),
    )
    parser.add_argument("--protocol", type=Path, default=BASE / "single_protocol.json")
    parser.add_argument(
        "--output", type=Path, default=BASE / "results/single_0110_20261104"
    )
    parser.add_argument("--phase", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args.protocol)
        return
    torch.set_num_threads(2)
    plan = json.loads((args.output / "protocol.json").read_text())
    if args.stage == "collect":
        collect(plan, args.output, args.phase)
    elif args.stage == "fine_tune":
        fine_tune(
            plan,
            args.output,
            args.phase,
            torch.device(args.device),
            args.shard,
            args.shards,
        )
    elif args.stage == "select":
        select(plan, args.output, args.phase)
    elif args.stage == "freeze":
        freeze(plan, args.output)
    elif args.stage == "fresh_calibrate":
        fresh_calibrate(plan, args.output, torch.device(args.device))
    else:
        evaluate(plan, args.output, torch.device(args.device))


if __name__ == "__main__":
    main()

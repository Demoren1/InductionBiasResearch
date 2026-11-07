"""Confirm prior settings and tune numerical fitting without target feedback."""

import argparse
import gzip
import itertools
import json

import numpy as np
import torch

from experiments.sine_ar_current_v1.run import fit
from experiments.sine_ar_repair_v1.models import Prior
from experiments.sine_ar_repair_v1.run import (
    BASE,
    FIT_PLAN,
    OUTPUT,
    PLAN,
    ROOT,
    SOURCE,
    calibration_tasks,
    configurations,
    dump,
    load_models,
    query_scores,
    solve,
    source_masks,
    tasks,
)

SELECTION_PLAN = json.loads((BASE / "selection_protocol.json").read_text())


class EmpiricalPrior:
    def __init__(self, mode, bank):
        self.mode = mode
        self.bank = bank

    @torch.no_grad()
    def sample(self, count, scale, bank, sampling_noise=False):
        bank = self.bank
        device = bank.device
        if self.mode == "frequency":
            return (
                torch.rand(count, 16, device=device) < bank.float().mean(0)
            ).long()
        if self.mode == "uniform":
            indices = torch.rand(count, 16, device=device).argsort(dim=1)[
                :, :2
            ]
            return torch.zeros(
                count, 16, device=device, dtype=torch.long
            ).scatter_(1, indices, 1)
        if self.mode == "replay":
            return bank[torch.randint(len(bank), (count,), device=device)]
        parts = [
            bank[torch.randint(len(bank), (count,), device=device), j::2]
            for j in range(2)
        ]
        return torch.stack(parts, dim=-1).reshape(count, 16)


class MixedPrior:
    def __init__(
        self, joint, block, joint_scale, block_scale, block_probability
    ):
        self.joint = joint
        self.block = block
        self.joint_scale = joint_scale
        self.block_scale = block_scale
        self.block_probability = block_probability

    def sample(self, count, scale, bank, sampling_noise=False):
        joint = self.joint.sample(count, self.joint_scale, bank)
        block = self.block.sample(count, self.block_scale, bank)
        choose = (
            torch.rand(count, 1, device=bank.device) < self.block_probability
        )
        return torch.where(choose, block, joint)


def generators(setting, device):
    if "mixture" in setting:
        joint_setting, block_setting = setting["mixture"]
        joint = load_models(joint_setting, device)
        block = load_models(block_setting, device)
        return [
            MixedPrior(
                j,
                b,
                joint_setting["scale"],
                block_setting["scale"],
                setting["block_probability"],
            )
            for j, b in zip(joint, block)
        ]
    if "empirical" in setting:
        bank = source_masks(device)
        return [EmpiricalPrior(setting["empirical"], bank) for _ in range(3)]
    if "frozen" in setting:
        if setting["frozen"] == "strict":
            folder = SOURCE
        else:
            folder = (
                ROOT
                / "experiments/sine_ar_current_v1/results"
                / "diagnosis_20261103/threshold_0.35"
            )
        result = []
        for index in range(3):
            model = Prior({"family": "joint", "width": 128}).to(device)
            model.net.load_state_dict(
                torch.load(
                    folder / f"prior_{index}.pt",
                    map_location=device,
                    weights_only=True,
                )
            )
            result.append(model.eval().requires_grad_(False))
        return result
    return load_models(setting, device)


def policy(numerical=None, allocation=None):
    numerical = numerical or SELECTION_PLAN["numerical_grid"][0]
    allocation = allocation or SELECTION_PLAN["allocation_grid"][0]
    return {
        **numerical,
        **allocation,
        "proposals": 65536 // numerical["steps"] // 2 // allocation["repeats"],
    }


def save_selected(path, result):
    ti, si = np.indices(result["selected"].shape)
    chosen = result["selected"]
    np.savez_compressed(
        path,
        query=result["query"],
        masks=result["masks"].astype(np.uint8),
        selected=chosen,
        selected_query=result["selected_query"],
        selected_population=result["selected_population"],
        selected_weights=result["weights"][ti, si, chosen],
        selected_amplitude=result["amplitude"][ti, si, chosen],
    )


def candidates():
    rows = []
    for config in configurations():
        folder = OUTPUT / config["id"]
        if not (folder / "SCREEN_COMPLETE.json").exists():
            raise RuntimeError(f"Screen incomplete: {config['id']}")
        rows.extend(json.loads((folder / "screen.json").read_text()))
    chosen = {}
    for family in sorted({r["family"] for r in rows}):
        order = sorted(
            (r for r in rows if r["family"] == family),
            key=lambda r: r["mean_query"],
        )
        chosen[family] = order[
            : SELECTION_PLAN["confirmation_candidates_per_family"]
        ]
    dump(OUTPUT / "screen_summary.json", rows)
    dump(OUTPUT / "confirmation_candidates.json", chosen)
    dump(OUTPUT / "selection_protocol.json", SELECTION_PLAN)
    print(json.dumps(chosen), flush=True)


def tune_family(family, device):
    options = json.loads(
        (OUTPUT / "confirmation_candidates.json").read_text()
    )[family]
    folder = OUTPUT / "confirmation" / family
    folder.mkdir(parents=True)
    specs = calibration_tasks(PLAN["confirmation_functions"])
    searches = PLAN["confirmation_searches_per_seed"]
    seed = PLAN["confirmation_seed"]
    confirmed = []
    for index, setting in enumerate(options):
        result, _ = solve(
            generators(setting, device), setting, specs, searches, seed, device
        )
        confirmed.append({"setting": setting, **query_scores(result)})
        save_selected(folder / f"prior_{index}.npz", result)
    best = min(confirmed, key=lambda r: r["mean_query"])
    setting = best["setting"]
    models = generators(setting, device)
    numerical_rows = []
    for index, numerical in enumerate(SELECTION_PLAN["numerical_grid"]):
        current = policy(numerical)
        if index == 0:
            row = {
                "policy": current,
                **{
                    key: value
                    for key, value in best.items()
                    if key != "setting"
                },
            }
        else:
            result, _ = solve(
                models, setting, specs, searches, seed, device, current
            )
            row = {"policy": current, **query_scores(result)}
            save_selected(folder / f"numerical_{index}.npz", result)
        numerical_rows.append(row)
    best_numerical = min(numerical_rows, key=lambda r: r["mean_query"])
    allocation_rows = []
    for index, allocation in enumerate(SELECTION_PLAN["allocation_grid"]):
        current = policy(
            {k: best_numerical["policy"][k] for k in ["steps", "lr", "scale"]},
            allocation,
        )
        if index == 0:
            row = best_numerical
        else:
            result, _ = solve(
                models, setting, specs, searches, seed, device, current
            )
            row = {"policy": current, **query_scores(result)}
            save_selected(folder / f"allocation_{index}.npz", result)
        allocation_rows.append(row)
    selected = min(allocation_rows, key=lambda r: r["mean_query"])
    record = {
        "family": family,
        "setting": setting,
        **selected,
        "confirmed_priors": confirmed,
        "numerical_grid": numerical_rows,
        "allocation_grid": allocation_rows,
    }
    dump(folder / "selection.json", record)
    print(
        json.dumps({"selected": family, "prior": setting, **selected}),
        flush=True,
    )


def tune_controls(device):
    specs = calibration_tasks(PLAN["confirmation_functions"])
    searches = PLAN["confirmation_searches_per_seed"]
    seed = PLAN["confirmation_seed"]
    folder = OUTPUT / "confirmation" / "controls"
    folder.mkdir(parents=True)
    families = json.loads(
        (OUTPUT / "confirmation_candidates.json").read_text()
    )
    selected = {
        name: json.loads(
            (OUTPUT / "confirmation" / name / "selection.json").read_text()
        )
        for name in families
    }
    joint = selected["joint"]["setting"]
    block = selected["block"]["setting"]
    mixture_rows = []
    for probability in SELECTION_PLAN["mixture_block_probabilities"]:
        setting = {
            "mixture": [joint, block],
            "block_probability": probability,
            "scale": 1.0,
        }
        current = selected["block"]["policy"]
        result, _ = solve(
            generators(setting, device),
            setting,
            specs,
            searches,
            seed,
            device,
            current,
        )
        mixture_rows.append(
            {"setting": setting, "policy": current, **query_scores(result)}
        )
        save_selected(folder / f"mixture_{probability}.npz", result)
    dense = {}
    for method in ["sgd", "adam"]:
        base = [
            {"optimizer": method, "lr": lr, "scale": scale}
            for lr, scale in itertools.product(
                FIT_PLAN[f"{method}_rates"], FIT_PLAN["dense_scales"]
            )
        ]
        repetitions = SELECTION_PLAN[
            "dense_calibration_repetitions_per_setting"
        ]
        configs = [c for c in base for _ in range(repetitions)]
        masks = np.ones((len(specs), len(configs), 64, 8, 2), dtype=np.uint8)
        result, _ = fit(FIT_PLAN, specs, masks, configs, seed, device)
        query = (
            result["query"].min(2).reshape(len(specs), len(base), repetitions)
        )
        scores = query.mean((0, 2))
        rows = [
            {"config": c, "mean_query": float(score)}
            for c, score in zip(base, scores)
        ]
        dense[method] = {
            "selected": min(rows, key=lambda r: r["mean_query"]),
            "grid": rows,
        }
        np.savez_compressed(
            folder / f"dense_{method}_query.npz",
            query=result["query"],
            population=result["population"],
        )
    soft = {"frozen": "relaxed", "scale": 1.0}
    soft_rows = []
    for index, allocation in enumerate(SELECTION_PLAN["allocation_grid"]):
        current = policy(allocation=allocation)
        result, _ = solve(
            generators(soft, device),
            soft,
            specs,
            searches,
            seed,
            device,
            current,
        )
        soft_rows.append(
            {"setting": soft, "policy": current, **query_scores(result)}
        )
        save_selected(folder / f"relaxed_allocation_{index}.npz", result)
    dump(
        folder / "selection.json",
        {
            "mixture": min(mixture_rows, key=lambda r: r["mean_query"]),
            "mixture_grid": mixture_rows,
            "dense": dense,
            "soft_selection": min(soft_rows, key=lambda r: r["mean_query"]),
            "soft_grid": soft_rows,
        },
    )


def freeze():
    families = json.loads(
        (OUTPUT / "confirmation_candidates.json").read_text()
    )
    methods = []
    for name in families:
        selected = json.loads(
            (OUTPUT / "confirmation" / name / "selection.json").read_text()
        )
        methods.append(
            {
                "name": name,
                "setting": selected["setting"],
                "policy": selected["policy"],
            }
        )
        methods.append(
            {
                "name": f"{name}_prior_only",
                "setting": selected["setting"],
                "policy": policy(),
            }
        )
    controls = json.loads(
        (OUTPUT / "confirmation/controls/selection.json").read_text()
    )
    for name, key in [
        ("joint_block_mixture", "mixture"),
        ("relaxed_selection_tuned", "soft_selection"),
    ]:
        selected = controls[key]
        methods.append(
            {
                "name": name,
                "setting": selected["setting"],
                "policy": selected["policy"],
            }
        )
    for name, frozen, scale in [
        ("strict_t2", "strict", 2),
        ("strict_t6", "strict", 6),
        ("relaxed_t1", "relaxed", 1),
    ]:
        methods.append(
            {
                "name": name,
                "setting": {"frozen": frozen, "scale": scale},
                "policy": policy(),
            }
        )
    for mode in ["frequency", "product", "replay", "uniform"]:
        methods.append(
            {
                "name": mode,
                "setting": {"empirical": mode, "scale": 1.0},
                "policy": policy(),
            }
        )
    for method in ["sgd", "adam"]:
        methods.append(
            {
                "name": f"dense_{method}",
                "dense": controls["dense"][method]["selected"]["config"],
            }
        )
    methods.append(
        {
            "name": "oracle",
            "dense": {"optimizer": "sgd", "lr": 0.05, "scale": 2.0},
        }
    )
    dump(
        OUTPUT / "selection_frozen.json",
        {"methods": methods, "target_functions_not_generated_yet": True},
    )
    for condition, remainders in [("novel", [3]), ("familiar", [0, 1])]:
        folder = OUTPUT / condition
        folder.mkdir()
        dump(
            folder / "tasks.json", tasks(PLAN["target_functions"], remainders)
        )


def evaluate(method, device):
    for condition, searches in [("novel", 16), ("familiar", 8)]:
        folder = OUTPUT / condition
        specs = json.loads((folder / "tasks.json").read_text())
        seed = PLAN["evaluation_seed"]
        if "dense" in method:
            masks = np.ones(
                (len(specs), searches * 3, 64, 8, 2), dtype=np.uint8
            )
            if method["name"] == "oracle":
                masks.fill(0)
                for t, spec in enumerate(specs):
                    masks[t, ..., spec["rows"], [0, 1]] = 1
            result, history = fit(
                FIT_PLAN, specs, masks, [method["dense"]], seed, device
            )
            result["selected"] = result["query"].argmin(2)
            ti, si = np.indices(result["selected"].shape)
            chosen = result["selected"]
            result["selected_query"] = result["query"][ti, si, chosen]
            result["selected_population"] = result["population"][
                ti, si, chosen
            ]
            result["exploration_fits"] = np.asarray(64)
        else:
            setting = method["setting"]
            result, history = solve(
                generators(setting, device),
                setting,
                specs,
                searches,
                seed,
                device,
                method["policy"],
                True,
            )
        np.savez_compressed(folder / f"{method['name']}.npz", **result)
        with gzip.open(
            folder / f"{method['name']}_history.json.gz", "wt"
        ) as stream:
            json.dump(history, stream)
        print(
            json.dumps(
                {
                    "condition": condition,
                    "method": method["name"],
                    "successes": int(
                        (result["selected_population"] < 0.1).sum()
                    ),
                    "searches": result["selected_query"].size,
                    "mse": float(result["selected_population"].mean()),
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=["candidates", "families", "controls", "freeze", "evaluate"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "candidates":
        candidates()
    elif args.phase == "controls":
        tune_controls(args.device)
    elif args.phase == "freeze":
        freeze()
    else:
        if args.phase == "families":
            entries = list(
                json.loads(
                    (OUTPUT / "confirmation_candidates.json").read_text()
                )
            )
            action = tune_family
        else:
            entries = json.loads(
                (OUTPUT / "selection_frozen.json").read_text()
            )["methods"]
            action = evaluate
        for index, entry in enumerate(entries):
            if index % args.shards == args.shard:
                action(entry, args.device)


if __name__ == "__main__":
    main()

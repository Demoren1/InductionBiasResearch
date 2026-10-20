"""CUDA transfer to unseen functions whose correct masks occurred in training."""

import argparse
import fcntl
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F

from .io import PACKAGE as HERE
from .io import (
    checked_load,
    load_decoder,
    load_manifest,
    save_json,
    save_torch,
    sha256,
    snapshot,
    source_hashes,
    validate_config,
)
from .masks import structural_metrics
from .models.task import bce, fit, imp
from .models.vae import Decoder, topk
from .problem import observations, reference, teacher
from .runtime import configure_cuda
from .search import (
    Fitness,
    budget,
    canonical_mask,
    hard_scores,
    repeated_template,
    search,
    templates,
)

FUNCTIONS = {"adaptation": [4, 5], "evaluation": [6, 7]}


@contextmanager
def locked(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def fresh_observations(root, manifest, family, functions, device):
    """The data generator knows the teacher; search receives only x/y."""
    collected = {
        key: []
        for key in (
            ["support", "query"]
            if functions == [4, 5]
            else ["support", "query", "test"]
        )
    }
    for function in functions:
        directory = (
            root / "observations" / f"family_{family:02d}_function_{function}"
        )
        metadata = {
            "generator_seed": manifest["config"]["seed"],
            "family": family,
            "function": function,
            "split_sizes": manifest["config"]["data"],
            "splits": list(collected),
        }
        with locked(directory / "write.lock"):
            marker = directory / "complete.json"
            if marker.exists():
                existing = json.loads(marker.read_text())
                if existing["metadata"] != metadata:
                    raise ValueError(
                        "New-function observation cache has different settings"
                    )
                payload = checked_load(
                    directory / "observations.pt",
                    existing["observations_sha256"],
                )
            else:
                state = teacher(
                    manifest["config"],
                    manifest["family_definitions"][family],
                    function,
                    device,
                )
                data = observations(
                    manifest["config"],
                    manifest["family_definitions"][family],
                    function,
                    state,
                    device,
                    list(collected),
                )
                payload = {
                    key: {"x": pair[0].cpu(), "y": pair[1].cpu()}
                    for key, pair in data.items()
                }
                save_torch(directory / "observations.pt", payload)
                save_torch(
                    directory / "teacher.pt",
                    {key: value.cpu() for key, value in state.items()},
                )
                save_json(
                    marker,
                    {
                        "metadata": metadata,
                        "observations_sha256": sha256(
                            directory / "observations.pt"
                        ),
                        "teacher_sha256": sha256(directory / "teacher.pt"),
                    },
                )
        for split in collected:
            collected[split].append(payload[split])
    return {
        split: tuple(
            torch.stack([record[key] for record in records]).to(device)
            for key in ["x", "y"]
        )
        for split, records in collected.items()
    }


def make_selection(
    decoder,
    initial_z,
    data,
    template,
    options,
    controls,
    destination,
    sampler_seed,
):
    masks, latent, scores, budgets = {}, {}, {}, {}
    starts = None

    def progress(method, history):
        save_json(destination / f"{method}_history.json", history)
        if history[-1]["step"] % 10 == 0:
            print(json.dumps({"method": method, **history[-1]}), flush=True)

    for method in ["gumbel", "evolution"]:
        result = search(
            decoder,
            initial_z,
            Fitness(data, template, options),
            options,
            method,
            sampler_seed,
            progress,
        )
        index = result["selected"]
        masks[method] = result["masks"][index]
        latent[method] = result["z"][index]
        scores[method] = float(result["scores"][index])
        budgets[method] = result["budget"]
        save_torch(destination / f"{method}_search.pt", result)
        if starts is None:
            starts = result["initial_scores"]
    index = int(starts.argmin())
    with torch.no_grad():
        masks["best_initial"] = canonical_mask(
            topk(decoder(initial_z[index : index + 1]))
        )[0].cpu()
    latent["best_initial"] = initial_z[index].cpu()
    scores["best_initial"] = float(starts[index])
    generator = torch.Generator().manual_seed(sampler_seed + 719)
    random_z = torch.cat(
        (
            initial_z.cpu(),
            torch.randn(
                budget(options) - len(initial_z),
                initial_z.shape[-1],
                generator=generator,
            ),
        )
    ).to(initial_z.device)
    random_fitness = Fitness(data, template, options)
    random_scores = hard_scores(
        decoder, random_z, random_fitness, options["hard_batch"]
    )
    index = int(random_scores.argmin())
    with torch.no_grad():
        masks["random_budget"] = canonical_mask(
            topk(decoder(random_z[index : index + 1]))
        )[0].cpu()
    latent["random_budget"] = random_z[index].cpu()
    scores["random_budget"] = float(random_scores[index])
    budgets["random_budget"] = random_fitness.budget()
    save_torch(
        destination / "random_search.pt",
        {
            "z": random_z.cpu(),
            "scores": random_scores.cpu(),
            "selected": index,
        },
    )
    pool = controls["bank_masks"].to(initial_z.device)
    bank_fitness = Fitness(data, template, options)
    bank_scores = torch.cat(
        [bank_fitness(chunk) for chunk in pool.split(options["hard_batch"])]
    )
    index = int(bank_scores.argmin())
    masks["training_bank"] = pool[index].cpu()
    scores["training_bank"] = float(bank_scores[index])
    budgets["training_bank"] = bank_fitness.budget()
    save_torch(
        destination / "training_bank_selection.pt",
        {"scores": bank_scores.cpu(), "selected": index},
    )
    with torch.no_grad():
        masks["zero_z"] = canonical_mask(
            topk(decoder(torch.zeros_like(initial_z[:1])))
        )[0].cpu()
    latent["zero_z"] = torch.zeros_like(initial_z[0]).cpu()
    masks["source_frequency"] = controls["frequency_mask"]
    masks["dense"] = torch.ones_like(masks["zero_z"])
    fixed_names = ["zero_z", "source_frequency", "dense"]
    fixed = Fitness(data, template, options)
    fixed_scores = fixed(
        torch.stack([masks[name] for name in fixed_names]).to(initial_z.device)
    )
    scores.update(
        {name: float(score) for name, score in zip(fixed_names, fixed_scores)}
    )
    budgets["fixed_controls"] = fixed.budget()
    for method in ["gumbel", "evolution", "random_budget", "training_bank"]:
        if budgets[method]["candidate_fits"] != budget(options):
            raise RuntimeError("Candidate-fit budgets are unequal")
    return {
        "masks": masks,
        "z": latent,
        "adaptation_query_bce": scores,
        "budgets": budgets,
        "adaptation_query_entropy": float(
            F.binary_cross_entropy(data["query"][1], data["query"][1])
        ),
        "adaptation_functions": [4, 5],
        "evaluation_functions": [6, 7],
    }


def final_evaluation(
    selection,
    args,
    manifest,
    config,
    controls,
    decoder,
    source_data,
    source_template,
    device,
):
    if not (args.run / "selection_seal.json").exists():
        raise RuntimeError(
            "Evaluation functions must remain hidden until masks are sealed"
        )
    options = config["evaluation"]
    data = fresh_observations(args.root, manifest, args.family, [6, 7], device)
    target = reference(manifest["family_definitions"][args.family], "cpu")
    masks_by_method = {
        **selection["masks"],
        "correct_mask": canonical_mask(target[None])[0],
    }
    names = list(masks_by_method)
    template, seeds = templates(
        args.family,
        [6, 7],
        options["replicas"],
        config["seed"] + 1300000000,
        device,
    )
    masks = torch.stack([masks_by_method[name] for name in names]).to(device)
    masks = masks.repeat_interleave(options["replicas"], 0)[None].expand(
        2, -1, -1, -1
    )
    state, steps, history = fit(
        repeated_template(template, len(names)), masks, data, options
    )
    test, query = (
        bce(state, masks, data["test"]),
        bce(state, masks, data["query"]),
    )
    entropy = F.binary_cross_entropy(
        data["test"][1], data["test"][1], reduction="none"
    ).mean(-1)
    structural, _ = structural_metrics(
        torch.stack([masks_by_method[name] for name in names]), target
    )
    rows = []

    def append_rows(
        method,
        losses,
        query_losses,
        selected_steps,
        mask_metrics,
        source_score=None,
    ):
        for task_index, function in enumerate([6, 7]):
            for replica in range(options["replicas"]):
                metric = (
                    mask_metrics[task_index * options["replicas"] + replica]
                    if len(mask_metrics) > 1
                    else mask_metrics[0]
                )
                rows.append(
                    {
                        "family": args.family,
                        "function": function,
                        "representation": "wg",
                        "decoder_seed": args.seed,
                        "method": method,
                        "replica": replica,
                        "mlp_seed": seeds[
                            task_index * options["replicas"] + replica
                        ],
                        "test_bce": float(losses[task_index, replica]),
                        "test_excess_bce": max(
                            0,
                            float(
                                losses[task_index, replica]
                                - entropy[task_index]
                            ),
                        ),
                        "query_bce": float(query_losses[task_index, replica]),
                        "selected_mlp_step": int(
                            selected_steps[task_index, replica]
                        ),
                        "adaptation_query_bce": source_score,
                        **metric,
                    }
                )

    for index, method in enumerate(names):
        indices = slice(
            index * options["replicas"], (index + 1) * options["replicas"]
        )
        append_rows(
            method,
            test[:, indices],
            query[:, indices],
            steps[:, indices],
            [structural[index]],
            selection["adaptation_query_bce"].get(method),
        )
    # Ordinary per-function IMP; identical initializations to the dense control.
    cache = args.root / "imp_controls" / f"family_{args.family:02d}.pt"
    metadata = {
        "family": args.family,
        "functions": [6, 7],
        "config": options,
        "bank_sha256": controls["bank_sha256"],
        "seeds": seeds,
        "mlp_sha256": sha256(HERE / "models/task.py"),
    }
    with locked(cache.with_suffix(".lock")):
        if cache.exists():
            imp_payload = checked_load(cache)
            if imp_payload["metadata"] != metadata:
                raise ValueError("IMP cache has a different protocol")
        else:
            index = names.index("dense")
            indices = slice(
                index * options["replicas"], (index + 1) * options["replicas"]
            )
            dense_state = {
                key: value[:, indices] for key, value in state.items()
            }
            dense_history = [
                {
                    "step": row["step"],
                    "support_bce": [
                        values[indices] for values in row["support_bce"]
                    ],
                    "query_bce": [
                        values[indices] for values in row["query_bce"]
                    ],
                }
                for row in history
            ]
            fitted, imp_masks, rounds = imp(
                template,
                dense_state,
                steps[:, indices],
                dense_history,
                data,
                options,
                manifest["config"]["imp"],
                lambda text: print(text, flush=True),
            )
            imp_steps = torch.tensor(
                rounds[-1]["selected_steps"], dtype=torch.long
            )
            imp_payload = {
                "metadata": metadata,
                "state": {key: value.cpu() for key, value in fitted.items()},
                "masks": imp_masks.cpu(),
                "selected_steps": imp_steps.cpu(),
                "rounds": rounds,
            }
            save_torch(cache, imp_payload)
    imp_state = {
        key: value.to(device) for key, value in imp_payload["state"].items()
    }
    imp_masks = imp_payload["masks"].to(device)
    imp_metrics, _ = structural_metrics(imp_masks.flatten(0, 1), target)
    append_rows(
        "imp",
        bce(imp_state, imp_masks, data["test"]),
        bce(imp_state, imp_masks, data["query"]),
        imp_payload["selected_steps"],
        imp_metrics,
    )
    # Correct-mask short fitness is a diagnostic, after all decisions were sealed.
    diagnostic_fitness = Fitness(
        source_data, source_template, config["search"]
    )
    oracle_score = float(
        diagnostic_fitness(canonical_mask(target.to(device)[None]))[0]
    )
    bank_selection = checked_load(args.run / "training_bank_selection.pt")
    bank_metrics, _ = structural_metrics(controls["bank_masks"], target)
    exact_indices = [
        i for i, metric in enumerate(bank_metrics) if metric["exact_mask"]
    ]
    random = checked_load(args.run / "random_search.pt")
    with torch.no_grad():
        random_masks = canonical_mask(
            topk(decoder(random["z"].to(device)))
        ).cpu()
    random_metrics, _ = structural_metrics(random_masks, target)
    diagnostics = {
        "oracle_short_query_bce": oracle_score,
        "training_bank_exact_candidates": len(exact_indices),
        "training_bank_best_exact_short_bce": min(
            (float(bank_selection["scores"][i]) for i in exact_indices),
            default=None,
        ),
        "random_pool_exact_candidates": sum(
            metric["exact_mask"] for metric in random_metrics
        ),
        "random_pool_max_iou": max(metric["iou"] for metric in random_metrics),
        "scope": "Post-seal diagnostic only; analytical references do not affect selection.",
    }
    save_torch(
        args.run / "evaluation_models.pt",
        {
            "state": {key: value.cpu() for key, value in state.items()},
            "masks": masks.cpu(),
            "selected_steps": steps.cpu(),
            "methods": names,
            "seeds": seeds,
        },
    )
    save_json(args.run / "evaluation_history.json", history)
    save_json(args.run / "diagnostics.json", diagnostics)
    save_json(args.run / "results.json", rows)
    return rows


def run(args):
    torch.set_num_threads(1)
    args.root, args.run, args.bank, args.decoder_run = (
        path.resolve()
        for path in [args.root, args.run, args.bank, args.decoder_run]
    )
    config = json.loads(args.config.read_text())
    validate_config(config)
    manifest = load_manifest(args.bank)
    source, payload = load_decoder(
        args.decoder_run,
        sha256(args.bank / "manifest.json"),
        args.seed,
        [args.family],
    )
    controls = checked_load(args.root / "training_controls.pt")
    if controls["bank_sha256"] != source["bank_sha256"]:
        raise ValueError("Training-mask library belongs to another bank")
    if len(controls["bank_masks"]) != budget(config["search"]):
        raise ValueError("Training-mask pool must match the search budget")
    protocol = {
        "schema": "overlapping-groups.known-mask.v1",
        "family": args.family,
        "representation": "wg",
        "decoder_seed": args.seed,
        "config": config,
        "functions": FUNCTIONS,
        "bank_sha256": source["bank_sha256"],
        "decoder_sha256": sha256(args.decoder_run / "decoder.pt"),
        "controls_sha256": sha256(args.root / "training_controls.pt"),
        "source_hashes": source_hashes(),
        "gpu_memory_gib": args.gpu_memory_gib,
        "selection": "Adaptation functions 4/5 only, no target-family identifier or reference mask supplied to search.",
        "evaluation": "Functions 6/7 hidden until selection seal; functions 8+ untouched.",
        "budget_per_search": budget(config["search"]),
    }
    if (args.run / "COMPLETE.json").exists():
        if json.loads((args.run / "protocol.json").read_text()) != protocol:
            raise ValueError("Existing completed run has a different protocol")
        return
    sealed = (args.run / "selection_seal.json").exists()
    if args.run.exists() and (
        not sealed
        or json.loads((args.run / "protocol.json").read_text()) != protocol
    ):
        raise ValueError(
            "Unsealed or incompatible run exists; choose a new output root"
        )
    device = configure_cuda(args.device, args.gpu_memory_gib)
    decoder = (
        Decoder(payload["config"]).to(device).eval().requires_grad_(False)
    )
    decoder.load_state_dict(payload["model"])
    frozen = {
        key: value.cpu().clone() for key, value in decoder.state_dict().items()
    }
    if not sealed:
        args.run.mkdir(parents=True)
        snapshot(args.run / "source", args.config)
        save_json(args.run / "protocol.json", protocol)
    started = time.monotonic()
    try:
        data = fresh_observations(
            args.root, manifest, args.family, [4, 5], device
        )
        template, _ = templates(
            args.family,
            [4, 5],
            config["search"]["replicas"],
            config["seed"] + 1100000000,
            device,
        )
        if sealed:
            seal = json.loads((args.run / "selection_seal.json").read_text())
            selection = checked_load(args.run / "selection.pt", seal["sha256"])
        else:
            sampler_seed = (
                config["seed"] + 100003 * args.family + 17 * args.seed
            )
            generator = torch.Generator().manual_seed(sampler_seed)
            z = torch.randn(
                config["search"]["starts"],
                payload["config"]["latent_dim"],
                generator=generator,
            ).to(device)
            selection = make_selection(
                decoder,
                z,
                data,
                template,
                config["search"],
                controls,
                args.run,
                sampler_seed,
            )
            save_torch(args.run / "selection.pt", selection)
            save_json(
                args.run / "selection_seal.json",
                {
                    "sha256": sha256(args.run / "selection.pt"),
                    "functions": FUNCTIONS,
                    "evaluation_functions_accessed_at_seal": False,
                },
            )
        if any(
            not torch.equal(value.cpu(), frozen[key])
            for key, value in decoder.state_dict().items()
        ):
            raise RuntimeError("Frozen decoder changed")
        rows = final_evaluation(
            selection,
            args,
            manifest,
            config,
            controls,
            decoder,
            data,
            template,
            device,
        )
        save_json(
            args.run / "COMPLETE.json",
            {
                "status": "complete",
                "family": args.family,
                "seed": args.seed,
                "rows": len(rows),
                "budgets": selection["budgets"],
                "selection_sha256": sha256(args.run / "selection.pt"),
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(
                    device
                )
                / 1024**3,
                "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(
                    device
                )
                / 1024**3,
            },
        )
        (args.run / "FAILED.json").unlink(missing_ok=True)
    except BaseException as error:
        save_json(
            args.run / "FAILED.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--decoder-run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--family", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--gpu-memory-gib", type=float, default=12)
    run(parser.parse_args())

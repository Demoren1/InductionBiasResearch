"""Freeze the decoder, select z on source tasks, then evaluate sealed masks."""

import argparse
from pathlib import Path

import torch

from pattern.bank.imp import QueryCheckpoint
from pattern.io import ROOT, device_name, new_directory

from .common import (
    HIDDEN,
    ROWS,
    K,
    Model,
    bce,
    digest,
    fit,
    initial,
    load_bank,
    logits,
    progress,
    references,
    save_history,
    save_json,
    save_maps,
    save_torch,
    target,
    tasks,
    topk,
)


def repeated_initial(task_ids, candidates, replicas, seed, device):
    seeds = [
        seed + 1009 * (int(task[2:], 2) + 16 * (task[0] == "B")) + replica * 13
        for replica in range(replicas)
        for task in task_ids
    ]
    template = initial(seeds, device)
    return {
        key: value.repeat(candidates, *([1] * (value.ndim - 1)))
        for key, value in template.items()
    }


def observations(data, task_ids, device):
    x = {
        key: value.to(device) for key, value in data.items() if key != "probe"
    }
    return x, {
        key: torch.stack([target(value, task) for task in task_ids])
        for key, value in x.items()
    }


def search(
    decoder, task_ids, data, config, options, destination, device, enabled
):
    x, y = observations(
        {key: data[key] for key in ("support", "query")}, task_ids, device
    )
    generator = torch.Generator().manual_seed(
        config["seed"] + 9001 + (task_ids[0][0] == "B")
    )
    z = (
        torch.randn(
            options["z_starts"],
            config["model"]["latent_dim"],
            generator=generator,
        )
        .to(device)
        .requires_grad_()
    )
    with torch.no_grad():
        initial_scores = decoder(z).cpu()
    optimizer = torch.optim.Adam([z], lr=options["z_lr"])
    checkpoint = QueryCheckpoint({"z": z}, (len(z),))
    template = repeated_initial(
        task_ids, len(z), 1, config["seed"] + 310007, device
    )
    ys, yq = (y[key].repeat(len(z), 1) for key in ("support", "query"))
    history = []
    gradient_observed = False
    loop = progress(
        range(options["z_steps"] + 1),
        enabled,
        desc=f"Search z / {task_ids[0][0]}",
        unit="step",
    )
    for step in loop:
        optimizer.zero_grad(set_to_none=True)
        masks = topk(decoder(z), True).repeat_interleave(len(task_ids), 0)
        state = {
            key: value.detach().requires_grad_()
            for key, value in template.items()
        }
        for _ in range(options["inner_steps"]):
            objective = bce(state, masks, x["support"], ys).sum()
            gradients = torch.autograd.grad(
                objective, tuple(state.values()), create_graph=True
            )
            state = {
                key: value - options["inner_lr"] * gradient
                for (key, value), gradient in zip(state.items(), gradients)
            }
        loss = bce(state, masks, x["query"], yq).reshape(len(z), -1).mean(1)
        if not torch.isfinite(loss).all():
            raise RuntimeError("nonfinite source objective")
        checkpoint.update({"z": z}, loss.detach(), step)
        history.append(
            {
                "step": step,
                "source_query_bce": float(loss.detach().mean()),
                "best_restart_bce": float(loss.detach().min()),
                "restart_bce": loss.detach().cpu().tolist(),
            }
        )
        save_json(destination / "search/history.json", history)
        loop.set_postfix(bce=f"{history[-1]['source_query_bce']:.5f}")
        if step < options["z_steps"]:
            loss.sum().backward()
            if z.grad is None or not torch.isfinite(z.grad).all():
                raise RuntimeError("invalid z gradient")
            gradient_observed |= bool(z.grad.abs().sum() > 0)
            torch.nn.utils.clip_grad_norm_([z], 5.0, error_if_nonfinite=True)
            optimizer.step()
    if not gradient_observed:
        raise RuntimeError("source BCE did not produce a z gradient")
    save_history(destination / "search", history, "step")
    with torch.no_grad():
        scores = decoder(checkpoint.state["z"]).cpu()
    return (
        scores,
        initial_scores,
        checkpoint.state["z"].cpu(),
        checkpoint.steps.cpu(),
    )


def fresh(masks, task_ids, data, options, seed, device, enabled, final=False):
    """Match fresh seeds across methods, independently for each task."""
    # The caller seals candidate masks before passing held-out tasks.
    needed = ("support", "query", "test" if final else "validation")
    x, y = observations({key: data[key] for key in needed}, task_ids, device)
    count = len(masks)
    replicas = options["replicas"]
    shape = (count, replicas, len(task_ids))
    expanded = (
        masks.to(device)[:, None, None]
        .expand(*shape, ROWS, HIDDEN)
        .reshape(-1, ROWS, HIDDEN)
    )
    ys, yq = (
        y[key].repeat(count * replicas, 1) for key in ("support", "query")
    )
    original = repeated_initial(task_ids, count, replicas, seed, device)
    state, steps, history = fit(
        original,
        expanded,
        (x["support"], ys),
        (x["query"], yq),
        options["child_steps"],
        options["child_lr"],
        select_every=options["select_every"],
        enabled=enabled,
        log_every=options["log_every"],
    )
    split = "test" if final else "validation"
    expected = y[split].repeat(count * replicas, 1)
    with torch.no_grad():
        output = logits(state, expanded, x[split])
        risk = bce(state, expanded, x[split], expected).reshape(shape).cpu()
        oracle = (
            torch.nn.functional.binary_cross_entropy(
                expected, expected, reduction="none"
            )
            .mean(-1)
            .reshape(shape)
            .cpu()
        )
        accuracy = (
            ((output > 0) == (expected > 0.5))
            .float()
            .mean(-1)
            .reshape(shape)
            .cpu()
        )
    return {
        "bce": risk,
        "excess_bce": risk - oracle,
        "accuracy": accuracy,
        "selected_steps": steps.reshape(shape),
        "oracle_bce": oracle,
        "history": history,
    }


def frequency(cards, family=None):
    return topk(
        torch.cat(
            [
                card["masks"][card["split"] == 0]
                for task, card in cards.items()
                if family is None or task[0] == family
            ]
        ).mean(0, keepdim=True)
    )[0]


def evaluate(run, evaluation_id, device, enabled, overrides):
    checkpoint = torch.load(
        run / "checkpoints/best.pt", weights_only=True, map_location="cpu"
    )
    config = checkpoint["config"]
    reference = checkpoint["bank"]
    if (
        digest(Path(reference["path"]) / "manifest.json")
        != reference["sha256"]
    ):
        raise ValueError("training bank changed")
    _, _, cards, data = load_bank(reference["path"], config)
    options = {
        **config["evaluation"],
        **{
            key: value for key, value in overrides.items() if value is not None
        },
    }
    for key, value in options.items():
        if value <= 0:
            raise ValueError(f"evaluation {key} must be positive")
    model = Model(tasks(config), config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    decoder = model.decoder.eval().requires_grad_(False)
    frozen = {
        key: value.detach().cpu().clone()
        for key, value in decoder.state_dict().items()
    }
    destination = new_directory(run / "evaluations", evaluation_id)
    report = {"checkpoint_epoch": checkpoint["epoch"], "families": {}}
    save_json(
        destination / "protocol.json",
        {
            "config": config,
            "options": options,
            "bank": reference,
            "outer_trainable": "z only",
            "outer_loss": "source-query soft-target BCE through fresh SGD",
            "selection": "source validation BCE; held-out tasks excluded",
            "accuracy_target": "p > 0.5",
            "reference_set": "four Bayes-optimal masks, modulo hidden columns",
        },
    )
    for family in "AB":
        output = destination / family
        output.mkdir()
        source = tasks(config, family=family)
        heldout = tasks(config, "test", family)
        scores, initial_scores, z, steps = search(
            decoder, source, data, config, options, output, device, enabled
        )
        masks = topk(scores)
        selection = fresh(
            torch.cat((masks, topk(initial_scores))),
            source,
            data,
            options,
            config["seed"] + 710009,
            device,
            enabled,
        )
        risks = selection["bce"].mean((1, 2))
        selected = int(risks[: len(masks)].argmin())
        best_random = int(risks[len(masks) :].argmin())
        save_history(output / "selection", selection.pop("history"), "step")
        save_torch(
            output / "selection.pt",
            {
                **selection,
                "source_tasks": source,
                "z": z,
                "search_steps": steps,
                "selected_index": selected,
                "best_random_index": best_random,
                "candidate_order": "optimized restarts, then initial restarts",
            },
        )
        artifacts = save_maps(
            output, "latent_search", scores, source[0], selected
        )
        initial_artifacts = save_maps(
            output, "initial_z", initial_scores, source[0]
        )
        candidate_masks = torch.cat(
            (
                masks[selected : selected + 1],
                topk(initial_scores),
                torch.stack(
                    (
                        frequency(cards),
                        frequency(cards, family),
                        references("A_0000")[0],
                        references("B_0000")[0],
                    )
                ),
                references(source[0]),
            )
        )
        names = [
            "optimized_z",
            *[f"random_z_{index}" for index in range(len(initial_scores))],
            "global_frequency",
            "family_frequency",
            "fixed_A",
            "fixed_B",
            *[f"oracle_{index}" for index in range(4)],
        ]
        save_torch(
            output / "sealed_masks.pt",
            {
                "masks": candidate_masks,
                "methods": names,
                "z": z[selected],
                "selected_logits": scores[selected],
                "probabilities": scores[selected].sigmoid(),
                "source_validation_bce": float(risks[selected]),
            },
        )
        # First held-out target call, after masks and z are fixed on disk.
        result = fresh(
            candidate_masks,
            heldout,
            data,
            options,
            config["seed"] + 910019,
            device,
            enabled,
            final=True,
        )
        save_history(output / "test", result.pop("history"), "step")
        save_torch(
            output / "test/values.pt",
            {**result, "tasks": heldout, "methods": names},
        )
        records = {
            name: {
                key: float(result[key][index].mean())
                for key in ("bce", "excess_bce", "accuracy")
            }
            for index, name in enumerate(names)
        }
        records["random_z_mean"] = {
            key: sum(
                records[f"random_z_{index}"][key]
                for index in range(len(initial_scores))
            )
            / len(initial_scores)
            for key in ("bce", "excess_bce", "accuracy")
        }
        records["best_random_z"] = records[f"random_z_{best_random}"]
        iou = artifacts["iou"]
        exact = iou == 1
        counts = torch.bincount(artifacts["variants"][exact], minlength=4)
        report["families"][family] = {
            "source_tasks": source,
            "heldout_tasks": heldout,
            "selected_index": selected,
            "selected_iou": float(iou[selected]),
            "initial_mean_iou": float(initial_artifacts["iou"].mean()),
            "optimized_mean_iou": float(iou.mean()),
            "exact_fraction": float(exact.float().mean()),
            "exact_variant_counts": counts.tolist(),
            "covered_variants": int((counts > 0).sum()),
            "source_validation_bce": risks[: len(masks)].tolist(),
            "initial_source_validation_bce": risks[len(masks) :].tolist(),
            "best_random_index": best_random,
            "methods": records,
        }
        print(
            f"{family}: selected IoU={iou[selected]:.4f}; "
            f"held-out BCE={records['optimized_z']['bce']:.5f}; "
            f"excess={records['optimized_z']['excess_bce']:.5f}",
            flush=True,
        )
    if any(
        not torch.equal(value.cpu(), frozen[key])
        for key, value in decoder.state_dict().items()
    ):
        raise RuntimeError("frozen decoder changed")
    report.update(decoder_frozen=True, z_gradient_observed=True, k=K)
    save_json(destination / "metrics.json", report)
    save_json(
        destination / "COMPLETE.json",
        {"families": ["A", "B"], "decoder_frozen": True},
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "multimask/runs/nf_vae_v1"
    )
    parser.add_argument("--evaluation-id")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    integers = (
        "z-starts",
        "z-steps",
        "inner-steps",
        "child-steps",
        "select-every",
        "replicas",
        "log-every",
    )
    floats = ("z-lr", "inner-lr", "child-lr")
    for flag in integers:
        parser.add_argument("--" + flag, type=int)
    for flag in floats:
        parser.add_argument("--" + flag, type=float)
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    overrides = {
        flag.replace("-", "_"): getattr(args, flag.replace("-", "_"))
        for flag in (*integers, *floats)
    }
    print(f"Device: {device}", flush=True)
    destination = evaluate(
        args.run, args.evaluation_id, device, not args.no_progress, overrides
    )
    print(f"Evaluation saved: {destination}", flush=True)


if __name__ == "__main__":
    main()

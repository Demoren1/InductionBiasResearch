"""Select a decoder latent on source tasks, then evaluate a sealed mask on held-out tasks."""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .datasets import input_table, labels_for, balanced_rows
from .io import new_directory, save_json, save_torch, digest
from .masks import exact_topk, analytical_mask, align_to_reference, toeplitz_metrics
from .models.shared_decoder import SharedDecoder
from .reporting import progress, save_losses, plot_panels


DEFAULTS = dict(z_starts=8, z_steps=100, z_lr=.05, inner_steps=64,
                inner_lr=.2, support_count=256, query_count=64,
                child_steps=500, child_lr=.03, replicas=3, log_every=10)


def observations(tasks, manifest, support_count, query_count, device, final=False):
    """Only called for held-out task labels after source-only selection is saved."""
    _, x = input_table()
    parts = manifest["partitions"]; seed = manifest["config"]["seed"]
    xs, ys, xq, yq, ids = [], [], [], [], []
    for task in tasks:
        y = labels_for(x, task)
        support = balanced_rows(y, torch.tensor(parts["support"]), support_count,
                                seed + 117 + int(task, 2))
        query = (torch.tensor(parts["test"]) if final else
                 balanced_rows(y, torch.tensor(parts["query"]), query_count, seed + 219 + int(task, 2)))
        xs.append(x[support]); ys.append(y[support]); xq.append(x[query]); yq.append(y[query])
        ids.append({"task": task, "support": support.tolist(), "query": query.tolist()})
    return tuple(torch.stack(v).to(device) for v in (xs, ys, xq, yq)), ids


def child_initial(tasks, candidates, replicas, seed, device):
    """Identical weight seeds across candidate masks, independent across tasks/replicas."""
    weights, readouts, seeds = [], [], []
    for replica in range(replicas):
        w, a, task_seeds = [], [], []
        for task in tasks:
            network_seed = seed + 2000003 * (int(task, 2) + 1) + replica
            generator = torch.Generator().manual_seed(network_seed)
            w.append(torch.randn(11, 8, generator=generator) * .1)
            a.append(torch.randn(8, generator=generator) * .1)
            task_seeds.append(network_seed)
        weights.append(torch.stack(w)); readouts.append(torch.stack(a)); seeds.append(task_seeds)
    w = torch.stack(weights).repeat(candidates, 1, 1, 1).to(device)
    a = torch.stack(readouts).repeat(candidates, 1, 1).to(device)
    state = {"w": w, "b": torch.zeros_like(a), "a": a, "c": torch.zeros_like(a[..., 0])}
    return {key: value.requires_grad_() for key, value in state.items()}, seeds


def child_logits(state, masks, x):
    hidden = F.relu(torch.matmul(x.unsqueeze(0), state["w"] * masks[:, None]) + state["b"][:, :, None])
    return (hidden * state["a"][:, :, None]).sum(-1) + state["c"][:, :, None]


def child_bce(state, masks, x, y):
    predictions = child_logits(state, masks, x)
    return F.binary_cross_entropy_with_logits(predictions, y.unsqueeze(0).expand_as(predictions),
                                             reduction="none").mean(-1)


def source_objective(decoder, z, tasks, data, seed, steps, lr):
    """Unroll fresh child SGD; only z is an outer optimization variable.

    The forward mask has exactly 32 edges. Its surrogate derivative and the
    derivatives through child fitting carry source-query BCE back to z.
    """
    mask = exact_topk(decoder(z), straight_through=True)
    state, _ = child_initial(tasks, len(z), 1, seed, z.device)
    xs, ys, xq, yq = data
    for _ in range(steps):
        loss = child_bce(state, mask, xs, ys).sum()
        grads = torch.autograd.grad(loss, tuple(state.values()), create_graph=True)
        state = {key: value - lr * grad for (key, value), grad in zip(state.items(), grads)}
    return child_bce(state, mask, xq, yq).mean(-1), mask


def fit_candidates(masks, tasks, data, seed, steps, lr, replicas, device,
                   show_progress=True, log_every=10, description="Fresh networks"):
    """Fit all candidates/tasks/replicas in one batch with independent Adam states."""
    masks = masks.to(device).repeat_interleave(replicas, dim=0)
    state, seeds = child_initial(tasks, len(masks)//replicas, replicas, seed, device)
    optimizer = torch.optim.Adam(state.values(), lr=lr)
    xs, ys, xq, yq = data
    history = []
    loop = progress(range(1, steps + 1), show_progress, desc=description, unit="step")
    for step in loop:
        optimizer.zero_grad(set_to_none=True)
        values = child_bce(state, masks, xs, ys)
        values.sum().backward(); optimizer.step()
        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                train = child_bce(state, masks, xs, ys).cpu()
            for sample in range(len(masks)):
                for t, task in enumerate(tasks):
                    history.append({"step": step, "method": f"candidate_{sample//replicas}",
                                    "task": task, "replica": sample % replicas,
                                    "support_bce": float(train[sample, t])})
            loop.set_postfix(bce=f"{float(train.mean()):.5f}")
    with torch.no_grad():
        bce = child_bce(state, masks, xq, yq).reshape(-1, replicas, len(tasks)).cpu()
        prediction = child_logits(state, masks, xq)
        accuracy = ((prediction > 0) == (yq[None] > .5)).float().mean(-1).reshape_as(bce).cpu()
    return bce, accuracy, history, seeds


def save_masks(destination, z, masks, selected, initial_masks):
    reference = analytical_mask()
    pairs = [align_to_reference(mask, reference) for mask in masks]
    aligned = torch.stack([p[0] for p in pairs])
    save_torch(destination/"generated_masks/latent_search.pt", {
        "z": z.cpu(), "canonical": masks.cpu(), "initial_masks": initial_masks.cpu(),
        "selected_index": selected, "selected_mask": masks[selected].cpu(),
        "analytical": reference, "aligned_to_analytical": aligned,
        "alignment_permutations": torch.stack([p[1] for p in pairs]),
        "alignment_scope": "visualization only; raw masks used for BCE and selection"})
    heatmaps = destination/"heatmaps"; heatmaps.mkdir(exist_ok=True)
    initial_aligned, _ = align_to_reference(initial_masks[selected], reference)
    panels = {"Analytical Toeplitz": reference, "Initial aligned": initial_aligned,
              "Selected aligned": aligned[selected], "Restart frequency": aligned.mean(0)}
    plot_panels(heatmaps/"latent_search", panels, "Source-only latent selection", continuous=True)
    save_torch(heatmaps/"latent_search_values.pt", panels)


def evaluate_latent(run, device="cpu", evaluation_id=None, show_progress=True, **overrides):
    run = Path(run)
    checkpoint = torch.load(run/"checkpoints/best.pt", weights_only=True, map_location="cpu")
    config = checkpoint["config"]; reference = checkpoint["bank_reference"]
    bank = Path(reference["bank_path"])
    if digest(bank/"manifest.json") != reference["manifest_sha256"]:
        raise ValueError("bank manifest changed after training")
    manifest = json.loads((bank/"manifest.json").read_text())
    tasks = config["tasks"]
    heldout = config.get("test_tasks", [f"{i:04b}" for i in range(16) if f"{i:04b}" not in tasks])
    if not heldout or set(tasks) & set(heldout) or len(set(heldout)) != len(heldout):
        raise ValueError("evaluation requires disjoint nonempty training and held-out tasks")
    options = {**DEFAULTS, **config.get("evaluation", {}), **{k:v for k,v in overrides.items() if v is not None}}
    for key in ("z_starts", "z_steps", "inner_steps", "child_steps", "replicas", "log_every"):
        if options[key] < 1: raise ValueError(f"{key} must be positive")
    for key in ("z_lr", "inner_lr", "child_lr"):
        if options[key] <= 0: raise ValueError(f"{key} must be positive")
    decoder = SharedDecoder(config["model"]["latent_dim"], config["model"]["decoder_width"]).to(device)
    decoder.load_state_dict({key.removeprefix("decoder."): value for key,value in checkpoint["model_state"].items()
                             if key.startswith("decoder.")})
    decoder.eval(); decoder.requires_grad_(False)
    frozen_state = {key:value.detach().cpu().clone() for key,value in decoder.state_dict().items()}
    source, source_ids = observations(tasks, manifest, options["support_count"], options["query_count"], device)
    destination = new_directory(run/"evaluations", evaluation_id)
    save_json(destination/"protocol.json", {"mode":"latent_transfer", "train_tasks":tasks,"test_tasks":heldout,
        "options":options,"bank_reference":reference,"checkpoint_epoch":checkpoint["epoch"],
        "source_observation_ids":source_ids,"outer_trainable_parameters":"z only",
        "outer_loss":"unweighted mean source-query BCE through fresh child SGD",
        "selection":"source validation BCE after fresh Adam fits; held-out tasks excluded"})
    generator = torch.Generator().manual_seed(config["seed"]+9927)
    z = torch.randn(options["z_starts"],config["model"]["latent_dim"], generator=generator).to(device).requires_grad_()
    initial_z = z.detach().cpu().clone()
    with torch.no_grad(): initial_masks = exact_topk(decoder(z)).cpu()
    optimizer = torch.optim.Adam([z], lr=options["z_lr"])
    history = []; best_bce = torch.full((len(z),), float("inf"), device=device)
    best_z = z.detach().clone(); best_steps = torch.zeros(len(z), dtype=torch.long, device=device)
    gradient_observed = False
    loop = progress(range(options["z_steps"]+1), show_progress, desc=f"Latent search ({device})", unit="step")
    for step in loop:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = source_objective(decoder, z, tasks, source, config["seed"]+310007,
                                   options["inner_steps"], options["inner_lr"])
        if not torch.isfinite(loss).all(): raise RuntimeError("nonfinite latent objective")
        improved = loss.detach() < best_bce
        best_bce = torch.minimum(best_bce, loss.detach())
        best_z[improved] = z.detach()[improved]; best_steps[improved] = step
        for restart, value in enumerate(loss.detach().cpu()):
            history.append({"step":step,"method":f"restart_{restart}","bce":float(value)})
        save_json(destination/"latent_history.json", history)
        loop.set_postfix(bce=f"{float(loss.detach().mean()):.5f}")
        if step < options["z_steps"]:
            loss.sum().backward()
            if z.grad is None or not torch.isfinite(z.grad).all(): raise RuntimeError("invalid z gradient")
            gradient_observed |= bool(z.grad.abs().sum() > 0)
            torch.nn.utils.clip_grad_norm_([z], 5., error_if_nonfinite=True)
            optimizer.step()
    if not gradient_observed: raise RuntimeError("source BCE produced no gradient to z")
    for key,value in decoder.state_dict().items():
        if not torch.equal(value.detach().cpu(),frozen_state[key]): raise RuntimeError("frozen decoder changed")
    with torch.no_grad(): masks = exact_topk(decoder(best_z)).cpu()
    # Restart selection uses new source network weights and reserved source observations.
    xs,ys,_,_ = source; _,x = input_table(); rows = torch.tensor(manifest["partitions"]["validation"])
    selection = (xs,ys,x[rows].expand(len(tasks),-1,-1).to(device),
                 torch.stack([labels_for(x,t)[rows] for t in tasks]).to(device))
    scores,_,selection_history,selection_seeds = fit_candidates(masks,tasks,selection,config["seed"]+710009,
        options["child_steps"],options["child_lr"],options["replicas"],device,show_progress,
        options["log_every"],"Source-only restart selection")
    mean_scores = scores.mean((1,2)); selected = int(mean_scores.argmin())
    save_json(destination/"selection.json", {"selected_index":selected,"source_validation_bce":mean_scores.tolist(),
        "per_restart_replica_task_bce":scores.tolist(),"source_tasks":tasks,"validation_ids":rows.tolist(),
        "child_seeds":selection_seeds,"best_search_steps":best_steps.cpu().tolist()})
    save_torch(destination/"selected_mask.pt", {"z":best_z[selected].cpu(),"mask":masks[selected],
        "selected_index":selected,"source_validation_bce":float(mean_scores[selected])})
    save_torch(destination/"latent_codes.pt",{"initial":initial_z,"best":best_z.cpu(),"best_steps":best_steps.cpu()})
    save_masks(destination,best_z.detach(),masks,selected,initial_masks)
    save_losses(destination,history,x_key="step")
    selection_dir = destination/"selection_losses"; selection_dir.mkdir()
    save_json(selection_dir/"history.json",selection_history); save_losses(selection_dir,selection_history,x_key="step")
    # The selected mask is now sealed. Held-out labels appear for the first time here.
    test_data,test_ids = observations(heldout,manifest,options["support_count"],options["query_count"],device,final=True)
    test_masks = torch.stack([masks[selected], analytical_mask()])
    test_bce,test_accuracy,test_history,test_seeds = fit_candidates(test_masks,heldout,test_data,config["seed"]+910019,
        options["child_steps"],options["child_lr"],options["replicas"],device,show_progress,
        options["log_every"],"Held-out tasks: fixed generated / analytical masks")
    for row in test_history: row["method"] = ("generated","analytical")[int(row["method"].split("_")[-1])]
    test_dir = destination/"test_losses"; test_dir.mkdir()
    save_json(test_dir/"history.json",test_history); save_losses(test_dir,test_history,x_key="step")
    report = {"mode":"latent_transfer","train_tasks":tasks,"test_tasks":heldout,"selected_index":selected,
              "decoder_frozen":True,"z_gradient_observed":gradient_observed,"k":32,
              "structure_canonical":toeplitz_metrics(masks[selected]),
              "structure_aligned":toeplitz_metrics(align_to_reference(masks[selected],analytical_mask())[0]),
              "test_observation_ids":test_ids,"test_child_seeds":test_seeds,"tasks":{}}
    for t,task in enumerate(heldout):
        report["tasks"][task] = {method:{"test_bce":float(test_bce[m,:,t].mean()),
            "test_accuracy":float(test_accuracy[m,:,t].mean()),"replica_bce":test_bce[m,:,t].tolist()}
            for m,method in enumerate(("generated","analytical"))}
    report["macro"] = {method:{"test_bce":float(test_bce[m].mean()),"test_accuracy":float(test_accuracy[m].mean())}
                       for m,method in enumerate(("generated","analytical"))}
    save_json(destination/"metrics.json",report)
    save_json(destination/"COMPLETE.json",{"selected_index":selected,"test_tasks":heldout})
    return destination

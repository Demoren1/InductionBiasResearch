"""Select a decoder latent on source tasks, then evaluate a sealed mask on held-out tasks."""
from pathlib import Path
import torch
from torch.nn import functional as F
from .bank.imp import initialization, mean_bce, l2_penalty, QueryCheckpoint
from .datasets import input_table, labels_for, balanced_rows
from .io import new_directory, save_json, save_torch, load_run
from .masks import exact_topk, analytical_mask, align_to_reference, align_masks, toeplitz_metrics,aligned_iou
from .models.shared_decoder import SharedDecoder
from .reporting import progress, save_losses, save_loss_history, plot_panels


DEFAULTS = dict(z_starts=8, z_steps=100, z_lr=.05, inner_steps=64,
                inner_lr=.2, support_count=256, query_count=64,
                child_steps=500, child_lr=.03, replicas=3, log_every=10)


def observations(tasks, manifest, support_count, query_count, device, final=False, full_support=False):
    """Only called for held-out task labels after source-only selection is saved."""
    _, x = input_table()
    parts = manifest["partitions"]; seed = manifest["config"]["seed"]
    samples,ids=[],[]
    for task in tasks:
        y = labels_for(x, task)
        support = (torch.tensor(parts["support"]) if full_support else
                   balanced_rows(y, torch.tensor(parts["support"]), support_count,seed + 117 + int(task,2)))
        if full_support and len(support)!=support_count:raise ValueError("full support budget must match partition")
        query = (torch.tensor(parts["test"]) if final else
                 balanced_rows(y,torch.tensor(parts["query"]),query_count,
                               seed+91009 if full_support else seed+219+int(task,2)))
        samples.append((x[support],y[support],x[query],y[query]))
        ids.append({"task": task, "support": support.tolist(), "query": query.tolist()})
    return tuple(torch.stack(values).to(device) for values in zip(*samples)),ids


def child_initial(tasks, candidates, replicas, seed, device):
    """Identical weight seeds across candidate masks, independent across tasks/replicas."""
    seeds=[[seed+2000003*(int(task,2)+1)+replica for task in tasks] for replica in range(replicas)]
    templates=[initialization(seed) for replica in seeds for seed in replica]
    state={}
    for key in templates[0]:
        values=torch.stack([template[key] for template in templates])
        repeated=values.repeat(candidates,*([1]*(values.ndim-1)))
        state[key]=repeated.reshape(-1,len(tasks),*values.shape[1:]).to(device).requires_grad_()
    return state,seeds


def child_logits(state, masks, x):
    hidden = F.relu(torch.matmul(x.unsqueeze(0), state["w"] * masks[:, None]) + state["b"][:, :, None])
    return (hidden * state["a"][:, :, None]).sum(-1) + state["c"][:, :, None]


def child_bce(state, masks, x, y, class_balanced=False):
    predictions = child_logits(state, masks, x)
    return mean_bce(predictions,y.unsqueeze(0).expand_as(predictions),class_balanced)


def source_objective(decoder, z, tasks, data, seed, steps, lr, initial=None, class_balanced=False,l2=0.):
    """Unroll fresh child SGD; only z is an outer optimization variable.

    The forward mask has exactly 32 edges. Its surrogate derivative and the
    derivatives through child fitting carry source-query BCE back to z.
    """
    mask = exact_topk(decoder(z), straight_through=True)
    if initial is None:initial,_=child_initial(tasks,len(z),1,seed,z.device)
    # SGD is functional: fresh leaves may share the unchanged cached tensor storage.
    state={key:value.detach().requires_grad_() for key,value in initial.items()}
    xs, ys, xq, yq = data
    for _ in range(steps):
        loss = (child_bce(state,mask,xs,ys,class_balanced)+l2_penalty(state,mask[:,None],l2)).sum()
        grads = torch.autograd.grad(loss, tuple(state.values()), create_graph=True)
        state = {key: value - lr * grad for (key, value), grad in zip(state.items(), grads)}
    return child_bce(state, mask, xq, yq).mean(-1), mask


def fit_candidates(masks, tasks, data, seed, steps, lr, replicas, device,
                   show_progress=True, log_every=10, description="Fresh networks",
                   class_balanced=False,l2=0.,select_every=0,selection_data=None):
    """Fit all candidates/tasks/replicas in one batch with independent Adam states."""
    masks = masks.to(device).repeat_interleave(replicas, dim=0)
    state, seeds = child_initial(tasks, len(masks)//replicas, replicas, seed, device)
    optimizer = torch.optim.Adam(state.values(), lr=lr)
    xs, ys, xq, yq = data
    if select_every and selection_data is None:raise ValueError("child checkpoint selection needs separate query data")
    checkpoint = QueryCheckpoint(state, (len(masks),len(tasks))) if select_every else None
    history = []
    loop = progress(range(1, steps + 1), show_progress, desc=description, unit="step")
    for step in loop:
        optimizer.zero_grad(set_to_none=True)
        values = child_bce(state,masks,xs,ys,class_balanced)+l2_penalty(state,masks[:,None],l2)
        values.sum().backward(); optimizer.step()
        if select_every and (step%select_every==0 or step==steps):
            with torch.no_grad():
                score=child_bce(state,masks,*selection_data)
                checkpoint.update(state, score, step)
        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                train = child_bce(state,masks,xs,ys,class_balanced).cpu()
            for sample in range(len(masks)):
                for t, task in enumerate(tasks):
                    history.append({"step": step, "method": f"candidate_{sample//replicas}",
                                    "task": task, "replica": sample % replicas,
                                    "support_bce": float(train[sample, t])})
            loop.set_postfix(bce=f"{float(train.mean()):.5f}")
    if select_every:
        state=checkpoint.state
        # Last-step rows follow sample/task order; transfer the whole table once.
        selections = zip(checkpoint.steps.cpu().flatten().tolist(), checkpoint.loss.cpu().flatten().tolist())
        for row,(selected_step,score) in zip(history[-len(masks)*len(tasks):],selections):
            row.update(selected_step=selected_step, selected_query_bce=score)
    with torch.no_grad():
        bce = child_bce(state, masks, xq, yq).reshape(-1, replicas, len(tasks)).cpu()
        prediction = child_logits(state, masks, xq)
        accuracy = ((prediction > 0) == (yq[None] > .5)).float().mean(-1).reshape_as(bce).cpu()
    return bce, accuracy, history, seeds


def save_masks(destination, z, masks, selected, initial_masks):
    reference = analytical_mask()
    aligned,orders=align_masks(masks,reference)
    save_torch(destination/"generated_masks/latent_search.pt", {
        "z": z.cpu(), "canonical": masks.cpu(), "initial_masks": initial_masks.cpu(),
        "selected_index": selected, "selected_mask": masks[selected].cpu(),
        "analytical": reference, "aligned_to_analytical": aligned,
        "alignment_permutations": orders,
        "alignment_scope": "visualization only; raw masks used for BCE and selection"})
    heatmaps = destination/"heatmaps"; heatmaps.mkdir(exist_ok=True)
    initial_aligned, _ = align_to_reference(initial_masks[selected], reference)
    panels = {"Analytical Toeplitz": reference, "Initial aligned": initial_aligned,
              "Selected aligned": aligned[selected], "Restart frequency": aligned.mean(0)}
    plot_panels(heatmaps/"latent_search", panels, "Source-only latent selection", continuous=True)
    save_torch(heatmaps/"latent_search_values.pt", panels)


def search_latents(decoder,tasks,source,config,options,destination,device,show_progress):
    """Optimize source-only BCE; neither held-out tasks nor their labels enter here."""
    frozen_state={key:value.detach().cpu().clone() for key,value in decoder.state_dict().items()}
    generator = torch.Generator().manual_seed(config["seed"]+9927)
    z = torch.randn(options["z_starts"],config["model"]["latent_dim"], generator=generator).to(device).requires_grad_()
    initial_z = z.detach().cpu().clone()
    with torch.no_grad(): initial_masks = exact_topk(decoder(z)).cpu()
    optimizer = torch.optim.Adam([z], lr=options["z_lr"])
    initial,_=child_initial(tasks,len(z),1,config["seed"]+310007,device)
    history = []; checkpoint = QueryCheckpoint({"z":z}, (len(z),))
    gradient_observed = False
    loop = progress(range(options["z_steps"]+1), show_progress, desc=f"Latent search ({device})", unit="step")
    for step in loop:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = source_objective(decoder, z, tasks, source, config["seed"]+310007,
                                   options["inner_steps"],options["inner_lr"],initial=initial,
                                   class_balanced=options.get("support_loss")=="class-balanced BCE",
                                   l2=options.get("child_l2",0.))
        if not torch.isfinite(loss).all(): raise RuntimeError("nonfinite latent objective")
        checkpoint.update({"z":z}, loss.detach(), step)
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
    best_z=checkpoint.state["z"]
    with torch.no_grad(): masks = exact_topk(decoder(best_z)).cpu()
    return {"z":best_z,"masks":masks,"initial_z":initial_z,"initial_masks":initial_masks,
            "best_steps":checkpoint.steps.cpu(),"history":history}


def evaluate_latent(run, device="cpu", evaluation_id=None, show_progress=True, **overrides):
    run=Path(run);checkpoint,_,manifest=load_run(run)
    config=checkpoint["config"];reference=checkpoint["bank_reference"];tasks=config["tasks"]
    heldout=config.get("test_tasks",[f"{i:04b}" for i in range(16) if f"{i:04b}" not in tasks])
    if not heldout or set(tasks)&set(heldout) or len(set(heldout))!=len(heldout):
        raise ValueError("evaluation requires disjoint nonempty training and held-out tasks")
    options={**DEFAULTS,**config.get("evaluation",{}),**{k:v for k,v in overrides.items() if v is not None}}
    for key in ("z_starts","z_steps","inner_steps","child_steps","replicas","log_every"):
        if options[key]<1:raise ValueError(f"{key} must be positive")
    for key in ("z_lr","inner_lr","child_lr"):
        if options[key]<=0:raise ValueError(f"{key} must be positive")
    if options.get("child_l2",0.)<0 or options.get("child_select_every",0)<0:raise ValueError("invalid child regularization/selection")
    decoder=SharedDecoder(config["model"]["latent_dim"],config["model"]["decoder_width"]).to(device)
    decoder.load_state_dict({key.removeprefix("decoder."):value for key,value in checkpoint["model_state"].items()
                             if key.startswith("decoder.")})
    decoder.eval();decoder.requires_grad_(False)
    full_support=options.get("support_sampling")=="all support observations"
    fit_options={"class_balanced":options.get("support_loss")=="class-balanced BCE",
                 "l2":options.get("child_l2",0.),"select_every":options.get("child_select_every",0)}
    source,source_ids=observations(tasks,manifest,options["support_count"],options["query_count"],device,full_support=full_support)
    destination=new_directory(run/"evaluations",evaluation_id)
    save_json(destination/"protocol.json",{"mode":"latent_transfer","train_tasks":tasks,"test_tasks":heldout,
        "options":options,"bank_reference":reference,"checkpoint_epoch":checkpoint["epoch"],
        "source_observation_ids":source_ids,"outer_trainable_parameters":"z only",
        "outer_loss":"unweighted mean source-query BCE through fresh child SGD",
        "selection":"source validation BCE after fresh Adam fits; held-out tasks excluded"})
    search=search_latents(decoder,tasks,source,config,options,destination,device,show_progress)
    best_z,masks=search["z"],search["masks"]
    # Restart selection uses new source network weights and reserved source observations.
    xs,ys,_,_ = source; _,x = input_table(); rows = torch.tensor(manifest["partitions"]["validation"])
    selection = (xs,ys,x[rows].expand(len(tasks),-1,-1).to(device),
                 torch.stack([labels_for(x,t)[rows] for t in tasks]).to(device))
    scores,_,selection_history,selection_seeds = fit_candidates(masks,tasks,selection,config["seed"]+710009,
        options["child_steps"],options["child_lr"],options["replicas"],device,show_progress,
        options["log_every"],"Source-only restart selection",selection_data=source[2:],**fit_options)
    mean_scores = scores.mean((1,2)); selected = int(mean_scores.argmin())
    save_json(destination/"selection.json", {"selected_index":selected,"source_validation_bce":mean_scores.tolist(),
        "per_restart_replica_task_bce":scores.tolist(),"source_tasks":tasks,"validation_ids":rows.tolist(),
        "child_seeds":selection_seeds,"best_search_steps":search["best_steps"].tolist()})
    save_torch(destination/"selected_mask.pt", {"z":best_z[selected].cpu(),"mask":masks[selected],
        "selected_index":selected,"source_validation_bce":float(mean_scores[selected])})
    save_torch(destination/"latent_codes.pt",{"initial":search["initial_z"],"best":best_z.cpu(),"best_steps":search["best_steps"]})
    save_masks(destination,best_z,masks,selected,search["initial_masks"])
    save_losses(destination,search["history"],x_key="step")
    save_loss_history(destination/"selection_losses",selection_history)
    # The selected mask is now sealed. Held-out labels appear for the first time here.
    test_data,test_ids = observations(heldout,manifest,options["support_count"],options["query_count"],device,
                                      final=True,full_support=full_support)
    checkpoint_data,checkpoint_ids=observations(heldout,manifest,options["support_count"],options["query_count"],device,
                                                full_support=full_support)
    test_masks = torch.stack([masks[selected], analytical_mask()])
    test_bce,test_accuracy,test_history,test_seeds = fit_candidates(test_masks,heldout,test_data,config["seed"]+910019,
        options["child_steps"],options["child_lr"],options["replicas"],device,show_progress,
        options["log_every"],"Held-out tasks: fixed generated / analytical masks",
        selection_data=checkpoint_data[2:],**fit_options)
    for row in test_history: row["method"] = ("generated","analytical")[int(row["method"].split("_")[-1])]
    save_loss_history(destination/"test_losses",test_history)
    report = {"mode":"latent_transfer","train_tasks":tasks,"test_tasks":heldout,"selected_index":selected,
              "decoder_frozen":True,"z_gradient_observed":True,"k":32,
              "analytical_aligned_iou":float(aligned_iou(masks[selected])),
              "structure_canonical":toeplitz_metrics(masks[selected]),
              "structure_aligned":toeplitz_metrics(align_to_reference(masks[selected],analytical_mask())[0]),
              "test_observation_ids":test_ids,"child_checkpoint_observation_ids":checkpoint_ids,
              "test_child_seeds":test_seeds,"tasks":{}}
    for t,task in enumerate(heldout):
        report["tasks"][task] = {method:{"test_bce":float(test_bce[m,:,t].mean()),
            "test_accuracy":float(test_accuracy[m,:,t].mean()),"replica_bce":test_bce[m,:,t].tolist()}
            for m,method in enumerate(("generated","analytical"))}
    report["macro"] = {method:{"test_bce":float(test_bce[m].mean()),"test_accuracy":float(test_accuracy[m].mean())}
                       for m,method in enumerate(("generated","analytical"))}
    save_json(destination/"metrics.json",report)
    save_json(destination/"COMPLETE.json",{"selected_index":selected,"test_tasks":heldout})
    return destination

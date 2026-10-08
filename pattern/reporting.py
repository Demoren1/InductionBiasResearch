"""Loss tables, analytical masks and aligned publication artifacts."""
import csv
import io
import os
from pathlib import Path
import tempfile
import torch
from tqdm.auto import tqdm
from .io import atomic, save_json, save_torch
from .masks import analytical_mask, exact_topk, align_to_reference


def progress(iterable, enabled=True, **kwargs):
    return tqdm(iterable, disable=not enabled, dynamic_ncols=True, mininterval=.3, **kwargs)


def pyplot():
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"pattern-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_losses(destination, history, x_key="epoch"):
    """Store flat CSV rows and PNG/PDF plots; retain original JSON separately."""
    rows=[]
    for row in history:
        flat={key:value for key,value in row.items() if isinstance(value,(int,float,str))}
        for group in ("per_task_train","per_task_validation"):
            for task,values in row.get(group,{}).items():
                flat.update({f"{group}.{task}.{key}":value for key,value in values.items()})
        rows.append(flat)
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
    atomic(Path(destination)/"losses.csv",lambda handle:handle.write(stream.getvalue()),False)
    plt=pyplot();fig,ax=plt.subplots(figsize=(7,4),constrained_layout=True)
    groups=sorted({row.get("method","") for row in rows})
    for group in groups:
        for key in ("train_loss","validation_loss","loss","reconstruction","bce","hard_mse","kl","support_bce","objective"):
            grouped={}
            for row in rows:
                if row.get("method","")==group and key in row and x_key in row:
                    grouped.setdefault(row[x_key],[]).append(row[key])
            if grouped:
                ticks=sorted(grouped)
                ax.plot(ticks,[sum(grouped[t])/len(grouped[t]) for t in ticks],
                        label=f"{group}/{key}" if group else key)
    ax.set_xlabel(x_key);ax.set_ylabel("Loss");ax.grid(alpha=.2)
    if ax.lines:ax.legend()
    for suffix in ("png","pdf"):fig.savefig(Path(destination)/f"losses.{suffix}",dpi=160)
    plt.close(fig)


def plot_panels(path, panels, title, continuous=False):
    plt=pyplot();fig,axes=plt.subplots(1,len(panels),figsize=(3*len(panels),4),constrained_layout=True)
    for ax,(label,mask) in zip(axes,panels.items()):
        picture=ax.imshow(mask.detach().float().cpu().numpy(),vmin=0,vmax=1,cmap="Blues",origin="upper",aspect="equal")
        ax.set_title(label);ax.set_xlabel("Hidden column");ax.set_ylabel("Input position")
        ax.set_xticks(range(8));ax.set_yticks(range(11))
    if continuous:fig.colorbar(picture,ax=list(axes),label="Connection frequency",shrink=.8)
    fig.suptitle(title)
    for suffix in ("png","pdf"):fig.savefig(Path(path).with_suffix("."+suffix),dpi=160)
    plt.close(fig)


def save_task_masks(destination, task, data, output, centroid, split):
    """Align each mask independently to the analytical window reference.

    Canonical arrays remain untouched. Metrics use those canonical arrays.
    Alignment is a separate visual diagnostic, never a training target transform.
    """
    destination=Path(destination);directory=destination/"generated_masks"
    directory.mkdir(parents=True,exist_ok=True)
    reference=analytical_mask()
    canonical={"generated":output["mask"].detach().cpu(),"imp":data["targets"],
               "functional":exact_topk(data["functional_scores"])}
    aligned={};orders={}
    for name,batch in canonical.items():
        pairs=[align_to_reference(mask,reference) for mask in batch]
        aligned[name]=torch.stack([p[0] for p in pairs]);orders[name]=torch.stack([p[1] for p in pairs])
    aligned_centroid,centroid_order=align_to_reference(centroid,reference)
    save_torch(directory/f"{task}.pt",{"split":split,"network_seeds":data["seeds"],
        "logits":output["logits"].detach().cpu(),"canonical":canonical,
        "analytical":reference,"aligned_to_analytical":aligned,
        "alignment_permutations":orders,"functional_column_orders":data["column_orders"],
        "train_latent_centroid":centroid.detach().cpu(),
        "train_latent_centroid_aligned":aligned_centroid,"centroid_permutation":centroid_order,
        "alignment_scope":"independent column assignment for visualization only"})
    heatmaps=destination/"heatmaps";heatmaps.mkdir(parents=True,exist_ok=True)
    plot_panels(heatmaps/f"{task}_example",{
        "Analytical Toeplitz":reference,"IMP aligned":aligned["imp"][0],
        "Generated aligned":aligned["generated"][0],"Centroid aligned":aligned_centroid},
        f"{task}: {split} example; independently aligned columns")
    means={"Analytical Toeplitz":reference,"IMP frequency":aligned["imp"].mean(0),
           "Generated frequency":aligned["generated"].mean(0),"Functional frequency":aligned["functional"].mean(0)}
    plot_panels(heatmaps/f"{task}_mean",means,
        f"{task}: mean over {len(data['seeds'])} {split} networks; independently aligned",True)
    save_torch(heatmaps/f"{task}_values.pt",means)


def save_generation(model, datasets, destination, device, seed, split="validation", enabled=True):
    model.eval()
    with torch.no_grad():
        for task,values in progress(datasets.items(),enabled,desc="Saving masks / heatmaps",unit="task"):
            data=values[split]
            output=model(task,data["features"].to(device),sample=False)
            mu,_=model.encoders[task](values["train"]["features"].to(device))
            centroid=exact_topk(model.decoder(mu.mean(0,keepdim=True)))[0].cpu()
            save_task_masks(destination,task,data,output,centroid,split)
    save_prior(model,destination,device,seed)


def save_prior(model,destination,device,seed):
    generator=torch.Generator().manual_seed(seed+9927)
    with torch.no_grad():
        z=torch.randn(128,model.decoder.body[0].in_features,generator=generator).to(device)
        generated=exact_topk(model.decoder(z)).cpu()
        pairs=[align_to_reference(mask,analytical_mask()) for mask in generated]
        aligned=torch.stack([p[0] for p in pairs])
        save_torch(Path(destination)/"generated_masks/shared_prior.pt",{"z":z.cpu(),"canonical":generated,
            "aligned_to_analytical":aligned,"alignment_permutations":torch.stack([p[1] for p in pairs]),
            "analytical":analytical_mask()})
        plot_panels(Path(destination)/"heatmaps/shared_prior",{
            "Analytical Toeplitz":analytical_mask(),"Prior frequency aligned":aligned.mean(0)},
            "Shared prior: 128 samples; independently aligned columns",True)
        return z, generated

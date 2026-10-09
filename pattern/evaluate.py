"""Evaluate source-only latent transfer or reconstruction on unseen bank networks."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from .bank.imp import initialization, fit_fixed_mask, logits
from .datasets import load_task, input_table, labels_for, balanced_rows
from .io import new_directory, save_json, save_torch, load_run, device_name
from .masks import exact_topk, toeplitz_metrics, mean_structure
from .models.task_model import Experiment
from .losses import mask_vae_loss
from .reporting import progress, save_losses, save_loss_history, save_task_masks, save_prior
from .latent_evaluate import evaluate_latent


def reconstruction_metrics(predicted,target):
    overlap=(predicted*target).sum((-2,-1))
    return {"edge_f1":float((overlap/32).mean()),"iou":float((overlap/(64-overlap)).mean()),
            "exact_mask_fraction":float((predicted==target).flatten(1).all(1).float().mean()),
            **mean_structure(predicted)}


def fresh_quality(task, masks, manifest, steps, replicas, device, destination=None, show_progress=True, log_every=10):
    config=manifest["config"]; _,x=input_table();y=labels_for(x,task)
    pool=torch.tensor(manifest["partitions"]["support"])
    rows=(pool if config["bank"].get("support_sampling")=="all support observations" else
          balanced_rows(y,pool,config["bank"]["support_count"],config["seed"]+117))
    query=torch.tensor(manifest["partitions"]["test"])
    xs,ys,xq,yq=[v.to(device) for v in (x[rows],y[rows],x[query],y[query])]
    records=[]; loss_records=[]
    fits=progress(range(len(masks["prediction"])*replicas*len(masks)),show_progress,desc=f"{task} fresh networks",unit="fit",leave=False)
    for sample in range(len(masks["prediction"])):
        for replica in range(replicas):
            seed=config["seed"]+2000003*(sample+1)+replica
            initial=initialization(seed,device)
            for method,batch in masks.items():
                mask=batch[sample].to(device)
                loss_history=[]
                state=fit_fixed_mask(initial,mask,xs,ys,steps,.03,.001,loss_history=loss_history,
                    log_every=log_every,show_progress=show_progress,description=f"{task} {method} {sample}/{replica}",
                    class_balanced=config["bank"].get("support_loss")=="class-balanced BCE")
                loss_records.extend({"task":task,"method":method,"sample":sample,"replica":replica,**row} for row in loss_history)
                fits.update(1)
                with torch.no_grad():
                    prediction=logits(state,mask,xq)
                    loss=float(F.binary_cross_entropy_with_logits(prediction,yq))
                    accuracy=float(((prediction>0)==(yq>.5)).float().mean())
                records.append({"method":method,"sample":sample,"replica":replica,
                                "initialization_seed":seed,"test_bce":loss,"test_accuracy":accuracy})
    fits.close()
    if destination is not None:
        save_loss_history(Path(destination)/"child_losses"/task,loss_records)
    summary={method:{"test_bce":sum(r["test_bce"] for r in records if r["method"]==method)/(len(masks[method])*replicas),
                     "test_accuracy":sum(r["test_accuracy"] for r in records if r["method"]==method)/(len(masks[method])*replicas)}
             for method in masks}
    return {"summary":summary,"records":records,"steps":steps,"replicas":replicas,
            "test_rows":len(query),"scope":"same task, fresh weights, unseen observations"}


def evaluate(run, split="test", device="cpu", evaluation_id=None, child_steps=0, replicas=3, show_progress=True, log_every=10):
    if split not in ("validation","test") or child_steps<0 or replicas<1 or log_every<1:
        raise ValueError("invalid evaluation split or child budget")
    run=Path(run)
    checkpoint,bank,manifest=load_run(run)
    config=checkpoint["config"];reference=checkpoint["bank_reference"]
    model=Experiment(config["tasks"],config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state"]);model.eval()
    # Check all data before opening an output directory.
    datasets={task:{role:load_task(bank,task,role,manifest=manifest) for role in ("train",split)}
              for task in config["tasks"]}
    destination=new_directory(run/"evaluations",evaluation_id)
    report={"checkpoint_epoch":checkpoint["epoch"],"split":split,"k":32,
            "bank_reference":reference,"tasks":{},"child_steps":child_steps}
    generator=torch.Generator().manual_seed(config["seed"]+9927)
    loss_rows=[]
    with torch.no_grad():
        for task_number,(task,values) in enumerate(progress(datasets.items(),show_progress,desc=f"Evaluation ({device})",unit="task"),1):
            data=values[split];train=values["train"]
            if set(data["seeds"]) & set(train["seeds"]):raise ValueError("evaluation networks overlap training")
            output=model(task,data["features"].to(device),sample=False)
            predicted=output["mask"].cpu()
            loss,parts=mask_vae_loss(output,data["targets"].to(device),config["training"]["beta"],config["training"]["hard_loss_weight"])
            loss_row={"task":task,"task_index":task_number,"loss":float(loss),**{key:float(value) for key,value in parts.items()}}
            loss_rows.append(loss_row)
            save_json(destination/"loss_history.json",loss_rows)
            target=data["targets"]
            average_mask=exact_topk(train["targets"].mean(0)).expand_as(predicted).clone()
            functional=exact_topk(data["functional_scores"])
            random=exact_topk(torch.rand(predicted.shape,generator=generator))
            train_mu,_=model.encoders[task](train["features"].to(device))
            centroid=exact_topk(model.decoder(train_mu.mean(0,keepdim=True)))[0].cpu()
            methods={"prediction":predicted,"train_majority":average_mask,
                     "functional_top32":functional,"random32":random}
            task_metrics={name:reconstruction_metrics(masks,target) for name,masks in methods.items()}
            task_metrics["vae_loss"]=loss_row
            task_metrics["target_structure"]=mean_structure(target)
            task_metrics["train_latent_centroid_structure"]=toeplitz_metrics(centroid)
            report["tasks"][task]=task_metrics
            save_torch(destination/f"predicted_masks/{task}.pt",{
                "network_seeds":data["seeds"],"logits":output["logits"].cpu(),
                "predicted":predicted,"target":target,"column_orders":data["column_orders"],
                "train_latent_centroid":centroid,"canonical_order":"functional coordinate centroid"})
            save_task_masks(destination,task,data,output,centroid,split)
            if child_steps:
                methods["train_latent_centroid"]=centroid.expand_as(predicted).clone()
                methods["dense"]=torch.ones_like(predicted)
                # Re-enable gradients only for newly initialized downstream networks.
                with torch.enable_grad():
                    report["tasks"][task]["fresh_quality"]=fresh_quality(task,methods,manifest,child_steps,replicas,device,destination,show_progress,log_every)
        z,prior=save_prior(model,destination,device,config["seed"])
        save_torch(destination/"shared_prior_masks.pt",{"z":z.cpu(),"masks":prior})
        report["shared_prior"]={"count":len(prior),"distinct_masks":len(torch.unique(prior.flatten(1),dim=0)),
            "toeplitz_mse":sum(toeplitz_metrics(mask)["toeplitz_mse"] for mask in prior)/len(prior)}
    report["macro_prediction"]={key:sum(v["prediction"][key] for v in report["tasks"].values())/len(report["tasks"])
                                for key in report["tasks"][config["tasks"][0]]["prediction"]}
    save_losses(destination,loss_rows,x_key="task_index")
    save_json(destination/"metrics.json",report)
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--mode",choices=("latent","reconstruction"),default="latent")
    parser.add_argument("--split",choices=("validation","test"),default="test")
    parser.add_argument("--evaluation-id")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--no-progress",action="store_true")
    parser.add_argument("--loss-log-every",type=int)
    parser.add_argument("--child-steps",type=int,help="fresh-network steps; reconstruction accepts 0 to skip")
    parser.add_argument("--replicas",type=int)
    for flag in ("z-starts","z-steps","inner-steps","support-count","query-count"):
        parser.add_argument("--"+flag,type=int)
    for flag in ("z-lr","inner-lr","child-lr"):
        parser.add_argument("--"+flag,type=float)
    parser.add_argument("--child-l2",type=float)
    parser.add_argument("--child-select-every",type=int)
    parser.add_argument("--support-sampling",choices=("all support observations","balanced subset"))
    parser.add_argument("--support-loss",choices=("class-balanced BCE","mean BCE"))
    args=parser.parse_args()
    if args.threads<1:raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    device=device_name(args.device)
    print(f"Device: {device}",flush=True)
    if args.mode == "latent":
        options={key:getattr(args,key) for key in ("z_starts","z_steps","z_lr","inner_steps","inner_lr",
                    "support_count","query_count","child_steps","child_lr","replicas",
                    "child_l2","child_select_every","support_sampling","support_loss")}
        options["log_every"]=args.loss_log_every
        destination=evaluate_latent(args.run,device,args.evaluation_id,not args.no_progress,**options)
    else:
        destination=evaluate(args.run,args.split,device,args.evaluation_id,
            args.child_steps if args.child_steps is not None else 0,
            args.replicas if args.replicas is not None else 3,not args.no_progress,
            args.loss_log_every if args.loss_log_every is not None else 10)
    print(f"Evaluation saved: {destination}")

if __name__=="__main__":main()

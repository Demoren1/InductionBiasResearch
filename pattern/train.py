"""Train task-specific NF/VAE encoders and one shared mask decoder."""
import argparse
import json
import math
from pathlib import Path
import torch
from .bank.collect import DEFAULT_CONFIG
from .datasets import load_task
from .io import ROOT, load_config, new_directory, save_json, save_torch, digest, device_name
from .models.task_model import Experiment
from .losses import mask_vae_loss
from .reporting import progress, save_losses, save_generation


def train(config, bank, run_id=None, device="cpu", parent=None, show_progress=True):
    bank=Path(bank).resolve()
    manifest=json.loads((bank/"manifest.json").read_text())
    if manifest["config"]["bank"] != config["bank"] or manifest["config"]["seed"] != config["seed"]:
        raise ValueError("bank protocol and seed must match training config")
    data={task:{split:load_task(bank,task,split) for split in ("train","validation")}
          for task in config["tasks"]}
    for task, values in data.items():
        if set(values["train"]["seeds"]) & set(values["validation"]["seeds"]):
            raise ValueError(f"network seeds overlap for {task}")
    torch.manual_seed(config["seed"])
    model=Experiment(config["tasks"],config["model"]).to(device)
    options=config["training"]
    optimizer=torch.optim.Adam(model.parameters(),lr=options["lr"])
    destination=new_directory(parent or ROOT/"pattern/runs",run_id)
    reference={"bank_path":str(bank),"manifest_sha256":digest(bank/"manifest.json"),
               "splits":{task:{split:values[split]["seeds"] for split in values} for task,values in data.items()}}
    save_json(destination/"config.json",config)
    save_json(destination/"bank_reference.json",reference)
    best=float("inf"); history=[]
    generator=torch.Generator().manual_seed(config["seed"]+17)
    epochs=progress(range(1,options["epochs"]+1),show_progress,desc=f"Training ({device})",unit="epoch")
    for epoch in epochs:
        model.train()
        beta=options["beta"]*min(1,epoch/max(1,options["kl_warmup_epochs"]))
        permutations={task:torch.randperm(len(values["train"]["features"]),generator=generator)
                      for task,values in data.items()}
        batches=max(math.ceil(len(p)/options["batch_size"]) for p in permutations.values())
        total=0.; train_parts={task:{} for task in data}
        minibatches=progress(range(batches),show_progress,desc="Train batches",unit="batch",leave=False)
        for batch in minibatches:
            optimizer.zero_grad(set_to_none=True)
            objective=0.
            for task,values in data.items():
                indices=permutations[task]
                start=(batch*options["batch_size"])%len(indices)
                rows=indices[start:start+options["batch_size"]]
                x=values["train"]["features"][rows].to(device)
                target=values["train"]["targets"][rows].to(device)
                output=model(task,x,sample=True)
                loss,parts=mask_vae_loss(output,target,beta,options["hard_loss_weight"])
                for key,value in {"loss":loss,**parts}.items():
                    train_parts[task][key]=train_parts[task].get(key,0.)+float(value.detach())/batches
                objective=objective+loss/len(data)
            if not torch.isfinite(objective): raise RuntimeError("nonfinite VAE loss")
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step(); total+=float(objective.detach())
            minibatches.set_postfix(loss=f"{float(objective.detach()):.5f}")
        model.eval(); validations={}
        with torch.no_grad():
            for task,values in progress(data.items(),show_progress,desc="Validation",unit="task",leave=False):
                output=model(task,values["validation"]["features"].to(device),sample=False)
                loss,parts=mask_vae_loss(output,values["validation"]["targets"].to(device),
                                       options["beta"],options["hard_loss_weight"])
                validations[task]={"loss":float(loss),**{key:float(value) for key,value in parts.items()}}
        validation=sum(v["loss"] for v in validations.values())/len(validations)
        row={"epoch":epoch,"train_loss":total/batches,"beta":beta,
             "validation_loss":validation,"per_task_train":train_parts,"per_task_validation":validations}
        history.append(row);save_json(destination/"history.json",history)
        checkpoint={"config":config,"model_state":model.state_dict(),"epoch":epoch,
                    "validation_loss":validation,"bank_reference":reference}
        save_torch(destination/"checkpoints/last.pt",checkpoint)
        if validation < best:
            best=validation
            save_torch(destination/"checkpoints/best.pt",checkpoint)
            save_torch(destination/"checkpoints/shared_decoder.pt",model.decoder.state_dict())
            for task,encoder in model.encoders.items():
                save_torch(destination/f"checkpoints/task_encoders/{task}.pt",encoder.state_dict())
        epochs.set_postfix(train=f"{total/batches:.5f}",validation=f"{validation:.5f}")
        if not show_progress:
            print(f"epoch {epoch}/{options['epochs']}: train={total/batches:.5f}, validation={validation:.5f}",flush=True)
    save_losses(destination,history)
    selected=torch.load(destination/"checkpoints/best.pt",weights_only=True,map_location="cpu")
    model.load_state_dict(selected["model_state"])
    save_generation(model,data,destination,device,config["seed"],enabled=show_progress)
    save_json(destination/"generation.json",{"checkpoint_epoch":selected["epoch"],"split":"validation",
        "alignment":"independent column matching to analytical mask; canonical arrays also saved"})
    save_json(destination/"COMPLETE.json",{"best_validation_loss":best,"epochs":options["epochs"],
                                          "tasks":config["tasks"]})
    return destination


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--bank",type=Path,required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--threads",type=int,default=1)
    parser.add_argument("--no-progress",action="store_true")
    for flag in ("epochs","batch-size","kl-warmup-epochs","nf-channels","latent-dim","encoder-width","decoder-width"):
        parser.add_argument("--"+flag,type=int)
    for flag in ("lr","beta","hard-loss-weight"):
        parser.add_argument("--"+flag,type=float)
    args=parser.parse_args()
    if args.threads < 1: raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    config=load_config(args.config)
    for section,keys in (("training",("epochs","batch_size","lr","beta","kl_warmup_epochs","hard_loss_weight")),
                         ("model",("nf_channels","latent_dim","encoder_width","decoder_width"))):
        for key in keys:
            if getattr(args,key) is not None:config[section][key]=getattr(args,key)
    options=config["training"]
    if min(options["epochs"],options["batch_size"])<1 or options["lr"]<=0 or options["beta"]<0 or options["kl_warmup_epochs"]<0 or not 0<=options["hard_loss_weight"]<=1:
        raise ValueError("invalid training overrides")
    if min(config["model"].values())<1:raise ValueError("model dimensions must be positive")
    device=device_name(args.device)
    print(f"Device: {device}",flush=True)
    print(f"Run saved: {train(config,args.bank,args.run_id,device,show_progress=not args.no_progress)}")

if __name__=="__main__": main()

"""Create exclusively new IMP banks beneath root data/pattern/banks."""
import argparse
from pathlib import Path
import torch
from ..datasets import input_table, labels_for, partitions, balanced_rows
from ..io import ROOT, load_config, new_directory, save_json, save_torch, digest, device_name
from .imp import run_imp
from .functional_maps import extract_maps, extract_nf_channels, NF_CHANNELS, NF_BANK_SCHEMA

DEFAULT_CONFIG=ROOT/"pattern/configs/experiment.json"

def collect(config, parent=None, bank_id=None, device="cpu"):
    parent=Path(parent) if parent is not None else ROOT/"data/pattern/banks"
    data_root=(ROOT/"data").resolve()
    if not parent.resolve().is_relative_to(data_root):
        raise ValueError("functional banks must be saved under the repository data directory")
    options=config["bank"]; ids,x=input_table(); pools=partitions(config["seed"])
    query_rows={}; labels={task:labels_for(x,task) for task in config["tasks"]}
    for task,y in labels.items():
        balanced_rows(y,pools["support"],options["support_count"],config["seed"])
        query_rows[task]=balanced_rows(y,pools["query"],options["query_count"],config["seed"]+91009)
    destination=new_directory(parent,bank_id)
    manifest={"schema":NF_BANK_SCHEMA,"status":"running","k":32,
        "bank_id":destination.name,"config":config,"tasks":{},
        "input_representation":{"channels":list(NF_CHANNELS),"shape":[11,8,3],
            "normalization":"per-network, per-channel max(abs), clamped at 1e-8",
            "gradient":{"loss":"mean BCE, excluding L2","observations":"network support_ids",
                        "wrt":"first-layer W in W*IMP_mask","point":"terminal sparse trained state"},
            "functional_map":"q_abs = E_probe |x_i * d(psi_j)/dx_i|"},
        "partitions":{key:value.tolist() for key,value in pools.items()}}
    save_json(destination/"manifest.json",manifest)
    probe=pools["probe"][:options["probe_count"]]; probe_x=x[probe]
    save_torch(destination/"probe.pt",{"x":probe_x,"ids":ids[probe]})
    count=options["maps_per_task"]
    nval=max(1,count//8); ntest=max(1,count//8)
    for task_index,task in enumerate(config["tasks"]):
        y=labels[task]; task_seed=config["seed"]+100003*task_index
        allocation=torch.randperm(count,generator=torch.Generator().manual_seed(task_seed+71)).tolist()
        roles={index:("test" if rank<ntest else "validation" if rank<ntest+nval else "train")
               for rank,index in enumerate(allocation)}
        manifest["tasks"][task]=[]
        for number in range(count):
            seed=task_seed+1000003*(number+1)
            rows=balanced_rows(y,pools["support"],options["support_count"],seed)
            query=query_rows[task]
            state,mask,history,initial=run_imp(seed,x[rows],y[rows],x[query],y[query],
                steps=options["steps_per_round"],prune_fraction=options["prune_fraction"],
                lr=options["lr"],l2=options["l2"],device=device)
            target=destination/task/str(seed); target.mkdir(parents=True,exist_ok=False)
            raw=extract_maps(state,mask,probe_x)
            channels=extract_nf_channels(state,mask,x[rows],y[rows],raw["q_abs"])
            artifacts={"nf_features.pt":{**channels,"support_ids":ids[rows],"channels":list(NF_CHANNELS)},
                "functional_map.pt":raw,"imp_mask.pt":mask.bool(),
                "network_state.pt":{"state_dict":state,"initial_state":initial,
                    "network_seed":seed,"support_ids":ids[rows],"query_ids":ids[query]},
                "pruning_history.json":history}
            for name,value in artifacts.items():
                (save_json if name.endswith(".json") else save_torch)(target/name,value)
            manifest["tasks"][task].append({"network_seed":seed,"split":roles[number],
                "path":str(target.relative_to(destination)),"hashes":{name:digest(target/name) for name in artifacts}})
            save_json(destination/"manifest.json",manifest)
            print(f"{task}: {number+1}/{count} weight/gradient/functional maps, active connections={int(mask.sum())}",flush=True)
    manifest["status"]="complete"; save_json(destination/"manifest.json",manifest)
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    parser.add_argument("--parent",type=Path,default=ROOT/"data/pattern/banks",
                        help="bank parent directory beneath repository data")
    parser.add_argument("--bank-id",help="new identifier; existing banks cannot be overwritten")
    parser.add_argument("--device",default="auto")
    parser.add_argument("--threads",type=int,default=1)
    args=parser.parse_args()
    if args.threads<1: raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    bank=collect(load_config(args.config),parent=args.parent,bank_id=args.bank_id,device=device_name(args.device))
    print(f"Bank saved: {bank}")

if __name__=="__main__": main()

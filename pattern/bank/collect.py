"""Create exclusively new IMP banks beneath root data/pattern/banks."""
import argparse
from pathlib import Path
import torch
from ..datasets import input_table, labels_for, partitions, balanced_rows
from ..io import ROOT, load_config, new_directory, save_json, save_torch, digest, device_name
from ..reporting import progress
from .imp import run_imp_batch
from .functional_maps import extract_maps, extract_nf_channels, NF_CHANNELS, NF_BANK_SCHEMA

DEFAULT_CONFIG=ROOT/"pattern/configs/experiment.json"

def collect(config, parent=None, bank_id=None, device="cpu", network_batch_size=128, show_progress=True):
    if network_batch_size < 1:
        raise ValueError("network batch size must be positive")
    parent=Path(parent) if parent is not None else ROOT/"data/pattern/banks"
    data_root=(ROOT/"data").resolve()
    if not parent.resolve().is_relative_to(data_root):
        raise ValueError("functional banks must be saved under the repository data directory")
    options=config["bank"]; ids,x=input_table(); pools=partitions(config["seed"])
    full_support=options.get("support_sampling")=="all support observations"
    class_balanced=options.get("support_loss")=="class-balanced BCE"
    select_every=options.get("select_every",0)
    if full_support and options["support_count"]!=len(pools["support"]):
        raise ValueError("full-support budget must match the support partition")
    query_rows={}; labels={task:labels_for(x,task) for task in config["tasks"]}
    for task,y in labels.items():
        if not full_support:balanced_rows(y,pools["support"],options["support_count"],config["seed"])
        query_rows[task]=balanced_rows(y,pools["query"],options["query_count"],config["seed"]+91009)
    destination=new_directory(parent,bank_id)
    manifest={"schema":NF_BANK_SCHEMA,"status":"running","k":32,
        "bank_id":destination.name,"config":config,"tasks":{},
        "collection":{"network_batch_size":network_batch_size,"independent_networks":True},
        "input_representation":{"channels":list(NF_CHANNELS),"shape":[11,8,3],
            "normalization":"per-network, per-channel max(abs), clamped at 1e-8",
            "gradient":{"loss":"class-balanced mean support BCE, excluding L2" if class_balanced else "mean BCE, excluding L2",
                        "observations":"network support_ids","wrt":"first-layer W in W*IMP_mask",
                        "point":"query-selected terminal sparse state" if select_every else "terminal sparse trained state"},
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
        query=query_rows[task]
        for start in progress(range(0,count,network_batch_size),show_progress,desc=f"Collect {task}",unit="batch"):
            stop=min(count,start+network_batch_size)
            seeds=[task_seed+1000003*(number+1) for number in range(start,stop)]
            support=(pools["support"].expand(len(seeds),-1) if full_support else
                     torch.stack([balanced_rows(y,pools["support"],options["support_count"],seed) for seed in seeds]))
            states,masks,histories,initials=run_imp_batch(seeds,x[support],y[support],x[query],y[query],
                steps=options["steps_per_round"],prune_fraction=options["prune_fraction"],
                lr=options["lr"],l2=options["l2"],device=device,show_progress=show_progress,
                class_balanced=class_balanced,select_every=select_every)
            for index,seed in enumerate(seeds):
                # Clone slices so a per-network file never serializes the entire batch storage.
                state={key:value[index].clone() for key,value in states.items()}
                initial={key:value[index].clone() for key,value in initials.items()}
                rows,mask=support[index],masks[index].clone()
                target=destination/task/str(seed); target.mkdir(parents=True,exist_ok=False)
                raw=extract_maps(state,mask,probe_x)
                channels=extract_nf_channels(state,mask,x[rows],y[rows],raw["q_abs"],class_balanced)
                artifacts={"nf_features.pt":{**channels,"support_ids":ids[rows],"channels":list(NF_CHANNELS)},
                    "functional_map.pt":raw,"imp_mask.pt":mask.bool(),
                    "network_state.pt":{"state_dict":state,"initial_state":initial,
                        "network_seed":seed,"support_ids":ids[rows],"query_ids":ids[query]},
                    "pruning_history.json":histories[index]}
                for name,value in artifacts.items():
                    (save_json if name.endswith(".json") else save_torch)(target/name,value)
                manifest["tasks"][task].append({"network_seed":seed,"split":roles[start+index],
                    "path":str(target.relative_to(destination)),"hashes":{name:digest(target/name) for name in artifacts}})
                print(f"{task}: {start+index+1}/{count} weight/gradient/functional maps, active connections=32",flush=True)
            save_json(destination/"manifest.json",manifest)
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
    parser.add_argument("--network-batch-size",type=int,default=128,help="independent MLPs trained together")
    parser.add_argument("--no-progress",action="store_true")
    args=parser.parse_args()
    if args.threads<1: raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    bank=collect(load_config(args.config),parent=args.parent,bank_id=args.bank_id,device=device_name(args.device),
                 network_batch_size=args.network_batch_size,show_progress=not args.no_progress)
    print(f"Bank saved: {bank}")

if __name__=="__main__": main()

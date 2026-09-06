"""Independent witness audit and paired hard/soft evaluation of normalized search."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from data.generate import configure_compute_device, ideal_mask
from evaluation.eval_generated_masks import _load_model
from evaluation.structural import best_permutation_iou
from evaluation.report_combined_task_agreement import (
    _decode, _check_binary_topk, _check_soft, _assert_close, _crossed_ci,
    sha, tensor_sha, write_json_atomic,
)

METHODS = ["task_only", "normalized_0p1", "normalized_0p3", "normalized_1", "fixed_0p01",
           "agreement_30", "agreement_1000", "prior", "ideal"]
SOFT_METHODS = METHODS[:5] + ["prior"]
DEFAULT_OUT = ROOT / "outputs/normalized_task_agreement/20260906"
OLD = ROOT / "outputs/combined_task_agreement/20260906"


def audit(root, protocol, profile, gap, shard, models, device):
    spec = protocol["profiles"][profile]
    folder = root / profile / f"gap{gap}_shard{shard}"
    saved = torch.load(folder / "search.pt", weights_only=True, map_location="cpu")
    meta = saved["metadata"]
    done = json.loads((folder / "done.json").read_text())
    tasks = spec["tasks"][str(gap)]
    assert len(tasks) == 8 and all(config.parse_task(t).gap == gap for t in tasks)
    start, stop = shard*32, (shard+1)*32
    expected = {"profile": profile, "gap": gap, "start": start, "stop": stop,
                "tasks": tasks, "methods": METHODS, "protocol_sha256": sha(root / "protocol.json")}
    for key, value in expected.items():
        assert meta[key] == value, (folder, key)
        assert done[key] == value, (folder, "done", key)
    assert done["decoder_unchanged"]
    assert done["search_sha256"] == sha(folder / "search.pt")
    previous = torch.load(OLD / profile / f"gap{gap}_shard{shard}" / "search.pt", weights_only=True)
    assert torch.equal(saved["initial_z"], previous["initial_z"])
    parent = torch.load(spec["prior"], weights_only=True)
    index = parent["gaps"].tolist().index(gap)
    original_z = torch.stack([parent["latents"]["prior"][key].reshape(8,64,32)[index,start:stop]
                              for key in ("z1", "z2")])
    assert torch.equal(saved["initial_z"], original_z)
    for control in ("agreement_30", "agreement_1000"):
        for key in ("z", "soft", "hard"):
            assert torch.equal(saved["controls"][control][key], previous["controls"][control][key])
    assert len(saved["combined"]["history"]) == 30
    for row in saved["combined"]["history"]:
        ratio=torch.tensor(row["weighted_agreement_ratio"])
        valid=torch.tensor(row["valid_gradient_norms"])
        coefficient=torch.tensor(row["effective_coefficient"])
        cosine=torch.tensor(row["gradient_cosine"])
        assert ratio.shape==valid.shape==coefficient.shape==cosine.shape==(5,32)
        assert torch.isfinite(ratio).all() and torch.isfinite(cosine).all()
        assert (cosine.abs()<=1.00001).all()
        assert (coefficient[0]==0).all() and (coefficient[4]==.01).all()
        for i,alpha in ((1,.1),(2,.3),(3,1.)):
            assert torch.allclose(ratio[i][valid[i]],torch.full_like(ratio[i][valid[i]],alpha),atol=2e-5)
    z = torch.cat([saved["combined"]["z"], saved["controls"]["agreement_30"]["z"],
                   saved["controls"]["agreement_1000"]["z"], saved["initial_z"][None]])
    assert z.shape == (8,2,32,32) and torch.isfinite(z).all()
    assert (z.norm(dim=-1) <= 8.00002).all()
    soft_saved = torch.cat([saved["combined"]["soft"], saved["controls"]["agreement_30"]["soft"],
                            saved["controls"]["agreement_1000"]["soft"], saved["prior"]["soft"]])
    hard_saved = torch.cat([saved["combined"]["hard"], saved["controls"]["agreement_30"]["hard"],
                            saved["controls"]["agreement_1000"]["hard"], saved["prior"]["hard"]])
    condition = models[0].condition([tasks[0]], device=device)
    soft, hard = [], []
    for d, model in enumerate(models):
        assert torch.equal(condition, model.condition([tasks[0]], device=device))
        s, h = _decode(model, z[:,d].to(device), condition, temperature=.5, k=96)
        _assert_close(s, soft_saved[:,d], f"{folder} soft decoder{d}", cpu=device.type == "cpu")
        assert torch.equal(h, hard_saved[:,d]), (folder, "hard redecode", d)
        soft.append(s); hard.append(h)
    soft = torch.stack(soft,dim=1)
    hard = torch.stack(hard,dim=1)
    target = ideal_mask(tasks[0]).float()
    hard = torch.cat([hard, target[None,None,None].expand(1,2,32,16,16)])
    _check_binary_topk(hard, str(folder))
    _check_soft(soft, str(folder))
    assert tensor_sha(hard) == done["hard_masks_sha256"]
    # Hash the original saved soft values, since CPU re-decoding is tolerant.
    soft_subset = soft_saved[[0,1,2,3,4,7]]
    results = {}
    for kind, names, masks in (("hard", METHODS, hard), ("soft", SOFT_METHODS, soft_subset)):
        rows=[]
        for seed in (0,1,2):
            name = f"evaluation_seed{seed}.pt" if kind == "hard" else f"evaluation_soft_seed{seed}.pt"
            row = torch.load(folder / name, weights_only=True, map_location="cpu")
            emeta = row["metadata"]
            assert row["seed"] == seed
            assert emeta["search_sha256"] == sha(folder / "search.pt")
            assert emeta["protocol_sha256"] == expected["protocol_sha256"]
            assert emeta["methods"] == names
            assert emeta[kind + "_masks_sha256"] == tensor_sha(masks)
            for key in ("profile", "gap", "start", "stop", "tasks"):
                assert emeta[key] == expected[key]
            for key in ("acc", "bce"):
                assert row[key].shape == (len(names),2,8,32)
                assert torch.isfinite(row[key]).all()
            assert ((row["acc"] >= 0) & (row["acc"] <= 1)).all()
            assert (row["bce"] >= 0).all()
            rows.append(row)
        results[kind] = {key:torch.stack([row[key] for row in rows]) for key in ("acc", "bce")}
    iou = torch.tensor([[[best_permutation_iou(mask, target)["iou"] for mask in arm]
                         for arm in method] for method in hard], dtype=torch.float64)
    pair_iou = torch.tensor([[best_permutation_iou(method[0,n],method[1,n])["iou"] for n in range(32)]
                             for method in hard], dtype=torch.float64)
    old_rows = [torch.load(OLD / profile / f"gap{gap}_shard{shard}" / f"evaluation_seed{s}.pt",
                           weights_only=True) for s in (0,1,2)]
    old_acc = torch.stack([row["acc"][0] for row in old_rows])
    reproduction = {"task_only_max_z_delta": float((saved["combined"]["z"][0]-previous["combined"]["z"][0]).abs().max()),
                    "task_only_hard_equal": torch.equal(saved["combined"]["hard"][0],previous["combined"]["hard"][0]),
                    "task_only_max_acc_delta": float((results["hard"]["acc"][:,0]-old_acc).abs().max())}
    return {"tasks":tasks, "hard_masks":hard, "soft_masks":soft_subset, "iou":iou,
            "pair_iou":pair_iou, "evaluations":results,
            "history":saved["combined"]["history"], "reproduction":reproduction}


def comparisons(acc, tasks, names, references):
    pairs = sorted({(config.parse_task(t).a,config.parse_task(t).b) for t in tasks})
    gaps = sorted({config.parse_task(t).gap for t in tasks})
    order = {(config.parse_task(t).a,config.parse_task(t).b,config.parse_task(t).gap):i for i,t in enumerate(tasks)}
    result={}
    for candidate in names[1:5]:
        c=names.index(candidate)
        for reference in references:
            r=names.index(reference)
            delta=(acc[:,c]-acc[:,r]).mean((0,1)).double()
            grid=torch.stack([torch.stack([delta[order[a,b,g]] for g in gaps]) for a,b in pairs])
            result[f"{candidate}_minus_{reference}"]={"delta":float(grid.mean()),
                                                     "ci95":_crossed_ci(grid,seed=20260906+c*100+r)}
    return result


def metrics(evaluation, iou, pair_iou, tasks):
    result={}
    gaps=sorted({config.parse_task(t).gap for t in tasks})
    for kind,names in (("hard",METHODS),("soft",SOFT_METHODS)):
        acc,bce=evaluation[kind]["acc"],evaluation[kind]["bce"]
        rows={name:{"accuracy":float(acc[:,i].double().mean()),"bce":float(bce[:,i].double().mean())}
              for i,name in enumerate(names)}
        if kind=="hard":
            for i,name in enumerate(names):
                rows[name].update(iou=float(iou[i].mean()), max_iou=float(iou[i].max()),
                                  exact_ideal_count=int((iou[i]==1).sum()),pair_iou=float(pair_iou[i].mean()))
        by_gap={}
        for gi,gap in enumerate(gaps):
            indexes=[i for i,t in enumerate(tasks) if config.parse_task(t).gap==gap]
            by_gap[str(gap)]={name:{"accuracy":float(acc[:,i,:,indexes].double().mean()),
                                    "bce":float(bce[:,i,:,indexes].double().mean())}
                              for i,name in enumerate(names)}
            if kind=="hard":
                for i,name in enumerate(names):
                    by_gap[str(gap)][name].update(iou=float(iou[i,:,gi].mean()),
                                                  exact_ideal_count=int((iou[i,:,gi]==1).sum()))
        result[kind]={"methods":rows,"by_gap":by_gap,
                      "comparisons":comparisons(acc,tasks,names,["task_only","prior"] + (["agreement_30"] if kind=="hard" else []))}
    return result


def diagnostics(histories):
    # Histories retain per-ordinal raw measurements for independently checking
    # any interpretation of scale, conflict, and the realized Adam step.
    result={}
    scalar_keys=[key for key,value in histories[0][0].items()
                 if isinstance(value,list) and len(value)==5]
    for key in scalar_keys:
        values=torch.tensor([[row[key] for row in hist] for hist in histories],dtype=torch.float64)
        if values.ndim==5 and values.shape[2:]==(5,2,32):
            values=values.mean(3)
        if values.ndim!=4 or values.shape[2:]!=(5,32):
            continue
        result[key]={METHODS[i]:{"mean":float(values[:,:,i].mean()),
                                "initial_mean":float(values[:,0,i].mean()),
                                "final_mean":float(values[:,-1,i].mean())}
                     for i in range(5)}
    for representation in ("soft","hard"):
        before=torch.tensor([[r[f"validation_{representation}_bce_before"] for r in h] for h in histories])
        after=torch.tensor([[r[f"validation_{representation}_bce_after"] for r in h] for h in histories])
        delta=(after-before).mean(3)
        result[f"validation_{representation}_same_MLP_step_delta"]={METHODS[i]:{
            "mean":float(delta[:,:,i].mean()),"fraction_improved":float((delta[:,:,i]<0).float().mean())}
            for i in range(5)}
    for name in ("projected_step_dot_task_grad","projected_step_dot_agreement_grad"):
        values=torch.tensor([[r[name] for r in h] for h in histories])
        result[name+"_fraction_positive"]={METHODS[i]:float((values[:,:,i]>0).float().mean()) for i in range(5)}
    return result


def plot(root,profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    labels=["α=0.1","α=0.3","α=1","λ=0.01"]
    for ax,(profile,row) in zip(axes,profiles.items()):
        comps=row["metrics"]["hard"]["comparisons"]
        for offset,reference,color in ((-.12,"task_only","#2774ae"),(.12,"prior","#c66720")):
            entries=[comps[f"{name}_minus_{reference}"] for name in METHODS[1:5]]
            y=[100*e["delta"] for e in entries]
            error=[[100*(e["delta"]-e["ci95"][0]) for e in entries],
                   [100*(e["ci95"][1]-e["delta"]) for e in entries]]
            ax.errorbar([i+offset for i in range(4)],y,yerr=error,fmt="o",capsize=3,color=color,label=reference)
        ax.axhline(0,color="black",lw=.8)
        ax.set_xticks(range(4),labels)
        ax.set_title(profile)
        ax.set_ylabel("Hard-mask accuracy difference, percentage points")
        ax.grid(axis="y",alpha=.2);ax.legend()
    fig.suptitle("Normalized agreement: paired differences, conditional 95% intervals")
    path=root/"accuracy_differences.png"
    fig.savefig(path,dpi=180);plt.close(fig)
    return path


def run(root,device):
    root=root.resolve()
    protocol=json.loads((root/"protocol.json").read_text())
    assert protocol["settings"]["methods"]==METHODS
    for path,digest in protocol["source_hashes"].items():
        assert sha(path)==digest,("source",path)
    for path,digest in protocol["control_source_hashes"].items():
        assert sha(path)==digest,("control",path)
    assert sha(OLD/"protocol.json")==protocol["combined_protocol_sha256"]
    configure_compute_device(str(device))
    profiles={};aggregates={}
    for profile,spec in protocol["profiles"].items():
        for key in ("checkpoint","checkpoint2","prior","split"):
            assert sha(spec[key])==spec[key+"_sha256"]
        models=[_load_model(Path(spec[key]),device)[0] for key in ("checkpoint","checkpoint2")]
        for model in models:
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
        gaps=[]
        for gap in sorted(spec["heldout_gaps"]):
            rows=[audit(root,protocol,profile,gap,s,models,device) for s in (0,1)]
            assert rows[0]["tasks"]==rows[1]["tasks"]
            gaps.append({"tasks":rows[0]["tasks"],
                         "iou":torch.cat([r["iou"] for r in rows],dim=2),
                         "pair_iou":torch.cat([r["pair_iou"] for r in rows],dim=1),
                         "evaluations":{kind:{key:torch.cat([r["evaluations"][kind][key] for r in rows],dim=-1)
                                               for key in ("acc","bce")} for kind in ("hard","soft")},
                         "histories":[r["history"] for r in rows],
                         "reproductions":[r["reproduction"] for r in rows]})
            print(f"Audited {profile} gap{gap}",flush=True)
        tasks=[t for g in gaps for t in g["tasks"]]
        iou=torch.stack([g["iou"] for g in gaps],dim=2)
        pair_iou=torch.stack([g["pair_iou"] for g in gaps],dim=1)
        evaluations={kind:{key:torch.cat([g["evaluations"][kind][key] for g in gaps],dim=3)
                           for key in ("acc","bce")} for kind in ("hard","soft")}
        assert evaluations["hard"]["acc"].shape==(3,9,2,16,64)
        assert evaluations["soft"]["acc"].shape==(3,6,2,16,64)
        histories=[h for g in gaps for h in g["histories"]]
        profiles[profile]={"tasks":tasks,"metrics":metrics(evaluations,iou,pair_iou,tasks),
                           "diagnostics":diagnostics(histories),
                           "reproduction":[r for g in gaps for r in g["reproductions"]]}
        aggregates[profile]={"tasks":tasks,"evaluations":evaluations,"iou":iou,
                             "pair_iou":pair_iou,"histories":histories}
    torch.save(aggregates,root/"audited_aggregate.pt")
    figure=plot(root,profiles)
    summary={"audited":True,"protocol_sha256":sha(root/"protocol.json"),
             "report_source_hashes":{str(Path(__file__).resolve()):sha(__file__),
                                     str(ROOT/"evaluation/report_combined_task_agreement.py"):sha(ROOT/"evaluation/report_combined_task_agreement.py")},
             "aggregate_sha256":sha(root/"audited_aggregate.pt"),
             "profiles":profiles,"figure":str(figure),
             "scope":"supervised adaptation on fixed target tasks; separate draws from same finite input population; conditional bootstrap on motif pairs and ordinal; no alpha/test-based selection"}
    write_json_atomic(root/"summary.json",summary)
    print("Normalized experiment audit complete.",flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(2)
    run(args.out,torch.device(args.device))

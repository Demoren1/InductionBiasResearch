"""Separately disclosed dense-strength control after initial short comparison.

Tune only meta-validation; freeze before this control's target test scoring.
Report support-only fixed weights and separately query-selected checkpoints.
"""
import argparse
import itertools
import json
from pathlib import Path
import torch
import numpy as np
from .graph_flow_run import episodes,write_json
from .graph_flow_child import fit_short
from .stripe_debug import OLD
from .eval_run import _candidate_init_seed
from .meta import _atomic_save,_file_sha256
from .core import build_test_pool
from .evaluate import score_children_batched

GRID=list(itertools.product((.003,.01,.03,.1),(.0001,.001,.01)))


def batch(eps,configs,device):
    specs=[];x=[];y=[];qx=[];qy=[];seeds=[]
    for ep,config,r in itertools.product(eps,configs,range(2,6)):
        lr,l2=config
        specs.append({'task_id':ep['task_id'],'lr':lr,'l2':l2,'init_id':r})
        x.append(ep['support']['x']);y.append(ep['support']['y']);qx.append(ep['query']['x']);qy.append(ep['query']['y'])
        seeds.append(_candidate_init_seed(ep['task_id'],128,r))
    # Independent Adam fits must share LR/L2 within each vectorized call.
    fits=[]
    for lr,l2 in configs:
        ids=[i for i,s in enumerate(specs) if s['lr']==lr and s['l2']==l2]
        fit=fit_short(torch.ones(len(ids),11,8),torch.stack([x[i] for i in ids]),torch.stack([y[i] for i in ids]),
                      torch.stack([qx[i] for i in ids]),torch.stack([qy[i] for i in ids]),[seeds[i] for i in ids],
                      device=device,lr=lr,l2=l2,fixed_horizon=True)
        fit['specs']=[specs[i] for i in ids];fits.append(fit)
    return fits


def tune(root,seed,device):
    torch.set_num_threads(1);folder=root/f'seed_{seed}'/'dense_control';folder.mkdir(exist_ok=True)
    bank=torch.load(OLD/f'seed_{seed}/bank/bank.pt',map_location='cpu',weights_only=False)
    eps=episodes(bank,seed,'val');fits=batch(eps,GRID,device);scores=[]
    for fit in fits:
        tag=f"lr{fit['lr']}_l2{fit['l2']}";_atomic_save(fit,folder/f'tune_{tag}.pt')
        scores.append({'lr':fit['lr'],'l2':fit['l2'],'terminal_query':float(fit['best_query'].mean()),
                       'trajectory_best_query':float(fit['trajectory_best_query'].mean()),
                       'plateau':int(fit['converged'].sum()),'fits':len(fit['masks'])})
    write_json(folder/'tuning.json',{'seed':seed,'scores':scores,'uses_test':False})
    print(json.dumps({'seed':seed,'stage':'dense_tune','best_terminal':min(scores,key=lambda x:x['terminal_query']),
                      'best_query':min(scores,key=lambda x:x['trajectory_best_query'])}),flush=True)


def freeze(root):
    vals=[json.loads((root/f'seed_{s}'/'dense_control/tuning.json').read_text()) for s in (8100,8102)]
    scores=[]
    for j,(lr,l2) in enumerate(GRID):
        rows=[v['scores'][j] for v in vals];assert all((r['lr'],r['l2'])==(lr,l2) for r in rows)
        scores.append({'lr':lr,'l2':l2,'terminal_query':float(np.mean([r['terminal_query'] for r in rows])),
                       'trajectory_best_query':float(np.mean([r['trajectory_best_query'] for r in rows]))})
    selection={'dense_tuned_support':min(scores,key=lambda x:x['terminal_query']),
               'dense_tuned_query':min(scores,key=lambda x:x['trajectory_best_query'])}
    write_json(root/'dense_control_selection.json',{'selection':selection,'scores':scores,'uses_test_for_selection':False,
                'disclosure':'Additional dense-strength control after first primary test scores were inspected; tuning uses meta-validation only.',
                'tuning_hashes':{str(s):_file_sha256(root/f'seed_{s}'/'dense_control/tuning.json') for s in (8100,8102)}})


def evaluate(root,seed,device):
    torch.set_num_threads(1);folder=root/f'seed_{seed}'/'dense_control'
    selected=json.loads((root/'dense_control_selection.json').read_text())['selection']
    configs=list(dict.fromkeys((s['lr'],s['l2']) for s in selected.values()))
    bank=torch.load(OLD/f'seed_{seed}/bank/bank.pt',map_location='cpu',weights_only=False)
    eps=episodes(bank,seed,'test');fits=batch(eps,configs,device);rows=[]
    for fit in fits:
        _atomic_save(fit,folder/f"test_lr{fit['lr']}_l2{fit['l2']}_frozen.pt")
    write_json(folder/'frozen_before_test.json',{'selection_sha256':_file_sha256(root/'dense_control_selection.json'),
                'fit_hashes':{p.name:_file_sha256(p) for p in folder.glob('test_*_frozen.pt')},'test_labels_materialized':False})
    for method,config in selected.items():
        fit=next(f for f in fits if f['lr']==config['lr'] and f['l2']==config['l2'])
        key='best_params' if method=='dense_tuned_support' else 'trajectory_best_params'
        steps=fit['best_steps'] if method=='dense_tuned_support' else fit['trajectory_best_steps']
        for ep in eps:
            ids=[i for i,s in enumerate(fit['specs']) if s['task_id']==ep['task_id']]
            pool=build_test_pool(ep['task_id'].split(':')[-1])
            assert set(pool['ids'].tolist()).isdisjoint(ep['support']['ids'].tolist())
            assert set(pool['ids'].tolist()).isdisjoint(ep['query']['ids'].tolist())
            params={k:v[ids].to(device) for k,v in fit[key].items()}
            metrics=score_children_batched(params,torch.ones(len(ids),11,8),pool,device=device)
            for i,metric in zip(ids,metrics):
                rows.append({'seed':seed,'split':'test','method':method,**fit['specs'][i],**metric,
                             'best_step':int(steps[i]),'converged':bool(fit['converged'][i]),
                             'stopping_step':1400,'mask':torch.ones(11,8,dtype=torch.int).tolist(),
                             'support_ids':ep['support']['ids'].tolist(),'query_ids':ep['query']['ids'].tolist(),
                             'score_ids':pool['ids'].tolist()})
    write_json(folder/'records.json',rows)
    print(json.dumps({'seed':seed,'stage':'dense_control_evaluated',
                      'test_mean':{m:float(np.mean([r['balanced_bce'] for r in rows if r['method']==m])) for m in selected}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('tune','freeze','evaluate'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int);p.add_argument('--device',default='cpu');a=p.parse_args()
    if a.stage=='freeze':freeze(a.root)
    else:(tune if a.stage=='tune' else evaluate)(a.root,a.seed,a.device)

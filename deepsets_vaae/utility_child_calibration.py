"""Pre-test boundary expansion of the fixed-horizon target child solver."""
import argparse
import json
import time
import torch
import numpy as np

from .core import load_data
from .utility_graph_context import ROOT,task_sets
from .utility_graph_models import exact_topk
from .utility_graph_child import fit_children
from .rebuilt_bank_run import OUT,costs
from .run import write_json


def extend():
    device=torch.device('cuda:0');torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    data=load_data(ROOT/'datasets/mnist8m',4100,device)
    sets=task_sets(data,costs()['validation'],4100+820000)
    context=torch.load(OUT/'seed_4100/functional_context.pt',map_location=device,weights_only=False)
    base=torch.cat([exact_topk(context['mean_score'][None],7526),
                    exact_topk(context['source_mean_scores'][:1],7526),torch.ones((1,784,32),device=device)])
    out=OUT/'new_child_pilot';records=json.loads((out/'grid_extended.json').read_text())
    start=time.monotonic();start_index=max(r['grid'] for r in records)+1
    for j,(lr,l2) in enumerate([(.03,.3),(.03,1.),(.1,.1),(.1,.3),(.1,1.)],start=start_index):
        def cb(row):
            if row['step']%400==0:print(json.dumps(dict(grid=j,lr=lr,l2=l2,step=row['step'],query=float(row['queryNMSE'].mean()),elapsed=time.monotonic()-start)),flush=True)
        result=fit_children(base.repeat_interleave(2,0),torch.stack([r['x'] for r in sets]),
                    torch.stack([r['y'] for r in sets]),torch.stack([r['qx'] for r in sets]),
                    torch.stack([r['qy'] for r in sets]),[4100+830000+i*3001 for i in range(2)],
                    [0,1]*3,steps=2000,lr=lr,l2=l2,device=device,chunk_size=256,checkpoint_every=100,checkpoint_callback=cb)
        torch.save(result,out/f'grid_{j}.pt');q=result['query_loss'].reshape(2,3,2)
        for k,name in enumerate(['functional','source_functional','dense']):
            records.append(dict(grid=j,method=name,lr=lr,l2=l2,query=float(q[:,k].mean()),
                                all_plateau=bool(result['plateau_flags'][:,2*k:2*k+2].all())))
        write_json(out/'grid_extended.json',records)
    dense=min([r for r in records if r['method']=='dense'],key=lambda r:r['query'])
    sparse=[]
    for j in sorted({r['grid'] for r in records}):
        rows=[r for r in records if r['grid']==j and r['method']!='dense']
        sparse.append(dict(grid=j,lr=rows[0]['lr'],l2=rows[0]['l2'],query=np.mean([r['query'] for r in rows]),
                           all_plateau=all(r['all_plateau'] for r in rows)))
    selection=dict(mode='new_child',steps=2000,lr_decay_every=200,sparse=min(sparse,key=lambda r:r['query']),dense=dense,
                   selection='terminal meta-query; boundary expansions before target test; convergence reported separately',
                   cost_tasks='two held-out meta-validation cost vectors; bankseed4100 only',grid_total=len(sparse))
    write_json(out/'selection.json',selection);print(selection,flush=True)
    (out/'BOUNDARY_COMPLETE').write_text('extended boundary control completed\n')


if __name__=='__main__':extend()

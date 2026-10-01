"""Rebuild source solutions with stronger training and an auditable bank."""
from __future__ import annotations

import argparse
import json
import time
import os
from pathlib import Path

import numpy as np
import torch

from .core import _sets, centred_costs, load_data
from .run import write_json
from .utility_graph_child import fit_children
from .utility_graph_context import BANK, ROOT, task_sets
from .utility_graph_models import exact_topk
from .adaptive_data import _original_pools, _read_new, _provenance
from .rebuilt_bank_population import fit_population

OUT = ROOT / 'outputs/deepsets_vaae/20261001_rebuilt_functional_bank'
RHOS = (.1,.3,.5,.7,.9)


def new_audit_data(seed, device):
    """Reserve different parts inside the eight available MNIST8m blocks."""
    _, rows, hashes = _original_pools(seed,device)
    images=np.load(ROOT/'datasets/mnist8m/images.npy',mmap_mode='r')
    labels=np.load(ROOT/'datasets/mnist8m/labels.npy',mmap_mode='r')
    ignored={}; dup={}
    for name,block,count,part in (('old_selection_train',3,1000,0),('old_selection_checkpoint',4,300,0),
                                ('old_selection_score',5,300,0),('old_confirmation_train',6,1000,0),
                                ('old_confirmation_query',7,300,0),('old_confirmation_test',7,300,1)):
        _read_new(images,labels,rows,hashes,seed=seed,device=device,name=name,block=block,
                  per_digit=count,part=part,destination=ignored,duplicates=dup)
    splits={}
    _read_new(images,labels,rows,hashes,seed=seed,device=device,name='audit',block=5,
              per_digit=300,part=2,destination=splits,duplicates=dup)
    meta=_provenance(splits,stage='rebuilt source audit',identity_limit='raw rows and exact pixels only')
    meta.update(block=5,part=2,exact_pixel_duplicates_excluded=dup)
    return splits['audit'],meta


@torch.no_grad()
def evaluate_state(state,x,y,device,chunk=128):
    total=None
    w=(state['weight']*state['masks']).to(device)
    b=state['bias'].to(device); a=state['readout'].to(device); o=state['per_image_offset'].to(device)
    for start in range(0,len(x),chunk):
        h=torch.tanh(torch.einsum('bsi,mih->mbsh',x[start:start+chunk],w)+b[:,None,None,:])
        pred=((h*a[:,None,None,:]).sum(-1)+o[:,None,None]).sum(-1)
        value=(pred-y[None,start:start+chunk]).square().sum(-1)
        total=value if total is None else total+value
    return (total/len(x)/5).cpu()


@torch.no_grad()
def dense_function_score(state,probe,device):
    w=(state['weight']*state['masks']).to(device); a=state['readout'].to(device)
    h=torch.tanh(torch.einsum('pf,mfh->mph',probe,w)+state['bias'].to(device)[:,None,:])
    gain=((1-h.square())*a[:,None,:]).abs()
    q=w.abs()*torch.einsum('pf,mph->mfh',probe,gain)/len(probe)
    return q/q.flatten(1).amax(1).clamp_min(1e-8)[:,None,None]


def worker(seed,device):
    from .rebuilt_bank_functional import extract_full_functional
    out=OUT/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    spec=json.loads((OUT/'protocol.json').read_text())
    selected=json.loads((OUT/'population_solver_selection.json').read_text())
    data=load_data(ROOT/'datasets/mnist8m',seed,device)
    write_json(out/'data_provenance.json',{'split_hashes':data['split_hashes'],'row_ids_pairwise_disjoint':data['row_ids_pairwise_disjoint']})
    audit,audit_meta=new_audit_data(seed,device);write_json(out/'audit_provenance.json',audit_meta)
    probe_gen=torch.Generator(device=device).manual_seed(seed+870001)
    probe_rows=torch.randperm(len(data['source_train'].features),generator=probe_gen,device=device)[:128]
    probe=data['source_train'].features[probe_rows];started=time.monotonic()
    paths=[]
    for task,cost in enumerate(spec['task_vectors']['source']):
        bp=out/f'bank_{task}.pt'; dp=out/f'dense_{task}.pt'
        if bp.exists():paths.append(bp);continue
        sets=task_sets(data,[cost],seed+880000+3001*task,train_name='source_train',query_name='source_validation',train_count=2048,query_count=512)[0]
        c=centred_costs(cost,device)
        gen=torch.Generator(device=device).manual_seed(seed+890000+3001*task)
        def sampler(step):
            xx,yy=_sets(data['source_train'],c,128,5,gen);return xx[None],yy[None]
        ax,ay=_sets(audit,c,2048,5,torch.Generator(device=device).manual_seed(seed+900000+task))
        init_seed=seed+910000+3001*task
        def fit(name,masks,replicas,setting):
            progress(out,stage=name,task=task,elapsed=time.monotonic()-started)
            checkpoint_path=out/f'{name}_{task}_progress.pt'
            checkpoint=None
            if checkpoint_path.exists():
                checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
            def cb(row):
                progress(out,stage=name,task=task,step=row['step'],train=float(row['trainNMSE'].mean()),
                         query=float(row['queryNMSE'].mean()),plateau=int((~row['active']).sum()),
                         total=len(masks),elapsed=time.monotonic()-started)
                if row['step']%500==0:torch.save(row,checkpoint_path)
            result=fit_population(masks,data['source_train'].features,c[data['source_train'].digits],
                                  sets['qx'],sets['qy'],init_seed,replicas,lr=setting['lr'],l2=setting['l2'],
                                  cap=spec['population_cap'],minimum=spec['population_minimum'],device=device,
                                  checkpoint_every=50,lr_decay_every=spec['population_decay_every'],
                                  checkpoint_callback=cb,resume=checkpoint)
            # Slow stochastic fluctuations are not silently called convergence.
            for extension in range(3):
                if result['plateau_flags'].all():break
                progress(out,stage=name+'_support_polish',task=task,extension=extension,
                         plateau=int(result['plateau_flags'].sum()),total=len(masks))
                result=fit_population(masks,data['source_train'].features,c[data['source_train'].digits],
                                      sets['qx'],sets['qy'],init_seed,replicas,lr=setting['lr'],l2=setting['l2'],
                                      cap=result['terminal_step']+1200,minimum=spec['population_minimum'],device=device,
                                      checkpoint_every=50,lr_decay_every=spec['population_decay_every'],
                                      checkpoint_callback=cb,resume=result)
            state=result['state_dict']
            result['audit_nmse']=evaluate_state(state,ax,ay,device)
            result['source_state_dict']=state
            return result
        if dp.exists():dense=torch.load(dp,map_location=device,weights_only=False)
        else:
            dense=fit('dense',torch.ones((8,784,32),device=device),list(range(8)),selected['dense'])
            torch.save(dense,dp)
        scores=dense_function_score(dense['source_state_dict'],probe,device)
        generator=torch.Generator(device=device).manual_seed(seed+920000+task)
        masks=[];recipes=[];replicas=[]
        for rho in RHOS:
            for recipe in ('random','dense_functional'):
                for variant in range(4):
                    for replica in range(8):
                        if recipe=='random':score=torch.rand((1,784,32),device=device,generator=generator)
                        else:
                            base=scores[replica:replica+1]
                            noise=torch.randn(base.shape,device=device,generator=generator)
                            score=base+(.0,.2,.5,1.0)[variant]*base.std()*noise
                        masks.append(exact_topk(score,round(25088*rho))[0]);replicas.append(replica)
                        recipes.append(dict(rho=rho,recipe=recipe,variant=variant,replica=replica))
        sparse=fit('sparse',torch.stack(masks),replicas,selected['sparse'])
        sparse.update(candidate_recipe=recipes,source_task=task,retention='all solutions, no quality threshold',
                      initialization_seed=init_seed,train_sets_monitor=2048,query_sets=512,audit_sets=2048,
                      settings=selected['sparse'],state_dict=sparse.pop('source_state_dict'))
        sparse['paired_dense_audit_nmse']=dense['audit_nmse'][torch.tensor(replicas)]
        sparse['paired_difference']=sparse['audit_nmse']-sparse['paired_dense_audit_nmse']
        torch.save(sparse,bp);paths.append(bp)
        for name in ('dense','sparse'):
            temp=out/f'{name}_{task}_progress.pt'
            if temp.exists():temp.unlink()
    progress(out,stage='functional_profiles',elapsed=time.monotonic()-started)
    extract_full_functional(paths,probe,data['source_train'].source_ids[probe_rows],out,seed,device)
    write_json(out/'timing.json',dict(seconds=time.monotonic()-started,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
    (out/'COMPLETE').write_text('source banks and full functional profiles completed\n')
    progress(out,stage='complete',elapsed=time.monotonic()-started)


def progress(out, **kw):
    write_json(Path(out) / 'status.json', kw)
    print(json.dumps(kw), flush=True)


def costs():
    return json.loads((BANK / 'protocol.json').read_text())['task_vectors']


def masks_for_pilot(device, multi=False):
    p = torch.load(BANK / 'seed_4100/masks.pt', map_location=device, weights_only=True)
    g = torch.Generator(device=device).manual_seed(202610013)
    score = torch.rand((1, 784, 32), device=device, generator=g)
    flat = score.flatten(1); rand = torch.zeros_like(flat).scatter_(1, flat.topk(7526, dim=1).indices, 1).reshape_as(score)
    if not multi:
        return torch.cat((p['functional_mean_large'][:1], rand, torch.ones_like(rand))), ['functional', 'random', 'dense']
    with np.load(BANK / 'seed_4100/functional/functional_vae_arrays.npz') as arrays:
        prior = torch.from_numpy(np.asarray(arrays['function_train_aligned']).copy()).to(device).mean((0,1))[None]
    masks=[]; names=[]
    for rho in (.1,.3,.5,.7,.9):
        masks.extend((exact_topk(prior,round(25088*rho)),exact_topk(score,round(25088*rho))))
        names.extend((f'functional_{rho}',f'random_{rho}'))
    masks.append(torch.ones_like(rand));names.append('dense')
    return torch.cat(masks),names


def pilot(mode, device):
    out = OUT / f'{mode}_pilot'; out.mkdir(parents=True, exist_ok=True)
    data = load_data(ROOT / 'datasets/mnist8m', 4100, device)
    fresh_mode=mode in ('bank_v2','bank_v3')
    pilot_costs=costs()['validation'] if mode=='bank_v3' else costs()['source'][:2]
    if mode in ('bank','bank_v2','bank_v3'):
        sets = task_sets(data, pilot_costs, 4100+810000,
                         train_name='source_train', query_name='source_validation',
                         train_count=2048, query_count=512)
        steps = 8000 if fresh_mode else 2400; batch_size = 128
    elif mode=='new_child':
        data=load_data(ROOT/'datasets/mnist8m',4100,device)
        sets=task_sets(data,costs()['validation'],4100+820000)
        steps=2000;batch_size=None
    else:
        sets = task_sets(data, costs()['validation'], 4100+820000)
        steps = 1600; batch_size = None
    if mode=='new_child':
        context=torch.load(OUT/'seed_4100/functional_context.pt',map_location=device,weights_only=False)
        mean=context['mean_score'][None]
        scores=context['source_mean_scores'][0:1]
        base=torch.cat((exact_topk(mean,7526),exact_topk(scores,7526),torch.ones_like(mean)))
        names=['functional','source_functional','dense']
    else:base, names = masks_for_pilot(device,multi=fresh_mode)
    grid = [(lr, l2) for lr in ((.002,.005,.01) if mode=='new_child' else (.002, .005)) for l2 in ((0., .0001, .001) if fresh_mode else (.0001, .001, .01))]
    records = []; started = time.monotonic()
    for j, (lr, l2) in enumerate(grid):
        masks = base.repeat_interleave(2, 0)
        sampler=None
        if fresh_mode:
            generators=[torch.Generator(device=device).manual_seed(202610010+i) for i in range(2)]
            task_costs=[centred_costs(c,device) for c in pilot_costs]
            def sampler(step):
                batches=[_sets(data['source_train'],c,128,5,g) for c,g in zip(task_costs,generators)]
                return torch.stack([r[0] for r in batches]),torch.stack([r[1] for r in batches])
        progress(out, stage='solver_grid', mode=mode, index=j, settings=[lr,l2], elapsed=time.monotonic()-started)
        def cb(row):
            progress(out, stage='solver_grid', mode=mode, index=j, step=row['step'],
                     train=float(row['trainNMSE'].mean()), query=float(row['queryNMSE'].mean()),
                     plateau=int(row['plateau_flags'].sum()), elapsed=time.monotonic()-started)
        result = fit_children(masks, torch.stack([r['x'] for r in sets]), torch.stack([r['y'] for r in sets]),
                              torch.stack([r['qx'] for r in sets]), torch.stack([r['qy'] for r in sets]),
                              [4100+830000+i*3001 for i in range(len(sets))], [0,1]*len(names),
                              steps=steps, lr=lr, l2=l2, device=device, chunk_size=256,
                              checkpoint_every=100, checkpoint_callback=cb,
                              batch_size=batch_size, lr_decay_every=2000 if fresh_mode else (400 if mode=='bank' else 200),
                              support_sampler=sampler)
        torch.save(result, out / f'grid_{j}.pt')
        q = result['query_loss'].reshape(2,len(names),2)
        for k,name in enumerate(names):
            records.append(dict(grid=j,method=name,lr=lr,l2=l2,query=float(q[:,k].mean()),
                                all_plateau=bool(result['plateau_flags'][:,2*k:2*k+2].all())))
        write_json(out / 'grid.json', records)
    # Dense gets its own tuning; shared sparse solver chosen across the two priors.
    valid_dense = [r for r in records if r['method']=='dense']
    sparse_grid = [dict(grid=j,lr=grid[j][0],l2=grid[j][1],
                        query=np.mean([r['query'] for r in records if r['grid']==j and r['method']!='dense']),
                        all_plateau=all(r['all_plateau'] for r in records if r['grid']==j and r['method']!='dense')) for j in range(len(grid))]
    valid_sparse = sparse_grid
    if not valid_dense or not valid_sparse:
        write_json(out/'NOT_CONVERGED.json', dict(dense=valid_dense,sparse=valid_sparse))
        raise RuntimeError('No plateau-certified solver; inspect before bank/model training')
    selection = dict(mode=mode,steps=steps,batch_size=batch_size,lr_decay_every=2000 if fresh_mode else (400 if mode=='bank' else 200),
                     sparse=min(valid_sparse,key=lambda r:r['query']), dense=min(valid_dense,key=lambda r:r['query']),
                     selection='terminal query loss; plateau reported separately, no plateau-based quality filtering',
                     cost_tasks='two held-out meta-validation cost vectors' if mode=='bank_v3' else mode,
                     sets=[dict(x=r['x'].cpu(),y=r['y'].cpu(),qx=r['qx'].cpu(),qy=r['qy'].cpu()) for r in sets])
    torch.save(selection, out / 'selection.pt')
    write_json(out/'selection.json', {k:v for k,v in selection.items() if k!='sets'})
    (out/'COMPLETE').write_text('solver pilot completed\n')
    progress(out, stage='complete', elapsed=time.monotonic()-started)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--pilot', choices=['bank','bank_v2','bank_v3','child','new_child'])
    parser.add_argument('--seed',type=int)
    args=parser.parse_args(); torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.pilot:pilot(args.pilot,device)
    elif args.seed is not None:worker(args.seed,device)
    else:parser.error('supply --pilot or --seed')


if __name__=='__main__': main()

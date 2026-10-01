"""Paired full-batch L2 child validation and frozen held-out scoring.

All methods share initialization, support sample, schedule and tuning grid.
This is a new regularized protocol, not a replacement for the old scores.
"""
from pathlib import Path
import argparse
import json
import itertools
import torch
import torch.nn.functional as F
import numpy as np
from .core import build_experiment_data,build_task_support_query,build_test_pool
from .generator import Generator
from .functional_pool import FunctionalPool
from .repaired_generator import make_model
from .repaired_run import SEEDS
from .stripe_debug import OLD
from .meta import _init_children,child_logits_batch,_atomic_save,_file_sha256
from .eval_run import _draw_support,_stable_seed,_candidate_init_seed
from .evaluate import functional_centroid_mean_mask,score_children_batched
from .convergence import loss_plateau

METHODS=('legacy','legacy_fixedmass','column_set_fixedmass','functional_set_pool_bounded','uniform_functional','dense')
RATES=(.001,.003,.01)
L2=(.0001,.001,.01)
BUDGET=128
MAX_STEPS=32000


def load_models(root,seed,device):
    bank=torch.load(OLD/f'seed_{seed}/bank/bank.pt',map_location='cpu',weights_only=False)
    models={};hashes={}
    for method in METHODS:
        if method in ('uniform_functional','dense'):continue
        if method=='bank_evolution_fixed':
            from .bank_evolution import FixedProposal
            path=root/f'seed_{seed}/bank_evolution_fixed/best.pt';m=FixedProposal(torch.zeros(11,8))
        elif method=='legacy':
            path=OLD/f'seed_{seed}/transformer_mask/meta/best.pt';m=Generator()
        else:
            path=root/f'seed_{seed}'/method/'best.pt'
            m=FunctionalPool(bank,bounded_weights=True) if method=='functional_set_pool_bounded' else make_model(method)
        cp=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(cp['model_state']);m=m.to(device).eval()
        models[method]=m;hashes[method]=_file_sha256(path)
    return bank,models,hashes


def prepare(root,seed,stage,selection=None,device='cpu'):
    bank,models,hashes=load_models(root,seed,device)
    data=build_experiment_data(probe_seed=seed)
    uniform=torch.as_tensor(functional_centroid_mean_mask(bank['edge_q'])['mask']).float()
    specs=[];masks=[];xs=[];ys=[];qx=[];qy=[];conditions={}
    for task in data['splits']['val' if stage=='tune' else 'test']:
        tid=task.task_id
        pools=build_task_support_query(task,bank['probe_ids'])
        tag='val-support' if stage=='tune' else 'test-support'
        sample=_draw_support(pools,_stable_seed(tag,seed,tid,BUDGET),BUDGET)
        assert set(sample['ids'].tolist()).isdisjoint(pools['query']['ids'].tolist())
        conditions[tid]={'support_ids':sample['ids'].tolist(),'query_ids':pools['query']['ids'].tolist()}
        with torch.no_grad():
            chosen={method:m(bank['feature'].to(device),sample['x'].to(device),sample['y'].to(device))[0][0].cpu() for method,m in models.items()}
        chosen['uniform_functional']=uniform;chosen['dense']=torch.ones(11,8)
        for method in METHODS:
            rates=RATES if stage=='tune' else (selection[method]['lr'],)
            strengths=L2 if stage=='tune' else (selection[method]['l2'],)
            for lr,l2,init in itertools.product(rates,strengths,range(4)):
                specs.append({'method':method,'task_id':tid,'seed':seed,'init_id':init,'lr':lr,'l2':l2,'budget':BUDGET})
                masks.append(chosen[method]);xs.append(sample['x']);ys.append(sample['y']);qx.append(pools['query']['x']);qy.append(pools['query']['y'])
    payload={'specs':specs,'masks':torch.stack(masks),'x':torch.stack(xs),'y':torch.stack(ys),
             'qx':torch.stack(qx),'qy':torch.stack(qy),'conditions':conditions,'model_hashes':hashes,
             'bank_sha256':_file_sha256(OLD/f'seed_{seed}/bank/bank.pt'),
             'test_labels_materialized':False}
    folder=root/f'seed_{seed}'/('evolution_eval' if METHODS==('bank_evolution_fixed',) else 'regularized_eval');folder.mkdir(parents=True,exist_ok=True)
    manifest_path=folder/f'{stage}_manifest.pt'
    if manifest_path.exists():
        old=torch.load(manifest_path,map_location='cpu',weights_only=False)
        for k,v in payload.items():
            assert torch.equal(old[k],v) if isinstance(v,torch.Tensor) else old[k]==v, ('Manifest changed',k)
    else:_atomic_save(payload,manifest_path)
    return payload


def balanced(logits,y):
    losses=F.binary_cross_entropy_with_logits(logits,y,reduction='none')
    pos=y>.5;neg=~pos
    return .5*((losses*pos).sum(1)/pos.sum(1).clamp_min(1)+(losses*neg).sum(1)/neg.sum(1).clamp_min(1))


def fit(manifest,folder,stage,device,cap=MAX_STEPS):
    specs=manifest['specs'];count=len(specs)
    masks=manifest['masks'].to(device);x=manifest['x'].to(device);y=manifest['y'].to(device)
    qx=manifest['qx'].to(device);qy=manifest['qy'].to(device)
    rates=torch.tensor([s['lr'] for s in specs],device=device)
    reg=torch.tensor([s['l2'] for s in specs],device=device)
    seeds=[_candidate_init_seed(s['task_id'],BUDGET,s['init_id']) for s in specs]
    params=_init_children(seeds,torch.device(device))
    opt=torch.optim.Adam(params.values(),lr=1.)
    best=torch.full((count,),float('inf'),device=device);best_steps=torch.zeros(count,dtype=torch.int64,device=device)
    best_params={k:v.detach().clone() for k,v in params.items()}
    active=torch.ones(count,dtype=torch.bool,device=device);passes=torch.zeros(count,dtype=torch.int64,device=device)
    stopping=torch.full((count,),cap,dtype=torch.int64,device=device)
    history=[];step=0
    path=folder/f'{stage}_fit.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['specs']==specs
        for k in params:params[k].data.copy_(saved['last_params'][k].to(device))
        opt.load_state_dict(saved['optimizer_state']);best=saved['best_query'].to(device);best_steps=saved['best_steps'].to(device)
        best_params={k:v.to(device) for k,v in saved['best_params'].items()}
        active=saved['active'].to(device);passes=saved['passes'].to(device);stopping=saved['stopping_steps'].to(device)
        history=saved['history'];step=saved['step']
        stopping[active]=cap
    def regularizer():
        return .5*reg*((params['w']*masks).square().sum((1,2))+params['b'].square().sum(1)+params['a'].square().sum(1)+params['c'].square())
    while step<cap and bool(active.any()):
        opt.zero_grad(set_to_none=True)
        logit=child_logits_batch(x,masks,params)
        objectives=balanced(logit,y)+regularizer()
        (objectives*active).sum().backward()
        if not all(torch.isfinite(v.grad).all() for v in params.values()):raise RuntimeError('Nonfinite child gradient')
        with torch.no_grad():
            gradients=torch.sqrt(sum(v.grad.square().reshape(count,-1).sum(1) for v in params.values()))
        # Adam moment updates are independent per child; scaled per-run lr is
        # applied to its parameter displacement. Paired models share seeds.
        before={k:v.detach().clone() for k,v in params.items()}
        opt.step()
        lr=rates*(.5**(step//1000))
        floor_factor=64 if step<16000 else min(4096,64*2**(1+(step-16000)//1000))
        lr=torch.maximum(lr,rates/floor_factor)*active
        with torch.no_grad():
            for k,v in params.items():
                shape=(count,)+(1,)*(v.ndim-1)
                v.copy_(before[k]+(v-before[k])*lr.reshape(shape))
        step+=1
        if step%50:continue
        with torch.no_grad():
            train=balanced(child_logits_batch(x,masks,params),y)
            objective=train+regularizer()
            query=balanced(child_logits_batch(qx,masks,params),qy)
            improved=(query<best)&active
            best[improved]=query[improved];best_steps[improved]=step
            for k in params:best_params[k][improved]=params[k][improved]
            history.append({'step':step,'support_bce':train.cpu(),'objective':objective.cpu(),'query_bce':query.cpu(),
                            'gradient_norm':gradients.cpu(),'active':active.cpu().clone()})
            if step>=2000:
                objflat=loss_plateau(torch.stack([r['objective'] for r in history]),width=8,tolerance=.001).to(device)
                qflat=loss_plateau(torch.stack([r['query_bce'] for r in history]),width=8,tolerance=.001).to(device)
                trainflat=loss_plateau(torch.stack([r['support_bce'] for r in history]),width=8,tolerance=.001).to(device)
                passes=torch.where(objflat&qflat&trainflat,passes+1,torch.zeros_like(passes))
                done=active&(passes>=3);stopping[done]=step;active[done]=False
        if step%500==0 or not bool(active.any()) or step==cap:
            payload={'specs':specs,'masks':manifest['masks'],'best_params':{k:v.cpu() for k,v in best_params.items()},
                     'last_params':{k:v.detach().cpu() for k,v in params.items()},
                     'optimizer_state':opt.state_dict(),'best_query':best.cpu(),'best_steps':best_steps.cpu(),
                     'active':active.cpu(),'passes':passes.cpu(),'stopping_steps':stopping.cpu(),'history':history,'step':step,
                     'full_batch':True,'step_cap':cap,'lr_continuation':'after16000, floor base/64 halves every1000 to base/4096','regularization':'0.5*l2*(||Weff||^2+||b||^2+||a||^2+c^2)',
                     'convergence':'support BCE, regularized objective and query BCE: adjacent8 windows and slope <=.1%, 3 passes, min2000',
                     'selected_on':'query','frozen_before_test':True,'test_labels_used_for_selection':False}
            _atomic_save(payload,path)
            print(json.dumps({'stage':stage,'step':step,'remaining':int(active.sum()),'runs':count,'mean_best_query':float(best.mean())}),flush=True)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert saved['step']==step
    return saved


def tune(root,seed,device):
    m=prepare(root,seed,'tune',device=device);folder=root/f'seed_{seed}'/('evolution_eval' if METHODS==('bank_evolution_fixed',) else 'regularized_eval')
    r=fit(m,folder,'tune',device)
    scores={}
    for method in METHODS:
        scores[method]={}
        for lr,l2 in itertools.product(RATES,L2):
            ids=[i for i,s in enumerate(m['specs']) if s['method']==method and s['lr']==lr and s['l2']==l2]
            assert len(ids)==8
            scores[method][f'{lr:g}|{l2:g}']=float(r['best_query'][ids].mean())
    payload={'seed':seed,'scores':scores,'used_test':False,'tuning_sha256':_file_sha256(folder/'tune_fit.pt'),
             'candidate_count':len(m['specs']),'plateau_count':int((~r['active']).sum()),'cap_count':int(r['active'].sum())}
    (folder/'tuning.json').write_text(json.dumps(payload,indent=2)+'\n')


def freeze(root):
    data=[json.loads((root/f'seed_{s}/regularized_eval/tuning.json').read_text()) for s in SEEDS]
    assert all(not d['used_test'] for d in data)
    extras=[json.loads((root/f'seed_{seed}/evolution_eval/tuning.json').read_text()) for seed in SEEDS]
    assert all(not d['used_test'] for d in extras)
    for d,e in zip(data,extras):
        assert d['seed']==e['seed']
        d['scores'].update(e['scores'])
    selected={}
    for method in METHODS+('bank_evolution_fixed',):
        scores={key:float(np.mean([d['scores'][method][key] for d in data])) for key in data[0]['scores'][method]}
        key=min(scores,key=scores.get);lr,l2=map(float,key.split('|'))
        selected[method]={'lr':lr,'l2':l2,'validation_scores':scores}
    payload={'protocol':'regularized_child_v1','selection':selected,'used_test':False,'seeds':list(SEEDS),
             'tuning_provenance':{str(d['seed']):d['tuning_sha256'] for d in data},
             'evolution_tuning_provenance':{str(d['seed']):d['tuning_sha256'] for d in extras},'support_budget':128,
             'note':'Exploratory reuse of previously inspected test task split; no new test metrics used to select this run.'}
    (root/'regularized_selection.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


def test(root,seed,device):
    selection_path=root/'regularized_selection.json'
    selected=json.loads(selection_path.read_text())['selection']
    m=prepare(root,seed,'test',selected,device)
    folder=root/f'seed_{seed}'/('evolution_eval' if METHODS==('bank_evolution_fixed',) else 'regularized_eval');r=fit(m,folder,'test',device)
    # Freeze all masks, hyperparameters and child states BEFORE test labels.
    frozen={'selection_sha256':_file_sha256(selection_path),'child_sha256':_file_sha256(folder/'test_fit.pt'),
            'manifest_sha256':_file_sha256(folder/'test_manifest.pt'),'test_labels_materialized':False}
    (folder/'frozen_before_test.json').write_text(json.dumps(frozen,indent=2)+'\n')
    rows=[]
    for task_id in sorted({s['task_id'] for s in m['specs']}):
        pool=build_test_pool(task_id.split(':')[-1]);idx=[i for i,s in enumerate(m['specs']) if s['task_id']==task_id]
        assert set(pool['ids'].tolist()).isdisjoint(m['conditions'][task_id]['support_ids'])
        assert set(pool['ids'].tolist()).isdisjoint(m['conditions'][task_id]['query_ids'])
        p={k:v[idx].to(device) for k,v in r['best_params'].items()}
        metrics=score_children_batched(p,m['masks'][idx],pool,device=device)
        for i,score in zip(idx,metrics):
            rows.append({**m['specs'][i],**score,'converged':not bool(r['active'][i]),
                         'best_step':int(r['best_steps'][i]),'stopping_step':int(r['stopping_steps'][i]),
                         'mask':m['masks'][i].int().tolist(),'support_ids':m['conditions'][task_id]['support_ids'],
                         'test_ids':pool['ids'].tolist()})
    (folder/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps({'stage':'test','seed':seed,'records':len(rows),'converged':sum(r['converged'] for r in rows)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('tune','freeze','test'));p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int);p.add_argument('--device',default='cpu');p.add_argument('--evolution',action='store_true')
    a=p.parse_args();torch.set_num_threads(1)
    if a.evolution:METHODS=('bank_evolution_fixed',)
    if a.stage=='freeze':freeze(a.root)
    elif a.stage=='tune':tune(a.root,a.seed,a.device)
    else:test(a.root,a.seed,a.device)

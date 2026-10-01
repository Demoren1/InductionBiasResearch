"""Short source-utility distillation: deterministic GNN vs graph flow.

All proposal targets come from whole-mask, fresh-weight retraining. No
top-k surrogate is differentiated. Test metrics are a separate final stage.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .core import build_experiment_data, build_task_support_query, build_test_pool
from .stripe_debug import OLD
from .evaluate import functional_centroid_mean_mask, score_children_batched
from .meta import _atomic_save, _file_sha256
from .eval_run import _draw_support, _stable_seed, _candidate_init_seed
from .convergence import loss_plateau
from .graph_flow_child import fit_short
from .graph_flow_models import GraphVelocity, score_to_mask, flow_matching_coupling, sample_flow_endpoint

METHODS = ('gnn_direct','gnn_flow_single','gnn_search8','gnn_flow_search8',
           'functional_search8','uniform_functional','dense')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def bank_context(bank):
    """Explicit centroid prior; retain full functional features before pooling."""
    functional = functional_centroid_mean_mask(bank['edge_q'])
    order = torch.as_tensor(functional['column_order'])
    tokens = bank['feature'].gather(1, order[:,:,None].expand(-1,-1,187)).clone()
    # Source banks were query-curated, but do not feed teacher quality labels
    # as a "functional" feature of this comparison.
    tokens[:,:,-1] = 0
    node = torch.cat((tokens.mean(0),tokens.std(0,unbiased=False)), -1)
    raw = torch.stack([bank[k] for k in ('q_signed','q_abs','q_variance')],-1)
    raw = raw.gather(2,order[:,None,:,None].expand(-1,11,-1,3))
    occupancy = bank['masks'].gather(2,order[:,None,:].expand(-1,11,-1)).mean(0)
    edge = torch.cat((raw.mean(0),raw.std(0,unbiased=False),occupancy[...,None]),-1)
    # Featurewise scale over vertices/edges preserves hidden relabeling.
    node = (node-node.mean(0))/node.std(0,unbiased=False).clamp_min(.1)
    edge = (edge-edge.mean((0,1)))/edge.std((0,1),unbiased=False).clamp_min(.001)
    aligned_abs = raw[...,1]
    return {'node':node.float(),'edge':edge.float(),
            'uniform':torch.tensor(functional['mask']).float(),
            'scores':torch.tensor(functional['scores']).float(),
            'aligned_abs':aligned_abs,'order':order}


def task_context(support):
    x,y = support['x'],support['y']
    contrast = x[y>.5].mean(0)-x[y<=.5].mean(0)
    return torch.cat((x.mean(0),contrast,y.mean()[None],torch.ones(1))).float()


def episodes(bank,seed,split):
    data=build_experiment_data(probe_seed=seed)
    result=[]
    for task in data['splits'][split]:
        pools=build_task_support_query(task,bank['probe_ids'])
        support=_draw_support(pools,_stable_seed('graph-support',seed,task.task_id),128)
        result.append({'task_id':task.task_id,'support':support,'query':pools['query'],
                       'context':task_context(support)})
    return result


def evaluate_candidates(candidates, eps, solver, device, replicas=2, offset=0):
    """Candidates [tasks,population,11,8]; all candidates share child RNG."""
    tasks,pop = candidates.shape[:2]
    masks=candidates[:,:,None].expand(-1,-1,replicas,-1,-1).reshape(-1,11,8)
    x=[];y=[];qx=[];qy=[];seeds=[]
    for ep in eps:
        for _ in range(pop):
            for init in range(offset,offset+replicas):
                x.append(ep['support']['x']);y.append(ep['support']['y'])
                qx.append(ep['query']['x']);qy.append(ep['query']['y'])
                seeds.append(_candidate_init_seed(ep['task_id'],128,init))
    fit=fit_short(masks,torch.stack(x),torch.stack(y),torch.stack(qx),torch.stack(qy),
                  seeds,device=device,lr=solver['lr'],l2=solver['l2'],fixed_horizon=True)
    fit['task_ids']=[ep['task_id'] for ep in eps]
    fit['support_ids']={ep['task_id']:ep['support']['ids'] for ep in eps}
    fit['query_ids']={ep['task_id']:ep['query']['ids'] for ep in eps}
    fit['init_offset']=offset;fit['replicas']=replicas
    utility=fit['best_query'].reshape(tasks,pop,replicas).mean(-1)
    return utility,fit


def initial_candidates(ctx,tasks,rng):
    u=ctx['uniform'].numpy();pop=[]
    for _ in range(tasks):
        masks=[u.copy()]
        for index in rng.choice(len(ctx['aligned_abs']),6,replace=False):
            masks.append(score_to_mask(ctx['aligned_abs'][index]).numpy())
        active=np.flatnonzero(u.ravel());inactive=np.flatnonzero(1-u.ravel())
        for swaps in (1,2,3,4,6):
            m=u.copy().ravel();m[rng.choice(active,swaps,False)]=0;m[rng.choice(inactive,swaps,False)]=1
            masks.append(m.reshape(11,8))
        pop.append(np.stack(masks))
    return torch.tensor(np.stack(pop)).float()


def make_model():
    return GraphVelocity(374,7,24,width=32,layers=3)


def fit_model(kind, masks, utility, contexts, ctx, seed, stage, device, folder,
              initial_state=None, max_steps=600, base_lr=.002):
    """Same weighted elite archive, separate BCE and independent-coupling FM."""
    started=time.monotonic();torch.manual_seed(seed+30000+stage)
    model=make_model().to(device)
    if initial_state is not None:model.load_state_dict(initial_state)
    # Restart each stage: targets changed; prior stage losses are not comparable.
    best_ids=[]
    for task in range(len(masks)):
        seen=set();indices=[]
        for index in utility[task].argsort().tolist():
            key=masks[task,index].numpy().tobytes()
            if key not in seen:
                seen.add(key);indices.append(index)
            if len(indices)==4:break
        assert len(indices)==4, 'Need four distinct whole-mask elites per task'
        best_ids.append(indices)
    best_ids=torch.tensor(best_ids)
    elite=masks.gather(1,best_ids[:,:,None,None].expand(-1,-1,11,8))
    loss_values=utility.gather(1,best_ids)
    weights=(-(loss_values-loss_values.min(1,keepdim=True).values)/.03).softmax(1).flatten().to(device)
    weights=weights/weights.sum()
    target=elite.flatten(0,1).to(device)
    task=contexts[:,None,:].expand(-1,4,-1).reshape(-1,24).to(device)
    n=len(target);node=ctx['node'].to(device)[None].expand(n,-1,-1)
    edge=ctx['edge'].to(device)[None].expand(n,-1,-1,-1)
    generator=torch.Generator(device=device).manual_seed(seed+32000+stage)
    fixed_noise=torch.randn((n,11,8),generator=generator,device=device)
    fixed_time=torch.linspace(.02,.98,n,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=base_lr)
    def losses(noise=None,tm=None,augment=False):
        # One common hidden relabeling per update; no positional hidden IDs.
        perm=torch.randperm(8,device=device,generator=generator) if augment else torch.arange(8,device=device)
        tar=target[:,:,perm];nd=node[:,perm];ed=edge[:,:,perm]
        if kind=='gnn':
            pred=model(torch.zeros_like(tar),torch.zeros(n,device=device),nd,ed,task)
            per=F.binary_cross_entropy_with_logits(pred,tar,reduction='none').mean((1,2))
        else:
            if noise is not None:noise=noise[:,:,perm]
            state,t,vel=flow_matching_coupling(tar,noise=noise,time=tm,generator=generator)
            per=(model(state,t,nd,ed,task)-vel).square().mean((1,2))
        return (per*weights).sum()
    curves=[];best=float('inf');best_state=None;best_step=0;passes=0;reason='step_cap'
    for step in range(1,max_steps+1):
        opt.param_groups[0]['lr']=max(base_lr/8,base_lr*(.5**((step-1)//200)))
        model.train();opt.zero_grad(set_to_none=True)
        loss=losses(augment=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5)
        opt.step()
        if step%25:continue
        model.eval()
        with torch.no_grad():monitor=float(losses(fixed_noise,fixed_time))
        curves.append({'step':step,'stochastic_loss':float(loss),'fixed_monitor_loss':monitor})
        if monitor<best:
            best=monitor;best_step=step;best_state=copy.deepcopy(model.state_dict())
        if step>=300:
            flat=bool(loss_plateau(torch.tensor([r['fixed_monitor_loss'] for r in curves]),width=4,tolerance=.01))
            passes=passes+1 if flat else 0
            if passes>=3:reason='empirical_fixed_objective_plateau';break
        if time.monotonic()-started>=180:
            reason='wall_time_cap';break
    payload={'kind':kind,'seed':seed,'stage':stage,'model_state':{k:v.cpu() for k,v in best_state.items()},
             'last_model_state':{k:v.cpu() for k,v in model.state_dict().items()},
             'elite_masks':elite,'elite_query_bce':loss_values,'weights':weights.cpu(),
             'best_step':best_step,'best_loss':best,'steps':step,'stop_reason':reason,
             'elapsed_seconds':time.monotonic()-started,'curves':curves,
             'monitor':'same elite masks, independently fixed Gaussian coupling/time for FM; not unseen solutions',
             'uses_test':False,'uses_meta_validation_labels':False,
             'warm_start':initial_state is not None,'max_steps':max_steps,'base_lr':base_lr,
             'param_count':sum(p.numel() for p in model.parameters())}
    _atomic_save(payload,folder/f'{kind}_stage_{stage}.pt')
    write_json(folder/f'{kind}_stage_{stage}_curves.json',curves)
    model.load_state_dict(best_state);model.eval()
    print(json.dumps({k:payload[k] for k in ('seed','kind','stage','steps','stop_reason','elapsed_seconds','best_loss')}),flush=True)
    return model,payload


@torch.no_grad()
def model_candidates(models,ctx,contexts,seed,device,pop=8):
    count=len(contexts);node=ctx['node'].to(device)[None].expand(count,-1,-1)
    edge=ctx['edge'].to(device)[None].expand(count,-1,-1,-1)
    task=contexts.to(device);z=torch.zeros(count,11,8,device=device)
    scores=models['gnn'](z,torch.zeros(count,device=device),node,edge,task)
    rng=np.random.default_rng(seed)
    scale=scores.std((1,2),unbiased=False).clamp_min(.1)[:,None,None]
    variants=[scores]
    functional=[ctx['uniform'].to(device)[None].expand(count,-1,-1)]
    norm=ctx['scores'];norm=(norm-norm.mean())/norm.std().clamp_min(.001)
    for k in range(1,pop):
        noise=torch.tensor(rng.gumbel(size=(count,11,8)),device=device,dtype=torch.float32)
        variants.append(scores+(.10,.20,.35)[(k-1)%3]*scale*noise)
        functional.append(score_to_mask(norm.to(device)[None]+(.10,.20,.35)[(k-1)%3]*noise))
    gnn=score_to_mask(torch.stack(variants,1)).cpu()
    noise=torch.tensor(rng.normal(size=(count*pop,11,8)),device=device,dtype=torch.float32)
    flow_scores=sample_flow_endpoint(models['fm'],node.repeat_interleave(pop,0),
                                    edge.repeat_interleave(pop,0),task.repeat_interleave(pop,0),
                                    noise=noise,solver='euler',steps=12).reshape(count,pop,11,8)
    flow=score_to_mask(flow_scores).cpu()
    return {'gnn_search8':gnn,'gnn_flow_search8':flow,
            'functional_search8':torch.stack(functional,1).cpu()}, {'gnn':scores.cpu(),'fm':flow_scores.cpu()}


def train(root,seed,device):
    started=time.monotonic();torch.set_num_threads(1)
    folder=root/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
    bankpath=OLD/f'seed_{seed}/bank/bank.pt'
    bank=torch.load(bankpath,map_location='cpu',weights_only=False);ctx=bank_context(bank)
    _atomic_save({k:v for k,v in ctx.items() if k!='aligned_abs'},folder/'context.pt')
    eps=episodes(bank,seed,'train');contexts=torch.stack([e['context'] for e in eps])
    solver=json.loads((root/'solver_selection.json').read_text())['chosen']
    masks=initial_candidates(ctx,len(eps),np.random.default_rng(seed+18000))
    utility,fit=evaluate_candidates(masks,eps,solver,device)
    _atomic_save(fit,folder/'source_candidates_initial.pt')
    models={};meta=[]
    for stage in range(3):
        for kind in ('gnn','fm'):
            models[kind],info=fit_model(kind,masks,utility,contexts,ctx,seed,stage,device,folder)
            meta.append({k:info[k] for k in ('kind','stage','steps','stop_reason','elapsed_seconds','best_loss')})
        if stage==2:break
        proposed,_=model_candidates(models,ctx,contexts,seed+19000+stage,device)
        extra=torch.cat((proposed['gnn_search8'][:,:4],proposed['gnn_flow_search8'][:,:4]),1)
        scores,fit=evaluate_candidates(extra,eps,solver,device)
        _atomic_save(fit,folder/f'source_candidates_feedback_{stage}.pt')
        masks=torch.cat((masks,extra),1);utility=torch.cat((utility,scores),1)
    _atomic_save({'masks':masks,'query_bce':utility,'task_context':contexts,
                 'task_ids':[e['task_id'] for e in eps]},folder/'archive.pt')
    for kind in ('gnn','fm'):
        source=folder/f'{kind}_stage_2.pt';target=folder/f'{kind}_frozen.pt'
        cp=torch.load(source,map_location='cpu',weights_only=False)
        cp.update({'bank_sha256':_file_sha256(bankpath),'selection':'stage2 fixed beforehand; within-stage fixed objective minimum',
                   'solver_sha256':_file_sha256(root/'solver_selection.json')})
        _atomic_save(cp,target)
    write_json(folder/'training_summary.json',{'seed':seed,'models':meta,'source_tasks':len(eps),
               'candidates_per_task':masks.size(1),'child_fits':sum(len(torch.load(p,map_location='cpu',weights_only=False)['masks']) for p in folder.glob('source_candidates_*.pt')),
               'elapsed_seconds':time.monotonic()-started,'uses_test':False})
    print(json.dumps({'stage':'training_complete','seed':seed,'seconds':time.monotonic()-started}),flush=True)


def refine(root,seed,device):
    """One short FM-only continuation if the final initial fit hit its cap.

    This uses the same utility archive, a new fixed noise monitor, and a fresh
    lower-rate optimizer. It is a disclosed warm restart, not exact resume.
    """
    torch.set_num_threads(1);folder=root/f'seed_{seed}'
    path=folder/'fm_frozen.pt';prior=torch.load(path,map_location='cpu',weights_only=False)
    if prior['stop_reason']=='empirical_fixed_objective_plateau':return
    assert not (folder/'records.json').exists(), 'No refinement after inspecting test'
    polishing=(folder/'fm_pre_refine.pt').exists()
    _atomic_save(prior,folder/('fm_pre_polish.pt' if polishing else 'fm_pre_refine.pt'))
    bank=torch.load(OLD/f'seed_{seed}/bank/bank.pt',map_location='cpu',weights_only=False)
    ctx=bank_context(bank);a=torch.load(folder/'archive.pt',map_location='cpu',weights_only=False)
    _,payload=fit_model('fm',a['masks'],a['query_bce'],a['task_context'],ctx,
                        seed,4 if polishing else 3,device,folder,initial_state=prior['model_state'],
                        max_steps=600 if polishing else 1200,base_lr=.0000625 if polishing else .0005)
    for k in ('bank_sha256','solver_sha256'):payload[k]=prior[k]
    payload['selection']='short pre-test warm restart after cap; fixed-noise objective minimum'
    payload['initial_checkpoint_sha256']=_file_sha256(folder/('fm_pre_polish.pt' if polishing else 'fm_pre_refine.pt'))
    _atomic_save(payload,path)
    write_json(folder/('polishing_summary.json' if polishing else 'refinement_summary.json'),{k:payload[k] for k in ('kind','stage','steps','stop_reason','elapsed_seconds','best_loss','warm_start','max_steps','base_lr')})


def evaluate(root,seed,device):
    """Freeze candidates, select on query, refit with independent child RNG."""
    started=time.monotonic();torch.set_num_threads(1);folder=root/f'seed_{seed}'
    bankpath=OLD/f'seed_{seed}/bank/bank.pt';bank=torch.load(bankpath,map_location='cpu',weights_only=False)
    ctx=bank_context(bank);solver=json.loads((root/'solver_selection.json').read_text())['chosen']
    models={};hashes={}
    for kind in ('gnn','fm'):
        path=folder/f'{kind}_frozen.pt';cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['bank_sha256']==_file_sha256(bankpath)
        assert cp['solver_sha256']==_file_sha256(root/'solver_selection.json')
        model=make_model().to(device);model.load_state_dict(cp['model_state']);models[kind]=model.eval();hashes[kind]=_file_sha256(path)
    eps=episodes(bank,seed,'val')+episodes(bank,seed,'test')
    contexts=torch.stack([e['context'] for e in eps]);candidates,raw=model_candidates(models,ctx,contexts,seed+25000,device)
    names=list(candidates);pool=torch.cat([candidates[k] for k in names],1)
    _atomic_save({'candidates':candidates,'raw_scores':raw,'model_hashes':hashes,
                 'task_ids':[e['task_id'] for e in eps], 'uses_test_labels':False},folder/'candidates_frozen.pt')
    scores,selection_fit=evaluate_candidates(pool,eps,solver,device,replicas=2,offset=0)
    _atomic_save(selection_fit,folder/'candidate_selection_fit.pt')
    selected={};indexes={}
    for index,name in enumerate(names):
        losses=scores[:,index*8:(index+1)*8];best=losses.argmin(1)
        selected[name]=candidates[name][torch.arange(len(eps)),best]
        indexes[name]=best.tolist()
    selected['gnn_direct']=candidates['gnn_search8'][:,0]
    selected['gnn_flow_single']=candidates['gnn_flow_search8'][:,0]
    selected['uniform_functional']=ctx['uniform'][None].expand(len(eps),-1,-1)
    selected['dense']=torch.ones(len(eps),11,8)
    masks=torch.stack([selected[m] for m in METHODS],1)
    path=folder/'selected_masks_frozen.pt'
    _atomic_save({'masks':masks,'methods':METHODS,'selection_indexes':indexes,
                 'selection_scores':scores,'model_hashes':hashes,
                 'candidate_file_sha256':_file_sha256(folder/'candidates_frozen.pt'),
                 'uses_test_labels':False},path)
    _,fit=evaluate_candidates(masks,eps,solver,device,replicas=4,offset=2)
    _atomic_save(fit,folder/'final_children_frozen.pt')
    write_json(folder/'frozen_before_test.json',{'models':hashes,'selected_masks_sha256':_file_sha256(path),
               'children_sha256':_file_sha256(folder/'final_children_frozen.pt'),
               'solver_sha256':_file_sha256(root/'solver_selection.json'),'test_labels_materialized':False})
    rows=[]
    for t,ep in enumerate(eps):
        split='val' if t<2 else 'test'
        # Validation rows retain query metrics; test rows use independent IDs.
        pool=ep['query'] if split=='val' else build_test_pool(ep['task_id'].split(':')[-1])
        assert set(ep['support']['ids'].tolist()).isdisjoint(pool['ids'].tolist())
        if split=='test':assert set(ep['query']['ids'].tolist()).isdisjoint(pool['ids'].tolist())
        ids=list(range(t*len(METHODS)*4,(t+1)*len(METHODS)*4))
        params={k:v[ids].to(device) for k,v in fit['best_params'].items()}
        metrics=score_children_batched(params,fit['masks'][ids],pool,device=device)
        for j,metric in enumerate(metrics):
            i=ids[j];method=METHODS[j//4]
            rows.append({'seed':seed,'split':split,'task_id':ep['task_id'],'method':method,
                         'init_id':2+j%4,**metric,'converged':bool(fit['converged'][i]),
                         'best_step':int(fit['best_steps'][i]),'stopping_step':int(fit['stopping_steps'][i]),
                         'mask':fit['masks'][i].int().tolist(),'support_ids':ep['support']['ids'].tolist(),
                         'query_ids':ep['query']['ids'].tolist(),'score_ids':pool['ids'].tolist()})
    write_json(folder/'records.json',rows)
    write_json(folder/'evaluation_summary.json',{'seed':seed,'records':len(rows),
               'candidate_selection_fits':len(selection_fit['masks']),'candidate_selection_plateau':int(selection_fit['converged'].sum()),
               'final_fits':len(fit['masks']),'final_plateau':int(fit['converged'].sum()),
               'elapsed_seconds':time.monotonic()-started})
    print(json.dumps({'stage':'evaluation_complete','seed':seed,'seconds':time.monotonic()-started,
                      'test_mean':{m:float(np.mean([r['balanced_bce'] for r in rows if r['split']=='test' and r['method']==m])) for m in METHODS}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('train','refine','evaluate'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    {'train':train,'refine':refine,'evaluate':evaluate}[args.stage](args.root,args.seed,args.device)

"""Whole-mask utility distillation on the rebuilt functional DeepSets bank."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import load_data
from .adaptive_data import _original_pools, _read_new, _provenance
from .utility_graph_context import ROOT, task_sets
from .utility_graph_models import UtilityGraphField, exact_topk, flow_matching_losses, sample_flow
from .utility_graph_child import fit_children
from .rebuilt_bank_run import OUT as BANK, evaluate_state, costs
from .run import write_json

OUT=ROOT/'outputs/deepsets_vaae/20261001_rebuilt_utility_graph_flow'
K=7526


def hash_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def progress(out,stage,**extra):
    row=dict(stage=stage,**extra);write_json(Path(out)/'status.json',row)
    print(json.dumps(row),flush=True)


def final_data(seed,device,include_test=False):
    _,rows,hashes=_original_pools(seed,device)
    images=np.load(ROOT/'datasets/mnist8m/images.npy',mmap_mode='r')
    labels=np.load(ROOT/'datasets/mnist8m/labels.npy',mmap_mode='r')
    ignored={};duplicates={}
    for name,block,n,part in [('selection_train',3,1000,0),('selection_checkpoint',4,300,0),
                              ('selection_score',5,300,0),('old_confirmation_train',6,1000,0),
                              ('old_confirmation_query',7,300,0),('old_confirmation_test',7,300,1),
                              ('rebuilt_source_audit',5,300,2)]:
        _read_new(images,labels,rows,hashes,seed=seed,device=device,name=name,block=block,
                  per_digit=n,part=part,destination=ignored,duplicates=duplicates)
    splits={}
    definitions=[('train',6,1000,1),('query',7,300,2)]
    if include_test:definitions.append(('test',7,300,3))
    for name,block,n,part in definitions:
        _read_new(images,labels,rows,hashes,seed=seed,device=device,name=name,block=block,
                  per_digit=n,part=part,destination=splits,duplicates=duplicates)
    meta=_provenance(splits,stage='new target confirmation',identity_limit='distinct rows and exact pixels; no augmentation-group identity')
    meta.update(blocks={name:[block,part] for name,block,n,part in definitions},duplicates_excluded=duplicates)
    return splits,meta


def fit_sets(masks,sets,replicas,seed,settings,device,path):
    masks=torch.as_tensor(masks,device=device).float()
    result=fit_children(masks,torch.stack([r['x'] for r in sets]),torch.stack([r['y'] for r in sets]),
                        torch.stack([r['qx'] for r in sets]),torch.stack([r['qy'] for r in sets]),
                        [seed+3001*i for i in range(len(sets))],replicas,device=device,
                        steps=settings['steps'],lr=settings['lr'],l2=settings['l2'],
                        chunk_size=256,checkpoint_every=100,lr_decay_every=settings['lr_decay_every'])
    torch.save(result,path)
    return result


def elite_archive(masks,quality,count=4):
    """Utility scores are from fresh paired children, never teacher losses."""
    rows=[]
    for task,q in enumerate(quality):
        selected=[];seen=set()
        for index in q.argsort().tolist():
            h=hash_tensor(masks[index])
            if h in seen:continue
            seen.add(h);selected.append(index)
            if len(selected)==count:break
        idx=torch.tensor(selected,device=masks.device)
        utility=q[idx.cpu()].to(masks.device)
        weights=torch.softmax(-(utility-utility.min())/.03,dim=0)
        for i,w in zip(selected,weights):
            rows.append(dict(task=task,mask=masks[i],weight=w,quality=float(q[i]),candidate=i))
    return rows


def train_field(kind,context,sets,archive,out,seed,device,warm=None,max_steps=1200,stage=0):
    torch.manual_seed(seed)
    model=UtilityGraphField(node_dim=context['node'].shape[-1],edge_dim=6,task_dim=1570,
                            width=24,edge_width=8,depth=2).to(device)
    if warm is not None:model.load_state_dict(warm)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    masks=torch.stack([r['mask'] for r in archive]).to(device)
    task_ids=torch.tensor([r['task'] for r in archive],device=device)
    tasks=torch.stack([r['task'] for r in sets]).to(device)[task_ids]
    weights=torch.stack([r['weight'] for r in archive]).to(device)
    node=context['node'].to(device);edge=context['edge'].to(device)
    prior=context['mean_score'].to(device)
    prior_state=(prior-prior.mean())/prior.std().clamp_min(1e-4)
    generator=torch.Generator(device=device).manual_seed(seed+1)
    validation_generator=torch.Generator(device=device).manual_seed(seed+2)
    val_noise=torch.randn(masks.shape,device=device,generator=validation_generator)
    val_time=torch.rand(len(masks),device=device,generator=validation_generator)
    history=[];best=float('inf');best_state=None;flat_count=0;start=time.monotonic()
    def objective(indices,validation=False):
        n=node[None].expand(len(indices),-1,-1);e=edge[None].expand(len(indices),-1,-1,-1)
        t=tasks[indices];target=masks[indices]
        if kind=='gnn':
            score=model(prior_state[None].expand(len(indices),-1,-1),torch.zeros(len(indices),device=device),n,e,t)
            per=F.binary_cross_entropy_with_logits(score,target,reduction='none').flatten(1).mean(1)
        else:
            kw=dict(noise=val_noise[indices],time=val_time[indices]) if validation else dict(generator=generator)
            per=flow_matching_losses(model,2*target-1,n,e,t,**kw)
        if validation:return (per*weights[indices]).sum()/weights[indices].sum()
        return (per*weights[indices]).mean()*len(masks)/weights.sum()
    for step in range(1,max_steps+1):
        optimizer.param_groups[0]['lr']=.001*max(1/16,.5**((step-1)//300))
        indices=torch.randint(len(masks),(8,),device=device,generator=generator)
        optimizer.zero_grad(set_to_none=True);loss=objective(indices);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%25:continue
        with torch.no_grad():
            validation=0.
            # Every task gets its full archive weight, independent of minibatch composition.
            for begin in range(0,len(masks),4):
                ii=torch.arange(begin,min(begin+4,len(masks)),device=device)
                validation+=float(objective(ii,True))*float(weights[ii].sum()/weights.sum())
        history.append(dict(step=step,loss=float(loss),validation=validation))
        if validation<best:
            best=validation;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if step>=400 and len(history)>=5:
            values=[r['validation'] for r in history[-5:]]
            flat=(max(values)-min(values))/max(abs(np.mean(values)),1e-6)<.01
            flat_count=flat_count+1 if flat else 0
        if step%100==0:progress(out,f'train_{kind}',step=step,loss=float(loss),held_noise_loss=validation,seconds=time.monotonic()-start)
        if flat_count>=3:break
    payload=dict(state_dict=best_state,last_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                 kind=kind,history=history,plateau=flat_count>=3,steps=step,
                 objective='utility-weighted elite whole-mask distillation',
                 best_source_validation=best,seconds=time.monotonic()-start)
    torch.save(payload,out/f'{kind}_model_stage{stage}.pt')
    torch.save(payload,out/f'{kind}_model.pt');model.load_state_dict(best_state)
    return model,payload


@torch.no_grad()
def propose(kind,model,context,task,draws,seed,device):
    n=context['node'].to(device)[None].expand(draws,-1,-1)
    e=context['edge'].to(device)[None].expand(draws,-1,-1,-1)
    t=task[None].to(device).expand(draws,-1)
    g=torch.Generator(device=device).manual_seed(seed)
    if kind=='flow':scores=sample_flow(model,n,e,t,steps=12,generator=g)
    else:
        prior=context['mean_score'].to(device)
        prior=(prior-prior.mean())/prior.std().clamp_min(1e-4)
        scores=model(prior[None].expand(draws,-1,-1),torch.zeros(draws,device=device),n,e,t)
        if draws>1:
            u=torch.rand(scores.shape,device=device,generator=g).clamp(1e-6,1-1e-6)
            noise=-torch.log(-torch.log(u));noise[0]=0
            scores=scores+.15*noise
    return exact_topk(scores,K),scores


def worker(seed,device):
    out=OUT/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    spec=json.loads((OUT/'protocol.json').read_text())
    parent=BANK/f'seed_{seed}'
    if not (parent/'COMPLETE').exists():raise RuntimeError('Rebuilt bank is not complete')
    context=torch.load(parent/'functional_context.pt',map_location='cpu',weights_only=False)
    data=load_data(ROOT/'datasets/mnist8m',seed,device)
    source=task_sets(data,spec['task_vectors']['source'],seed+920000,train_name='source_train',query_name='source_validation')
    selection=json.loads((BANK/'new_child_pilot/selection.json').read_text())
    sparse=dict(steps=selection['steps'],lr_decay_every=selection['lr_decay_every'],**selection['sparse'])
    dense=dict(steps=selection['steps'],lr_decay_every=selection['lr_decay_every'],**selection['dense'])
    g=torch.Generator(device=device).manual_seed(seed+930000)
    mean=context['mean_score'].to(device)
    scores=[mean,*context['source_mean_scores'].to(device)]
    teacher=context['teacher_scores'].to(device)
    scores.extend([teacher[i]+.03*mean for i in (0,8,16,24)])
    scores.append(mean+.3*mean.std()*torch.randn(mean.shape,device=device,generator=g))
    pixel_score=mean.mean(1,keepdim=True).expand(-1,32)
    scores.append(pixel_score)
    random_score=torch.rand(mean.shape,device=device,generator=g)
    scores.append(random_score)
    candidates=exact_topk(torch.stack(scores),K)
    progress(out,'source_archive',candidates=len(candidates))
    fits=fit_sets(candidates.repeat_interleave(2,0),source,[0,1]*len(candidates),seed+940000,sparse,device,out/'source_children.pt')
    quality=fits['query_loss'].reshape(4,len(candidates),2).mean(2)
    archive=elite_archive(candidates,quality)
    torch.save(dict(masks=candidates.cpu(),quality=quality,elite=archive),out/'archive.pt')
    models={};train_records={}
    for kind in ('gnn','flow'):
        model,train=train_field(kind,context,source,archive,out,seed+950000+(kind=='flow'),device)
        models[kind]=model;train_records[kind]=train
    # One shared source-only feedback round; the two fields get identical utility labels.
    feedback=[]
    for task,row in enumerate(source):
        for kind in ('gnn','flow'):
            proposed,_=propose(kind,models[kind],context,row['task'],4,seed+960000+task*71+(kind=='flow'),device)
            feedback.extend(proposed.unbind())
    feedback_masks=torch.stack(feedback)
    feedback_fit=fit_sets(feedback_masks.repeat_interleave(2,0),source,[0,1]*len(feedback_masks),seed+940000,sparse,device,out/'feedback_children.pt')
    feedback_quality=feedback_fit['query_loss'].reshape(4,len(feedback_masks),2).mean(2)
    candidates=torch.cat((candidates,feedback_masks));quality=torch.cat((quality,feedback_quality),dim=1)
    archive=elite_archive(candidates,quality)
    torch.save(dict(masks=candidates.cpu(),quality=quality,elite=archive),out/'archive_final.pt')
    for kind in ('gnn','flow'):
        models[kind],train_records[kind]=train_field(kind,context,source,archive,out,
            seed+970000+(kind=='flow'),device,warm=train_records[kind]['state_dict'],max_steps=800,stage=1)
        for stage in range(2,6):
            if train_records[kind]['plateau']:break
            models[kind],train_records[kind]=train_field(kind,context,source,archive,out,
                seed+970000+1000*stage+(kind=='flow'),device,
                warm=train_records[kind]['state_dict'],max_steps=800,stage=stage)
        if not train_records[kind]['plateau']:
            write_json(out/'MODEL_NOT_CONVERGED.json',dict(kind=kind,steps=train_records[kind]['steps'],
                       reason='Source-only field objective has not plateaued; target test remains unopened'))
            raise RuntimeError('Final source field not plateau-certified; no target tests opened')
    # Freeze source artifacts before opening any final task examples/labels.
    write_json(out/'source_frozen.json',dict(bank_sha256=hashlib.sha256((parent/'functional_context.pt').read_bytes()).hexdigest(),
               models={k:hashlib.sha256((out/f'{k}_model.pt').read_bytes()).hexdigest() for k in models},
               seed=seed,target_tasks=spec['task_vectors']['new_test']))
    final,meta=final_data(seed,device);write_json(out/'target_provenance.json',meta)
    # test_name=None: test tensors are made only after final masks and children freeze.
    targets=task_sets(final,spec['task_vectors']['new_test'],seed+980000,train_name='train',query_name='query')
    records=[]
    for task,row in enumerate(targets):
        folder=out/f'task_{task}';folder.mkdir(exist_ok=True)
        progress(out,'target_mask_selection',task=task)
        bundles={};bundle_scores={}
        for kind in models:
            bundles[kind],bundle_scores[kind]=propose(kind,models[kind],context,row['task'],8,seed+990000+task*3001+(kind=='flow'),device)
        noise=torch.randn((8,784,32),device=device,generator=torch.Generator(device=device).manual_seed(seed+995000+task))
        noise[0]=0
        bundles['functional']=exact_topk(mean[None]+.15*mean.std()*noise,K)
        masks=torch.cat([bundles[k] for k in ('gnn','flow','functional')])
        choices=fit_sets(masks.repeat_interleave(2,0),[row],[0,1]*len(masks),seed+1000000+task*3001,sparse,device,folder/'selection_children.pt')
        q=choices['query_loss'][0].reshape(3,8,2).mean(2)
        selected={kind:int(q[i].argmin()) for i,kind in enumerate(('gnn','flow','functional'))}
        direct_gnn,_=propose('gnn',models['gnn'],context,row['task'],1,seed+990000+task*3001,device)
        methods=['functional','pixel_prior','random','gnn_single','flow_single','gnn_search8','flow_search8','functional_search8']
        chosen=torch.stack([exact_topk(mean[None],K)[0],exact_topk(pixel_score[None],K)[0],exact_topk(random_score[None],K)[0],direct_gnn[0],bundles['flow'][0],
                            bundles['gnn'][selected['gnn']],bundles['flow'][selected['flow']],bundles['functional'][selected['functional']]])
        torch.save(dict(methods=methods,masks=chosen.cpu(),candidate_masks={k:v.cpu() for k,v in bundles.items()},
                        candidate_query=q,selected=selected,
                        distinct={k:len({hash_tensor(m) for m in v}) for k,v in bundles.items()},
                        mask_sha256={k:hash_tensor(m) for k,m in zip(methods,chosen)}),folder/'masks_frozen.pt')
        child=fit_sets(chosen.repeat_interleave(4,0),[row],[2,3,4,5]*len(chosen),seed+1000000+task*3001,sparse,device,folder/'final_sparse_children.pt')
        control=fit_sets(torch.ones((4,784,32),device=device),[row],[2,3,4,5],seed+1000000+task*3001,dense,device,folder/'final_dense_children.pt')
        write_json(folder/'children_frozen.json',dict(masks_sha256=hashlib.sha256((folder/'masks_frozen.pt').read_bytes()).hexdigest(),
                   sparse_sha256=hashlib.sha256((folder/'final_sparse_children.pt').read_bytes()).hexdigest(),
                   dense_sha256=hashlib.sha256((folder/'final_dense_children.pt').read_bytes()).hexdigest(),
                   test_opened=False))
        torch.save(dict(x=row['x'].cpu(),y=row['y'].cpu(),qx=row['qx'].cpu(),qy=row['qy'].cpu()),folder/'sets.pt')
    # Every target mask and child state is fixed before materializing NEW test labels.
    write_json(out/'all_children_frozen.json',dict(tasks=len(targets),test_opened=False,
        children_sha256={str(t):hashlib.sha256((out/f'task_{t}/children_frozen.json').read_bytes()).hexdigest() for t in range(len(targets))}))
    final_with_test,test_meta=final_data(seed,device,include_test=True)
    if any(meta['split_hashes'][k]!=test_meta['split_hashes'][k] for k in ('train','query')):
        raise ValueError('Pre-freeze target support/query splits changed')
    write_json(out/'target_test_provenance.json',test_meta)
    for task,row in enumerate(targets):
        folder=out/f'task_{task}'
        frozen=torch.load(folder/'masks_frozen.pt',map_location='cpu',weights_only=False)
        methods=frozen['methods']
        child=torch.load(folder/'final_sparse_children.pt',map_location='cpu',weights_only=False)
        control=torch.load(folder/'final_dense_children.pt',map_location='cpu',weights_only=False)
        from .core import _sets,centred_costs
        tx,ty=_sets(final_with_test['test'],centred_costs(spec['task_vectors']['new_test'][task],device),512,5,
                    torch.Generator(device=device).manual_seed(seed+1010000+task*3001))
        torch.save(dict(x=row['x'].cpu(),y=row['y'].cpu(),qx=row['qx'].cpu(),qy=row['qy'].cpu(),tx=tx.cpu(),ty=ty.cpu()),folder/'sets.pt')
        for names,result in ((methods,child),(['dense_tuned'],control)):
            state={k:v[0] for k,v in result['state_dict'].items()}
            loss=evaluate_state(state,tx,ty,device).reshape(len(names),4)
            for i,name in enumerate(names):
                for r in range(4):
                    records.append(dict(seed=seed,task=task,method=name,replica=r+2,nmse=float(loss[i,r]),
                                        query_nmse=float(result['query_loss'][0,i*4+r]),
                                        support_nmse=float(result['support_loss'][0,i*4+r]),
                                        plateau=bool(result['plateau_flags'][0,i*4+r])))
        write_json(out/'results.json',records)
    write_json(out/'training_summary.json',{k:{name:value for name,value in v.items() if name not in ('state_dict','last_state_dict')} for k,v in train_records.items()})
    (out/'COMPLETE').write_text('utility graph comparison completed\n');progress(out,'complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    worker(args.seed,torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


if __name__=='__main__':main()

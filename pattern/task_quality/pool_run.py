"""Learn positive functional-bank pooling with the same task-quality objective."""
from pathlib import Path
import argparse
import json
import math
import time
import torch
import torch.nn.functional as F
from .core import build_experiment_data
from .generator import permute_hidden_columns
from .repaired_generator import fixed_mass_soft
from .functional_pool import FunctionalPool
from .meta import (MetaConfig,_fixed_train_episodes,_fixed_validation_episodes,
                   _stack_episode_samples,_init_child_batch,_inner_adapt,_batched_logits,
                   _run_validation,_cpu_state,_cpu_optimizer_state,_atomic_save,_file_sha256)
from .convergence import loss_plateau
from .stripe_debug import OLD,additive_stats

VARIANTS=('functional_set_pool','functional_set_pool_bounded')
SEEDS=(8100,8102)


def train(out, seed, variant, device, cap=1200):
    torch.set_num_threads(1);device=torch.device(device)
    bank_path=OLD/f'seed_{seed}/bank/bank.pt'
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    feature=bank['feature'].to(device)
    data=build_experiment_data(probe_seed=seed)
    config=MetaConfig(max_steps=cap)
    train_ids=[t.task_id for t in data['splits']['train']]
    train_mid,ts,tq=_fixed_train_episodes(data,config,seed+601)
    val_ids,vs,vq=_fixed_validation_episodes(data,config,seed+701)
    torch.manual_seed(seed+31)
    model=FunctionalPool(bank,bounded_weights=variant.endswith('_bounded')).to(device)
    rng=torch.Generator().manual_seed(seed+1201)
    aug=torch.Generator().manual_seed(seed+6201)
    opt=torch.optim.Adam(model.parameters(),lr=config.outer_learning_rate)
    run=out/f'seed_{seed}'/variant;run.mkdir(parents=True,exist_ok=True)
    metadata={'protocol':'pattern_functional_pool_meta_v1','seed':seed,'variant':variant,
              'bank_path':str(bank_path),'bank_sha256':_file_sha256(bank_path),
              'meta_config':config.__dict__,'train_ids':train_ids,'val_ids':val_ids,
              'uses_oracle_for_training_or_selection':False,'warm_start':'uniform functional baseline',
              'alignment':'fixed source sensitivity centroid, explicit coordinate prior',
              'pool_weight_bound':2 if variant.endswith('_bounded') else None,
              'fixed_mask_edges':32,'inner_solver':'SGD momentum .9 lr .1, 64 steps',
              'inner_solver_converged_claim':False,'torch':torch.__version__,
              'cuda':torch.version.cuda,'device_name':torch.cuda.get_device_name() if device.type=='cuda' else 'cpu'}
    (run/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    curves=[];stochastic=[];best=math.inf;best_step=0;step=0;tp=vp=0;reason='step_cap'
    latest=run/'last.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=False)
        assert saved['metadata']==metadata,'Protocol changed during resume'
        model.load_state_dict(saved['model_state']);opt.load_state_dict(saved['optimizer_state'])
        rng.set_state(saved['episode_rng']);aug.set_state(saved['augmentation_rng'])
        curves=saved['curves'];stochastic=saved['stochastic'];best=saved['best_val'];best_step=saved['best_step'];step=saved['step'];tp=saved['train_plateau_passes'];vp=saved['val_plateau_passes']
        if saved['stop_reason']=='empirical_train_val_plateau':return run/'best.pt'
    if step==0 and not latest.exists():
        initial_tr=_run_validation(model,feature,ts,tq,seed+8000,config,device)
        initial_va=_run_validation(model,feature,vs,vq,seed+9000,config,device)
        best=float(initial_va['query_balanced_bce']);best_step=0
        _atomic_save({'metadata':metadata,'model_state':_cpu_state(model),'step':0,
                     'val':initial_va,'train_monitor':initial_tr,'best_val':best,
                     'val_support_ids':vs['ids'],'val_query_ids':vq['ids']},run/'initial.pt')
        _atomic_save(torch.load(run/'initial.pt',weights_only=False),run/'best.pt')
        print(json.dumps({'variant':variant,'seed':seed,'initial_val':best,
                          'initial_train':float(initial_tr['query_balanced_bce'])}),flush=True)
    start=time.monotonic()
    while step<cap:
        chosen=torch.randperm(len(train_ids),generator=rng)[:config.meta_batch_tasks]
        ids=[train_ids[int(i)] for i in chosen]
        s,q=_stack_episode_samples(data['pools'],ids,128,128,rng)
        x,y=s['x'].to(device),s['y'].to(device)
        qx,qy=q['x'].to(device),q['y'].to(device)
        order=torch.stack([torch.randperm(8,generator=aug) for _ in range(feature.size(0))])
        current=permute_hidden_columns(feature,order)
        map_order=torch.randperm(feature.size(0),generator=aug).to(device)
        current=(current[map_order],model.proposals[map_order])
        mask,logits=model(current,x,y)
        params=_init_child_batch(4,2,rng,device)
        params,_=_inner_adapt(x,y,mask,params,steps=64,learning_rate=.1,momentum=.9,create_graph=True)
        pred=_batched_logits(qx,mask,params)
        loss=F.binary_cross_entropy_with_logits(pred,qy[:,None,:].expand_as(pred))
        opt.zero_grad(set_to_none=True);loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite hypergradient')
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.0))
        opt.step();step+=1;stochastic.append(float(loss.detach()))
        if step%500==0:
            for group in opt.param_groups:group['lr']=max(.0001,.002*(.5**(step//500)))
        if step%20:continue
        tr=_run_validation(model,feature,ts,tq,seed+8000,config,device)
        va=_run_validation(model,feature,vs,vq,seed+9000,config,device)
        with torch.no_grad():
            masks,scores=model(feature,vs['x'].to(device),vs['y'].to(device))
            soft=fixed_mass_soft(scores)
        row={'step':step,'train_query_bce':float(tr['query_balanced_bce']),
             'val_query_bce':float(va['query_balanced_bce']),'hypergradient_norm':grad_norm,
             'learning_rate':opt.param_groups[0]['lr'],
             'surrogate_mean_p_times_1mp':float((soft*(1-soft)).mean()),
             'unique_val_masks':int(torch.unique(masks.flatten(1),dim=0).size(0)),
             **additive_stats(scores)}
        curves.append(row)
        if row['val_query_bce']<best:
            best=row['val_query_bce'];best_step=step
            _atomic_save({'metadata':metadata,'model_state':_cpu_state(model),'step':step,
                         'val':va,'train_monitor':tr,'best_val':best,
                         'val_support_ids':vs['ids'],'val_query_ids':vq['ids']},run/'best.pt')
        if step>=400:
            tp=tp+1 if bool(loss_plateau(torch.tensor([r['train_query_bce'] for r in curves]),width=8,tolerance=.01)) else 0
            vp=vp+1 if bool(loss_plateau(torch.tensor([r['val_query_bce'] for r in curves]),width=8,tolerance=.01)) else 0
        if tp>=3 and vp>=3:reason='empirical_train_val_plateau'
        payload={'metadata':metadata,'model_state':_cpu_state(model),'optimizer_state':_cpu_optimizer_state(opt),
                 'episode_rng':rng.get_state(),'augmentation_rng':aug.get_state(),
                 'curves':curves,'stochastic':stochastic,'step':step,'best_val':best,'best_step':best_step,
                 'train_plateau_passes':tp,'val_plateau_passes':vp,
                 'stop_reason':reason if reason!='step_cap' else ('step_cap' if step==cap else 'running')}
        _atomic_save(payload,latest)
        (run/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
        print(json.dumps({'variant':variant,'seed':seed,'best_val':best,'best_step':best_step,'seconds':time.monotonic()-start,**row}),flush=True)
        if reason!='step_cap':break
    summary={'variant':variant,'seed':seed,'step':step,'best_val':best,'best_step':best_step,
             'stop_reason':reason,'empirical_outer_plateau':reason!='step_cap',
             'inner_solver_converged_claim':False,'bank_sha256':metadata['bank_sha256']}
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return run/'best.pt'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--device',default='cpu');p.add_argument('--cap',type=int,default=1200)
    a=p.parse_args();print(train(a.root,a.seed,a.variant,a.device,a.cap),flush=True)

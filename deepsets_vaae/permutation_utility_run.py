"""Source-only integration of actual hard-mask utility and input consistency.

This is a new experiment; completed utility comparisons are left immutable.
Target/test examples are not loaded. Four arms separate architectural set
invariance from the effect of an explicit regularizer on an order-sensitive
control. The fixed child solver is the existing source-meta-calibrated solver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .core import _read_split, _hash_ids
from .rebuilt_bank_run import OUT as BANK, costs
from .utility_graph_context import ROOT, task_sets
from .utility_graph_child import fit_children
from .utility_graph_models import exact_topk
from .permutation_bank_encoder import TrainOnlyFunctionalTeacherBank, RawFunctionalBankEncoder, permute_teacher_hidden_columns
from .permutation_utility_loss import sample_ordered_topk, dual_objective

OUT=ROOT/'outputs/deepsets_vaae/20261001_permutation_dual_loss'
ARMS={'set_joint':(False,1.),'set_quality':(False,0.),
      'position_joint':(True,1.),'position_quality':(True,0.)}


def load_source_only(seed,device):
    """Read source blocks only, without materializing target/test observations."""
    images=np.load(ROOT/'datasets/mnist8m/images.npy',mmap_mode='r')
    labels=np.load(ROOT/'datasets/mnist8m/labels.npy',mmap_mode='r')
    rows=set();pixels=set();splits={};excluded={}
    for name,block,count in [('source_train',0,1000),('source_validation',1,300)]:
        splits[name],excluded[name]=_read_split(images,labels,block_id=block,per_digit=count,
            seed=seed,part=0,device=device,used_rows=rows,used_pixel_hashes=pixels)
    splits['split_hashes']={k:_hash_ids(v.source_ids) for k,v in splits.items()}
    splits['source_duplicate_exclusions']=excluded
    return splits


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def fit_policy_draws(masks,sets,seed,settings,device):
    """Batched source children; each draw is evaluated on every source task.

    Only diagonal (task, masks generated for that task) entries train that
    task's policy. Off-diagonal fits do not become invented policy samples.
    """
    tasks,draws=masks.shape[:2]
    fitted=fit_children(masks.flatten(0,1).repeat_interleave(2,0).detach(),
        torch.stack([s['x'] for s in sets]),torch.stack([s['y'] for s in sets]),
        torch.stack([s['qx'] for s in sets]),torch.stack([s['qy'] for s in sets]),
        [seed+3001*i for i in range(tasks)],[0,1]*(tasks*draws),
        steps=settings['steps'],lr=settings['sparse']['lr'],l2=settings['sparse']['l2'],
        device=device,chunk_size=256,checkpoint_every=100,
        lr_decay_every=settings['lr_decay_every'])
    all_query=fitted['query_loss'].reshape(tasks,tasks,draws,2)
    own=torch.stack([all_query[t,t].mean(-1) for t in range(tasks)]).to(device)
    return own,fitted


def worker(arm,seed,max_updates,min_updates,teacher_count,device):
    position_bias,coefficient=ARMS[arm]
    out=OUT/f'seed_{seed}'/arm;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    if not (OUT/'protocol.json').exists():raise RuntimeError('Freeze protocol before training')
    protocol=json.loads((OUT/'protocol.json').read_text())
    expected=dict(max_updates=max_updates,min_updates=min_updates,teacher_count=teacher_count)
    if any(protocol[k]!=v for k,v in expected.items()):raise ValueError('CLI differs from frozen protocol')
    settings=json.loads((BANK/'new_child_pilot/selection.json').read_text())
    bank=TrainOnlyFunctionalTeacherBank(BANK/f'seed_{seed}/functional_context.pt',
                                       include_source_query_quality=True,include_training_masks=True)
    data=load_source_only(seed,device)
    # Only original source pools enter this run. No final_data or target/test calls.
    sets=task_sets(data,costs()['source'],seed+920000,
                   train_name='source_train',query_name='source_validation')
    task=torch.stack([r['task'] for r in sets])
    torch.manual_seed(seed+1100000)
    model=RawFunctionalBankEncoder(token_dim=bank.token_dim,task_context_dim=task.shape[-1],
         features=784,hidden=32,width=32,column_position_bias=position_bias).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    cpu_rng=torch.Generator(device='cpu').manual_seed(seed+1100001)
    gpu_rng=torch.Generator(device=device).manual_seed(seed+1100002)
    fixed=bank.sample(batch_size=4,teacher_count=teacher_count,generator=torch.Generator().manual_seed(seed+1100003))
    fixed_tokens=fixed.tokens.to(device);fixed_quality=fixed.source_query_quality.to(device)
    fixed_perm,_=permute_teacher_hidden_columns(fixed_tokens,generator=torch.Generator(device=device).manual_seed(seed+1100004))
    u=torch.rand((4,2,784,32),device=device,generator=torch.Generator(device=device).manual_seed(seed+1100005)).clamp(1e-7,1-1e-7)
    fixed_noise=-torch.log(-torch.log(u))
    history=[];monitor=[];best=float('inf');best_state=None;flat_count=0;child_count=0;child_flat=0
    started=time.monotonic()
    save_json(out/'input_provenance.json',dict(train_teacher_rows=bank.train_references.tolist(),
        functional_context_sha256=hashlib.sha256((BANK/f'seed_{seed}/functional_context.pt').read_bytes()).hexdigest(),
        sampled_teacher_count=teacher_count,source_query_quality_input=True,source_audit_input=False,
        test_opened=False,source_split_hashes={k:data['split_hashes'][k] for k in ('source_train','source_validation')},
        fixed_monitor_teacher_refs=fixed.references.tolist()))

    def measure(update):
        nonlocal best,best_state,flat_count,child_count,child_flat
        model.eval()
        with torch.no_grad():
            e,z=model(fixed_tokens,task,teacher_quality=fixed_quality)
            ep,zp=model(fixed_perm,task,teacher_quality=fixed_quality)
            masks,lp,_=sample_ordered_topk(z[:,None].expand(-1,2,-1,-1),7526,gumbel=fixed_noise)
        q,children=fit_policy_draws(masks,sets,seed+940000,settings,device)
        child_count+=children['plateau_flags'].numel();child_flat+=int(children['plateau_flags'].sum())
        obj=dual_objective(lp,q,e,ep,z,zp,coefficient=coefficient)
        value=float(obj['quality_nmse']+coefficient*obj['consistency'])
        hard_original=exact_topk(z,7526);hard_perm=exact_topk(zp,7526)
        row=dict(update=update,quality_nmse=float(obj['quality_nmse']),
                 encoder_consistency=float(obj['encoder_consistency']),response_consistency=float(obj['response_consistency']),
                 objective=value,hard_mask_agreement=float((hard_original==hard_perm).float().mean()),
                 child_plateau=int(children['plateau_flags'].sum()),child_fits=children['plateau_flags'].numel())
        monitor.append(row)
        torch.save(dict(masks=masks.cpu(),own_query=q.cpu(),children=children,
                        logits=z.cpu(),permuted_logits=zp.cpu(),embedding=e.cpu(),permuted_embedding=ep.cpu()),out/'monitor_last.pt')
        if value<best:
            best=value;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(state_dict=best_state,monitor=row,architecture=dict(token_dim=bank.token_dim,
               task_context_dim=task.shape[-1],features=784,hidden=32,width=32,column_position_bias=position_bias)),out/'best_model.pt')
        if update>=min_updates and len(monitor)>=4:
            values=[v['objective'] for v in monitor[-4:]]
            flat=(max(values)-min(values))/max(abs(np.mean(values)),1e-6)<.01
            flat_count=flat_count+1 if flat else 0
        save_json(out/'status.json',row)
        print(json.dumps(dict(arm=arm,stage='source_monitor',**row)),flush=True)
        model.train()

    measure(0)
    for update in range(1,max_updates+1):
        optimizer.param_groups[0]['lr']=.001*max(1/64,.5**((update-1)//24))
        teacher_batch=bank.sample(batch_size=4,teacher_count=teacher_count,generator=cpu_rng)
        tokens=teacher_batch.tokens.to(device);quality=teacher_batch.source_query_quality.to(device)
        augmented,_=permute_teacher_hidden_columns(tokens,generator=gpu_rng)
        e,z=model(tokens,task,teacher_quality=quality)
        ep,zp=model(augmented,task,teacher_quality=quality)
        masks,lp,_=sample_ordered_topk(z[:,None].expand(-1,2,-1,-1),7526,generator=gpu_rng)
        q,children=fit_policy_draws(masks,sets,seed+940000,settings,device)
        child_count+=children['plateau_flags'].numel();child_flat+=int(children['plateau_flags'].sum())
        obj=dual_objective(lp,q,e,ep,z,zp,coefficient=coefficient)
        optimizer.zero_grad(set_to_none=True);obj['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(norm):raise RuntimeError('nonfinite dual-objective gradient')
        optimizer.step()
        history.append(dict(update=update,quality_nmse=float(obj['quality_nmse']),
             policy_gradient_loss=float(obj['policy_gradient_loss']),consistency=float(obj['consistency']),
             encoder_consistency=float(obj['encoder_consistency']),response_consistency=float(obj['response_consistency']),
             gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
             child_plateau=int(children['plateau_flags'].sum()),child_fits=children['plateau_flags'].numel()))
        save_json(out/'training_history.json',history)
        if update%8==0:measure(update)
        if flat_count>=3:break
    if monitor[-1]['update']!=update:measure(update)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    optimizer_state=optimizer.state_dict(),update=update),out/'terminal_model.pt')
    torch.save(dict(fixed_tokens=fixed_tokens.cpu(),fixed_teacher_quality=fixed_quality.cpu(),
                    fixed_permuted_tokens=fixed_perm.cpu(),task_context=task.cpu(),fixed_gumbel=fixed_noise.cpu(),
                    source_sets=[{k:r[k].cpu() for k in ('x','y','qx','qy')} for r in sets]),out/'replay_inputs.pt')
    save_json(out/'monitor_history.json',monitor)
    save_json(out/'summary.json',dict(arm=arm,seed=seed,updates=update,plateau=flat_count>=3,
         capped=flat_count<3,child_fits=child_count,child_plateau=child_flat,
         best_source_objective=best,coefficient=coefficient,column_position_bias=position_bias,
         seconds=time.monotonic()-started,test_opened=False,
         target_quality_claim=False,task_family='four previously used sourcecost vectors',
         consistency_zero_expected=not position_bias))
    (out/'COMPLETE').write_text('Source-only dual-loss run completed; convergence flag is separate.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=list(ARMS),required=True)
    p.add_argument('--seed',type=int,default=4100);p.add_argument('--max-updates',type=int,default=96)
    p.add_argument('--min-updates',type=int,default=32);p.add_argument('--teacher-count',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    worker(a.arm,a.seed,a.max_updates,a.min_updates,a.teacher_count,torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))


if __name__=='__main__':main()

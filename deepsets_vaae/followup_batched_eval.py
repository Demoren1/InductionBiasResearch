"""Fit independent target task/budget conditions in larger GPU batches.

Each condition keeps its own initialization, minibatch RNG, Adam parameters,
and validation checkpoint. Batching changes execution, not the experiment.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .core import (MaskedDeepSets, Split, _make_target_sets, _mask_items,
                   centred_costs)


class BatchedConditions(nn.Module):
    def __init__(self, masks, condition_seeds, replicas, reference_models=20,
                 kernel_mode='reference'):
        super().__init__()
        self.kernel_mode=kernel_mode
        self.register_buffer('masks', masks[None].expand(len(condition_seeds),-1,-1,-1))
        states=[]
        exemplar={}
        for i,r in enumerate(replicas):exemplar.setdefault(r,i)
        source=torch.tensor([exemplar[r] for r in replicas],device=masks.device)
        for seed in condition_seeds:
            model=MaskedDeepSets(masks,seed=seed,initialization_reference_models=reference_models)
            states.append({name:p.detach()[source].clone() for name,p in model.named_parameters()})
        for name in states[0]:
            self.register_parameter(name,nn.Parameter(torch.stack([s[name] for s in states])))

    def forward(self,x):
        if self.kernel_mode=='reference':
            # Preserve the pilot's individual GEMM shapes on CUDA. A larger
            # strided GEMM can use a different accumulation order; tiny first
            # step errors can alter later checkpoint selection. Optimizer and
            # loss work still operate on all conditions together.
            values=[]
            for j in range(len(x)):
                hidden=torch.tanh(torch.einsum('bsi,mih->mbsh',x[j],self.weight[j]*self.masks[j])
                                  +self.bias[j,:,None,None,:])
                values.append(((hidden*self.readout[j,:,None,None,:]).sum(-1)
                               +self.per_image_offset[j,:,None,None]).sum(-1))
            return torch.stack(values)
        # One strided batched GEMM computes all independent conditions. Within
        # a condition all methods see exactly the same images.
        c,b,s,f=x.shape
        _,m,_,h=self.weight.shape
        weight=(self.weight*self.masks).permute(0,2,1,3).reshape(c,f,m*h)
        pre=torch.bmm(x.reshape(c,b*s,f),weight).reshape(c,b,s,m,h).permute(0,3,1,2,4)
        hidden=torch.tanh(pre+self.bias[:,:,None,None,:])
        return ((hidden*self.readout[:,:,None,None,:]).sum(-1)+self.per_image_offset[:,:,None,None]).sum(-1)


@torch.no_grad()
def losses(model,x,y,set_size,chunk=128):
    total=None
    for start in range(0,x.shape[1],chunk):
        values=(model(x[:,start:start+chunk])-y[:,None,start:start+chunk]).square().sum(-1)
        total=values if total is None else total+values
    return total/x.shape[1]/set_size


def evaluate_masks_batched(data,costs,masks,seed,device,*,support_sizes=(32,64,128,256),
                           steps=800,batch_size=32,set_size=5,validation_sets=128,
                           test_sets=512,artifact_dir=None,initialization_reference_models=20,
                           batch_conditions=8, kernel_mode='reference'):
    device=torch.device(device)
    if batch_conditions<1:raise ValueError('batch_conditions must be positive')
    if not support_sizes or min(support_sizes)<2 or min(steps,batch_size,set_size,validation_sets,test_sets)<1:
        raise ValueError('support sizes >=2 and all other sizes positive required')
    if initialization_reference_models<1:raise ValueError('reference model count must be positive')
    if kernel_mode not in ('reference','bmm'):raise ValueError('unknown kernel mode')
    names,stacked,_=_mask_items(masks,device)
    replicas=[];counts={}
    for name in names:
        replicas.append(counts.get(name,0));counts[name]=counts.get(name,0)+1
    task_values=torch.as_tensor(costs,dtype=torch.float32,device=device)
    if task_values.ndim==1:task_values=task_values[None]
    task_values=torch.stack([centred_costs(row,device) for row in task_values])
    splits={}
    for name in ('target_train','target_validation','target_test'):
        value=data[name]
        value=value if isinstance(value,Split) else Split(*value)
        splits[name]=Split(*(v.to(device) for v in value))
    valid_counts={int(n):min(validation_sets,max(1,round(.2*n))) for n in support_sizes}
    train_counts={int(n):int(n)-valid_counts[int(n)] for n in support_sizes}
    prepared={}
    for task,cost in enumerate(task_values):
        gen=torch.Generator(device=device).manual_seed(seed+10007*(task+1))
        sx,sy=_make_target_sets(splits['target_train'],cost,max(support_sizes),set_size,gen)
        vx,vy=_make_target_sets(splits['target_validation'],cost,max(valid_counts.values()),set_size,gen)
        tx,ty=_make_target_sets(splits['target_test'],cost,test_sets,set_size,gen)
        prepared[task]=(sx,sy,vx,vy,tx,ty)
    conditions=[(task,int(budget)) for task in range(len(task_values)) for budget in support_sizes]
    output=None if artifact_dir is None else Path(artifact_dir)
    if output:output.mkdir(parents=True,exist_ok=True)
    records=[];check_every=max(1,min(50,steps//8))
    for start in range(0,len(conditions),batch_conditions):
        group=conditions[start:start+batch_conditions]
        print(f'batched target conditions={group}: {len(group)} × {len(stacked)} models',flush=True)
        model=BatchedConditions(stacked,[seed+3001*t+b for t,b in group],replicas,
                                initialization_reference_models,kernel_mode).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=2e-3)
        generators=[torch.Generator(device=device).manual_seed(seed+70001*(t+1)+b) for t,b in group]
        c=len(group);m=len(stacked)
        best_val=torch.full((c,m),float('inf'),device=device)
        best_train=torch.full_like(best_val,float('nan'));best_step=torch.zeros(c,m,dtype=torch.long,device=device)
        best_state={name:p.detach().clone() for name,p in model.named_parameters()}
        for step in range(1,steps+1):
            examples=[];labels=[]
            for (task,budget),gen in zip(group,generators):
                indices=torch.randint(train_counts[budget],(batch_size,),device=device,generator=gen)
                examples.append(prepared[task][0][indices]);labels.append(prepared[task][1][indices])
            x=torch.stack(examples);y=torch.stack(labels)
            per_model=(model(x)-y[:,None]).square().mean(-1)/set_size
            optimizer.zero_grad(set_to_none=True)
            (per_model.sum()/initialization_reference_models).backward();optimizer.step()
            if step==1 or step%check_every==0 or step==steps:
                train_rows=[];val_rows=[]
                # Group equal budgets to avoid padding entering a mean or
                # changing checkpoint selection for a small-data condition.
                with torch.no_grad():
                    for j,(task,budget) in enumerate(group):
                        sx,sy,vx,vy,_,_=prepared[task]
                        view=_ConditionView(model,j)
                        train_rows.append(losses(view,sx[None,:train_counts[budget]],sy[None,:train_counts[budget]],set_size)[0])
                        val_rows.append(losses(view,vx[None,:valid_counts[budget]],vy[None,:valid_counts[budget]],set_size)[0])
                    train_loss=torch.stack(train_rows);val_loss=torch.stack(val_rows)
                    if step==1:initial_train=train_loss.clone();initial_val=val_loss.clone()
                    improved=val_loss<best_val
                    for name,p in model.named_parameters():best_state[name][improved]=p.detach()[improved]
                    best_train[improved]=train_loss[improved];best_step[improved]=step
                    best_val=torch.minimum(best_val,val_loss)
                    final_train=train_loss;final_val=val_loss
        with torch.no_grad():
            for name,p in model.named_parameters():p.copy_(best_state[name])
            for j,(task,budget) in enumerate(group):
                _,_,_,_,tx,ty=prepared[task]
                predictions=[]
                for k in range(0,len(tx),128):predictions.append(_ConditionView(model,j)(tx[None,k:k+128])[0])
                prediction=torch.cat(predictions,dim=1)
                mse=(prediction-ty[None]).square().mean(-1)/set_size
                mae=(prediction-ty[None]).abs().mean(-1)
                condition_records=[]
                for i,(method,replica) in enumerate(zip(names,replicas)):
                    row={'task':task,'support_size':budget,'method':method,'init':replica,
                         'train_sets':train_counts[budget],'validation_sets':valid_counts[budget],
                         'total_labeled_sets':budget,'mse':float(mse[i]),'mae':float(mae[i]),
                         'best_step':int(best_step[j,i]),'validation_mse':float(best_val[j,i]),
                         'best_train_mse':float(best_train[j,i]),'final_train_mse':float(final_train[j,i]),
                         'final_validation_mse':float(final_val[j,i]),
                         'early_train_normalized_mse':float(initial_train[j,i]),
                         'early_validation_normalized_mse':float(initial_val[j,i]),
                         'best_validation_normalized_mse':float(best_val[j,i])}
                    records.append(row);condition_records.append(row)
                if output:
                    state={name:p[j].detach().cpu().clone() for name,p in model.named_parameters()}
                    state['masks']=stacked.detach().cpu().clone()
                    torch.save({'schema':'deepsets_vaae.target_checkpoint.v1','task':task,'support_size':budget,
                                'set_size':set_size,'method_names':names,'replica_indices':replicas,
                                'methods':[{'model_index':i,'method':n,'init':r} for i,(n,r) in enumerate(zip(names,replicas))],
                                'records':condition_records,'state_dict':state,'masks':state['masks'],
                                'weight':state['weight'],'effective_weight':state['weight']*state['masks'],
                                'bias':state['bias'],'readout':state['readout'],
                                'per_image_offset':state['per_image_offset'],'offset':state['per_image_offset'],
                                'execution':{'batch_conditions':len(group),
                                             'kernel_mode':kernel_mode,
                                             'gradient_denominator':initialization_reference_models}},
                               output/f'target_task{task}_budget{budget}.pt')
    return records


class _ConditionView:
    """Evaluate one condition without allocating or mutating a new model."""
    def __init__(self,model,index):self.model,self.index=model,index
    def __call__(self,x):
        j=self.index;model=self.model
        c,b,s,f=x.shape;_,m,_,h=model.weight.shape
        weight=(model.weight[j:j+1]*model.masks[j:j+1]).permute(0,2,1,3).reshape(1,f,m*h)
        pre=torch.bmm(x.reshape(1,b*s,f),weight).reshape(1,b,s,m,h).permute(0,3,1,2,4)
        hidden=torch.tanh(pre+model.bias[j:j+1,:,None,None,:])
        return ((hidden*model.readout[j:j+1,:,None,None,:]).sum(-1)+model.per_image_offset[j:j+1,:,None,None]).sum(-1)

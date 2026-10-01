"""Source-only bank-guided discrete mask search; no mask hypergradient."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import torch.nn.functional as F
from .stripe_debug import OLD
from .core import build_experiment_data
from .meta import MetaConfig,_fixed_train_episodes,_fixed_validation_episodes,_init_child_batch,_inner_adapt,_batched_logits,_run_validation,_atomic_save,_file_sha256
from .evaluate import functional_centroid_mean_mask
from .convergence import loss_plateau

class FixedProposal(torch.nn.Module):
    def __init__(self,mask):super().__init__();self.register_buffer('mask',mask.float())
    def forward(self,bank,x,y):
        batch=x.size(0) if x.ndim==3 else 1
        return self.mask[None].expand(batch,-1,-1),self.mask[None].expand(batch,-1,-1)


def run(root,seed,device):
    torch.set_num_threads(1);device=torch.device(device)
    bankpath=OLD/f'seed_{seed}/bank/bank.pt'
    bank=torch.load(bankpath,map_location='cpu',weights_only=False)
    functional=functional_centroid_mean_mask(bank['edge_q'])
    parent=torch.as_tensor(functional['mask'],device=device,dtype=torch.float32)
    initial=parent.clone();score=np.asarray(functional['scores'])
    data=build_experiment_data(probe_seed=seed);cfg=MetaConfig()
    tids,ts,tq=_fixed_train_episodes(data,cfg,seed+601)
    vids,vs,vq=_fixed_validation_episodes(data,cfg,seed+701)
    metadata={'protocol':'source_only_fixed_objective_discrete_bank_search_v1','seed':seed,'variant':'bank_evolution_fixed',
              'bank_sha256':_file_sha256(bankpath),'bank_path':str(bankpath),'train_ids':tids,'val_ids':vids,
              'uses_test':False,'uses_gold':False,'alignment':'fixed source sensitivity centroid',
              'population':24,'max_rounds':64,'min_rounds':18,'inner_steps':64,'replicas_for_population':4,
              'proposal':'parent and uniform prior plus 1..4 exact-count swaps or Gumbel top-32 around bank scores',
              'acceptance':'at least .002 train-query BCE gain over paired parent, fixed 4 initializations and source episodes across rounds',
              'selection':'fixed meta-validation query BCE with 4 fresh initializations',
              'stop':'train and val monitor plateau <=1% for two windows8, three passes after18 rounds',
              'inner_solver_converged_claim':False}
    out=root/f'seed_{seed}/bank_evolution_fixed';out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2)+'\n')
    rng=np.random.default_rng(seed+51101);curves=[];best=float('inf');best_round=0;flat_passes=0
    def monitor(mask,step):
        nonlocal best,best_round
        model=FixedProposal(mask)
        tr=_run_validation(model,None,ts,tq,seed+8000,cfg,device)
        va=_run_validation(model,None,vs,vq,seed+9000,cfg,device)
        if float(va['query_balanced_bce'])<best:
            best=float(va['query_balanced_bce']);best_round=step
            _atomic_save({'metadata':metadata,'model_state':{'mask':mask.detach().cpu()},'step':step,
                         'best_val':best,'val':va,'train_monitor':tr,
                         'val_support_ids':vs['ids'],'val_query_ids':vq['ids']},out/'best.pt')
        return float(tr['query_balanced_bce']),float(va['query_balanced_bce'])
    tr,va=monitor(parent,0);curves.append({'round':0,'train_query_bce':tr,'val_query_bce':va,'accepted':False})
    stopping='round_cap'
    for step in range(1,65):
        candidates=[parent,initial]
        p=parent.cpu().numpy().reshape(-1)
        active=np.flatnonzero(p);inactive=np.flatnonzero(1-p)
        for k in range(22):
            if k%5==0:
                z=(score-score.mean())/max(score.std(),.01)
                temp=(.1,.2,.4)[(step+k)%3]
                logits=z.flatten()+temp*rng.gumbel(size=88)
                flat=np.zeros(88,np.float32);flat[np.argpartition(logits,-32)[-32:]]=1
            else:
                n=1+(k%4);flat=p.copy()
                flat[rng.choice(active,n,replace=False)]=0;flat[rng.choice(inactive,n,replace=False)]=1
            candidates.append(torch.as_tensor(flat.reshape(11,8),device=device))
        masks=torch.stack(candidates);n=len(candidates)
        x=ts['x'].to(device).repeat(n,1,1);y=ts['y'].to(device).repeat(n,1)
        qx=tq['x'].to(device).repeat(n,1,1);qy=tq['y'].to(device).repeat(n,1)
        params0=_init_child_batch(10,4,torch.Generator().manual_seed(seed+8000),device)
        params={key:value.detach()[None].expand(n,*value.shape).reshape(n*10,*value.shape[1:]).clone().requires_grad_() for key,value in params0.items()}
        stacked=masks[:,None].expand(-1,10,-1,-1).reshape(n*10,11,8)
        params,_=_inner_adapt(x,y,stacked,params,steps=64,learning_rate=.1,momentum=.9,create_graph=False)
        with torch.no_grad():
            preds=_batched_logits(qx,stacked,params)
            loss=F.binary_cross_entropy_with_logits(preds,qy[:,None,:].expand_as(preds),reduction='none').mean((1,2)).reshape(n,10).mean(1)
            chosen=int(loss.argmin());accepted=float(loss[chosen])+ .002 < float(loss[0])
            if accepted:parent=masks[chosen].clone()
        _atomic_save({'round':step,'masks':masks.cpu(),'candidate_query_bce':loss.cpu(),
                     'selected_candidate':chosen,'accepted':accepted,
                     'adapted_child_params':{k:v.detach().cpu() for k,v in params.items()}},out/f'round_{step:02}.pt')
        tr,va=monitor(parent,step)
        row={'round':step,'train_query_bce':tr,'val_query_bce':va,'accepted':accepted,
             'population_best_query_bce':float(loss.min()),'population_parent_query_bce':float(loss[0]),
             'best_val':best,'best_round':best_round}
        curves.append(row);(out/'curves.json').write_text(json.dumps(curves,indent=2)+'\n')
        print(json.dumps({'seed':seed,**row}),flush=True)
        if step>=18:
            a=bool(loss_plateau(torch.tensor([r['train_query_bce'] for r in curves]),width=8,tolerance=.01))
            b=bool(loss_plateau(torch.tensor([r['val_query_bce'] for r in curves]),width=8,tolerance=.01))
            flat_passes=flat_passes+1 if a and b else 0
            if flat_passes>=3:stopping='empirical_train_val_plateau';break
    _atomic_save({'metadata':metadata,'mask':parent.cpu(),'round':step,'rng_state':rng.bit_generator.state,'curves':curves},out/'last.pt')
    summary={'seed':seed,'variant':'bank_evolution_fixed','rounds':step,'best_round':best_round,'best_val':best,
             'stop_reason':stopping,'empirical_outer_plateau':stopping=='empirical_train_val_plateau',
             'accepted_rounds':sum(r['accepted'] for r in curves),'candidate_evaluations':step*24,'inner_solver_converged_claim':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();run(a.root,a.seed,a.device)

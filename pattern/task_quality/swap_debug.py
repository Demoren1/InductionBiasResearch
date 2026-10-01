"""Matched finite discrete swaps versus local continuous mask hypergradient."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .stripe_debug import OLD
from .core import build_experiment_data
from .generator import Generator
from .meta import MetaConfig,_fixed_train_episodes,_init_child_batch,_inner_adapt,_batched_logits

def run(out,device):
    torch.set_num_threads(1);dev=torch.device(device);seed=8100
    data=build_experiment_data(probe_seed=seed);cfg=MetaConfig()
    tids,s,q=_fixed_train_episodes(data,cfg,seed+601)
    x=s['x'].to(dev);y=s['y'].to(dev);qx=q['x'].to(dev);qy=q['y'].to(dev)
    bank=torch.load(OLD/'seed_8100/bank/bank.pt',map_location='cpu',weights_only=False)
    m=Generator().to(dev);m.load_state_dict(torch.load(OLD/'seed_8100/transformer_mask/meta/best.pt',map_location='cpu',weights_only=False)['model_state'])
    with torch.no_grad():base=m(bank['feature'].to(dev),x,y)[0][0]
    p0=_init_child_batch(10,4,torch.Generator().manual_seed(seed+8000),dev)
    mask=base.expand(10,-1,-1).clone().requires_grad_()
    p={k:v.detach().clone().requires_grad_() for k,v in p0.items()}
    p,_=_inner_adapt(x,y,mask,p,steps=64,learning_rate=.1,momentum=.9,create_graph=True)
    pred=_batched_logits(qx,mask,p)
    loss=F.binary_cross_entropy_with_logits(pred,qy[:,None,:].expand_as(pred))
    grad=torch.autograd.grad(loss,mask)[0].sum(0).flatten()
    rng=np.random.default_rng(2201)
    yes=np.flatnonzero(base.cpu().numpy().reshape(-1));no=np.flatnonzero(1-base.cpu().numpy().reshape(-1))
    pairs=list(zip(rng.choice(yes,64),rng.choice(no,64)))
    variants=[base]
    predicted=[]
    for a,b in pairs:
        v=base.clone().flatten();v[a]=0;v[b]=1;variants.append(v.reshape(11,8));predicted.append(float(grad[b]-grad[a]))
    variants=torch.stack(variants);n=variants.size(0)
    masks=variants[:,None].expand(-1,10,-1,-1).reshape(n*10,11,8)
    p={k:v.detach()[None].expand(n,*v.shape).reshape(n*10,*v.shape[1:]).clone().requires_grad_() for k,v in p0.items()}
    xx=x.repeat(n,1,1);yy=y.repeat(n,1)
    p,_=_inner_adapt(xx,yy,masks,p,steps=64,learning_rate=.1,momentum=.9,create_graph=False)
    with torch.no_grad():
        pred=_batched_logits(qx.repeat(n,1,1),masks,p)
        losses=F.binary_cross_entropy_with_logits(pred,qy.repeat(n,1)[:,None,:].expand_as(pred),reduction='none').mean((1,2)).reshape(n,10).mean(1)
    actual=(losses[1:]-losses[0]).cpu().numpy();predicted=np.array(predicted)
    assert abs(float(losses[0])-float(loss))<2e-6
    sig=(np.abs(actual)>1e-5)&(np.abs(predicted)>1e-7)
    report={'source_only':True,'seed':seed,'mask':'saved_transformer_best','task_ids':tids,'inner_steps':64,'replicas':4,
            'sampled_swaps':64,'base_query_bce':float(losses[0]),'baseline_reproduced_max_difference':abs(float(losses[0])-float(loss)),
            'pearson_local_gradient_vs_discrete_delta':float(np.corrcoef(actual,predicted)[0,1]),
            'sign_agreement_non_negligible':float(((actual[sig]>0)==(predicted[sig]>0)).mean()),
            'non_negligible_swaps':int(sig.sum()),'actual_improving_swaps':int((actual<0).sum()),
            'local_predicts_improving_swaps':int((predicted<0).sum()),
            'interpretation':'Finite binary swaps retrain children from matched initializations; local continuous mask derivative need not predict these non-infinitesimal changes.'}
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(out/'arrays.npz',pairs=np.array(pairs),predicted_delta=predicted,actual_delta=actual,mask_gradient=grad.cpu().numpy(),baseline_mask=base.cpu().numpy())
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args();run(a.out,a.device)

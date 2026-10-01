"""Non-invasive diagnosis of saved stripe masks; never changes old artifacts."""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from .core import build_experiment_data
from .generator import Generator, exact_topk_ste
from .meta import (MetaConfig, _fixed_train_episodes, _fixed_validation_episodes,
                   _run_validation, _file_sha256)

OLD = Path(__file__).resolve().parents[1] / 'outputs/task_quality_toeplitz_20261001'

def additive_stats(logits):
    x=logits.detach().cpu().double()
    centered=x-x.mean((-2,-1),keepdim=True)
    interaction=x-x.mean(-1,keepdim=True)-x.mean(-2,keepdim=True)+x.mean((-2,-1),keepdim=True)
    energy=float(centered.square().sum())
    return {'interaction_energy_fraction':float(interaction.square().sum())/max(energy,1e-30),
            'within_row_variance':float(x.var(-1,unbiased=False).mean()),
            'between_row_variance':float(x.mean(-1).var(-1,unbiased=False).mean()),
            'min':float(x.min()),'max':float(x.max())}

def run(out, device):
    out.mkdir(parents=True,exist_ok=True)
    records=[];arrays={}
    for seed in (8100,8101,8102,8103):
        bank_path=OLD/f'seed_{seed}/bank/bank.pt'
        bank=torch.load(bank_path,map_location='cpu',weights_only=False)
        data=build_experiment_data(probe_seed=seed)
        config=MetaConfig()
        _,sx,qx=_fixed_train_episodes(data,config,seed+601)
        for kind in ('init','best','last'):
            torch.manual_seed(seed+31)
            model=Generator().to(device)
            if kind!='init':
                cp=torch.load(OLD/f'seed_{seed}/transformer_mask/meta/{kind}.pt',map_location='cpu',weights_only=False)
                model.load_state_dict(cp['model_state'])
            with torch.no_grad():
                masks,logits=model(bank['feature'].to(device),sx['x'].to(device),sx['y'].to(device))
            l=logits[0].detach().requires_grad_()
            upstream=torch.linspace(-1,1,88,device=device).reshape(11,8)
            m=exact_topk_ste(l)
            g=torch.autograd.grad((m*upstream).sum(),l)[0]
            shifted=(l.detach()+15).requires_grad_()
            ms=exact_topk_ste(shifted)
            gs=torch.autograd.grad((ms*upstream).sum(),shifted)[0]
            rec={'seed':seed,'checkpoint':kind,'bank_sha256':_file_sha256(bank_path),
                 **additive_stats(logits),
                 'distinct_masks':int(torch.unique(masks.flatten(1),dim=0).size(0)),
                 'row_counts':masks[0].sum(-1).tolist(),
                 'shift_15_changed_edges':int((m.detach()!=ms.detach()).sum()),
                 'surrogate_gradient_norm':float(g.norm()),
                 'surrogate_gradient_shifted_norm':float(gs.norm()),
                 'hidden_query_norm':float(model.decoder_queries.norm(dim=-1).mean()),
                 'position_embedding_norm':float(model.input_position_embeddings.norm(dim=-1).mean())}
            records.append(rec)
            arrays[f'{seed}_{kind}_logits']=logits.cpu().numpy()
            arrays[f'{seed}_{kind}_masks']=masks.cpu().numpy()
    # Same masks, episodes, child RNG, solver: locate finite-horizon ranking.
    seed=8100;bank=torch.load(OLD/'seed_8100/bank/bank.pt',map_location='cpu',weights_only=False)
    model=Generator().to(device)
    cp=torch.load(OLD/'seed_8100/transformer_mask/meta/best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(cp['model_state'])
    data=build_experiment_data(probe_seed=seed)
    config=MetaConfig()
    _,train_s,train_q=_fixed_train_episodes(data,config,seed+601)
    _,val_s,val_q=_fixed_validation_episodes(data,config,seed+701)
    # Oracle is diagnostic only, never passed to a trainable model.
    oracle=torch.tensor([[float(0<=i-j<4) for j in range(8)] for i in range(11)],device=device)
    class FixedMask(torch.nn.Module):
        def __init__(self,mask):super().__init__();self.mask=mask
        def forward(self,bank,x,y):return self.mask.expand(x.size(0),-1,-1),None
    with torch.no_grad():
        stripe=model(bank['feature'].to(device),train_s['x'].to(device),train_s['y'].to(device))[0][0]
    comparisons=[]
    for steps in (64,256,1024):
        cfg=MetaConfig(inner_steps=steps)
        for split,s,q,rs in [('train',train_s,train_q,seed+8000),('val',val_s,val_q,seed+9000)]:
            for label,mask in [('saved_transformer',stripe),('oracle_diagnostic',oracle),('dense',torch.ones_like(oracle))]:
                result=_run_validation(FixedMask(mask),None,s,q,rs,cfg,torch.device(device))
                comparisons.append({'inner_steps':steps,'split':split,'mask':label,
                                    'query_bce':float(result['query_balanced_bce']),
                                    'per_task_bce':result['query_balanced_bce_per_task'].tolist()})
    payload={'protocol':'noninvasive_stripe_debug_v1','source_only':True,
             'original_artifacts_changed':False,'records':records,'matched_solver_comparisons':comparisons}
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    np.savez_compressed(out/'arrays.npz',**arrays)
    print(json.dumps(payload,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();torch.set_num_threads(1);run(a.out,a.device)

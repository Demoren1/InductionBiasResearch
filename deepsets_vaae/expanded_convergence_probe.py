"""Source-only longer-fit diagnostic; never evaluate or select on target tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .expanded_functional_vae import _metric_bundle, _reconstruct
from .followup_common import configure
from .masks import _fit_vae
from .run import write_json

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT/'outputs/deepsets_vaae/20261001_expanded_functional_vae'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    configure(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    source = EXPERIMENT/f'seed_{args.seed}'/'functional'
    arrays_path = source/'functional_vae_arrays.npz'
    with np.load(arrays_path) as archive:
        train_np = archive['function_train_aligned']
        valid_np = archive['function_validation_aligned']
    original = torch.load(source/'functional_vae_artifacts.pt',map_location='cpu',weights_only=False)
    original_states = original['vae_state_dicts']['functional_vae_large']
    device = torch.device('cuda:0')
    results=[]
    for task in range(4):
        train=torch.as_tensor(train_np[task],device=device)
        valid=torch.as_tensor(valid_np[task],device=device)
        model_seed=args.seed+50000+task*1009
        short,short_report=_fit_vae(train,valid,flat_dim=25088,latent=16,width=128,
                                  epochs=160,seed=model_seed,device=device)
        assert all(torch.equal(value.detach().cpu(),original_states[task][key])
                   for key,value in short.state_dict().items()), '160-step replay changed'
        short_metrics=_metric_bundle(_reconstruct(short,valid),valid,train.mean(0),source_k=5018)
        del short
        long,long_report=_fit_vae(train,valid,flat_dim=25088,latent=16,width=128,
                                epochs=1000,seed=model_seed,device=device)
        long_metrics=_metric_bundle(_reconstruct(long,valid),valid,train.mean(0),source_k=5018)
        torch.save({key:value.detach().cpu() for key,value in long.state_dict().items()},
                   args.out/f'functional_large_task{task}_1000.pt')
        results.append({'task':task,'seed':model_seed,'short160':short_report,'long1000':long_report,
                        'short_metrics':short_metrics,'long_metrics':long_metrics,'short_replay_exact':True})
        print(json.dumps({'seed':args.seed,'task':task,'best160':short_report['best_step'],
                          'best1000':long_report['best_step']}),flush=True)
        del long
    write_json(args.out/'results.json', {'seed':args.seed,'tasks':results,
               'input_arrays_sha256':hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
               'scope':'source-only diagnostic; no target labels, evaluation, or variant selection',
               'epochs':[160,1000], 'method':'functional_vae_large','primary160_unchanged':True})
    (args.out/'COMPLETE').write_text('complete\n')


if __name__=='__main__':
    main()

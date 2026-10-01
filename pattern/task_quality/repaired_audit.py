"""Independent NumPy plateau and frozen real-weight scoring audit."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .meta import _file_sha256
from .core import build_test_pool
from .evaluate import binary_metrics
from .repaired_run import SEEDS
from .stripe_debug import OLD


def flat(values):
    x=np.asarray(values,dtype=np.float64)[-16:]
    if len(x)<16:return False
    before,after=x[:8],x[8:]
    denom=max(abs(before.mean()),abs(after.mean()),.01)
    trend=abs((after*np.arange(-3.5,4.5)).sum()/42)*8/denom
    return bool(np.isfinite(x).all() and abs(after.mean()-before.mean())/denom<=.001 and trend<=.001)


def run(root):
    selection=json.loads((root/'regularized_selection.json').read_text());max_delta=0.;runs=0;plateau=0;records_count=0
    reports=[]
    for seed in SEEDS:
        bank_sha=_file_sha256(OLD/f'seed_{seed}/bank/bank.pt')
        for folder_name in ('regularized_eval','evolution_eval'):
            folder=root/f'seed_{seed}'/folder_name
            for stage in ('tune','test'):
                manifest=torch.load(folder/f'{stage}_manifest.pt',map_location='cpu',weights_only=False)
                fit=torch.load(folder/f'{stage}_fit.pt',map_location='cpu',weights_only=False)
                assert bank_sha==manifest['bank_sha256']
                for name,sha in manifest['model_hashes'].items():
                    if name=='legacy':cp=OLD/f'seed_{seed}/transformer_mask/meta/best.pt'
                    else:cp=root/f'seed_{seed}/{name}/best.pt'
                    assert _file_sha256(cp)==sha,(seed,name)
                assert len(fit['specs'])==len(manifest['specs']);assert fit['specs']==manifest['specs']
                assert torch.equal(fit['masks'],manifest['masks'])
                mask=fit['masks'].numpy();assert np.isin(mask,[0,1]).all()
                for i,s in enumerate(fit['specs']):
                    assert mask[i].sum()==(88 if s['method']=='dense' else 32)
                    if stage=='test':assert s['lr']==selection['selection'][s['method']]['lr'] and s['l2']==selection['selection'][s['method']]['l2']
                    if not bool(fit['active'][i]):
                        stop=int(fit['stopping_steps'][i]);h=[r for r in fit['history'] if r['step']<=stop]
                        assert stop>=2000 and len(h)>=18
                        for offset in (0,1,2):
                            hh=h if offset==0 else h[:-offset]
                            for k in ('support_bce','objective','query_bce'):
                                assert flat([float(r[k][i]) for r in hh]),(seed,folder_name,stage,i,k,stop)
                        plateau+=1
                    runs+=1
                for s_key in (('tuning_provenance' if folder_name=='regularized_eval' else 'evolution_tuning_provenance'),):
                    if stage=='tune':assert _file_sha256(folder/'tune_fit.pt')==selection[s_key][str(seed)]
                if stage=='test':
                    frozen=json.loads((folder/'frozen_before_test.json').read_text())
                    assert frozen['selection_sha256']==_file_sha256(root/'regularized_selection.json')
                    assert frozen['child_sha256']==_file_sha256(folder/'test_fit.pt')
                    assert frozen['manifest_sha256']==_file_sha256(folder/'test_manifest.pt')
                    assert frozen['test_labels_materialized'] is False
                    records=json.loads((folder/'records.json').read_text())
                    for i,s in enumerate(fit['specs']):
                        row=next(r for r in records if r['task_id']==s['task_id'] and r['method']==s['method'] and r['init_id']==s['init_id'])
                        pool=build_test_pool(s['task_id'].split(':')[-1])
                        assert row['test_ids']==pool['ids'].tolist()
                        assert len(pool['ids'])==414
                        assert set(pool['ids'].tolist()).isdisjoint(manifest['conditions'][s['task_id']]['support_ids'])
                        assert set(pool['ids'].tolist()).isdisjoint(manifest['conditions'][s['task_id']]['query_ids'])
                        p={k:v[i].numpy().astype(np.float64) for k,v in fit['best_params'].items()}
                        # Pure NumPy reconstruction of actual masked ReLU logits.
                        pred=np.maximum(pool['x'].numpy().astype(np.float64)@(p['w']*mask[i])+p['b'],0)@p['a']+p['c']
                        score=binary_metrics(torch.from_numpy(pred).float(),pool['y'])
                        for key in ('balanced_bce','balanced_accuracy','bce','accuracy'):
                            delta=abs(score[key]-row[key]);max_delta=max(max_delta,delta);assert delta<3e-6,(seed,s,key,delta)
                        assert row['converged']==(not bool(fit['active'][i]));records_count+=1
                reports.append({'seed':seed,'folder':folder_name,'stage':stage,'runs':len(fit['specs']),
                                'plateau':int((~fit['active']).sum()),'cap':int(fit['active'].sum()),'last_step':fit['step']})
    payload={'passed':True,'fitted_runs':runs,'plateau_runs':plateau,'test_records_recomputed':records_count,
             'max_test_metric_difference':max_delta,'checks':['unchanged source banks','unchanged meta checkpoints',
             'frozen hyperparameters and weights','exact-count binary masks','independent NumPy three-pass plateau',
             'support/query/test ID separation','full414-ID test scoring from actual weights'], 'stages':reports}
    (root/'independent_audit.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);run(a.root)

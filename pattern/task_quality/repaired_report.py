"""Scientific plots, numerical archives and observed proposal diagnostics."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .repaired_eval import METHODS,load_models
from .repaired_run import SEEDS
from .core import build_experiment_data,sample_balanced
from .functional_pool import FunctionalPool
from .generator import permute_hidden_columns
from .meta import _file_sha256
from .structure import align_mask_to_gold

LABELS={'legacy':'Исходный Transformer','legacy_fixedmass':'Исправленный top-K',
        'column_set_fixedmass':'Отдельные slots','functional_set_pool_bounded':'Обучаемый pooling',
        'uniform_functional':'Равномерное среднее','dense':'Dense'}
COLORS={'legacy':'#a93b3b','legacy_fixedmass':'#bf791c','column_set_fixedmass':'#8b51a6',
        'functional_set_pool_bounded':'#16799b','uniform_functional':'#22804f','dense':'#555555'}

def diagnostics(root):
    folder=root/'proposal_diagnostics';folder.mkdir(exist_ok=True)
    summaries=[];arrays={}
    for seed in SEEDS:
        bank,models,hashes=load_models(root,seed,'cpu')
        data=build_experiment_data(probe_seed=seed)
        ids=list(data['pools']);rng=torch.Generator().manual_seed(seed+313000)
        episodes=[sample_balanced(data['pools'][tid]['support'],128,rng) for tid in ids]
        x=torch.stack([e['x'] for e in episodes]);y=torch.stack([e['y'] for e in episodes])
        feature=bank['feature']
        hp=torch.stack([torch.randperm(8,generator=rng) for _ in range(960)])
        mp=torch.randperm(960,generator=rng)
        perm=permute_hidden_columns(feature,hp)[mp]
        for name,model in models.items():
            with torch.no_grad():
                mask,score=model(feature,x,y)
                pm,ps=model((perm,model.proposals[mp]) if isinstance(model,FunctionalPool) else perm,x,y)
                lm,ls=model(feature,x,1-y)
            structure=align_mask_to_gold(mask[0].numpy())
            row={'seed':seed,'method':name,'source_only':True,'checkpoint_sha256':hashes[name],
                 'distinct_masks_12_source_contexts':int(torch.unique(mask.flatten(1),dim=0).size(0)),
                 'permutation_changed_edges':int((mask!=pm).sum()),
                 'permutation_logit_max_difference':float((score-ps).abs().max()),
                 'flipped_labels_changed_edges':int((mask!=lm).sum()),
                 'flipped_labels_logit_max_difference':float((score-ls).abs().max()),
                 'iou_first_source_context':structure['iou'],
                 'all_windows_exact_first_source_context':structure['exact_recover_all_windows_once'],
                 'row_counts':mask[0].sum(-1).tolist(),
                 'column_counts':mask[0].sum(-2).tolist()}
            if isinstance(model,FunctionalPool):
                with torch.no_grad():_,_,weights=model.forward_with_weights(feature,x,y)
                entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
                row['effective_maps_exp_entropy_mean']=float(entropy.exp().mean());row['maximum_map_weight']=float(weights.max())
                arrays[f'{seed}_{name}_weights']=weights.numpy()
            summaries.append(row);arrays[f'{seed}_{name}_masks']=mask.numpy();arrays[f'{seed}_{name}_scores']=score.numpy()
            arrays[f'{seed}_{name}_aligned_mask']=structure['aligned_mask']
    (folder/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    np.savez_compressed(folder/'arrays.npz',**arrays)
    return summaries,arrays


def meta_plots(root):
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    diag=json.loads((root/'diagnosis/summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    stages=('init','best','last')
    for seed in (8100,8101,8102,8103):
        rows=[next(r for r in diag['records'] if r['seed']==seed and r['checkpoint']==st) for st in stages]
        axes[0].plot(stages,[r['interaction_energy_fraction'] for r in rows],'o-',label=f'seed {seed}')
        axes[1].plot(stages,[max(r['surrogate_gradient_norm'],1e-10) for r in rows],'o-',label=f'seed {seed}')
    axes[0].set_yscale('log');axes[0].set_title('Взаимодействие позиция × нейрон');axes[0].set_ylabel('Доля энергии в центрированных logits')
    axes[1].set_yscale('log');axes[1].set_title('Исходный sigmoid-surrogate');axes[1].set_ylabel('Норма градиента (нули показаны на 10⁻¹⁰)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(figdir/'original_failure.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    meta_arrays={}
    for n,seed in enumerate(SEEDS):
        for method in ('legacy_fixedmass','column_set_fixedmass','functional_set_pool_bounded'):
            curve=json.loads((root/f'seed_{seed}'/method/'curves.json').read_text())
            steps=np.array([r['step'] for r in curve]);train=np.array([r['train_query_bce'] for r in curve]);val=np.array([r['val_query_bce'] for r in curve])
            axes[n,0].plot(steps,train,'--',color=COLORS[method],alpha=.6)
            axes[n,0].plot(steps,val,label=LABELS[method],color=COLORS[method])
            axes[n,1].plot(steps,[r['surrogate_mean_p_times_1mp'] for r in curve],color=COLORS[method],label=LABELS[method])
            for key in ('train_query_bce','val_query_bce','hypergradient_norm','surrogate_mean_p_times_1mp','interaction_energy_fraction'):
                meta_arrays[f'{seed}_{method}_{key}']=np.array([r[key] for r in curve])
            meta_arrays[f'{seed}_{method}_step']=steps
        axes[n,0].set_title(f'seed {seed}: query BCE после 64 шагов');axes[n,0].set_ylabel('Balanced BCE');axes[n,0].legend(fontsize=8)
        axes[n,1].set_title(f'seed {seed}: мягкие gates не насыщены');axes[n,1].set_ylabel('Среднее p(1−p)')
        for ax in axes[n]:ax.set_xlabel('Outer-шаг');ax.grid(alpha=.2)
    fig.savefig(figdir/'rerun_meta_curves.png',dpi=170);plt.close(fig)
    np.savez_compressed(root/'meta_plot_arrays.npz',**meta_arrays)
    summaries,arrays=diagnostics(root)
    fig,axes=plt.subplots(2,4,figsize=(12,7),layout='constrained')
    for n,seed in enumerate(SEEDS):
        for j,method in enumerate(('legacy','legacy_fixedmass','column_set_fixedmass','functional_set_pool_bounded')):
            ax=axes[n,j];ax.imshow(arrays[f'{seed}_{method}_aligned_mask'],cmap='Blues',vmin=0,vmax=1,interpolation='nearest',aspect='auto')
            row=next(r for r in summaries if r['seed']==seed and r['method']==method)
            ax.set_title(LABELS[method]+f"\nseed {seed}, IoU={row['iou_first_source_context']:.3f}",fontsize=10)
            ax.set_xticks(range(8));ax.set_yticks(range(11));ax.set_xlabel('Нейрон j');ax.set_ylabel('Позиция входа i')
    fig.savefig(figdir/'rerun_masks.png',dpi=170);plt.close(fig)
    (figdir/'captions.md').write_text('''# Пояснения графиков

## original_failure.png

Слева — доля энергии взаимодействия строки и столбца после вычитания обоих главных эффектов и общего среднего из logits исходного decoder; Y логарифмический. По X — initialization, validation-selected best и last checkpoint. Справа — норма backward исходного sigmoid при фиксированном тестовом upstream; точные нули отображены на $10^{-10}$. Четыре линии — четыре исходных seeds. Практически аддитивные оценки и исчезновение surrogate-градиента наблюдались отдельно; этот график не доказывает, что только одна из этих причин вызвала итоговое качество.

## rerun_meta_curves.png

По X — outer-шаг. Слева balanced query BCE после одинаковых 64 inner-SGD шагов: сплошные линии meta-validation, пунктир — fixed train monitor. Seeds показаны раздельно. Это конечный inner-horizon, не сходимость дочерних сетей. Справа — среднее $p(1-p)$ нормированной fixed-mass relaxation, индикатор насыщения gates; это не норма полного гиперградиента. Ненасыщенный surrogate сам по себе не гарантирует изменения маски или её правильности.

## rerun_masks.png

Реальные маски best checkpoint на заранее фиксированном первом source-context (`0001`, 128 support-примеров). По X — hidden-нейрон, по Y — входная позиция; синий означает разрешённую связь, белый — запрещённую. Для изображения столбцы выровнены Hungarian относительно gold только после обучения и выбора checkpoint; oracle не входит в обучение. Каждая маска имеет 32 связи. Полосы сохраняются после исправления градиента и у нового decoder, тогда как pooling над выровненными source-предложениями сохраняет значительно больше оконной структуры. Последний использует явный координатный prior и начинает с равномерного functional baseline; это не открытие окон с нуля.
''')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);meta_plots(a.root)

"""Aggregate three follow-up directions and export fixed-example weight maps."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t

from .followup_common import FOLLOWUP, PILOT, SEEDS
from .run import write_json


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    margin = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
    return {'mean': mean, 'ci95': None if margin is None else [mean-margin, mean+margin],
            'repeat_values': values.tolist()}


def save_figure(fig, out: Path, name: str):
    fig.savefig(out/f'{name}.png', dpi=160, bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def aggregate(direction: Path, require_complete=True):
    payloads = [json.loads(path.read_text()) for path in sorted(direction.glob('seed_*/results.json'))]
    if require_complete and {x['seed'] for x in payloads} != set(SEEDS):
        raise ValueError(f'{direction.name}: expected seeds {SEEDS}, found {[x["seed"] for x in payloads]}')
    per_seed = {}
    audit_rows = []
    for payload in payloads:
        groups = {}
        records = payload['records']
        identities = {(r['task'],r['support_size'],r['method'],r['init']) for r in records}
        if len(identities) != len(records):
            raise ValueError('Duplicate records')
        for r in records:
            groups.setdefault((r['method'],int(r['support_size'])), []).append(float(r['mse']))
        if any(len(values)!=32 for values in groups.values()):
            raise ValueError('Each method/budget must cover all 8 tasks and 4 initializations')
        per_seed[payload['seed']] = {key:float(np.mean(values)) for key,values in groups.items()}
        audits = list((direction/f'seed_{payload["seed"]}').rglob('control_audit.json'))
        if len(audits)!=1:
            raise ValueError(f'Expected one control audit: {audits}')
        audit = json.loads(audits[0].read_text())
        if not audit['passed']:
            raise ValueError(f'Failed control audit {audits[0]}')
        states=list((direction/f'seed_{payload["seed"]}').rglob('target_task*_budget*.pt'))
        expected={(task,budget) for task in range(8) for budget in (32,64,128,256)}
        actual={(int(re.search(r'target_task(\d+)_budget',p.name)[1]),
                 int(re.search(r'_budget(\d+)\.pt',p.name)[1])) for p in states}
        if len(states)!=32 or actual!=expected:
            raise ValueError(f'Missing/duplicate target checkpoints in {direction}/seed_{payload["seed"]}')
        audit={**audit,'target_checkpoints':len(states)}
        audit_rows.append(audit)
    keys = sorted(next(iter(per_seed.values())))
    if any(set(rows)!=set(keys) for rows in per_seed.values()):
        raise ValueError('Method/budget coverage differs between seeds')
    rows = [{'method':method,'support_size':budget,
             **interval([per_seed[seed][(method,budget)] for seed in sorted(per_seed)])}
            for method,budget in keys]
    controls = {'agreement','mean','single_vae','random','dense'}
    new_methods = sorted({method for method,budget in keys}-controls)
    comparisons = []
    for method in new_methods:
        for baseline in ('agreement','random','dense','mean'):
            for budget in (32,64,128,256):
                comparisons.append({'method':method,'baseline':baseline,'support_size':budget,
                                    'direction':'negative favors new method',
                                    **interval([per_seed[seed][(method,budget)]-per_seed[seed][(baseline,budget)]
                                                for seed in sorted(per_seed)])})
    result = {'direction':direction.name,'seeds':sorted(per_seed),'aggregate':rows,
              'comparisons':comparisons,'control_audits':audit_rows,'new_methods':new_methods,
              'inference_unit':'repeat mean across fixed 8 target tasks and 4 initializations'}
    write_json(direction/'transfer_summary.json',result)
    return result


def export_heatmaps(out: Path, results: dict, locations: dict):
    chosen = {}
    for name,result in results.items():
        files = list((locations[name]/'seed_4100').rglob('target_task0_budget256.pt'))
        if len(files)!=1:
            raise ValueError(f'Fixed heatmap checkpoint ambiguous: {files}')
        checkpoint = torch.load(files[0],map_location='cpu',weights_only=False)
        rows = {}
        for method in ['dense','agreement',*result['new_methods']]:
            indices = [i for i,(m,r) in enumerate(zip(checkpoint['method_names'],checkpoint['replica_indices']))
                       if m==method and r==0]
            if len(indices)!=1:
                raise ValueError(f'Missing heatmap {name}/{method}')
            i = indices[0]
            state = checkpoint['state_dict']
            weight = state['weight'][i].numpy()
            mask = state['masks'][i].numpy()
            effective = checkpoint['effective_weight'][i].numpy()
            assert np.array_equal(weight*mask,effective)
            assert np.all(effective[mask==0]==0)
            rows[method] = {'weight':weight,'mask':mask,'effective':effective,
                            'record':checkpoint['records'][i]}
        chosen[name] = rows
    limit = max(float(np.abs(row['effective']).max()) for rows in chosen.values() for row in rows.values())
    metadata = {'selection':{'seed':4100,'task':0,'budget':256,'init':0},
                'selection_rule':'fixed before viewing quality; no column sorting',
                'signed_color_limits':[-limit,limit],'methods':{}}
    for direction,rows in chosen.items():
        destination = out/direction/'weight_heatmaps'
        destination.mkdir(exist_ok=True)
        arrays = {}
        captions = ['# Heatmaps обученных весов','',
                    'Пример выбран заранее: seed 4100, target task 0, бюджет 256 наборов, init 0. '
                    'Цвет — эффективный подписанный вес W×M; неактивные связи равны точно нулю. '
                    'Колонки сохранены в исходном порядке. Во всех направлениях одна цветовая шкала.','']
        for method,row in rows.items():
            slug = re.sub(r'[^a-zA-Z0-9_-]','_',method)
            fig,axes = plt.subplots(1,2,figsize=(9,10),sharey=True)
            for ax,value,title in zip(axes,(rows['dense']['effective'],row['effective']),('Dense W',f'{method}: W × M')):
                im=ax.imshow(value,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto',interpolation='nearest')
                ax.set_title(title);ax.set_xlabel('Hidden neuron (saved order)');ax.set_ylabel('Input pixel: row × 28 + col')
            fig.colorbar(im,ax=axes.tolist(),shrink=.7,label='Effective signed weight')
            save_figure(fig,destination,f'dense_vs_{slug}_weights')
            fig,axes=plt.subplots(4,8,figsize=(13,7))
            for j,ax in enumerate(axes.flat):
                im=ax.imshow(row['effective'][:,j].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                ax.set_title(f'h={j}',fontsize=8);ax.set_xticks([]);ax.set_yticks([])
            fig.suptitle(f'{method}: all 32 effective filters')
            fig.colorbar(im,ax=list(axes.flat),shrink=.65,label='Effective signed weight')
            save_figure(fig,destination,f'{slug}_filters')
            captions += [f'![{method}: dense comparison](dense_vs_{slug}_weights.png)',
                         f'Матрицы 784×32: строки — входные пиксели, колонки — скрытые нейроны. '
                         f'Слева dense, справа {method}. Это один парный пример, не средняя карта.', '',
                         f'![{method}: filters]({slug}_filters.png)',
                         'Каждый квадрат — одна колонка в геометрии 28×28. Показывает расположение '
                         'эффективных связей; без функциональной проверки не доказывает полезность фильтра.','']
            arrays.update({f'{slug}_{key}':row[key] for key in ('weight','mask','effective')})
            metadata['methods'][f'{direction}/{method}'] = row['record']
        np.savez_compressed(destination/'values.npz',**arrays)
        (destination/'README.md').write_text('\n'.join(captions))
    write_json(out/'weight_heatmap_metadata.json',metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=FOLLOWUP)
    parser.add_argument('--ae-dir',type=Path)
    parser.add_argument('--alignment-dir',type=Path)
    parser.add_argument('--importance-dir',type=Path)
    args=parser.parse_args();out=args.out.resolve()
    locations={name:(getattr(args,f'{name}_dir') or out/name).resolve()
               for name in ('ae','alignment','importance')}
    results={name:aggregate(locations[name]) for name in locations}
    write_json(out/'result_locations.json',{name:str(path.relative_to(out))
                                         for name,path in locations.items()})
    fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
    for ax,(direction,result) in zip(axes,results.items()):
        for method in [*result['new_methods'],'agreement','random','dense']:
            rows=[r for r in result['aggregate'] if r['method']==method]
            rows.sort(key=lambda r:r['support_size'])
            x=[r['support_size'] for r in rows];y=[r['mean'] for r in rows]
            ax.plot(x,y,marker='o',label=method)
            ax.fill_between(x,[r['ci95'][0] for r in rows],[r['ci95'][1] for r in rows],alpha=.07)
        ax.set_title(direction);ax.set_xscale('log',base=2);ax.set_xticks([32,64,128,256]);ax.set_xticklabels([32,64,128,256])
        ax.set_xlabel('Total labeled sets (train + validation)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Test MSE / 5 (lower is better)')
    save_figure(fig,out,'transfer_learning_curves')
    fig,axes=plt.subplots(1,3,figsize=(19,5),sharey=True)
    for ax,(direction,result) in zip(axes,results.items()):
        for method in result['new_methods']:
            rows=[r for r in result['comparisons'] if r['method']==method and r['baseline']=='random']
            rows.sort(key=lambda r:r['support_size'])
            y=np.array([r['mean'] for r in rows]);low=np.array([r['ci95'][0] for r in rows]);high=np.array([r['ci95'][1] for r in rows])
            ax.errorbar([r['support_size'] for r in rows],y,yerr=np.stack([y-low,high-y]),marker='o',capsize=3,label=method)
        ax.axhline(0,color='black',lw=1);ax.set_title(direction);ax.set_xscale('log',base=2)
        ax.set_xticks([32,64,128,256]);ax.set_xticklabels([32,64,128,256]);ax.set_xlabel('Total labeled sets');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_ylabel('Paired NMSE difference: new − random')
    save_figure(fig,out,'paired_effects_vs_random')
    export_heatmaps(out,results,locations)
    lines=['# DeepSets: три направления после диагностики','',
           'Все направления используют те же восемь исходных seeds, четыре source-задачи и восемь '
           'фиксированных target-задач. GPU распределены 3+3+2. Значения усредняются по задачам и '
           'инициализациям внутри seed; интервалы вычислены по восьми seeds. Это условная '
           'вариативность обучения на фиксированном наборе задач, а не независимые 256 повторов.','',
           'Каждый target-бюджет включает train и checkpoint validation. Test не используется '
           'для извлечения масок или выбора метода. Все sparse маски имеют 5018 связей. '
           'Численные веса обучаются заново; сохранены все выбранные checkpoints.','',
           'Замена VAE на детерминированные AE не устранила усреднение: разнообразие реконструкций '
           'внутри задачи составляет лишь около 13–20% разнообразия исходных карт. Flat AE и Set AE '
           'уступают random при бюджете 256. Функциональное выравнивание повышает воспроизводимость '
           'сопоставления нейронов, но VAE agreement после него почти не меняет перенос. '
           'Функциональные importance scores полезнее |W| для выбора значимых связей исходной модели '
           'и лучше raw-среднего на новых задачах. Однако ни один новый метод при бюджете 256 '
           'не показал преимущества над random с парным 95% интервалом, лежащим целиком ниже нуля. '
           'Это не доказывает невозможность переноса: проверены конкретные банки, задачи и протокол.','',
           '![Learning curves](transfer_learning_curves.png)','',
           'Ось X — общий бюджет размеченных наборов, Y — test MSE/5; меньше лучше. '
           'Три панели показывают направления, полосы — 95% t-интервалы по seeds.','',
           '![Paired effects](paired_effects_vs_random.png)','',
           'Разность нового метода и random на одинаковых seeds/задачах/инициализациях; '
           'отрицательное значение выгодно новому методу. Интервалы 95%, без коррекции '
           'множественных сравнений; выводы исследовательские.','',
           '[Ускорение вычислений, проверка совпадения и наблюдения загрузки GPU](performance/REPORT.md). '
           'Восемь условий обучаются вместе; очередь допускает два процесса на карту. '
           'На контрольном замере ускорение 1.56×, выбранные веса и test NMSE совпали точно.','']
    for direction,result in results.items():
        lines += [f'## {direction}','',
                  'Парное отличие от random при бюджете 256 наборов. Интервал, включающий ноль, '
                  'не подтверждает преимущество; положительное отличие означает более высокую ошибку.','',
                  '| Метод | NMSE − random | 95% CI |','|---|---:|---|']
        for r in result['comparisons']:
            if r['baseline']=='random' and r['support_size']==256:
                lines.append(f'| {r["method"]} | {r["mean"]:+.6f} | [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}] |')
        lines += ['', '| Метод | Бюджет | NMSE | 95% CI |','|---|---:|---:|---|']
        for r in result['aggregate']:
            lines.append(f'| {r["method"]} | {r["support_size"]} | {r["mean"]:.6f} | [{r["ci95"][0]:.6f}, {r["ci95"][1]:.6f}] |')
        maximum=max(a['max_absolute_mse_delta'] for a in result['control_audits'])
        lines += ['',f'Контроль воспроизведения исходных методов: максимальная абсолютная разница NMSE {maximum:.8g}.',
                  f'[Парные сравнения и значения по seeds]({locations[direction].relative_to(out)}/transfer_summary.json).',
                  f'[Dense и все маски с обученными весами]({direction}/weight_heatmaps/README.md).','']
        documents=sorted(p for p in locations[direction].glob('*.md'))
        for p in documents:
            lines.append(f'[Диагностика направления: {p.name}]({p.relative_to(out)}).')
        lines.append('')
    (out/'REPORT.md').write_text('\n'.join(lines))
    write_json(out/'summary.json',results)
    immutable=json.loads((out/'original_seed_artifact_hashes.json').read_text())
    changed=[name for name,digest in immutable.items() if hashlib.sha256((PILOT/name).read_bytes()).hexdigest()!=digest]
    audit={'original_artifacts_checked':len(immutable),'original_artifacts_changed':changed,
           'completed_directions':list(results),'seeds_per_direction':{n:r['seeds'] for n,r in results.items()},
           'all_control_reproduction_passed':True,'heatmaps_have_exact_masked_zeros':True}
    write_json(out/'audit.json',audit)
    if changed:raise AssertionError(f'Original artifacts changed: {changed}')
    files=[p for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json']
    write_json(out/'artifact_manifest.json',{'files':[{ 'path':str(p.relative_to(out)),
               'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]})
    print(json.dumps({'report':str(out/'REPORT.md'),'directions':list(results),'files':len(files)}))


if __name__=='__main__':
    main()

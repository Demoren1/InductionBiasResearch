"""Rebuild the September 8 report figures from committed evidence, without a GPU.

Run from any directory: python /path/to/repo/paper_plots/build_final_report_figures.py
Existing mask/example figures are verbatim artifact copies listed in manifest.json.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent / 'mds'
DATA = ROOT / 'data' / '2026-09-08'
OUT = ROOT / 'assets' / '2026-09-08'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 140, 'savefig.dpi': 180})
COLORS = ['#176b9a', '#59a5b5', '#c19235', '#8472b5', '#cd7662', '#929a9e', '#258465']

def read(name):
    return json.loads((DATA / name).read_text())

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    svg = OUT / f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines()) + '\n')
    plt.close(fig)

pattern = read('pattern_evidence.json')
meta = read('meta_interpolation.json')['models']
diag = read('meta_diagnosis.json')
motif = read('motif_evidence.json')

# Structural drawings are mathematical examples, not learned supports.
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout='constrained')
for ax, (n, h, k, gap, title) in zip(axes, [
    (8, 8, 4, None, 'Pattern-8: окно длины 4'),
    (16, 16, 3, 4, 'Motif-pair: два окна, gap=4'),
    (32, 32, 5, None, 'Pattern-32: окно длины 5'),
]):
    a = np.zeros((n, h))
    for j in range(h):
        start = j % (n-k+1) if gap is None else j
        for offset in ([0] if gap is None else [0, gap]):
            for t in range(k):
                a[(start+offset+t) % n, j] = 1
    ax.imshow(a, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set(title=title, xlabel='Скрытый нейрон', ylabel='Входная позиция')
    ax.set_xticks([0, h-1], [1, h]); ax.set_yticks([0, n-1], [1, n])
fig.suptitle('Аналитические поддержки: достаточная структура, не результат генератора', fontsize=13)
save(fig, 'task_supports')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
for ax, k in zip(axes, [5, 7]):
    d=pattern['length32_interpolation']['balanced_accuracy_percent'][f'k{k}']
    keys=['cvae', f'wrong_lower_k{k-1}', f'wrong_upper_k{k+1}', 'random', 'ideal']
    vals=[d[key] for key in keys]
    bars=ax.bar(range(5), vals, color=[COLORS[i] for i in [0,1,3,5,6]])
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    ax.set_xticks(range(5), ['CVAE\nверная k', f'Условие\n{k-1}', f'Условие\n{k+1}', 'Random', 'Ideal'])
    ax.set(title=f'Невидимая при обучении длина {k}', ylabel='Balanced test accuracy, %', ylim=(60, 77))
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.suptitle('Pattern-32: полезный перенос маски; преимущества верного условия не видно', fontsize=13)
save(fig, 'pattern32_transfer')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
for ax, regime, title in zip(axes, ['interpolation', 'extrapolation'], ['Интерполяция gaps 5, 8', 'Экстраполяция gaps 3, 4']):
    d=motif['two_decoder_and_task_search']['fresh_mlp_accuracy'][regime]
    keys=['prior42','pair42','single_z_task_loss','gap_mean','ideal']
    bars=ax.bar(range(5), [100*d[k] for k in keys], color=[COLORS[i] for i in [0,1,3,5,6]])
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
    ax.set_xticks(range(5), ['Prior', 'Agreement', 'Task z', 'Gap mean', 'Ideal'])
    ax.set(title=title, ylabel='Hard-mask accuracy, %', ylim=(65, 78))
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.suptitle('Motif-pair: исторический протокол с повторяющимися входами', fontsize=13)
save(fig, 'motif_search')

keys=list(meta)
labels=['1.1M', '4.5M', '9.3M', 'Без k\n9.3M', 'v=40\n18M', 'Random U', 'Ideal U']
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
for ax, k in zip(axes,[5,7]):
    vals=[100*meta[key]['metrics'][f'{k}/2000']['accuracy'] for key in keys]
    bars=ax.bar(range(7), vals, color=COLORS)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    ax.set_xticks(range(7), labels, fontsize=9)
    ax.set(title=f'Длина {k}; 2000 шагов свежего v', ylabel='Balanced test accuracy, %', ylim=(50, 80))
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
fig.suptitle('Meta-pattern: рост числа параметров не дал выигрыша в seed 42', fontsize=13)
save(fig, 'meta_capacity')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout='constrained')
budgets=[20,50,100,500,2000]
for ax, k in zip(axes,[5,7]):
    for key,label,color in [('phase1_small_seed42','Learned U, 1.1M',COLORS[0]),('baseline_random_seed42','Random U',COLORS[5]),('baseline_ideal_seed42','Analytic U + learned v',COLORS[6])]:
        ax.plot(budgets,[100*meta[key]['metrics'][f'{k}/{b}']['accuracy'] for b in budgets], 'o-', label=label, color=color)
    ax.axvline(50, linestyle=':', color='#555555', alpha=.6)
    ax.set(xscale='log', xticks=budgets, xticklabels=budgets, ylim=(48,80), title=f'Невидимая длина {k}', xlabel='Шаги обучения v (лог. шкала)', ylabel='Balanced test accuracy, %')
    ax.grid(alpha=.2); ax.legend(fontsize=9)
fig.suptitle('U заморожена: быстрый старт learned U и больший запас analytic U', fontsize=13)
save(fig, 'meta_adaptation')

fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
ax=axes[0]; x=np.arange(3)
for basis,color,shift,label in [('learned',COLORS[0],-.19,'Learned U'),('ideal',COLORS[6],.19,'Analytic U')]:
    vals=[100*r['query_accuracy'] for r in diag['fixed_u_2000_steps'] if r['basis']==basis]
    bars=ax.bar(x+shift, vals, .38, color=color, label=label)
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)
ax.set_xticks(x,['Adam .1\nсреднее 4', 'Лучший\nиз 4 стартов', 'Лучший\nиз 20 траекторий'])
ax.set(ylim=(65,100), ylabel='Known-validation query accuracy, %', title='Замороженная U; 2000 шагов v')
ax.legend(fontsize=9); ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
ax=axes[1]
for i,key in enumerate(['before','h200','h500']):
    d=diag['paired_bootstrap_vs_h50'][key]; val=100*d['candidate_minus_h50_query_accuracy']; lo,hi=[100*v for v in d['ci95']]
    ax.errorbar(val, i, xerr=[[val-lo],[hi-val]], fmt='o', color=COLORS[i], capsize=5)
    ax.text(hi+.018,i,f'{val:+.2f}',va='center',fontsize=10)
ax.axvline(0,color='#666666',linestyle=':')
ax.set_yticks(range(3),['Без дообучения','Горизонт 200','Горизонт 500'])
ax.set(xlim=(-.5,.37), ylim=(-.6,2.6), xlabel='Изменение относительно горизонта 50, п.п.',title='50 внешних обновлений; paired 95% CI')
ax.grid(axis='x',alpha=.2)
fig.suptitle('Диагностика 7 сентября: 12 задач известных длин; test и длины 5/7 не использованы', fontsize=12)
save(fig, 'meta_diagnosis')
print(f'Rebuilt 6 figures (PNG + SVG) in {OUT}')

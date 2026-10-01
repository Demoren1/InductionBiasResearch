"""Finalize preserved numerical results, scientific plots and Russian report."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .repaired_report import LABELS,COLORS,meta_plots
from .repaired_eval import METHODS
from .repaired_run import SEEDS
from .structure import align_mask_to_gold
from .meta import _file_sha256

ALL=METHODS+('bank_evolution_fixed',)
LABELS['bank_evolution_fixed']='Дискретный поиск'
COLORS['bank_evolution_fixed']='#4864b2'

def finalize(root):
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    rows=[];fits={};manifests={};structure=[]
    for seed in SEEDS:
        for folder in ('regularized_eval','evolution_eval'):
            p=root/f'seed_{seed}'/folder
            rows.extend(json.loads((p/'records.json').read_text()))
            fits[(seed,folder)]=torch.load(p/'test_fit.pt',map_location='cpu',weights_only=False)
            manifests[(seed,folder)]=torch.load(p/'test_manifest.pt',map_location='cpu',weights_only=False)
    assert len(rows)==224
    tasks=sorted({r['task_id'] for r in rows})
    dense={t:np.mean([r['balanced_bce'] for r in rows if r['task_id']==t and r['method']=='dense']) for t in tasks}
    summary={}
    for method in ALL:
        found=[r for r in rows if r['method']==method]
        bce=float(np.mean([r['balanced_bce'] for r in found]))
        tm={t:float(np.mean([r['balanced_bce'] for r in found if r['task_id']==t])) for t in tasks}
        masks=[]
        for row in found:
            audit=align_mask_to_gold(np.array(row['mask']))
            structure.append({k:row[k] for k in ('seed','task_id','method','init_id')} | {'iou':audit['iou'],'exact_windows':audit['exact_recover_all_windows_once']})
            masks.append(audit['iou'])
        summary[method]={'test_balanced_bce':bce,'test_balanced_accuracy':float(np.mean([r['balanced_accuracy'] for r in found])),
                         'per_task_bce':tm,'tasks_improved_vs_dense':int(sum(tm[t]<dense[t] for t in tasks)),
                         'mean_iou':float(np.mean(masks)),'records':len(found),
                         'converged_runs':sum(r['converged'] for r in found)}
    output={'protocol':'regularized_child_v1','seeds':list(SEEDS),'support':128,'test_tasks':tasks,'methods':summary,
            'fresh_test_split':False,'task_orbits':1,'selection':'meta-val hyperparameters, target query checkpoints',
            'independent_audit':json.loads((root/'independent_audit.json').read_text())}
    (root/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    (root/'records.json').write_text(json.dumps(rows,indent=2)+'\n')
    (root/'structure.json').write_text(json.dumps(structure,indent=2)+'\n')
    np.savez_compressed(root/'summary_arrays.npz',method=np.array(ALL),task=np.array(tasks),
                       bce=np.array([[summary[m]['per_task_bce'][t] for t in tasks] for m in ALL]),
                       accuracy=np.array([summary[m]['test_balanced_accuracy'] for m in ALL]),
                       mean_iou=np.array([summary[m]['mean_iou'] for m in ALL]))
    # Paired per-task BCE values: plotted results are all repeats, not selected examples.
    fig,ax=plt.subplots(figsize=(11.8,5.2),layout='constrained')
    xs=np.arange(4)
    for method in ALL:
        ax.plot(xs,[summary[method]['per_task_bce'][t] for t in tasks],'-o',label=LABELS[method],color=COLORS[method])
    ax.set_xticks(xs,[t.split(':')[-1] for t in tasks]);ax.set_xlabel('Отложенный pattern');ax.set_ylabel('Test balanced BCE, меньше лучше')
    ax.set_title('Новые свежие сети · support 128 · 2 seed-банка × 4 инициализации');ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9)
    fig.savefig(figdir/'per_task_quality.png',dpi=170);plt.close(fig)
    # Masks and signed actual weights, with the SAME post-hoc column ordering.
    fig,axes=plt.subplots(2,7,figsize=(16,6.4),layout='constrained')
    example_arrays={};lim=0.
    for method in ALL:
        folder='evolution_eval' if method=='bank_evolution_fixed' else 'regularized_eval'
        fit=fits[(8100,folder)];i=next(i for i,s in enumerate(fit['specs']) if s['method']==method and s['task_id']=='k4:0010' and s['init_id']==0)
        mask=fit['masks'][i].numpy();order=np.array(align_mask_to_gold(mask)['gold_to_candidate'])
        weff=fit['best_params']['w'][i].numpy()*mask
        example_arrays[method+'_mask']=mask;example_arrays[method+'_weff']=weff
        example_arrays[method+'_posthoc_order']=order
        example_arrays[method+'_a']=fit['best_params']['a'][i].numpy()
        example_arrays[method+'_b']=fit['best_params']['b'][i].numpy()
        example_arrays[method+'_c']=fit['best_params']['c'][i].numpy()
        lim=max(lim,float(abs(weff).max()))
    for j,method in enumerate(ALL):
        order=example_arrays[method+'_posthoc_order']
        mask=example_arrays[method+'_mask'][:,order];weight=example_arrays[method+'_weff'][:,order]
        axes[0,j].imshow(mask,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
        im=axes[1,j].imshow(weight,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto',interpolation='nearest')
        axes[0,j].set_title(LABELS[method],fontsize=10)
        for ax in axes[:,j]:ax.set_xticks([0,3,7]);ax.set_yticks([0,5,10]);ax.set_xlabel('Нейрон j')
        axes[0,j].set_ylabel('Вход i' if j==0 else '');axes[1,j].set_ylabel('Вход i' if j==0 else '')
    fig.colorbar(im,ax=list(axes[1]),label='Фактический W ⊙ M',shrink=.85)
    fig.suptitle('Выбранный заранее пример: seed 8100 · pattern 0010 · init 0 · сверху M, снизу реальные веса',fontsize=12)
    fig.savefig(figdir/'selected_masks_and_weights.png',dpi=170);plt.close(fig)
    np.savez_compressed(root/'selected_example.npz',**example_arrays)
    # Keep curve means honest: exclude frozen rows after their stopping steps.
    fig,axes=plt.subplots(3,1,figsize=(11.5,8),layout='constrained',sharex=True)
    history_arrays={}
    for method in ALL:
        grouped={}
        for (seed,folder),fit in fits.items():
            idx=[i for i,s in enumerate(fit['specs']) if s['method']==method]
            if not idx:continue
            for row in fit['history']:
                step=row['step'];valid=[i for i in idx if int(fit['stopping_steps'][i])>=step]
                if not valid:continue
                d=grouped.setdefault(step,{'support':[],'query':[],'objective':[]})
                d['support'].extend(row['support_bce'][valid].tolist());d['query'].extend(row['query_bce'][valid].tolist());d['objective'].extend(row['objective'][valid].tolist())
        steps=sorted(grouped);count=[len(grouped[s]['query']) for s in steps]
        for key in ('support','query','objective'):
            history_arrays[method+'_'+key]=np.array([np.mean(grouped[s][key]) for s in steps])
        history_arrays[method+'_steps']=np.array(steps);history_arrays[method+'_count']=np.array(count)
        axes[0].plot(steps,history_arrays[method+'_support'],color=COLORS[method],label=LABELS[method])
        axes[1].plot(steps,history_arrays[method+'_query'],color=COLORS[method])
        axes[2].plot(steps,count,color=COLORS[method])
    axes[0].set_ylabel('Support balanced BCE');axes[1].set_ylabel('Query balanced BCE');axes[2].set_ylabel('Число ещё наблюдаемых сетей');axes[2].set_xlabel('Шаг Adam')
    axes[0].legend(ncol=3,fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Полные траектории: mean учитывает только сети до их собственной остановки')
    fig.savefig(figdir/'child_convergence.png',dpi=170);plt.close(fig)
    np.savez_compressed(root/'child_plot_arrays.npz',**history_arrays)
    # Real swaps versus the local surrogate.
    swap=np.load(root/'diagnosis/discrete_swaps/arrays.npz')
    fig,ax=plt.subplots(figsize=(6.5,5),layout='constrained')
    ax.scatter(swap['predicted_delta'],swap['actual_delta'],s=25,alpha=.75)
    ax.axhline(0,color='grey',lw=1);ax.axvline(0,color='grey',lw=1)
    ax.set_xlabel('Изменение BCE по локальному гиперградиенту');ax.set_ylabel('Измеренное изменение BCE после замены связи')
    ax.set_title('64 парные замены · source-only · 4 инициализации');ax.grid(alpha=.2)
    fig.savefig(figdir/'discrete_swap_diagnostic.png',dpi=170);plt.close(fig)
    # Fixed-objective discrete search versus baseline at round zero.
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    evolution_arrays={}
    for ax,seed in zip(axes,SEEDS):
        c=json.loads((root/f'seed_{seed}/bank_evolution_fixed/curves.json').read_text())
        rounds=np.array([r['round'] for r in c])
        for key,label in [('train_query_bce','Fixed train query'),('val_query_bce','Meta-validation query')]:
            values=np.array([r[key] for r in c]);ax.plot(rounds,values,'o-',ms=3,label=label);evolution_arrays[f'{seed}_{key}']=values
        evolution_arrays[f'{seed}_round']=rounds
        ax.set_title(f'seed {seed}');ax.set_xlabel('Раунд дискретного поиска');ax.set_ylabel('Balanced BCE после 64 SGD-шагов');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(figdir/'discrete_search_curves.png',dpi=170);plt.close(fig)
    np.savez_compressed(root/'evolution_plot_arrays.npz',**evolution_arrays)
    # Preserve primary source snapshots for all direct jobs too.
    snap=root/'source_final';snap.mkdir(exist_ok=True);hashes={}
    for source in Path(__file__).parent.glob('*.py'):
        raw=source.read_bytes();(snap/source.name).write_bytes(raw);hashes[source.name]=hashlib.sha256(raw).hexdigest()
    (snap/'SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    captions=(figdir/'captions.md').read_text()
    captions+='''
## per_task_quality.png

По X — четыре прежние отложенные pattern-задачи, по Y — test balanced BCE на всех 414 test-входах, меньше лучше. Каждая точка — среднее двух source-банков и четырёх свежих инициализаций. Все методы независимо получили одну и ту же LR/L2-сетку; выбор по meta-validation, без новых test-метрик. Линии соединяют разные задачи только для сравнения методов, не изображают временную траекторию. Ошибки равномерного functional mean, ограниченного pooling и дискретного поиска точечно ниже dense на всех четырёх задачах. Это исследовательский повтор ранее использованного test-split; четыре задачи относятся к одной орбите, два seeds недостаточны для общего статистического утверждения.

## selected_masks_and_weights.png

Заранее заданный пример: seed 8100, pattern `0010`, support=128, init=0. Сверху бинарная маска, синий — разрешённая связь, белый — запрещённая. Снизу реальные подписанные $W\odot M$ query-selected checkpoint; красный — положительный вес, синий — отрицательный, белый — нуль; общая симметричная шкала и colorbar. X — hidden-нейрон, Y — входная координата. Столбцы каждой сети одинаково переставлены в обоих рядах по post-hoc Hungarian-сопоставлению маски с gold; при обучении и выборе checkpoint gold не используется. Веса не усреднялись и не заменялись шаблоном. Правильная или близкая к правильной поддержка не означает точного повторения одного signed ядра.

## child_convergence.png

По X — шаг нового full-batch Adam. Сверху mean support BCE, посередине mean query BCE, снизу число сетей, ещё входящих в среднее. Для каждой сети исключены точки после её собственной остановки; поэтому позднее среднее меняет состав, это показано нижней панелью. Фактические индивидуальные кривые и regularized objective сохранены в fit-checkpoints. У всех 224 итоговых траекторий support BCE, objective и query BCE прошли независимую трёхкратную plateau-проверку, но это не доказательство глобального минимума или нулевого градиента. Качество на test оценивается по лучшему query-checkpoint, который может предшествовать plateau; полные последние состояния также сохранены.

## discrete_swap_diagnostic.png

Каждая точка — удаление одной разрешённой связи и добавление одной запрещённой: сохраняются 32 связи. X — линейное предсказание изменения query BCE по локальному continuous mask hypergradient. Y — реальное изменение после полного повторения 64 inner-SGD шагов с одинаковыми support/query и четырьмя парными инициализациями на 10 source-задачах. Отрицательные значения означают улучшение. Маска — сохранённые полосы Transformer seed 8100. Корреляция −0,137, совпадение знака 32,8%; это ограниченная проверка конкретной маски и solver, а не универсальное утверждение о всех STE.

## discrete_search_curves.png

По X — раунд поисковой процедуры с population=24, по Y — query BCE после 64 SGD-шагов. Train-эпизоды и четыре initialization seeds фиксированы, поэтому train objective сравним между раундами; validation выбирает checkpoint отдельно. Раунд 0 — равномерное functional mean. Маска изменяется только при измеренном train-query улучшении минимум 0,002. Поиск остановился по эмпирическому train/validation plateau; отсутствие найденного улучшения в конечной population не доказывает глобального или полного локального оптимума. Возможен overfit фиксированных source-инициализаций; финальное сравнение использует другие свежие seeds.
'''
    (figdir/'captions.md').write_text(captions)
    table='\n'.join(f"| {LABELS[m]} | {summary[m]['test_balanced_bce']:.4f} | {summary[m]['test_balanced_accuracy']:.4f} | {summary[m]['tasks_improved_vs_dense']}/4 | {summary[m]['mean_iou']:.3f} |" for m in ALL)
    pertable='\n'.join('| '+t.split(':')[-1]+' | '+' | '.join(f"{summary[m]['per_task_bce'][t]:.4f}" for m in ('dense','functional_set_pool_bounded','uniform_functional','bank_evolution_fixed'))+' |' for t in tasks)
    body=fr'''# Pattern: диагностика Transformer и повторная проверка моделей

Дата: 1 октября 2026 года. Банк состоит из **полных функциональных карт и state сетей**. Новые обучаемые модели не восстанавливают карты через reconstruction: их objective — query-качество свежей сети после обучения с предложенной маской. Матрица $U$ не используется.

**Исправление градиента top-K оказалось необходимо, но недостаточно.** И прежний decoder с исправленным surrogate, и новый Set Transformer со slots снова дали полосы. Более подходящее ограничение пространства предложений — positive pooling выровненных функциональных карт банка — даёт test BCE **{summary['functional_set_pool_bounded']['test_balanced_bce']:.4f}** против **{summary['dense']['test_balanced_bce']:.4f}** у заново настроенного dense; точечно лучше на 4/4 задачах. Однако **равномерное functional mean ещё лучше: {summary['uniform_functional']['test_balanced_bce']:.4f}**. Следовательно, отдельное преимущество обучения pooling над простым средним не показано. Дискретный поиск также улучшает dense, но не обходит среднее.

Это новый full-batch L2-протокол с двумя seeds и четырьмя свежими инициализациями на задачу. Его числа нельзя напрямую смешивать с прежней четырёхseedовой таблицей minibatch-Adam. На meta-validation dense выиграл от дополнительного L2-подбора; улучшение его качества на test относительно прежнего протокола отдельно не установлено.

## Где локализована проблема

**Наблюдалось:** у best checkpoints исходного decoder доля взаимодействия строки и столбца после удаления их главных эффектов — порядка $10^{{-7}}$–$10^{{-6}}$. Сами scores почти аддитивны, поэтому top-K предпочитает целые строки. В last checkpoints всех четырёх исходных seeds sigmoid-surrogate имеет практически нулевой backward. Прибавление одной константы ко всем scores сохраняет маску, но может обнулить выбранный исходный градиент — воспроизводимый дефект surrogate.

**Наблюдалось:** при source-only контроле на seed 8100 правильные окна дают query BCE 0,5756 на train-monitor и 0,3694 на meta-validation после тех же 64 SGD-шагов; полосы — 0,6514 и 0,5656. Следовательно, хорошие ограничения различимы в исходном finite-horizon objective. Простое увеличение горизонта без изменения обучения не решает всё: при 1024 SGD-шагах query-ошибка растёт, особенно у dense.

**Наблюдалось:** для 64 реальных парных замен связей в полосатой маске локальный гиперградиент совпал со знаком измеренного изменения query BCE в 32,8% случаев; корреляция −0,137. Реально улучшали objective 58/64 замен, локально предсказывались улучшения лишь для 17/64. Совпадение baseline при повторном батчевом расчёте — точное в данном запуске. Это показывает плохое соответствие continuous-градиента конечным дискретным изменениям именно в этой проверке.

**Вывод с ограничением:** увеличить decoder или убрать насыщение недостаточно для текущей схемы обучения. Проектировать следующий метод нужно с учётом реального эффекта дискретного изменения связей и сохранения bank-derived структуры. Проверка swaps относится к одному seed, одной маске и конечному solver; универсальная непригодность STE не доказана. Общая маска для разных pattern-задач сама по себе допустима: полезная оконная структура тоже общая.

![Диагностика исходного decoder](figures/original_failure.png)

![Локальный градиент и реальные замены связей](figures/discrete_swap_diagnostic.png)

Объяснения осей, шкал и границ вывода — в [подписях графиков](figures/captions.md). Численные проверки: [исходная диагностика](diagnosis/summary.json), [дискретные замены](diagnosis/discrete_swaps/summary.json).

## Что перезапущено

1. **Исправленный прежний Transformer:** forward — тот же binary global top-32; backward — нормированная fixed-mass sigmoid-relaxation с implicit threshold derivative. Масса мягких gates — 32; общий сдвиг logits не меняет их градиент. STE остаётся приближением дискретной задачи. CPU gradcheck и проверка shift-invariance прошли.
2. **Column-preserving Set Transformer:** отдельные teacher-column tokens, восемь attention-slots и явные взаимодействия position × slot в decoder. Нет оконного шаблона или weight sharing. Архитектурная основа — [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html); это наша адаптация. Полосы вернулись после обучения на двух seeds.
3. **Обучаемый функциональный pooling:** модель читает полные 187-признаковые tokens, включая 128 значений $\psi_j(x)$, sensitivities, state и качество source-решения. Она выдаёт положительные веса source-картам и агрегирует их выровненные $\mathbb E|q_{{ij}}(x)|$. Итоговая маска — top-32 этого агрегата. Координатное centroid-выравнивание задано явно; обучаемой модели не приписываем его открытие. Инициализация — прежнее равномерное среднее.
4. **Дискретный поиск ограничений:** старт из того же функционального среднего; population из 24 exact-32 масок, парные изменения 1–4 связей или Gumbel top-K вокруг source-оценок. Реальное улучшение source-query после нового обучения сети определяет принятие маски. Гиперградиент через маску не используется. Метод формирует общий переносимый prior, не условную маску по новой задаче.

Неограниченный attention-pooling иногда концентрировал вес на одном teacher и терял градиент. Финальный pooling центрирует и нормирует scores и ограничивает их диапазон: в best checkpoints effective число source-карт по entropy — около 474 и 670 из 960. Ограничение поддерживает использование множества решений. Старый unbounded-вариант сохранён отдельно: один fit дошёл до cap, это не объявляется сходимостью.

Первый дискретный поиск со свежими случайными initialization seeds в каждом раунде часто принимал изменения, не дававшие стабильного улучшения fixed-monitor, и оба запуска дошли до cap. Для финального `bank_evolution_fixed` source objective зафиксирован на четырёх парных initialization seeds и одинаковых эпизодах; оба запуска вышли на plateau. Это снижает шум сравнения, но допускает overfit этих фиксированных инициализаций. Последующая оценка использует новые seeds.

![Кривые обучаемых моделей](figures/rerun_meta_curves.png)

![Маски перезапусков](figures/rerun_masks.png)

![Дискретный поиск](figures/discrete_search_curves.png)

## Результаты на прежних отложенных задачах

Support=128; 32 связи у всех sparse-методов, 88 у dense. Каждому методу независимо подобраны LR и L2 по одинаковой сетке на meta-validation; test-метрики не используются при подборе. Четыре прежних test-задачи принадлежат одной reversal/complement-орбите. По 32 свежих сети на метод: 2 source-банка × 4 задачи × 4 инициализации. IoU — post-hoc Hungarian относительно правильной поддержки, без участия в обучении или выборе моделей.

| Метод | Test balanced BCE ↓ | Test balanced accuracy ↑ | Задач лучше dense | Mean IoU |
|---|---:|---:|---:|---:|
{table}

| Pattern | Dense | Обучаемый pooling | Равномерное среднее | Дискретный поиск |
|---|---:|---:|---:|---:|
{pertable}

![Качество на каждой задаче](figures/per_task_quality.png)

![Бинарные маски и фактические веса, включая dense](figures/selected_masks_and_weights.png)

В этой проверке **обучаемое взвешивание улучшило равномерное среднее только на одной из четырёх задач** (`0010`); в среднем оно хуже на 0,0143 BCE. Ни один новый обучаемый вариант не восстановил точную поддержку всех окон на выбранных best checkpoints. Равномерный baseline сохранил точные окна в двух использованных банках. Близкая к правильной поддержка не означает, что actual signed weights стали точной теплицевой матрицей.

## Сходимость и воспроизводимость

**Все 1232 validation/test-траектории дочерних сетей прошли установленный empirical plateau**, включая все 224 финальные сети. Проверяются support BCE, regularized objective и query BCE: три последовательных проверки двух окон по 8 оценок, изменение среднего и тренд ≤0,1%, minimum 2000 steps. После первичного cap 16 000 активные validation-запуски продолжены с дополнительным понижением LR; ранние состояния и cap-статусы сохранены в `initial_regularized_tune_16000`. Это эмпирическая стабилизация кривых при данном LR schedule, **не доказательство нулевого градиента, глобального оптимума или оптимальности маски**.

Качество test относится к лучшему query-checkpoint; такой checkpoint может предшествовать plateau. Полные последние состояния и индивидуальные кривые сохранены отдельно. Для финального сравнения выполнено 1008 validation-кандидатов и 224 test-фита; каждый test оценивается на всех 414 входах. Восемь основных outer-процедур (две пары Transformer, bounded pooling и fixed discrete search) прошли свои source train/validation plateau-критерии. Negative диагностические запуски с cap показаны отдельно выше.

Независимая NumPy-проверка всех трёх plateau-прохождений для всех 1232 trajectories и пересчёт 224 test-метрик из реальных $W,M,b,a,c$ прошли. Максимальная разница метрик — $1{{,}}19\times10^{{-7}}$. Проверены hashes source-банков и model checkpoints, frozen LR/L2 и final-child states, разделение support/query/test IDs и exact число связей. Сравнение vectorized Adam с обычным независимым Adam дало максимальную разницу параметров $1{{,}}19\times10^{{-7}}$ на CPU-тесте.

![Кривые финальных дочерних сетей](figures/child_convergence.png)

[Независимая проверка](independent_audit.json) · [Зафиксированные параметры](regularized_selection.json) · [Точный протокол](PROTOCOL_RU.md) · [Все результаты](records.json) · [Численные массивы](summary_arrays.npz) · [Пояснения графиков](figures/captions.md).

## Ограничения и следующий осмысленный шаг

- Только два seeds и четыре test-задачи одной орбиты; test-split уже анализировался раньше. Это exploratory повтор, не новый независимый тест универсального переноса.
- Source-bank содержит решения без подтверждённого plateau. В этом опыте банк зафиксирован, его переобучение не проводилось.
- Coordinate alignment у functional mean, pooling и discrete search — явный prior. Отдельный вклад функционального сигнала сверх этого правила ещё требует null-контроля с тем же выравниванием.
- Outer objective — 64 SGD-шагов без L2, финальное обучение — full-batch Adam с L2. Solver mismatch сохранился; validation-выбор пары задач `0000`, `1111` может плохо представлять другие задачи. Ненасыщенный surrogate не устраняет mismatch дискретного forward и continuous backward.
- Простое mean остаётся лучшим по средней BCE. **Приоритет следующего метода — реальная query-оценка изменений bank-derived ограничений с более согласованным inner solver и разнообразными validation-задачами**, а не увеличение генератора ради генеративности. Новое обучение должно показать выигрыш сверх этого сильного functional baseline, а не только над dense.

Все новые результаты и source snapshots сохранены отдельно от исходного опыта. GPU-запуски завершены.
'''
    (root/'RESULTS_RU.md').write_text(body)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);finalize(a.root)

"""Saved scientific figures and captions for the completed mask-transfer pilot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t


def save_figure(fig, out: Path, name: str) -> None:
    fig.savefig(out / f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def extra_figures(out: Path, paths: list[Path], comparisons: list[dict], labels: dict) -> list[str]:
    """Save plots and their exact plotted values; return Markdown explanations."""
    lines = ['## Графики и пояснения', '',
             'Все графики сохранены в PNG и PDF. Для дополнительных графиков исходные '
             'агрегированные значения сохранены в `figure_data.json`; полные результаты '
             'и параметры каждого повтора — в его каталоге `seed_*`.', '',
             '### Качество на новых задачах', '',
             '![Кривые качества](learning_curves.png)', '',
             '**Как читать.** По горизонтали — полный бюджет размеченных наборов новой задачи, '
             'включая validation; каждый набор содержит пять изображений. По вертикали — '
             'средняя тестовая MSE, делённая на пять; меньше лучше. Сначала усредняются '
             'восемь задач и четыре обучения внутри повтора, затем независимые повторы. '
             'Полосы — 95% t-интервалы по повторам при фиксированном разбиении задач.', '',
             '**Вывод.** Agreement лучше средней карты, но его кривая близка к одной VAE и '
             'случайной маске. Этот график сам по себе не доказывает дополнительную пользу agreement.', '']
    data = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for ax, baseline in zip(axes.flat, ('mean', 'single_vae', 'random', 'dense')):
        rows = sorted((row for row in comparisons if row['baseline'] == baseline),
                      key=lambda row: row['support_size'])
        x = np.array([row['support_size'] for row in rows])
        y = np.array([row['mean'] for row in rows])
        bounds = [row['ci95'] for row in rows]
        if all(bound is not None for bound in bounds):
            errors = np.array([[v - b[0], b[1] - v] for v, b in zip(y, bounds)]).T
            ax.errorbar(x, y, yerr=errors, marker='o', capsize=4)
        else:
            ax.plot(x, y, marker='o')
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title('Agreement minus ' + labels[baseline])
        ax.set_xscale('log', base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_ylabel('Paired test MSE difference')
        ax.set_xlabel('Total labeled sets')
        ax.grid(alpha=.2)
    fig.tight_layout()
    save_figure(fig, out, 'paired_effects')
    data['paired_effects'] = comparisons
    lines += ['### Парные различия между методами', '',
              '![Парные эффекты](paired_effects.png)', '',
              '**Как читать.** Каждая панель сравнивает agreement с одним baseline. '
              'По горизонтали — бюджет меток; по вертикали — MSE(agreement) − MSE(baseline). '
              'Отрицательные значения выгодны agreement. Отрезки — 95% t-интервалы '
              'парных различий по независимым повторам; это точечные интервалы без '
              'поправки на множество бюджетов.', '',
              '**Вывод.** Разница со средней картой отрицательна на всех бюджетах. '
              'Интервалы сравнения с одной VAE включают ноль. Относительно random небольшой '
              'выигрыш наблюдается при 32 наборах, но не подтверждается на остальных бюджетах.', '']
    diagnostics_paths = [path.parent / 'mask_diagnostics.json' for path in paths]
    if all(path.exists() for path in diagnostics_paths):
        diagnostics = [json.loads(path.read_text()) for path in diagnostics_paths]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        for ax, metric, title in zip(axes, ('soft_loss', 'hard_iou', 'softness'),
                                     ('Soft decoder disagreement', 'Hard-mask IoU', 'Softness: mean p(1-p)')):
            values = np.array([[np.mean(d['agreement'][metric + '_' + moment])
                                for moment in ('initial', 'final')] for d in diagnostics])
            for row in values:
                ax.plot([0, 1], row, color='steelblue', alpha=.4, marker='o')
            ax.plot([0, 1], values.mean(0), color='black', linewidth=2.5, marker='o', label='Mean')
            ax.set_xticks([0, 1], ['Before', 'After'])
            ax.set_title(title)
            ax.grid(alpha=.2)
            if metric == 'soft_loss':
                ax.set_yscale('log')
            if metric == 'softness':
                density = diagnostics[0]['density']
                ax.axhline(density * (1 - density), linestyle='--', color='darkorange',
                           label='Uniform-mask maximum')
            ax.legend(fontsize=8)
            data[metric] = values.tolist()
        fig.tight_layout()
        save_figure(fig, out, 'agreement_diagnostics')
        lines += ['### Что меняет оптимизация agreement', '',
                  '![Диагностика agreement](agreement_diagnostics.png)', '',
                  '**Как читать.** Линии показывают каждый независимый повтор до и после поиска '
                  'латентных кодов; внутри повтора усредняются четыре старта. Слева — мягкое '
                  'расхождение, логарифмическая ось. В центре — IoU итоговых бинарных масок '
                  'после сопоставления столбцов, больше лучше. Справа — среднее p(1−p), '
                  'больше означает менее бинарные вероятности. Пунктир — максимум при '
                  'одинаковых вероятностях p=0,2 и фиксированной суммарной массе.', '',
                  '**Вывод.** Мягкое расхождение снижается примерно в 9,5 раза, но бинарный IoU '
                  'остаётся около 0,147. Вероятности приближаются к равномерным. Это согласуется '
                  'с уменьшением расхождения за счёт более однородных мягких масок; отдельная '
                  'причинная проверка этого объяснения пока не выполнена.', '']
        fig, ax = plt.subplots(figsize=(7, 4))
        vae_values = []
        for moment in ('initial', 'best', 'last'):
            vae_values.append([[entry['validation_loss_' + moment] for entry in d['vae']]
                               for d in diagnostics])
        vae_values = np.asarray(vae_values)
        for index, name in enumerate(('Initial', 'Best checkpoint', 'Last epoch')):
            ax.bar(np.arange(4) + .25 * (index - 1), vae_values[index].mean(0), width=.25, label=name)
        ax.set_xticks(np.arange(4), ['Source 0', 'Source 1', 'Source 2', 'Source 3'])
        ax.set_ylabel('Validation BCE-sum + 0.1 KL')
        ax.set_title('VAE fit on held-out source importance maps')
        ax.legend()
        fig.tight_layout()
        save_figure(fig, out, 'vae_validation')
        data['vae_validation'] = {'moments': ['initial', 'best', 'last'], 'values': vae_values.tolist()}
        lines += ['### Обучение VAE', '', '![Validation VAE](vae_validation.png)', '',
                  '**Как читать.** По горизонтали — четыре исходные задачи. Столбцы показывают '
                  'validation loss до обучения, на выбранном лучшем состоянии и в последнюю эпоху; '
                  'это средние по восьми повторам. Loss суммирует BCE по всем связям и добавляет '
                  '0,1 KL; сравнивать его численно с downstream MSE нельзя.', '',
                  '**Вывод.** VAE обучились реконструировать карты лучше начального состояния. '
                  'Реконструкция сама по себе не доказывает полезность генерируемой бинарной маски.', '']
    bank_paths = [[path.parent / f'bank_{task}.pt' for path in paths] for task in range(4)]
    if all(path.exists() for group in bank_paths for path in group):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
        selected = []
        for task, (ax, group) in enumerate(zip(axes.flat, bank_paths)):
            banks = [torch.load(path, map_location='cpu', weights_only=False) for path in group]
            curves = [bank['training_curves'] for bank in banks]
            steps = [entry['step'] for entry in curves[0]]
            selected.extend(value for bank in banks for value in bank['validation_losses'])
            data[f'bank_task_{task}'] = {'steps': steps}
            for key, name in (('train_normalized_mse', 'Train'),
                              ('validation_mean_normalized_mse', 'Validation')):
                values = np.array([[entry[key] for entry in curve] for curve in curves])
                line, = ax.plot(steps, values.mean(0), label=name)
                if len(values) > 1:
                    margin = t.ppf(.975, len(values) - 1) * values.std(0, ddof=1) / np.sqrt(len(values))
                    ax.fill_between(steps, values.mean(0) - margin, values.mean(0) + margin,
                                    color=line.get_color(), alpha=.15)
                data[f'bank_task_{task}'][key] = values.tolist()
            ax.set_title(f'Source task {task}')
            ax.set_xlabel('Bank training steps')
            ax.set_ylabel('MSE / set size')
            ax.grid(alpha=.2)
            ax.legend()
        fig.tight_layout()
        save_figure(fig, out, 'bank_learning')
        data['selected_bank_validation_mean'] = float(np.mean(selected))
        lines += ['### Качество исходных банков', '', '![Обучение банков](bank_learning.png)', '',
                  '**Как читать.** Четыре панели соответствуют исходным задачам. По горизонтали — '
                  'шаги обучения; по вертикали — нормированная MSE. Кривые усредняют все 128 '
                  'кандидатов банка, а затем восемь независимых повторов; полосы — t-интервалы '
                  'по повторам. Train измеряется на текущем minibatch, validation — на фиксированных '
                  'отложенных наборах, поэтому шум кривых различается.', '',
                  f'**Вывод.** Средняя validation MSE выбранных 32 сетей банка равна {np.mean(selected):.3f}. '
                  'Банки обучились, но почти оптимальное качество их решений не установлено. '
                  'Это ограничивает интерпретацию отсутствия выигрыша переноса.', '']
    mask_paths = [path.parent / 'masks.pt' for path in paths]
    if all(path.exists() for path in mask_paths):
        masks = [torch.load(path, map_location='cpu', weights_only=True) for path in mask_paths]
        fig, axes = plt.subplots(1, 5, figsize=(13, 3.3))
        for ax, method in zip(axes, ('agreement', 'mean', 'single_vae', 'random', 'dense')):
            pixels = torch.stack([value[method].sum(-1).mean(0) for value in masks]).mean(0).reshape(28, 28).numpy()
            im = ax.imshow(pixels, cmap='viridis', vmin=0, vmax=32)
            ax.set_title(labels[method], fontsize=10)
            ax.set_xlabel('Pixel column')
            ax.set_ylabel('Pixel row')
            data['pixel_degree_' + method] = pixels.tolist()
        fig.colorbar(im, ax=axes.tolist(), shrink=.7, label='Active hidden connections per pixel')
        save_figure(fig, out, 'mask_pixel_degree')
        lines += ['### Где расположены связи масок', '', '![Число связей на пиксель](mask_pixel_degree.png)', '',
                  '**Как читать.** Каждый квадрат имеет размер изображения 28×28. Цвет — число '
                  'разрешённых связей этого пикселя со скрытыми нейронами, усреднённое по четырём '
                  'маскам и восьми повторам. Шкала общая: от 0 до 32. Все разреженные методы '
                  'имеют одинаковое общее число связей, dense — все связи.', '',
                  '**Ограничение.** Это пространственная маргинальная статистика. Она не показывает '
                  'распределение связей между конкретными hidden units и не доказывает восстановление '
                  'локальных фильтров. Полные бинарные матрицы сохранены в `seed_*/masks.pt`.', '']
    (out / 'figure_data.json').write_text(json.dumps(data, indent=2, allow_nan=False))
    return lines

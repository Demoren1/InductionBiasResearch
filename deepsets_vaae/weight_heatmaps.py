"""Compare saved, paired dense and VAAE first-layer weights without sorting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch

from .figures import save_figure


def selected(checkpoint: dict, method: str, init: int = 0) -> dict:
    indices = [index for index, (name, replica) in enumerate(
        zip(checkpoint['method_names'], checkpoint['replica_indices']))
        if name == method and replica == init]
    if len(indices) != 1:
        raise ValueError(f'Expected one {method} init {init}, found {indices}')
    index = indices[0]
    state = checkpoint['state_dict']
    values = {name: value[index].numpy() for name, value in state.items()}
    values['effective_weight'] = checkpoint['effective_weight'][index].numpy()
    assert np.array_equal(values['effective_weight'], values['weight'] * values['masks'])
    values['record'] = checkpoint['records'][index]
    return values


def matrix_pair(out: Path, dense: dict, ours: dict, limit: float, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 10), sharey=True)
    for ax, values, title in zip(axes, (dense, ours), ('Dense: learned W', 'VAAE: learned W × M')):
        im = ax.imshow(values['effective_weight'], aspect='auto', cmap='RdBu_r',
                       vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Hidden neuron (saved order)')
        ax.set_ylabel('Input pixel: row × 28 + column')
    fig.colorbar(im, ax=axes.tolist(), shrink=.7, label='Effective signed first-layer weight')
    save_figure(fig, out, name)


def single_matrix(out: Path, values: dict, title: str, limit: float, name: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(values['effective_weight'], aspect='auto', cmap='RdBu_r',
                   vmin=-limit, vmax=limit, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Hidden neuron (saved order)')
    ax.set_ylabel('Input pixel: row × 28 + column')
    fig.colorbar(im, ax=ax, label='Effective signed first-layer weight')
    fig.tight_layout()
    save_figure(fig, out, name)


def filters(out: Path, values: dict, title: str, limit: float, name: str) -> None:
    fig, axes = plt.subplots(4, 8, figsize=(13, 7))
    for hidden, ax in enumerate(axes.flat):
        im = ax.imshow(values['effective_weight'][:, hidden].reshape(28, 28),
                       cmap='RdBu_r', vmin=-limit, vmax=limit, interpolation='nearest')
        ax.set_title(f'h={hidden}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.colorbar(im, ax=list(axes.flat), shrink=.6, label='Effective signed weight')
    save_figure(fig, out, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    exports = args.out / 'weighted_exports'
    folders = sorted(path for path in exports.glob('seed_*') if path.is_dir())
    if not folders:
        raise SystemExit('No exported weights')
    pairs = []
    for folder in folders:
        checkpoint = torch.load(folder / 'target_task0_budget256.pt', map_location='cpu', weights_only=False)
        dense, ours = selected(checkpoint, 'dense'), selected(checkpoint, 'agreement')
        pairs.append((folder, dense, ours))
    limit = max(float(np.abs(values['effective_weight']).max())
                for _, dense, ours in pairs for values in (dense, ours))
    for folder, dense, ours in pairs:
        matrix_pair(folder, dense, ours, limit, 'dense_vs_vaae_weight_heatmap')
    folder, dense, ours = pairs[0]
    matrix_pair(args.out, dense, ours, limit, 'dense_vs_vaae_weight_heatmap')
    single_matrix(args.out, dense, 'Dense: learned first-layer weights', limit, 'dense_weight_heatmap')
    single_matrix(args.out, ours, 'VAAE: learned effective weights W × M', limit, 'vaae_weight_heatmap')
    filters(args.out, dense, 'Dense: all 32 learned first-layer filters', limit, 'dense_weight_filters')
    filters(args.out, ours, 'VAAE: all 32 learned masked first-layer filters', limit, 'vaae_weight_filters')
    fig, axes = plt.subplots(1, 2, figsize=(9, 10), sharey=True)
    for ax, values, title in zip(axes, (dense, ours), ('Dense connectivity', 'VAAE binary mask')):
        im = ax.imshow(values['masks'], cmap='Greys', vmin=0, vmax=1,
                       aspect='auto', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('Hidden neuron (saved order)')
        ax.set_ylabel('Input pixel: row × 28 + column')
    fig.colorbar(im, ax=axes.tolist(), ticks=[0, 1], shrink=.7, label='Allowed connection')
    save_figure(fig, args.out, 'dense_vs_vaae_binary_masks')
    arrays = {}
    for name, values in (('dense', dense), ('vaae', ours)):
        arrays.update({name + '_' + key: value for key, value in values.items() if key != 'record'})
    np.savez_compressed(args.out / 'selected_weight_heatmap_values.npz', **arrays)
    metadata = {'selection': 'first repeat, first target task, budget 256, initialization 0; fixed before inspecting quality',
                'seed': folder.name, 'task': 0, 'budget': 256, 'init': 0,
                'color_limits': [-limit, limit], 'matrix_shape': [784, 32],
                'dense_record': dense['record'], 'vaae_record': ours['record'],
                'masked_values_are_exact_zero': bool(np.all(ours['effective_weight'][ours['masks'] == 0] == 0)),
                'repeated_export_count': len(pairs)}
    (args.out / 'weight_heatmap_metadata.json').write_text(json.dumps(metadata, indent=2))
    lines = ['# Heatmap обученных весов: dense и VAAE', '',
             f'Заранее выбранный пример: {folder.name}, первая тестовая задача, 256 размеченных наборов, init=0. '
             'Это 205 обучающих и 51 validation-набор. Чекпойнт каждого метода выбран только по validation. '
             'Для всех методов используются одни и те же данные и парная численная инициализация.', '',
             'Все веса исходного эксперимента воспроизведены и сохранены отдельно в `weighted_exports/`. '
             'Точность воспроизведения проверяется по исходным метрикам; результаты проверки лежат '
             'в `weighted_exports/seed_*/export_provenance.json`. Первоначальные результаты не изменены.', '',
             '## Сопоставление матриц', '',
             '![Dense и VAAE с весами](dense_vs_vaae_weight_heatmap.png)', '',
             '**Как читать.** По вертикали — 784 пикселя изображения в построчном порядке; по горизонтали — '
             '32 скрытых нейрона. Слева обученная матрица dense W, справа фактические веса нашей сети '
             'W⊙M. Красный — положительный вес, синий — отрицательный, белый — около нуля. '
             'У масочной сети запрещённые связи строго равны нулю. Общая симметричная цветовая шкала '
             'использует максимум абсолютного веса среди всех восьми выбранных пар; значения не обрезаны. '
             'Порядок нейронов сохранён, сортировка по качеству или визуальной форме не выполнялась.', '',
             '**Вывод и границы.** График показывает фактически обученные численные веса, а не только '
             'разрешённые связи. Матрица первого слоя не определяет всю функцию сети: biases и readout '
             'также сохранены в численных чекпойнтах. Сходство цветов между столбцами разных сетей '
             'не означает соответствие функций нейронов.', '',
             '## Dense отдельно', '', '![Dense weights](dense_weight_heatmap.png)', '',
             '**Как читать.** Это левая матрица предыдущего графика в отдельном файле, с теми же осями '
             'и цветовой шкалой. Все связи разрешены; величина и знак веса обучены на выбранной задаче.', '',
             '## VAAE отдельно', '', '![VAAE masked weights](vaae_weight_heatmap.png)', '',
             '**Как читать.** Это правая матрица сравнения: W⊙M. Белые запрещённые связи — точные нули; '
             'малые обученные активные веса тоже выглядят почти белыми, поэтому для различения '
             'разрешённости связи ниже показана сама бинарная маска.', '',
             '## Бинарные маски', '', '![Binary masks](dense_vs_vaae_binary_masks.png)', '',
             '**Как читать.** Те же оси матриц; чёрный означает разрешённую связь, белый — запрещённую. '
             'Dense разрешает все 25 088 связей, VAAE — ровно 5 018. Эта картинка показывает '
             'поддержку связей независимо от их обученных значений.', '',
             '## Фильтры dense в геометрии изображения', '', '![Dense filters](dense_weight_filters.png)', '',
             '**Как читать.** Каждый квадрат 28×28 — один столбец dense W, возвращённый в координаты '
             'изображения. Показаны все 32 нейрона в сохранённом порядке. Цветовая шкала та же, '
             'что на матричных heatmap. Красный и синий обозначают противоположные знаки весов.', '',
             '## Фильтры VAAE в геометрии изображения', '', '![VAAE filters](vaae_weight_filters.png)', '',
             '**Как читать.** Каждый квадрат — один столбец W⊙M нашей сети в координатах пикселей. '
             'Показаны все 32 нейрона; запрещённые пиксельные связи строго нулевые. '
             'Эти фильтры не следует называть обнаруженной свёрткой: веса разных нейронов не разделяются.', '',
             'Все изображения сохранены в PNG и PDF. Точные массивы выбранного примера — '
             '`selected_weight_heatmap_values.npz`; все параметры, включая biases и readout, — '
             'в `.pt`-чекпойнтах экспортированных моделей. Для каждого из восьми повторов '
             'дополнительно сохранена парная матричная heatmap в его каталоге.', '']
    (args.out / 'WEIGHT_HEATMAPS.md').write_text('\n'.join(lines))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

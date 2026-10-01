# Другие генеративные модели для функциональных карт DeepSets

Генераторы обучены на том же неизменённом банке: четыре source-задачи, по 205 aligned functional train-карт и 51 held-out source-карте на seed. Модель получает pooled source-карты с условием source-task; target cost-векторы, target validation и target test не используются для выбора checkpoint, маски или sparsity. Все пять моделей здесь — самостоятельные генераторы карт, не VAE и не PCA.

Все target-сравнения ниже exploratory: обе группы из восьми cost-задач уже были просмотрены в VAE-контроле. Интервалы — парные 95% Student t (df=7) по seed; множественные сравнения не корректировались.

## Схема и фиксированная sparsity

Размер карты — 784×32=25088; каждая новая маска проходит общий exact-top-7526 projection (30%). Adaptive sparsity намеренно отложена: её нельзя подбирать по target validation/test, иначе сравнение протекает. Следующий отдельный протокол может проверить source-only joint utility с penalty на число рёбер (L0/hard-concrete) и nested validation; это ещё не тестировалось. Прямая `functional_mean_large` остаётся сильным baseline.

Методы: diffusion и flow matching моделируют распределение карт; `gnn_flow` использует двудольный граф pixel–hidden; `set_transformer_flow` рассматривает hidden-columns как множество; GAN использует генератор/критик. Во всех вариантах task-condition и нормализация учатся только на pooled 4×205 source-train картах; held-out 4×51 используется для source-only checkpoint selection.

## Архитектурный охват и ограничения

Новые модели — один pooled conditional generator на четыре source-задачи; VAE-контроль состоит из четырёх отдельных моделей, а его маска выбирается agreement между decoder. Поэтому это сравнение полезно как проверка способов генерации и извлечения на одном банке, но не изолирует один только класс архитектуры. Число параметров: diffusion и flow matching — по 13 416 705, GNN-flow — 78 274, Set Transformer-flow — 549 521, GAN generator — 6 944 256 и critic — 6 886 657. У FM-backbone одинаковы задача, данные и batch 128, но не capacity.

Ранний flat diffusion/FM вариант с hidden width 256 оказался rank-ограниченным diagnostic control: его выходная вариация по шуму имеет rank не выше 256 в пространстве 25 088 координат. Source-only проверка зафиксировала это как obstruction, а не как вывод о diffusion/flow families. В итоговом main-five варианте добавлен residual `+a(t,c)·x` (257 добавочных параметров), сохраняющий зависимость от исходного шума во всём пространстве; первоначальные flat результаты сохранены как архивная диагностика и не входят в основное сравнение. См. [rank_obstruction.json](rank_obstruction.json).

Средний variance ratio generated samples по восьми seed: diffusion 0,054, flow matching 0,032, GNN-flow 1,007, Set Transformer-flow 0,082 и GAN 0,006. Это описывает разнообразие сэмплов, но не само по себе target utility. Все source networks банка имеют 20% плотности: такой банк не способен сам определить оптимальную ρ. Для variable sparsity понадобятся банк с несколькими плотностями либо source-utility при нескольких K с L0/expected-edge penalty и nested held-out selection; этого эксперимента здесь нет.

## Статус source-only попыток

| Метод | stable / все попытки | сохранённые статусы |
|---|---:|---|
| diffusion | 8 / 8 | converged: 8 |
| flow_matching | 8 / 8 | converged: 8 |
| gnn_flow | 8 / 8 | converged: 8 |
| set_transformer_flow | 8 / 8 | converged: 8 |
| gan | 0 / 8 | max_steps_unstable: 8 |

Лимит GPU updates сам по себе не означает сходимость. В частности, GAN может быть сохранён как exploratory unstable game; его результаты не заменяются и не дорисовываются, а статус остаётся в сводке. Для GAN слева на кривой показан source fixed-probe, справа — generator/critic objectives в разных единицах; направление «меньше лучше» относится к probe, но не к game losses.

## Target NMSE, fresh, budget 256

| Метод | NMSE, среднее [95% CI] | Δ к functional mean large [95% CI] |
|---|---:|---:|
| diffusion | 0.7260 [0.7141; 0.7378] | +0.0194 [+0.0083; +0.0304] |
| flow_matching | 0.7299 [0.7150; 0.7447] | +0.0232 [+0.0058; +0.0406] |
| gnn_flow | 0.7107 [0.6984; 0.7230] | +0.0041 [-0.0077; +0.0158] |
| set_transformer_flow | 0.7266 [0.7141; 0.7391] | +0.0200 [+0.0034; +0.0366] |
| gan (unstable: 0/8 plateau) | 0.7126 [0.7001; 0.7251] | +0.0059 [-0.0055; +0.0174] |

NMSE = MSE/5, меньше лучше. Значение Δ ниже нуля означает меньшую ошибку, но не доказывает обобщение вне этих восьми фиксированных задач. GAN включён как exploratory reference: ни один из 8 запусков не достиг объявленного stability/plateau criterion, поэтому его target-числа не являются результатом сошедшейся GAN-модели.

## Графики

- [Кривые target-качества](target_learning_curves.png): линии — средние по seed, полупрозрачные области — 95% t-CI.
- [Парные контрасты на fresh задачах](target_contrasts_fresh.png): каждый столбец — generator minus baseline; ниже нуля лучше генератора.
- [Разнообразие source samples](sample_diversity_metrics.png): variance ratio, парная IoU и расстояния до train/held-out карт.
- [Все 40 source loss curves](loss_curves/): train/held-out objectives, выбранный checkpoint и остановка. GAN-графики раздельно показывают fixed probe и generator/critic game losses.
- [W×M heatmap](weight_heatmaps/fresh_task0_budget256_effective_weights.png), [бинарные маски](weight_heatmaps/fresh_task0_budget256_binary_masks.png) и [scores генераторов](weight_heatmaps/seed4100_generated_scores.png): фиксированный seed 4100, fresh task 0, budget 256, init 0; шкала W×M общая и знаковая.

## Воспроизводимость

- [protocol.json](protocol.json) фиксирует банк, методы и target protocol; старый банк не изменялся.
- [summary.json](summary.json) содержит все seed means и paired contrasts; [figure_data.npz](figure_data.npz) — численные массивы графиков.
- Каждый `seed_<n>/<method>/fit.json` содержит source-only loss curve и sample diagnostics; `samples.pt` — сэмплы, score и exact-K masks. Target checkpoints находятся в `weights/` и `fresh_weights/`.

## Источники и адаптации

- [DDPM](https://arxiv.org/abs/2006.11239): прямое full-map x0 prediction без VAE/PCA компрессора.
- [Flow Matching](https://arxiv.org/abs/2210.02747): conditional linear-path velocity field; full-dimensional residual исправляет архивный low-rank flat control.
- [Set Transformer](https://arxiv.org/abs/1810.00825): SAB над 32 hidden-columns как equivariant conditional field.
- [WGAN-GP](https://arxiv.org/abs/1704.00028): conditional GAN на full functional-map logits.
- [Graph-flow preprint](https://arxiv.org/html/2609.32833v1) и [weight-space flow motivation](https://arxiv.org/abs/2504.03710): основание для малого bipartite pixel–hidden GNN, не воспроизведение этих работ.
- [HyperGAN](https://arxiv.org/abs/1901.11058) использован только как мотивация генеративной постановки; его архитектура и loss здесь не реализованы.
- Полные машинно-читаемые ссылки и оговорки сохранены в [article_notes.json](article_notes.json) и [protocol.json](protocol.json).

Метаданные отчёта: seeds=[4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107], budgets=[32, 64, 128, 256], methods=['functional_mean_small', 'functional_mean_large', 'functional_vae_small', 'functional_vae_large', 'raw_vae_large', 'random', 'dense', 'diffusion', 'flow_matching', 'gnn_flow', 'set_transformer_flow', 'gan'].

## Дополнительные материалы и итог исследования

GNN-flow лучше VAE, random и dense в точечном парном сравнении на additional target-задачах, но не показал убедительного выигрыша над прямой functional mean. Pixel-only prior достигает NMSE 0,7036 без убедительной разницы с полной functional mean 0,7066; GNN sample agreement ухудшает результат до 0,7368. Поэтому полезность генеративного моделирования межреберных зависимостей пока не установлена; высокая sample variance GNN сама по себе этого не доказывает.

- [Полные extraction-контроли, интервалы и W×M](../20261001_generative_extraction_controls/EXTRACTION_REPORT.md).
- [Фиксированные source/generated карты и распределение бюджета связей](sample_figures/CAPTIONS.md). На raw-картах общая шкала [0,1]; отдельно [логарифмический цветовой вариант](sample_figures/fixed_generated_maps_vs_source_log.png) со шкалой LogNorm [10⁻⁵,1], значения ниже нижней границы только визуально ограничены, численные массивы не менялись.
- [Распределение по hidden-нейронам](sample_figures/hidden_neuron_degree.png): X — hidden-нейрон, Y — доля разрешённых input-связей; общая шкала 0–1. Вместе с pixel degree это описывает распределение fixed-K бюджета, без причинного вывода о структуре.
- [Полный отчёт за 27 сентября](../../../mds/FINAL_REPORT_2026-09-27.md), включая пояснения каждого показанного графика и следующие проверки source utility/variable density.
- [Whole-device GPU observation](../20261001_other_generators/plots/gpu_utilization.png): X — секунды 1912-секундного окна, Y — загрузка GPU; среднее восьми карт 52,4%, максимум 100%. Измерение включает перекрывающиеся фазы и не является benchmark отдельной архитектуры.

Плотность source-банка равна 20% retained edges (sparsity 80%), а плотность оцениваемых sparse target-масок равна 30% (sparsity 70%). Variable density здесь не обучалась; её будущий критерий должен включать source utility и стоимость числа связей, например [L0/hard-concrete regularization](https://arxiv.org/abs/1712.01312), с nested source selection.

[Итоговый независимый аудит](independent_final_audit.json) · [контрольные суммы всех фаз](artifact_manifest.json) · [проверка локальных ссылок](local_link_audit.json).

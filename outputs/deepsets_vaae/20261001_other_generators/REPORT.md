# Другие генеративные модели для функциональных карт DeepSets

## Статус этой первой фазы

**Diffusion и flat flow matching в этой фазе — диагностические контроли с обнаруженным ограничением архитектуры. Их оценки не входят в итоговое сравнение семейств.** Выход поля проходит через 256-мерный слой при 25088 координатах карты; без прямого residual-пути поле не может выразить произвольное покоординатное удаление Gaussian noise. Низкий loss или остановка по плато не устраняют это ограничение. Source-only диагностика и исправление сохранены в [rank_obstruction.json](rank_obstruction.json).

Обе модели повторно обучены с нулево инициализированным членом `a(t,task) × x` (+257 параметров), с теми же банками и протоколом. **Итоговые результаты:** [сравнение после исправления](../20261001_other_generators_corrected/REPORT.md). GNN, Set Transformer и GAN перенесены без переобучения; исходные файлы этой фазы сохранены. GAN нестабилен во всех восьми запусках и не считается сошедшимся. Таблица ниже описывает только исходную фазу.

Генераторы обучены на том же неизменённом банке: четыре source-задачи, по 205 aligned functional train-карт и 51 held-out source-карте на seed. Модель получает pooled source-карты с условием source-task; target cost-векторы, target validation и target test не используются для выбора checkpoint, маски или sparsity. Все пять моделей здесь — самостоятельные генераторы карт, не VAE и не PCA.

Все target-сравнения ниже exploratory: обе группы из восьми cost-задач уже были просмотрены в VAE-контроле. Интервалы — парные 95% Student t (df=7) по seed; множественные сравнения не корректировались.

## Схема и фиксированная sparsity

Размер карты — 784×32=25088; каждая новая маска проходит общий exact-top-7526 projection (30%). Adaptive sparsity намеренно отложена: её нельзя подбирать по target validation/test, иначе сравнение протекает. Следующий отдельный протокол может проверить source-only joint utility с penalty на число рёбер (L0/hard-concrete) и nested validation; это ещё не тестировалось. Прямая `functional_mean_large` остаётся сильным baseline.

Методы: diffusion и flow matching моделируют распределение карт; `gnn_flow` использует двудольный граф pixel–hidden; `set_transformer_flow` рассматривает hidden-columns как множество; GAN использует генератор/критик. Во всех вариантах task-condition и нормализация учатся только на pooled 4×205 source-train картах; held-out 4×51 используется для source-only checkpoint selection.

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
| diffusion | 0.7110 [0.6986; 0.7234] | +0.0044 [-0.0048; +0.0135] |
| flow_matching | 0.7296 [0.7157; 0.7435] | +0.0229 [+0.0070; +0.0389] |
| gnn_flow | 0.7107 [0.6984; 0.7230] | +0.0041 [-0.0077; +0.0158] |
| set_transformer_flow | 0.7266 [0.7141; 0.7391] | +0.0200 [+0.0034; +0.0366] |
| gan | 0.7126 [0.7001; 0.7251] | +0.0059 [-0.0055; +0.0174] |

NMSE = MSE/5, меньше лучше. Значение Δ ниже нуля означает меньшую ошибку, но не доказывает обобщение вне этих восьми фиксированных задач.

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

- [article_notes.json](article_notes.json) — сохранённые источники/заметки, использованные для выбора архитектур: `{"purpose": "Primary papers read before architecture implementation; adaptations generate functional saliency maps rather than neural weights.", "articles": [{"model": "diffusion", "url": "https://arxiv.org/abs/2006.11239", "adaptation": "direct full functional-map x0-prediction DDPM; no VAE compressor"}, {"model": "flow_matching", "url": "https://arxiv.org/abs/2210.02747", "adaptation": "conditional independent linear-path flow"}, {"model": "gnn_flow", "url": "https://arxiv.org/html/2609.32833v1", "adaptation": "small bipartite neural graph edge/node velocity field; not paper reproduction"}, {"model": "flow_matching", "url": "https://arxiv.org/abs/2601.05052", "adaptation": "alignment motivates matched-bank control; no full-weight DeepWeightFlow reproduction"}, {"model": "gnn_flow", "url": "https://arxiv.org/abs/2504.03710", "adaptation": "weight-space graph flow motivation"}, {"model": "set_transformer_flow", "url": "https://arxiv.org/abs/1810.00825", "adaptation": "SAB over 32 hidden-neuron columns as equivariant flow field"}, {"model": "gan", "url": "https://arxiv.org/abs/1704.00028", "adaptation": "conditional WGAN-GP on full maps"}, {"model": "gan", "url": "https://arxiv.org/abs/1901.11058", "adaptation": "HyperGAN inspiration only; not its architecture/loss"}], "scope_notes": ["The September 2026 graph-flow preprint studies neural weights with permutation-equivariant parameter graphs; our bipartite saliency graph is an adaptation, not reproduction.", "Set Transformer supplies an equivariant attention architecture, made generative by the shared conditional flow-matching objective.", "HyperGAN does not equal WGAN-GP: our GAN uses the latter on functional-map logits.", "All density selections remain fixed at rho=0.3; adaptive sparsity not trained here.", "FM backbones share minibatch128, not parameter counts; checkpoint selection uses only source maps.", "The papers do not establish which model wins on our bank; actual target evaluation is required."], "additional_sparsity_reference": {"title": "Learning Sparse Neural Networks through L0 Regularization", "url": "https://arxiv.org/abs/1712.01312", "proposed_future_use": "task utility plus expected gate count penalty; not trained in this fixed-K study"}, "future_bank_constraint": "All source networks were trained at 20% retained edges. These maps alone do not identify the optimal sparsity-utility tradeoff. Adaptive K needs source evaluation at multiple K or banks covering multiple densities, with nested heldout selection."}`

Метаданные отчёта: seeds=[4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107], budgets=[32, 64, 128, 256], methods=['functional_mean_small', 'functional_mean_large', 'functional_vae_small', 'functional_vae_large', 'raw_vae_large', 'random', 'dense', 'diffusion', 'flow_matching', 'gnn_flow', 'set_transformer_flow', 'gan'].

# OOD transfer: 12 meta-train / 4 held-out patterns

Дата: 2026-08-23.

## Вывод

Zero-shot structural prior переносится на не виденные генератором задачи.
По пяти последовательным deterministic random split'ам CVAE достигает **0.9086**
downstream accuracy против **0.8821** у честного `random_exact32` baseline.
Средний transfer gain равен **+0.0266 ± 0.0046** (population standard
deviation по пяти split'ам). Все пять gain положительны; диапазон —
`+0.0203...+0.0344`.

Это подтверждает OOD-гипотезу внутри данной синтетической task family:
общая структурная Toeplitz-индукция, извлечённая из решений 12 задач,
улучшает обучение свежих MLP на четырёх held-out задачах. Поскольку
`CVAE_COND_DIM=0`, это перенос общего безусловного structural prior, а не
conditional extrapolation по идентичности паттерна.

## Протокол

- Пять последовательных split seeds без отбора: `42 43 44 45 46`.
- В каждом split: случайные 12 паттернов на meta-train, оставшиеся 4 на
  meta-test; split сохраняется до обучения в `split.json`.
- VAE обучается только на raw `importance.pt` meta-train задач: top-10% карт
  по validation BCE, BCE-sum, `β=0.1`, 80 эпох.
- `mean_imp` и `det_reg` также строятся только по тем же 12 meta-train задачам.
- На каждой held-out задаче для каждой маски обучается свежий MLP: 2 000 шагов,
  64 маски на метод.
- Primary baseline — `random_exact32`: равномерная случайная маска ровно с 32
  активными связями, как у CVAE. Bernoulli random приведён как secondary.
- Target-task `top10%` и bilevel z-optimization исключены: первое является
  within-task oracle, второе адаптируется на данных held-out задачи.
- Evaluation data имеет другой seed, чем validation data, использованный для
  отбора meta-train решений.

## Агрегированные результаты

Среднее и population standard deviation считаются по пяти split-level macro
means; всего выполнено 20 held-out task evaluations.

| источник маски | mean accuracy | σ по split'ам | mean BCE |
|---|---:|---:|---:|
| random Bernoulli(0.5) | 0.8787 | 0.0124 | 0.3008 |
| random exact-32 | 0.8821 | 0.0132 | 0.2977 |
| mean importance, train-only | 0.7871 | 0.0088 | 0.4063 |
| deterministic regressor, train-only | 0.7871 | 0.0089 | 0.4060 |
| **CVAE, train-only** | **0.9086** | **0.0088** | **0.2539** |
| ideal Toeplitz | 0.9240 | 0.0098 | 0.2226 |

CVAE выигрывает у `random_exact32` на **2.66 п.п.** и отстаёт от ideal на
**1.53 п.п.**

| split seed | CVAE | random exact-32 | gain |
|---:|---:|---:|---:|
| 42 | 0.9155 | 0.8912 | +0.0243 |
| 43 | 0.9078 | 0.8814 | +0.0264 |
| 44 | 0.9159 | 0.8955 | +0.0203 |
| 45 | 0.9121 | 0.8848 | +0.0273 |
| 46 | 0.8919 | 0.8575 | +0.0344 |

## Воспроизведение

Для одного split (пример: seed 42, GPU 3):

```bash
conda activate ras
SPLIT_SEED=42 GPU_ID=3 bash scripts/09_ood.sh
```

Результаты одного прогона записываются в
`outputs/ood/split_seed_<seed>/`. Агрегация пяти прогонов:

```bash
python evaluation/aggregate_ood.py \
  --summaries outputs/ood/split_seed_{42,43,44,45,46}/summary.json \
  --out outputs/ood/aggregate_seeds_42_46.json
```

Основной агрегированный артефакт:
`outputs/ood/aggregate_seeds_42_46.json`. В каждом split также сохранены
список задач, provenance VAE/det-reg, per-mask accuracy/BCE и графики.

## Ограничения вывода

- Все 16 задач имеют одну и ту же pattern-independent Toeplitz-структуру;
  эксперимент показывает перенос общего bias внутри специально построенной
  синтетической среды, но не перенос на другое семейство задач.
- Пять split'ов переиспользуют одни и те же 16 возможных задач, поэтому
  `±` выше — описательный разброс по split'ам, а не confidence interval для
  произвольной внешней task distribution.
- Для более сильного статистического вывода нужны дополнительные task
  instances (другие длины последовательности/паттерна или более богатое
  параметризованное семейство), а не только новые перестановки этих 16 задач.

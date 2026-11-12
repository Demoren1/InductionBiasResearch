# Omniglot: weighted SVD and the first geometric candidate

This is the real-data follow-up to `theorem_pair_recovery_v1`. The user
explicitly selected its **first geometric candidate**, rather than its
source-functional choice. See [PROTOCOL.md](PROTOCOL.md).

The original Aug-Omniglot data preparation, dense source learner, G34
projection, common source fitter, numerical adapter and final target adapter
remain unchanged in `paper_structure_benchmark_v1`. This package owns the new
signature extraction and its evaluation registration.

Source observations reproduce the historical panel hashes. Development
episodes use the unchanged historical seed and generator.
The 1000 final episodes have a new seed; their test class pool is the same.
The runtime files, images and complete dense source ensemble were removed
on 12 November at the user's request after exporting compact evidence to
`artifacts/omniglot_signature_2026-11-12/`. All episode logits and answers,
structural model states and one representative Dense state are retained.

## Sequence

1. `prepare` recreates the observations, creates a new target sample and
   separates support, numerical and structure panels.
2. The unchanged `fitting.real_dense` fits the source ensemble. The unchanged
   `methods.real_project` and `fitting.real_bank_fit` produce the G34 control.
3. `extract` uses the same two dense anchors, direction-dispersion uncertainties
   and four mask proposals as the existing real-data projection. It centers
   log magnitudes inside each neuron, weights source task columns by inverse
   RMS uncertainty and computes an SVD in float64. It scans every feasible
   rank and the original threshold grid, retaining only complete partitions
   with the correct calibration rank. Interrupted runs resume only missing
   rank jobs, with amendments preserving code hashes.
4. `rank` uses the existing robust log score and sign-disagreement penalty.
   It searches the same source sign starts/voting rounds, vectorized over
   independent groups, and freezes the first candidate for each available
   budget. No source M responses enter this step.
5. `validate` chooses the code budget and one support-only numerical rule on
   disjoint development tasks. It cannot replace a budget's frozen structure.
6. `pipeline` exports only structures and the numerical rule, adapts fresh
   target parameters, and freezes every method's predictions together.
7. `score` verifies the freeze and target observation commitment before opening
   target responses for scoring. It writes episode-paired comparisons.

The streaming pair implementation retains both theorem merge rules. A forest
represents zero-signature connected components exactly. For nonzero pairs,
early stopping uses only rigorous component/isolated-vertex certificates;
every retained partition exhausts its radius matches. CUDA matrix products
use conservative float64 lower bounds and directly verify candidate
distances. This avoids treating cancellation in a Gram-matrix distance as
an exact equal signature.

The current partial partition also bounds the remaining search. Coarsening
groups cannot increase the contrast-space rank. If its current rank already
falls below the SVD rank being examined, no subsequent merge can produce an
eligible candidate. Neurons with no nonzero pairs likewise force separate
calibration components. Both are exact rejection rules, not approximations.

Rank zero is permitted by the original method: it assigns one unsigned group
to each active neuron. Such a candidate is a valid degenerate output and
does not demonstrate recovery of cross-neuron sharing. Missing budgets are
reported; no k-means completion or forced component merging is applied.

No software tests are added or run. Scientific runs, audit receipts, frozen
hashes and targeted linting provide the relevant verification.

## Состояние данных после уборки

Полные ансамбли, сырые наблюдения и промежуточные численные кэши завершённых запусков удалены 12 ноября по запросу пользователя. Приведённые команды полного запуска требуют заново получить соответствующие входы; новые запуски используют новые каталоги. Итоговые структуры, метрики, точные предсказания переноса и данные аудита сохранены в artifacts/. [Состав уборки](../../archive/2026-11-12/README.md).

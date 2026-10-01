# Абляция permutation consistency на исходных задачах

**Статус:** ПОЛНЫЙ ОТЧЕТ: все четыре ветви завершены.

Четыре ветви обучались на seed `4100` с `32` train-only учителями и четырьмя исходными задачами. На каждой задаче policy выбирала `7526` связей из карты `784×32`; query NMSE измерялась на свежих child-моделях после фиксированных `2000` шагов по support.

Policy-gradient использовал два ordered Gumbel/Plackett–Luce draw на задачу, exact score-function estimator и leave-one-draw-out baseline. Полученный scalar — оценка градиента, а не сама NMSE. Consistency штраф — сумма embedding MSE и sigmoid-response MSE на фиксированных decoder slots. Фиксированный монитор использовал те же teacher refs и Gumbel noise на update 0 и далее каждые 8 updates.

Хэши image pools `source_train` и `source_validation` совпали с teacher-source хэшами ([сравнение](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/source_query_provenance_comparison.json)). Наборы при этом выбирались отдельно. Фиксированные 51 query set повторно использовались во внешнем обучении и при выборе monitor-best состояния, поэтому это source monitor, не независимая validation. Test не открывался.

| Ветвь | γ | Шаги encoder | Финальная source NMSE | Лучший monitor objective | Encoder / response consistency | Совпадение hard masks | Plateau-флаги child fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Набор, γ=1 | 1 | 48 | 0.815470 | 0.81456518 @ 24 | 2.27e-15 / 9.98e-17 | 1.000000 | 64/64 всего; 16/16 own-task |
| Набор, γ=0 | 0 | 48 | 0.815470 | 0.81456518 @ 24 | 2.42e-15 / 9.64e-17 | 1.000000 | 64/64 всего; 16/16 own-task |
| Смещение столбца, γ=1 | 1 | 48 | 0.817195 | 0.81524479 @ 0 | 4.71e-10 / 7.65e-14 | 0.999980 | 64/64 всего; 16/16 own-task |
| Смещение столбца, γ=0 | 0 | 48 | 0.817195 | 0.81524479 @ 0 | 4.72e-10 / 7.65e-14 | 1.000000 | 64/64 всего; 16/16 own-task |

Все четыре ветви достигли эмпирического encoder plateau на update 48. В каждой выполнено 3 520/3 520 child fit с support-plateau флагами; на финальном мониторе проверялись 64 fit, из них 16 диагональных own-task fit вошли в policy utility. Plateau-флаги — диагностика и не останавливают фиксированный 2 000-шаговый child horizon.

Source quality при γ=0 и γ=1 совпала до показанной точности в обеих архитектурах. Set-инвариантность заложена в архитектуру, поэтому consistency близка к нулю и loss не обучает эту инвариантность. В position-biased ветвях consistency также мала (начальный encoder MSE 9.6e-9); измеримого выигрыша от γ нет.

Независимый replay прошел: максимальная ошибка воспроизведения 64 child NMSE на ветвь `3.58e-7`, diagonal utility credit `1.79e-7`, все sampled masks имеют K=7 526, shared-Gumbel agreement — 100%. В таблице выше и на monitor-графике показан другой контроль: deterministic `exact_topk` сравнивает исходные logits; у `position_joint` он равен 0.999980 (2 из 100 352 позиций), что не противоречит совпадению sampled masks при общем шуме. Подробности — в независимом отчете ниже.

Hard-mask agreement на monitor-графике — детерминированное сравнение двух `exact_topk(logits)` карт. Независимый shared-Gumbel replay отдельно проверил случайные sampled masks: agreement 100%; это другой контроль и он не противоречит deterministic сравнению.

Source-only controls dense/random/functional дали средние NMSE свежих child-моделей: dense 0.844567, random 0.811098, functional 0.830246). Это не target/test результат и не независимая validation.

## Графики

- [NMSE и risk objective фиксированного source-монитора, consistency, совпадение deterministic exact_topk masks и gradient norm](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/plots/source_monitor_diagnostics.png). По X — шаг encoder; consistency показана в log масштабе, нули размещены на floor `1e-20`; gradient norm измерен до clipping.
- [Фактические policy masks и обученные signed W·M](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/plots/selected_policy_masks_and_weights.png): terminal encoder, draw 0, child initialization replica 0. Черный — активная связь; в W·M красный — положительный вес, синий — отрицательный. Шкала симметрична, предел — 99-й процентиль каждой панели.
- [Маски и signed W·M dense/random/functional контролей](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/plots/source_control_masks_and_weights.png): actual initialization replica 0; черный — активная связь, красный/синий — знак веса.
- [Численные данные графиков](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/figure_data.npz); [JSON-сводка и проверки](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/summary.json).
- [Независимая проверка child NMSE, hard masks и градиентов](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/independent_validation.md); [машиночитаемый результат](../../../outputs/deepsets_vaae/20261001_permutation_dual_loss/independent_validation.json).

Frozen protocol SHA256: `fe2d22eea8270516da4c487dbb7da85f6ea9daa7c67984ed3a7045e5c2543f8b`. Context, source banks, code snapshot, train rows, source-quality flags и настройки ветвей: **проверки пройдены**.

Это один bank seed и четыре исходные задачи. Результат проверяет обучение и абляцию только на source data; target transfer из него не следует.

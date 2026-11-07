# G12: полноценный совместный fusion-контроль

Это добавление известного objective-level PACS/convex-clustering контроля к уже открытым G4 данным, не новая независимая confirmation и не новый наш метод. Профили всех128 функций фьюзятся совместно; нули/суммы/разности имеют group-L2 penalties. Используется полный sample Gram. Native I читается по точным нулям proximal переменных, без k-means. Числа C на validation/transfer обучаются заново при выбранных равенствах.

## Выбор и численная достоверность

На4 прежних pilot-геометриях×2n проверены gamma0/1/2/4/8 и r0=.1/1, семь lambda. По mean source-validation выбраны gamma8,r0=.1. Каждое новое source-наблюдение выбирает lambda и сходящийся checkpoint по тем же32 full-C support4/query128 функциям, что G4. Учитель/target в выборе не участвуют. Все48 дополнительных source-моделей заморожены до structural scoring; парный перенос использует те же100 новых коэффициентных функций, что G4.

У выбранных состояний проверены primal/dual residual и feasible relative primal-dual gap<=1e−6. Глобальная оптимальность выпуклой цели не означает истинность I. Adaptive rho — численная эвристика; gap служит прямым ограничением погрешности цели, а не обещанием сходимости изменяющегося rho.

| Режим | Exact I | MSE переноса, support4 |
|---|---:|---:|
| primary, n8192 | 0/12 | 0.647633 |
| primary, n131072 | 12/12 | 0.000569849 |
| low_noise, n131072 | 12/12 | 0.000569849 |
| high_noise, n131072 | 1/12 | 0.628785 |

## Диагноз

Приn8192 pilot0/4: удалены все14 слабых позиций каждого conv и3 каждого signed. Приn131072 pilot4/4. Прямая/двойственная цель уже близки; потеря группы не объясняется лишь незавершённым solver. На6101,n8192 true-equality squaredRMS≈.001338 почти совпадает с estimated noise≈.001318; слабый код против0 после коррекции имеет≈.000642. Raw adaptive distances содержат substantial noise-floor.

По этому диагнозу THEORY§20 до G13 выводит noise-corrected fusion-веса. Нужны отдельные raw shrinking-floor и Gaussian-kernel контроли. Грубая lambda-сетка остаётся ограничением этой реализации: нельзя выдавать G12 за отрицание всех хорошо настроенных PACS/convex-clustering вариантов. Приоритет общего noise-correction/embedding+fusion принципа не заявляется; PCMF и другие noisy-clustering предшественники должны учитываться.

## Сравнение и ограничения

Joint-контроль сильнее прежнего независимого OSCAR при большом n, но не превосходит val-selected LC/Yeh/LUT-Q и whitening-PCA. Более сильный solver/адаптивные веса/совместная норма изменены одновременно; различие с прежним OSCAR нельзя приписать одной joint-норме. Вся серия остаётся линейной. Independent nonlinear native comparison, robust noise и приоритет собственного компонента ещё не закрыты.

- [Сравнение с неизменёнными G4 конкурентами](joint_fusion_comparison.png)
- [Реальные ADMM loss/recall/gap кривые](joint_fusion_loss_recall_gap.png)
- [Численные W и native I](joint_fusion_indexators.png)
- [Таблица](comparison.csv), [summary](summary.json), [pilot-диагноз](../pilot_diagnosis.json).

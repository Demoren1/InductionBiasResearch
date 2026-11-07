# G13: коррекция adaptive fusion-весов

Полный matched pilot на прежних четырёх геометриях6101/6102/6201/6202,128source-функций, n8192/131072. Это диагностическая серия, не independent confirmation. Source-validation, shape/K и solver одинаковы. Истинные I не участвуют в выборе. Все24 новых source-состояния сначала заморожены, затем открыты для аудита.

THEORY§20 до запуска выводит unbiased squared-profile-distance и достаточный asymptotic KKT-аргумент при bounded-curvature/separation. Веса всегда фиксированные и неотрицательные; прежняя выпуклая цель и native zero-prox readout сохранены. Это не finite-sample F-гарантия, и приоритет специализации не установлен.

| Метод | n8192, 7λ | n8192, 43λ | n131072, 7λ |
|---|---:|---:|---:|
| Raw adaptive PACS | 0/4 | 0/4 | 4/4 |
| Raw + shrinking floor | 0/4 | 0/4 | 4/4 |
| Noise-corrected distances | 2/4 | 2/4 | 4/4 |
| Known Gaussian-kernel weights | 0/4 | 0/4 | 4/4 |
| LC, frozen prior pilot settings | 3/4 | 3/4 | 4/4 |

## Что из этого следует

Уменьшение floor и известные Gaussian-kernel weights не исправили слабую группу. Noise-correction восстановила оба signed случая, но оба conv остались неточными. Это отдельный положительный эффект против raw fusion, а не преимущество над сильными конкурентами. Сохранённый LC на тех же четырёх задачах и с ранее выбранными настройками даёт3/4 приn8192; он не переобучался и не получал нового выбора по структурной истине.

После уплотнения7→43 lambda значений raw остаётся0/4, corrected2/4. Весовые gamma/bandwidth в этой дополнительной проверке фиксированы по предыдущей source-validation; это не полный повторный joint gamma×lambda HPO. Поэтому нельзя объявлять отрицание всех PACS/heat-kernel вариантов. Главные ограничения — оставшиеся conv ошибки, малый уже открытый pilot, отсутствие независимого nonlinear сравнения и неустановленный приоритет.

Заранее зарегистрированное расширение на G4 cohort ещё не запущено: кандидат сначала должен стать конкурентоспособнее на диагностике. Перенос нового corrected-I здесь не измерялся; G12 raw transfer находится в отдельном отчёте. Никакие target-метрики не использовались для настройки.

- [Точное I и разрешение lambda](calibrated_fusion_recovery.png)
- [Выбранные индексаторы и истинные](calibrated_fusion_indexators.png)
- [Измеренные loss/recall кривые](calibrated_fusion_loss_recall.png)
- [Таблица](comparison.csv), [summary](summary.json)

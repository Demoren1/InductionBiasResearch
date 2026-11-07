# Проверка требований G34 — промежуточное состояние

Это перечень требуемых доказательств; запись в перечне не заменяет сами исходники, квитанции и метрики.

| Требование | Текущий источник доказательства | Состояние |
|---|---|---|
| Размеры/K без численных teacher-параметров | blind_contract.load_sizes strict D/H/K/O; registered learner call-chain; ObservationBoundary | Исходники проверены; итог всех квитанций ожидается |
| b собственная обучаемая | blind_dense train parameters; blind_numeric / orientation initial_states получают source beta; GN обучает H независимых b | Реализовано |
| Шум не передаётся | blind_dense_extract.uncertainty по собственным остаткам и rank J; BIC профилирует свою дисперсию | Реализовано |
| Свободная W2 | blind_numeric.profile_head LS по S на каждом GN шаге | Реализовано |
| Нет известных масок/ролей | source-alignment и vote masks из собственных направлений/u; case-name используется как путь | Реализовано |
| Все3 семейства и оба шума | protocol:2Pattern,2DeepSets,6general;20jobs | Dense20/20 и joint20/20 завершены, дополнительные20rank контролей завершены |
| Корректный I | raw выбранный I, final teacher alignment по global code/sign и hidden permutation; mask/sharing отдельно | Подтверждено14/20; остальные6 противоречат полному успеху и требуют продолжения |
| Истина закрыта до всех выборов | Фазовый runtimeguard;20frozenpredictors перед evaluation/private | Проверено:20prediction-файлов и140 квитанций доteacher-аудита |
| Независимая оценка Q | blind_joint_report --phase freeze читает только Q-x; затем отдельный --phase audit | Выполнено:freeze затем отдельныйaudit,21private-commitment совпадает |
| Полная таблица и выбранные I | comparison.csv/summary.csv и selected_indices JSON/NPZ из frozen predictions | Сохранено вevaluation/ |
| Хитмапы реальных Adam весов/индексов, loss/recall | blind_joint_report.render; выбрать первыеcaseсемейств, не лучшиеposthoc | PNG сохранены и репрезентативные/кривые визуально проверены |
| Честный бюджет |16tasks×4096train вместо4×16384, одинаковый total122880 наблюдений/case | Проверено по protocol |
| GPU6/7 свободны от наших запусков | GPU_POLICY, explicit CUDA_VISIBLE_DEVICES, manager allowlist2/3 | Соблюдается |
| Нет software tests | Новые программы — научные эксперименты и отчёты; запускались только Ruff style checks | Соблюдается |

Новая схема предложения также отличается от G33, поэтому сравнение этих серий не является чистой причинной абляцией T. При неудаче не объявлять цель достигнутой по низкой MSE или одному правильному числу групп.

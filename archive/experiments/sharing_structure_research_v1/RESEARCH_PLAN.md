# Поиск общей структуры: программа исследования

Состояние: согласованный заключительный контроль G24 и отчёт завершены. G25: все12новыхI различны и не совпадают с прежними; primary calibrated single/fixed, frozen bootstrap и jointfixed дают12/12 при SD0.2/2. Перенос96новыхфункций/noise: cleanMSE9,07879e−7/1,08719e−4, excessкtrueIcontrol0. Предварительный банк уже точен, дополнительного выигрыша калибровки/joint нет.47параметров/7corehashes неизменны. Session45892/supervisor753528 terminal0, GPU-контексты освобождены; новых запусков нет. Старый широкий goal пользователь очистил; не восстанавливать и не продолжать экспериментальные гипотезы автоматически.

## Завершённый заключительный контроль

Текущий основной документ — [FINAL_REPORT.md](FINAL_REPORT.md), раздел8.5.
[Отчёт G25](results/20261105/joint_prefix_confirmation_g25/report/REPORT.md)
содержит полный source/target аудит, CSV, настоящие train/validation кривые
и хитмапы всех12структур. Редакции G23/G24 сохранены отдельно с исходными
контрольными суммами. Итоговая12/12оценка относится к общему выбору из банка;
19/384исходных и12/240новыхтраекторий закончились частичным отказом.
Плохие семейства не заменялись, перенастройки по контролю нет.

Широкая исследовательская программа ниже сохранена как история требований
и научных ограничений. Её неисполненные пункты не являются поручением
продолжать работу: новый query-based конкурент, свободные bias/readout,
другой шум и проверка приоритета остаются за границей доказанного.
Новая axis-profile схема §35 не реализована и не запускалась.

## Архив требований и доказательств до заключительного контроля

Актуализация 5 ноября 2026: G23 source/common/meta, полный T/V supplement
и перенос на восемь новых кодбуков завершены. По запросу пользователя
подготовлен [итоговый отчёт серии](FINAL_REPORT.md), который начинается
с подробного описания метода, всех определений, математического вывода
и псевдоалгоритма. Факторная абляция отделила расширение банка от критерия;
диагностика V2 разделила codebook drift и локальную integer-ловушку.
Общая цель независимого подтверждения и новизны остаётся active.

Завершённый этап G24: THEORY§34 до кода задаёт общие prefix-counts при
разных согласованных C_t. Старый H выбирает provisional bootstrap,
новый поиск начинает все подключения/группы заново. Свежие независимые
prefix-ответы отделяют joint-метрику от повторного использования шума;
single/joint, fixed/refit и unit/parent_pair разделены. Это открытая
диагностика прежних шести I, не новая confirmation. G24 завершён в
17:24:53 UTC: 288 новых поисков и 12 контролей, 32 пары «условие, I» ×4 независимых
окончательных C-fit из 32 знаковых стартов, полная фиксация предсказателей,
meta-выбор, фиксация I и поздний аудит. Bootstrap и старый осевой C дают
6/6 при SD0.2 и 5/6 при SD2. Калиброванные single-fixed, single-refit,
joint-fixed и joint-refit дают 6/6 при обоих шумах. Все common-маски точны;
условных ошибок выбора при наличии истины в банке нет.
Отдельный joint-эффект не выявлен: банк четырёх независимых single-поисков
после калибровки также даёт 6/6. Только source0: fixed даёт 5/6 при обоих
шумах, random9921 отказывает; refit при SD2 даёт 4/6.
Новый target-transfer и запуск конкурентов G24 не проводились.

Исполнение: CPU, восемь работников; чужие MPS на GPU0–3 не менялись,
карты 6/7 не занимались. Session84944 terminal130: остановлен только
проверенный ожидавший контроллер. Session72251 terminal1: все 300 выходов
поиска готовы, ошибка старого загрузчика до численного fit из-за отсутствующих
неиспользуемых axis_t/axis_vt. Отдельно зарегистрированный адаптер,
session61398 terminal0, прочёл четыре фактических fit-массива и вызвал
неизменённые GN API. Помощник session73015 terminal0 параллельно обработал
только не начатые кандидаты, сохранив текущее состояние прежнего процесса.
CPU serial202.906 → batch87.360 с, ускорение 2.3226×; C-difference6.66e−16.
Протоколы, снимки кода и freezes сохранены; Ruff пройден, software tests нет.
Актуальны FINAL_REPORT.md§4.4/5.2/8.4 и G24/report/REPORT.md.

THEORY§35 записан до следующего обучающего кода: калибровка по покоординатным
целочисленным histogram-профилям осей, согласование имён без предварительного
I, затем joint-prefix с пустого шаблона. Пока это непроверенная гипотеза.
Oracle-only диагностика профилей открытого G24: два парных класса в
local/mixed, четыре различных в random; 64/1 относительных согласований
при T=4. Истинные профили learner не передавались.

| Требование | Достаточное свидетельство | Текущее состояние |
|---|---|---|
| Реальный метод I/C | Воспроизводимый общий жёсткий I, отдельные свежие C задач | Реализованы прежний cross-fit/F-граф и текущий функциональный prefix-count extractor; METHOD.md разделяет их. G23 проверяет общий I четырёх функций, а не перенос численных W |
| Диагноз перед изменением | Измеренный сбой, проверяемая гипотеза, отдельный контроль | G1–G18: спектр, covariance, нули, nonlinear solver и наблюдаемость; G19–G22: C-error, integer feasibility, QP numeric failure и local discrete traps. G22 true-C18/18 против estimated-C12/18 при сильном шуме определяет следующий вопрос |
| Теория до опыта | Наблюдаемый инвариант, условия различимости, цель оптимизации и ограничения | THEORY.md теперь35разделов: axis/prefix count, full task metric, common-I selection, implicit T/V, факторная абляция, запас против C-error, joint-Gram и следующая axis-profile гипотеза. Нет доказательства глобальной сходимости полной процедуры |
| Точное восстановление | Exact mask, unsigned sharing и signed I после freeze; отдельно approximation MSE | Эти три метрики реализованы в G23lateaudit; source1/common4, bank recall и selection failure разделены. G23lateaudit завершён: prefix6/6 и5/6 signedI,6/6масок на обоих уровнях шума; falsebank/selectionfailure разделены |
| Устойчивость | Новые I, независимые C/seed, шум, изменения распределений | Линейные G1–G5 и nonlinear G6–G22 завершены. G22highprecision18/18 приSD.2 и12/18 приSD2. G23:6новыхI×4C с pairedSD.2/2; следующая независимая confirmation и freebias/readout/misspecifiednoise остаются незавершёнными |
| Преимущество над сильными конкурентами | Сильный matched-data native baseline, честное HPO и эффект по независимым I | G23 завершён: common4 prefix6/6 и5/6, denseGN+DP4/6 и4/6, LC3/6 и3/6; T/V direct и literalNeumann20 по1/6 на обоихnoise. Это наблюдаемое преимущество в данном6-I pilot, не доказанное общее превосходство. Нужны независимая confirmation, более полная настройка и matched-query reconstruction контроль |
| Содержательная новизна | Проверка первичных статей/кода, конкретное отличие и абляция его эффекта | Общая новизна не установлена. Finite-alphabet factorization, ReLU multiset recovery, confidence sets, PCA, STE и block optimization известны. Возможный вклад prefix compilation требует собственного эффекта и дальнейшей проверки приоритета |
| Перенос только I | Fresh target C, новые функции, predictor freeze до targettest | G23 завершён:8новыхC/I,48target-задач/метод/noise; оба fit-набора frozen до первогоtestaudit. PrefixcleanMSE8,618e−7 и0,00985693, true-Icontrol8,618e−7 и0,000119519. ПриSD2 excess сосредоточен в random9921 с4sharingerrors |
| Подробный отчёт всей истории | Единый связный отчёт, отрицательные результаты, источники, реальные кривые и I/W | FINAL_REPORT.md содержит метод/определения/математику/псевдоалгоритм,7методов,transfer,абляции,новизну и всю историю; REPORT.md сохраняет подробную хронологию. Новые сводные фигуры и все genuineoptimizer/heatmap artifacts готовы; history сохраняет46 прежних документов и82таблицы |
| Эффективное исполнение | Измеренная пропускная способность при сохранении научной цели | Fixed-I batch2,47×, innerAdam17,90×, targetbatch6,28×. TargetselectedCdiff1,33e−15,perstartvalidationrel8,87e−16. Это отдельные kernel/phase измерения. Supervisors91300/22916terminal0; GPU6/7зарезервированы, использованы только free0–3 |

## Этапы

1. **G0: аудит текущих результатов.** Сопоставить прежние алгоритмы и реальные артефакты. Не смешивать Pattern, nonlinear DeepSets и скрытую линейную свёртку.
2. **G1: информация о структуре в градиенте и спектре.** Сравнить начальный/оптимальный градиент, эмпирическое предобусловливание, градиентные профили, обычный и cross-split спектры. Ковариация входов, шум и возбуждение кодов меняются независимо. Это диагностические базовые алгоритмы, не наш новый метод.
3. **G2: выбрать механизм только после G1.** Сначала записать вывод: какую неоднозначность устраняет механизм, какой наблюдаемый инвариант использует, почему получится дискретный I. Провести аудит близких методов. Если механизм совпадает с известным, использовать его как контроль.
4. **G3: пилот метода и абляции.** Общий I, новые C, независимая selection. Данные не являются искусственным ограничением. Изменения после наблюдения пилота фиксировать до новых подтверждающих семейств.
5. **G4: устойчивость и подтверждение.** Непросмотренные геометрии и source-функции, различные шумы и кондиционирование, новые коэффициенты на переносе. Отдельно проверять маску, группы, знаки, подпространство и ошибки оптимизации C.
6. **G5: конкуренты.** Как минимум shared Yeh, LC/LUT-Q, MSR/MTSR и WSR; сильные прямые OLS/profile и спектральные контроли. Непрерывный U без дискретного выхода не объявлять восстановленным I. Postprocessing выводить отдельной строкой, предоставлять его одинаково сравниваемым представлениям.
7. **G6: полный отчёт и аудит завершения.** История всех методов, теория, воспроизводимость, шумовая устойчивость, ограничения новизны. Цель остаётся active, пока преимущество и остальные требования не подтверждены текущими артефактами.

## Принципы сравнения

- Учитель и роли координат недоступны обучающему алгоритму; известные shape/K раскрываются. Истины открываются для диагностик и после заморозки моделей.
- На target фиксируется I и заново обучаются все числа. Исходные C, optimizer states и meta-initialization не переносятся в основной контроль.
- Architecture-query является внешней обучающей целью. Независимая validation выбирает настройки/checkpoint; новые test-функции не участвуют в выборе.
- Нельзя объявлять новой идеей PCA весов, изменение названия латента, градиентную корреляцию, STE или nearest-code. Требуется содержательная разница с первоисточниками и её отдельный контроль.
- Положительная теория идентификации не является доказательством глобальной сходимости локального projector или k-means.
- Программные тесты не писать и не запускать. Допустимы необходимые научные эксперименты, численные проверки вывода, минимальный lint и просмотр фигур. Старые dirty-файлы не перезаписывать.
- Параллельные численные запуски использовать на свободных GPU. Не останавливать и не вытеснять чужие процессы. Субагенты без прямого разрешения не используются.

## Учёт прогресса

G10/G11 завершены по108 случаям/6 новым геометриям каждая: глобальная совместимость и независимый GLS-codebook улучшают confidence coverage относительно pairwise-графов, но уступают сильному zero-cleanup+DP по точному I на том же общем n. G11 приn131072:75/108 full GLS,72/108 global CI,66/108 full F,89/108 zero+DP. Ошибочных confidence-выдач не было. Общая новизна и сильная конкурентная победа не подтверждены.

Предыдущий завершённый исследовательский ход — progress: созданы и проверены 576 frozen-состояний, независимый перенос, диагностика и графики sharing_gradient_v1. Текущий ход начинает новую цель с сохранением её полного объёма. Ни подготовка этого документа, ни отдельная диагностическая победа не являются завершением цели.

G12 завершён: proper joint group-L2 PACS-контроль,70 settings/пути и solver gap; дополнительные48 source-fit на уже открытомG4. Exact I0/12 при8192/noise1,12/12 при131072/noise1 иnoise.2,1/12 приnoise2. Диагноз — noise-floor adaptive weights и удаление слабой группы. G13 после THEORY§20 завершён:24 pilot-fit, freeze/audit, затем16 matched43-lambda запусков. Raw/shrinking-floor/heat0/4 приn8192, debiased2/4; old frozen LC3/4 на тех же данных. Приn131072 все4/4. Новая широкая конкурентная победа отсутствует; дополнительный G4 cohort зарегистрирован, но не запущен. Большая цель остаётся active.

G14 завершён после THEORY§21:48 frozen состояний на открытом G4, conditional GLS/Schur score с полной covariance и spectral readout. Low-n conditional corrected/whitening даёт12/12 вместо0/12 native fusion, но прямой known whitening-PCA тоже12/12. Variance уменьшилась, превосходства нет. Новая A/B confirmation этого компонента не обоснована отдельным выигрышем. G15 после THEORY§22 завершён:16 frozen noisy G9 состояний,1088 single-step ReLU responses. Во всех12 warm полный GN-прогноз имеет неверный знак actual decrease, exact measured remainder объясняет разницу до1.45e−17; лучший source gain3.82e−12..6.99e−11. Cold8402 всё ещё недооптимизирован и учитывается отдельно. Старый gate/abstentions не менялись.

G16 после THEORY§23 завершён:16 frozen reference×3 независимыхB, studentization по двум noisy labels/x, точное pseudo-target распределение и отдельный oracle-bias аудит. Warm scalar-DP12/12 уже приB262144, graph6/12 иglobalCI9/12; приB1048576 оба12/12. Это4 геометрии×3 correlated warm reference, не12 новых структур. ColdDP2/4 при всехB, confidence0/4→2/4;2 плохих исходных конфигурации сохраняютbias484/385SE приB4194304. CIcoverpseudo48/48,true42/48. Полныеcost/readout/coverage ограничения вG16/report; конкурентная победа не заявлена.

G17 native nonlinear LC завершён после THEORY§24:primary48fit/HPO/freeze/audit, затем§24.5 и32same-data-warmupfit. НачальныйLC/penalty3/4cold, denseGN+DP4/4. При24изпрежних144L-solves наобучениепоB безштрафа LC/penaltyтоже4/4cold+4/4warm. B/validation SHA тежеG16; dense16неповторяли;actualL/refit/runtimeсохранены. Этоизвестныеконтролина4открытыхгеометриях,несобственнаяпобеда.

G18 завершён после THEORY §25: доказана точная неоднозначность Gaussian input-моментов степени≤2, включая covariance линейного sample-gradient при нуле. Два signed-4-code I shape16×16 имеют32/128 активных весов и разные функции, но совпадающие низкие моменты для любых output-transforms/разных source-C. Это не утверждение о всех нелинейных градиентах. Шесть новых перестановок×8C, local/mixed, noise.2/1/2, пять N;12 моделей/180 состояний frozen до oracle-аудита. Известный mixed functional-ANOVA/F-контраст на noise.2 даёт input-граф6/6 для обеих сетей приN8192; приnoise1 наmaxN131072 local4/6,mixed0/6, но mixed partition5/6. Приnoise2 local1/6,mixed0/6; ложных рёбер не было, пропуски велики. Cost до287309824 labels/режим. Это active-query input-граф, **не полный weight-I**, не новый метод или победа. Три фигуры/таблицы/отчёт готовы и просмотрены; fit/audit завершены.

G19 завершён после THEORY §26: direct signed prefix-count candidate, unknown C изeven-axis среза, затем линейныйfit целых multiplicities вместо кластеризацииW. Commonb/readout известны, offset0, одинsource/I — явные ограниченияpilot.18job/126условий frozen до factorial teacher-аудита. Primarynoiseless15/18 (local6,mixed6,random3), noisy0/108. Приnoise.2/r4096 known-C+observed16/18, estimated-C+clean0/18; known-C+clean18/18уникальныхструктур. Приnoise2known-C+observed0/18. Главный measured сбой — C-stage, отдельно amplification LS-rounding. Random8903codeerror.0565 приaxisMSE3e−24/однойточкемеждублизкимиkinks показывает coarse-grid ambiguity continuous amplitudes;8901/8905имеютещёsolver residual. Все fit/audit/report завершены,4фигуры просмотрены,42MiB compact artifacts.

G20 завершён после THEORY §27: integer-intercept DP для релаксированного кодбука, отдельный контроль DP-инициализации и fixed-M constraint, matched axis budget для density/SNR. Все18 запусков/540 состояний заморожены до аудита. Без шума на редкой сетке continuous15/18, seeded/integer17/18; на плотной все18/18. При шуме.2, большом бюджете и плотной сетке continuous14/18, integer15/18 (local4→6, mixed6→6, random4→3); на редкой seeded/integer14 против13. При малом бюджете и noise2 все0/18. Высокий бюджет до275548995584labels/условие; известные bias/readout, одна функция/I, без нового переноса/конкурентов. Отдельного устойчивого выигрыша integerM нет. Использованы5свободных GPU1/2/3/6/7; GPU0 к старту был занят другим процессом. Все fit/audit/report завершены, четыре фигуры просмотрены, подписи numerical W исправлены; около59MiB артефактов.

G21 выполнил этот шаг: feasible Euclidean, full function-QP, integer task-moves и C-refit на прежних C/slices. Новыхaxislabelsнет. Все360primary states и две source-only numerical continuation заморожены доteacher-аудита; допускинеослаблялись. SD.2/high:14→15→16→18/18, Crefit18; SD2/high:0→0→3→12/18,Crefit12. SD.2/low Crefit3→8/18. Итог179точных/145неверныхполных/36rank отказов: нельзя принимать fullI за certificate. Known-C controlSD2:0→0→4→16/18. В двухостаточныхknown-C ошибках teacher имеет меньший observed иvalidation loss, но одиночные ходы ухудшаютloss; это measured integer local trap. Формулаdirect=gᵀd+.5dᵀGd совпала до7.97e−15. До нового опыта теоретически обосновать согласованные discrete exchanges/более сильный integer solver; общий принцип известен, не объявлятьновизной. Затем независимый common-I multi-source/strong-native benchmark, а не новая бесконечнаяknown-bias абляция. Raw G19/G20/G21 сохранить, sourcebudget считать полностью. Нужны matched-query nativeLC/Yeh/LUT-Q, свободныеbias/readout, независимыегеометрии, misspecifiednoise ипереностолькоI. Большаяцельactive.

5 ноября 2026 Mac/SSH Basic Memory восстановлен: G17/G18 сохранены в отдельных тематических заметках, карта и текущее состояние обновлены, устаревший следующий шаг G16 исправлен. Все пять изменённых заметок перечитаны через MCP; синхронизация подтверждена. Копию Obsidian на сервере не создавали. Все научные fit/audit/report процессы G17/G18 завершены.

G19 также синхронизирован с Basic Memory: тематическая заметка «Подсчёт шаблонов индексатора», текущая сводка и карта обновлены и перечитаны. Предыдущий goal-turn (G18 report + восстановление памяти) — progress; текущий G19 — новый выполненный научный опыт с факторным диагнозом, а не повтор статуса. Нет live fit/audit/report процесса или внешнего blocker; следующая работа доступна, большая цель остаётся active.

Последний ход G20 — progress: выполнены новая кодовая абляция, фиксированный протокол, freeze/audit и отчёт с отрицательными контролями. Новая тематическая заметка памяти «Кодбук и целые количества», сводка, карта и следующий шаг G19 обновлены и перечитаны через MCP. Все процессы G20 терминальны; блокировки нет, следующая работа доступна. Общая цель остаётся active.

Текущий ход G21 — progress:360новыхmethodstates, numerically certified QP continuation безповторапринятыхprefix,108known-C diagnostic states, teacher-audit, диагноздвухinteger ловушек и5фигур. Всеfit/audit/oracle/diagnostic workers terminal. ПамятьMac/SSH доступна; новая тематическая заметкаG21, текущая сводка, карта и следующийшагG20 обновлены и перечитаны черезMCP. Отчёт results/20261105/count_metric_g21/report/REPORT.md. Превосходство надnative конкурентами/общийmulti-C перенос/новизна не подтверждены; общаяцельнеcomplete/blocked.

G22 завершён после THEORY§29:360новыхprimary states,wholefreeze доteacher,108known-C diagnostics,5просмотренныхфигур. Pair17/18 иparent17/18 устраняют разныеknown-C traps, объединение18/18; primaryсestimatedC12/18безвыигрыша. Lowaxis+Crefit регрессия8→4, highaxis/SD2+C2gain/2lossприaggregate12. Новыхlabels0,384.15mдискретныхcandidate evaluations,0QP/capfailures; всеworkersterminal. Текущийgoalturn — progress с новым отрицательным контролем, неwait/no-progress. Не блокировать/не завершать большую цель.

Дальше приоритет — независимый common-I multi-source benchmark с переносомI и сильными nativeLC/Yeh/LUT-Q/denseGN. Начальный содержательный вариант: несколько функций с разными численными C одного скрытогоI; whole-I кандидаты извлекаются из каждой функции, все C для проверки кандидата обучаются заново, общийI выбирается на отдельной meta-selection части, ещё не использованной для C-fits. Это известный hypothesis-selection контроль, не новая теория сам по себе; возможный вклад — функциональная count-факторизация безW-target. Не сообщать teacherперестановки/маску, не переноситьC. Нужны multi-C конкурентные objective-level адаптации с правильным shared assignment, сильным numericalwarmup/HPO и одинаковыми observations; их нельзя выдавать за неизменённые авторские скрипты. До запуска — теоретическое условие различимости/ошибкиdictionary, фиксированный protocol и отдельныйI-transfer. G23 ещё не зарегистрирован/не запущен.

Важные детали следующего benchmark: common whole-I/meta-selection outer pipeline применять также к per-source сильным nativeLC/Yeh/LUT-Q/denseGN candidates, чтобы не сравнить наш multi-source выбор с чужим одиночным fit. Отдельный joint multi-C LC — дополнительный сильный контроль, не обязательная подмена существующих per-source objective-level методов. На transfer/multi-C C должен быть свободно signed относительно gauge фиксированногоI; нынешний positive-only single-source refit_codes нельзя применять без изменения. Ни старыеC, ни teacher label matching не передавать. Проверка на новых геометриях/свежихC и новая meta-selection нужны до конкурентной claims.

Синхронизация памяти G22 5ноября не завершена: после успешного чтения существующих заметок обе записи Basic Memory вернули «Mac memory is offline; the request was not sent». Новая тематическая заметкаG22 и изменение текущей сводки НЕ записаны; карта/G21 также не редактировались. MCP exec завершён с этими ошибками, незавершённого запроса нет. Не создавать копию Obsidian на сервере. Источник истины — отчёт/summary G22 в репозитории; при восстановленииMCP создать тематическую заметку «Согласованные дискретные ходы», обновить сводку/карту/следующийшагG21 и перечитать. Это не блокировка большойцели: независимая G23 работа доступна, статусыcomplete/blockedнеставить.

G23 уже зарегистрирован и запущен после THEORY§30. Root results/20261105/shared_structure_g23. Шесть новыхI:local9901/9902,mixed9911/9912,random9921/9922;4sourceC/I;SD.2/2. Dataдлявсех6family созданы, data_frozen.json записан;65568trainqueries/source,275,196,674,048physical labels/source/condition,1,100,786,696,192/family/condition включаяaxis/prefix/fitval/meta. Меткиmeta в отдельномNPZ, никакойteacherI-аудит не открывался. Код shared_structure_data/numeric/native/extract/benchmark.py; все source-файлы и hyperparams snapshot до запуска.

Source extract stage:240jobs,3072траектории (prefix4/задачу, LC/denseDP/LUTQ по12, Yeh24 сnuclear). Все6workers наGPU0/1/2/3/6/7 запущены; чужие4/5не трогали. Живые exec session IDs:GPU0=47754,GPU1=58607,GPU2=48985,GPU3=43700,GPU6=17195,GPU7=12314. Последний specific poll подтвердил все6running,63/240jobs и796/3072trajectories complete. Это изменяющийся снимок; текущий счёт читать из extract/job_*.json. Не рестартовать только поtimeout, читатьтежеhandles/logs/extractfiles. ИдётдолгийnativeHPO; sourcefilesво времяработы не менять. ВсеoldG22workers terminal.

Общий phase-код shared_structure_common.py уже написан, пока НЕ выполнен/зарегистрирован: register послеextract_frozen, banks изtop4uniqueI/source/method,32freshsigned Cfits/каждыйcanonicalI/4задачи, fitfreeze передmetaY, select common4/source1, selectionfreeze передteacher-I audit. Его отдельные snapshots/протокол не меняют source-stage. Сейчаснеоткрыватьteacherрадипромежуточныхметрик. Transfer8newC и итоговыйreport теперь тоже реализованы, см. ниже. Новыйgoalturn — progress (теория/независимыеданные/сильныеконкуренты/живыерасчёты), неverified-wait-only и неblocker.

После вопроса пользователя о прогрессе подготовлены `shared_structure_transfer.py` и `shared_structure_report.py`. Transfer: register только после source-I selectionfreeze, восемь новых targetC/I, fresh32signed starts, фиксированный I, отдельный true-I numerical oracle; target test-Y открывается после whole fitfreeze. Код пока не зарегистрирован/не выполнялся. Report также ещё не выполнялся: читает только полностью frozen source/common/target, таблицы bank recall/selection failure/exact I/source1 vs common4/coverage/transfer excess, все шесть I, actual source optimizer W/Z, fresh-C train/validation и search accuracy/recall на настоящих шагах. LUT-Q Z явно обозначается latent, не quantized forward W. Ruff только новых файлов; software tests не создавались и не запускались. Source workers и их зарегистрированный код не менялись.

Пользователь затем явно подтвердил приоритет: «наша цель найти структуру», «продолжай сравнение». До teacher-аудита добавлена THEORY§30.7: основной критерий exact signed I, независимо точная binary mask и unsigned sharing; late audit разбивает ошибки на mask / sharing при точной mask / signs при точном unsigned I. Изменены только пока не зарегистрированные common/audit и report, source search/данные/HPO прежние. `shared_structure_report --source-only` публикует structure_report после common selectionfreeze/audit, не ожидая target fits. Transfer остаётся вторичной проверкой; MSE не подменяет точное восстановление I. На начало этого продолжения все6source sessions подтверждены running; предыдущий goalturn был progress (реальный код отчёта/переноса и успешная синхронизация памяти), не status-only.

Затем зарегистрирован и запущен `shared_structure_pipeline.py`, exec session70932. Его root comparison_pipeline.json и comparison_pipeline_snapshot фиксируют текущий source/common/report код; эти Python-файлы теперь не менять до завершения/диагностированного сбоя runner. Он НЕ перезапускает и НЕ останавливает source. Подтверждённые host PID шести source поGPU0/1/2/3/6/7:318940/318942/318933/318946/318938/318944. После всех240результатов и терминальных этих PID runner делает freeze_extract → common register/banks → fresh-C fit на свободныхGPU → freeze_fit → meta select → freeze_selection → teacher audit → source-only structure_report. Только затем завершается со state structural_comparison_report_ready (это не завершение общей цели). Transfer автоматически не запускается. Проверяются code hashes; только GPU без чужих compute processes, включая4/5, если к тому времени они освободятся. При worker error все остальные собственные workers доходят до конца, без автоматического повтора completed fits; ошибка записывается в comparison_pipeline_state.json. Последняя непосредственная проверка70932running и state waiting_for_source:83/240, все6source PIDlive. Sandbox обычных exec имеет отдельный PID namespace; pgrep/ps там не видят host CUDA workers, это не признак остановки. Авторитетны escalated host ps/nvidia и исходные exec handles.

До G23 teacher-аудита найдено существенное ограничение fidelity: `shared_structure_native.py` tag `yeh` — joint training P/C с entropy/nuclear, тогда как Algorithm1 и AppendixF.2 статьи требуют T/V bi-level гиперградиент. Это joint ablation, не полная реализация Yeh. Авторский публичный код прочитан через GitHub plugin, treeSHA164fe57aba0939e7db62daf72a899b2f25ce93c7,20Neumann и250inner; библиографическое название исправлено на Equivariance Discovery by Learned Parameter-Sharing. LITERATURE теперь различает G23joint, G4profile-T/T и прежний sharing_benchmark_v1/yeh_shared T/V (его6/6сохраняется). User получил прямое пояснение. До общей сильной claims требуется отдельный signed shared T/V/H bi-level baseline с nativeMAP. Он пока НЕ реализован/НЕ зарегистрирован/НЕ запущен; старый code/данные/метрики не подменять. Прочитаны author implicit_function/neumann_series/train_loops, в том числе фактический Neumann return без внешнего alpha; перед адаптацией проверить конвенцию grad_helpers и отделить literal implementation от exact4×4/корректно масштабированного numerical контроля. Это известный baseline, не новизна. Текущий pipeline продолжает исходные240jobs и первую structural таблицу, baseline дополнить отдельно.

По прямому запросу пользователя выполнена дополнительная очистка: удалены356старых untracked файлов,391.91MiB логических/392.87MiB allocated. Невыбранные SineAR checkpoints, untrained priors старых версий,4промежуточных image search snapshots и byte-identical MADE duplicate. Сохранены45selected/confirmation-candidate checkpoints (не только tracked),6final image selected artifacts, все I/метрики/истории/графики/код/данные, tracked и .git, вся текущая sharing_structure_research_v1. Manifest archive/2026-11-05/artifact_cleanup.json со всеми SHA и причинами; удаление356 и наличие51protected файла проверены один раз. Размер experiments4.6→4.2GiB при продолжающемся росте activeG23. Повторно подтверждён runner70932live; cleanup не останавливал сравнение. Старые image промежуточныеprobability/W replay snapshots удалены, конечные selected_states/selected_indexators/fits/history сохранены.

Mac/SSH Basic Memory снова доступен: созданы тематические заметки «Согласованные дискретные ходы» (G22 complete) и «Общая структура нескольких задач» (G23 in progress), обновлены текущая сводка, карта и следующий шаг G21. Все пять заметок перечитаны через MCP, синхронизация подтверждена. Предыдущая запись об offline сохранена как история сбоя, не текущий blocker. Копию Obsidian на сервере не создавали. Повторное чтение primary Rolnick–Kording подтвердило уже учтённую в LITERATURE G19 границу: query-based neuron reconstruction известен; ещё один такой readout нельзя объявить новым принципом. Новых структурных метрик G23 пока нет.

### Ограничение GPU по указанию пользователя, 5 ноября 2026

Пользователь запретил занимать GPU 6 и 7. Наши source PID 318938/318944 и ожидавший supervisor 403536 остановлены SIGINT; исходные sessions 17195/12314/70932 завершились с exit 130. Их отсутствие подтверждено host ps. Остальные процессы на GPU 6/7 чужие, их не трогали. Действующая политика `GPU_POLICY.json`: только GPU 0–3; GPU 4/5 заняты другой работой. Эта инструкция отменяет прежнее разрешение runner использовать все освободившиеся карты.

Завершённые source jobs сохранены. На момент регистрации продолжения их было 146/240; 34 незавершённых задания исходных shard 4/5 распределены между GPU 0–3. Настройки, seed, исходные наблюдения и зарегистрированные научные Python-файлы не менялись; потеряно только несохранённое промежуточное состояние прерванных jobs. Исходные shard 0–3 остаются в sessions 47754/58607/48985/43700, без перезапуска. Новый scheduling-only `shared_structure_gpu_continuation.py`, Ruff passed, session 82154, supervisor PID 520631; дополнительные source workers PID 520670/520671/520672/520673 на GPU 0/1/2/3. Работают рядом с нашими исходными workers, без чужих процессов на этих картах. Регистрация и снимок: root `gpu_restriction_continuation.json`, `gpu_restriction_continuation_snapshot.py`, идентичности процессов — `gpu_restriction_processes.json`. Старый session 70932 не возобновлять.

Новый supervisor выполняет тот же freeze/common/meta/selectionfreeze/audit/source-only report после всех 240 outputs и завершения source workers. GPU allowlist передаётся неизменённым `run_stage` явно как [0,1,2,3]; последующие фазы используют только свободные карты из этого списка. Общая цель остаётся active. Структурные метрики G23 пока не открыты; предыдущий ETA был рассчитан для шести карт и теперь не гарантируется.

### Полный supplemental T/V/H контроль реализован и зарегистрирован

THEORY теперь31раздел. §31.2/31.5 выводит exactlocalH и mixedVJP в ReLU-клетке; ridge1e−8 делает K4блоки SPD, finiteinnerAdam означает приблизительный implicit gradient текущего iterate, не global/stationary guarantee. `shared_structure_bilevel_optimization.py` содержит независимые scenarioAdam moments, аналитические data-gradients/H/mixed derivative, literal20/directinverse и одинаковый fixedI Cpolish. `shared_structure_bilevel.py` владеет registration/source/nativeMAP/common bank; report/pipeline отдельны. Новые4файла Ruffpassed, softwaretestsнеписались/незапускались. На реальных128queries/4sources CPUderivative diagnostic безteacher/meta дал H2.65e−16,mixed4.02e−16,directP4.22e−16 относительноautodiff; root local_derivative_diagnostic.json содержит наблюдаемыйinputhash/rows. Это не структурный результат.

Root `results/20261105/shared_bilevel_g23`, protocolsha e770660cbf6da5f572696bb81e0f1e5a64d5750b828313ff1097dbb96448f9c5. Регистрация до G23teacher-аудита: те же source6I/4C/noise.2/2, новыхlabels0, общиеdata черезreadonly-use symlink и прежнийdatafreezehash.24jobs(family/noise/inverse)×18configs=432trajectories; perjob batch18, pairs inverse modes/LR используют те же seeds/minibatches/starts. T65568, architectureV8192prefixvalidation, H2048Gaussian, meta2048отдельно. AdamC250steps/outer×500outer, uniformminibatch1024, innerLR.001; outerLR.003/.01/.03, entropy.025,nuclear.05, clipHG25, temp1. Init random.1 или24source0GN/smoothdistancelogitsfloor−20; otherC independent. Source codebooks не переносятся между задачами. Cinnerunbounded; commonfixedI fitter сохраняет прежнийcap16. Literal authorNeumann20 и directHsolve — раскрытыеизвестныеадаптации, не новизна; scaledNeumann residualdiagnosticнеотдельнаямодель. До16uniqueMAPI/mode поHmean; свежие32signedCfits, freezeпередmeta, selectionfreezeпередteacher; толькоscopecommon4, source1неоценивается.

Queue runner `shared_structure_bilevel_pipeline.py`, session84271, supervisorPID547723/startticks1405597348; `queued_pipeline.json/snapshot` фиксируют всекодhashes и parentidentity520631. Подтверждёнrunning, statewaiting_for_primary_structure_report: parentlive/reportfalse. Он НЕ занимаетGPU во времяожидания; ждёт terminal именноэтогоparent+готовыйfrozenreport, затемextract/freezes/commonfits/meta/lateaudit/reportнаfree0–3. Не возобновлятьстарый70932, не использовать6/7. Все зарегистрированные sciencePython включаяновые4файла до завершенияне менять; docsобновлятьможно. Parentна14:12UTC163/240jobs, teacherI/metaYещёнеоткрыты. Итог текущегоgoalturn — прогресс (теория, реализованныйсильныйконтроль, локальнаячисленнаядиагностика, проспективнаярегистрация и живаяочередь), не простоожидание. Fullgoalнеcomplete/blocked: конкурентное превосходство, independentconfirmation/freebias/readout/robustness и priorityещёнедоказаны.


### Действующее продолжение после измерения производительности

С15:10UTC актуален `shared_structure_performance_pipeline.py`, session91300,
PID623686: root `performance_continuation.json` и
`performance_continuation_snapshot/`. Он заменил только ожидавшие
CPU-supervisors520631/547723; sessions82154/84271 terminal130, старые очереди
не возобновлять. Два текущих source writers318933/318946 продолжаются.
На15:13UTC231/240sourcejobs. GPU0–3 только при отсутствии чужой работы,
6/7 всегда исключены. Схема source/HPO/data/seeds сохранена.

Новые зарегистрированные `shared_structure_numeric_batch.py` и
`numeric_batch_diagnostic.py` теперь заморожены вместе с performance runner.
Парное измерение полного fixed-I C-fit:15,78→6,39с (2,47×), одинаковые
выбранные старты, C maxdiff5,33e−15. Принята только быстрая версия с active
compaction, более медленные попытки сохранены вне исполнения. Common
execution регистрируется до первых fits/meta. Supplement inner kernel18×4
сценариев/задач,250Adamsteps:5,20→0,291с (17,90×), Cdiff0; это не full-job
speedup. CPU source diagnosis25,42с против22,54с GPU не обосновал перенос.
Подробности/границы измерений и ссылки — REPORT§30.

Source freeze → common banks/fresh32signedC batched → fit freeze → meta
selection → selection freeze → late teacher audit → source-only report →
прежний зарегистрированный bi-level supplement. Сохранять приоритет первой
структурной таблицы. Общая цель active, численные ускорения не считать
структурным результатом, не объявлять matched-FLOPs или novelty по ним.


### Синхронизация памяти после итогового отчёта

5 ноября 2026: тематическая Basic Memory прочитана, но заключительное обновление не отправлено: MCP сообщил, что Mac и SSH-канал недоступны. Копия Obsidian на сервере не создавалась. Актуальные завершённые результаты, статусы sessions91300/22916 и ограничения зафиксированы в FINAL_REPORT.md, REPORT.md§33 и G23/final_report/document.json. Следующему чату использовать эти файлы, пока общая память не будет доступна.

При завершении отчёта G24 Mac/SSH снова ответил: тематическая заметка
«InductionBiasResearch — общая структура нескольких задач» успешно обновлена
через Basic Memory, project codex. В неё внесены окончательный G23-transfer,
G24, ограничения, статусы процессов и следующий непроверенный вопрос.
Основной актуальный документ — FINAL_REPORT.md; исходный G23-снимок теперь
FINAL_REPORT_G23.md в корне эксперимента, чтобы сохранить его относительные
ссылки. Копия Obsidian на сервере не создавалась.

# Предыдущие эксперименты

Эти каталоги содержат прежние исходники, отчёты, настройки и компактные
результаты. Они выведены из рабочего Python-пакета. Исторические импорты
и команды сохранены как запись экспериментов; для повторного запуска
понадобится восстановить их прежнее окружение и расположение файлов.

| Направление | Каталоги |
|---|---|
| Ранний поиск структур | `random_structure_search_v1`, `structure_search_v2`, `structure_only_v1`, `mlp_structure_discovery_v1`, `mlp_logic_v1`, `hidden_convolution_v1` |
| Генераторы и перенос | `generative_structure_transfer_v1`, `sine_ar_current_v1`, `sine_ar_repair_v1`, `real_image_ar`, `dynamic_signal_v1`, `compositional_sine_v2` |
| Адаптация и оптимизационные режимы | `adaptive_indexators_v1`, `adaptive_tanh_v1`, `basin_structure_transfer_v1`, `basin_routing_v1`, `basin_release_v1` |
| DARTS, дискретизация и sharing | `darts_indexator_v1`, `darts_structured_tasks_v1`, `shared_codebook_mlp_v1`, `sharing_benchmark_v1`, `sharing_diagnostics_v1`, `sharing_gradient_v1`, `sharing_structure_research_v1` |
| Первые задачи | `pattern`, `overlapping_groups` |
| Старые диспетчеры итогового сравнения | `paper_structure_benchmark_v1` |
| Завершённые проверки G34 от 8 ноября | `g34_theory_v1`, `g34_robust_v1`, `g34_support_v1` |
| Контроль одного запуска из якоря от 10 ноября | `anchor_start_ablation_v1` |

`*.summary.npz` сохраняют ограниченные по размеру поля индексаторов, масок,
кодбуков и метрик. Они не заменяют исходные массивы и обучающие выборки.
Тяжёлые чекпойнты, необязательные состояния обучения и кэши удалены.
Большие JSON-записи сохранены отдельно в локальном архиве, исключённом
из Git при уборке 7 ноября; этот тяжёлый архив удалён 8 ноября.
[Протоколы уборки](../2026-11-08/README.md) описывают актуальные границы хранения.

У веток G34 от 8 ноября удалены сырые `results/`, но сохранены исходники
и [сжатые комплекты аудита](../2026-11-08/README.md). Их команды и
регистрации используют прежние имена `experiments.g34_*`. Для повторного
обучения нужно восстановить соответствующие три каталога в `experiments/`
и заново подготовить данные; архивные модули не входят в рабочий пакет.

Зарегистрированная часть G34 осталась в
[рабочем ядре](../../experiments/sharing_structure_research_v1/README.md).
Новейшее сравнение и его команды находятся в
[paper_structure_benchmark_v1](../../experiments/paper_structure_benchmark_v1/README.md).

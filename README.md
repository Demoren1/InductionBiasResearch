# Извлечение структурного inductive bias

Модель изучает функциональный банк решений и предлагает связность, которая помогает новой сети повысить качество относительно настроенного dense при ограниченном числе примеров. Численные веса после выбора маски обучаются заново.

## Текущие материалы

- [Основной отчёт](mds/FINAL_REPORT_2026-09-27.md).
- [История экспериментов и планы](mds/experiments_md/README.md).
- [План Transformer-генератора с суррогатным оценщиком](mds/experiments_md/2026-10-02/01_transformer_surrogate_plan.md).
- [Статья](paper/main.pdf) и [математическая схема генератора](paper/ALGORITHM_TRANSFORMER_RU.md).

## Оставленные проекты

- [pattern](pattern/README.md): обнаружение паттернов, функциональные банки и поиск полезной связности.
- [meta_pattern](meta_pattern/): общие модели и обучение на нескольких pattern-задачах.
- [deepsets_vaae](deepsets_vaae/README.md): DeepSets на MNIST8m, функциональные банки и сравнения моделей.
- [deepsets_z](deepsets_z/): сохранённый DeepSets-проект с латентным кодом.

Датасет сохранён в datasets/mnist8m/, результаты DeepSets — в outputs/deepsets_vaae/, результаты pattern — в pattern/outputs/. Материалы исторических финальных отчётов остаются в mds_archive/ и сохранённой части motif_pair/.

Старый корневой MA-проект и исполняемый код motif-pair удалены 2 октября. [Манифест очистки](mds/experiments_md/2026-10-02/cleanup_manifest.json) перечисляет удалённые файлы и проверки сохранности.

## Среда и короткие проверки

Conda environment ras; [зависимости](requirements.txt), [средства проверки](requirements-dev.txt).

Из корня проекта:

    python -m unittest deepsets_vaae.permutation_bank_encoder_tests deepsets_vaae.permutation_utility_loss_tests

Pattern использует локальные импорты config/data/models. Его проверки запускаются из каталога pattern:

    cd pattern
    python -m unittest discover -s tests -p 'test_data_generation.py'

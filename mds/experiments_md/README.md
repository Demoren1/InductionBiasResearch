# Подробные записи экспериментов

Здесь сохранена подробная история из отчёта за 27 сентября 2026 года, включая дополнения от 1–2 октября. Основной отчёт содержит актуальную постановку и краткие выводы. Перенесены все прежние тексты, числа, таблицы, объяснения графиков и ограничения; сами изображения, массивы и checkpoints остаются в исходных каталогах.

[Основной отчёт](../FINAL_REPORT_2026-09-27.md).

## Эксперименты из отчёта за 27 сентября

- [Функциональное представление, utility-постановка и pattern](2026-09-27/00_functional_method_pattern.md)
- [DeepSets: что дало положительный результат](2026-09-27/01_deepsets_functional_result.md)
- [MNIST8m: выравнивание карт и agreement](2026-09-27/02_mnist_alignment_agreement.md)
- [MNIST8m: ночной запуск и перенос на десять цифр](2026-09-27/03_mnist_night_transfer.md)
- [DeepSets: выравнивание, AE и функциональные baseline](2026-09-27/04_deepsets_alignment_baselines.md)
- [DeepSets: выбор числа связей](2026-09-27/05_deepsets_density_sweep.md)
- [DeepSets: расширенный банк и VAE на функциональных картах](2026-09-27/06_deepsets_expanded_functional_vae.md)
- [DeepSets: другие генераторы и extraction-контроли](2026-09-27/07_deepsets_other_generators.md)
- [DeepSets: аудит source-банка и остановленный подбор плотности](2026-09-27/08_deepsets_source_bank_audit.md)

В [MANIFEST.json](2026-09-27/MANIFEST.json) указаны исходные диапазоны разделов и контрольные суммы текста до переноса.

## Дополнительные проверки 1 октября

- [Pattern: диагностика полос Transformer, новые модели и дискретный поиск](2026-10-01/01_pattern_transformer_debug.md)
- [Pattern: короткая проверка GNN и GNN + flow matching по utility масок](2026-10-01/02_pattern_gnn_flow.md)

- [DeepSets: пересобранный банк нескольких плотностей и dense, полный функциональный state](2026-10-01/03_deepsets_rebuilt_functional_bank.md)
- [DeepSets: GNN и GNN + flow matching, обученные по полезности масок на новом банке](2026-10-01/04_deepsets_gnn_flow_utility.md)

- [DeepSets: отдельные функциональные решения, quality и permutation-consistency loss](2026-10-01/05_deepsets_permutation_dual_loss.md)

## План 2 октября

- [Transformer-генератор, суррогатный оценщик качества и периодические реальные проверки](2026-10-02/01_transformer_surrogate_plan.md).
- [Манифест очистки старого кода с сохранением отчётов, графиков и рабочих банков](2026-10-02/cleanup_manifest.json).

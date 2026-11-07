# Зарегистрированное ядро G34

Здесь оставлены исходные модули, контрольные суммы которых зарегистрированы
в завершённой серии сравнения. Их пути и содержимое сохранены: wrapper
`../paper_structure_benchmark_v1/current_method.py` проверяет эти суммы.
Этот каталог — фиксированная реализация, а не рабочая папка исследовательских
скриптов. Новая численная подгонка, конкуренты, данные и отчёты находятся в
[основном пакете](../paper_structure_benchmark_v1/README.md).

Основной runtime использует `blind_contract`, `blind_runtime`, `blind_dense`,
`blind_dense_extract`, `blind_dense_repair`, `blind_numeric`, `blind_relu`,
`blind_joint_bank`, `blind_orientation_refine` и
`shared_structure_bilevel_optimization`. Остальные сохранённые `blind_*`
участвуют в проверке исходной регистрации. Перечень —
[registered_g34_sources.json](../../archive/2026-11-07/registered_g34_sources.json).

Предшествующие разработки, планы и результаты перенесены в
[архив этой ветки](../../archive/experiments/sharing_structure_research_v1/README.md).
Большие исторические численные массивы усечены; сохранённые `*.summary.npz`
явно содержат только компактные структурные поля и метрики.

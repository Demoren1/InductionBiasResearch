# G23: отдельно эффект банка и общего критерия

Диагностика зарегистрирована до первичного teacher-аудита. Те же frozen I/C и meta-панели, без нового обучения или labels. Это разложение двух эффектов, не независимое подтверждение и не новый extractor.

| SD | Метод | Банк | Критерий | Точный I / 6 | Истина в банке / 6 | Ошибка выбора при наличии истины |
|---|---|---|---|---:|---:|---:|
| 0.2 | prefix | source0 | task0 | 5 | 5 | 0 |
| 0.2 | prefix | source0 | mean4 | 5 | 5 | 0 |
| 0.2 | prefix | all4 | task0 | 6 | 6 | 0 |
| 0.2 | prefix | all4 | mean4 | 6 | 6 | 0 |
| 0.2 | lc | source0 | task0 | 1 | 1 | 0 |
| 0.2 | lc | source0 | mean4 | 1 | 1 | 0 |
| 0.2 | lc | all4 | task0 | 3 | 3 | 0 |
| 0.2 | lc | all4 | mean4 | 3 | 3 | 0 |
| 0.2 | dense_dp | source0 | task0 | 3 | 3 | 0 |
| 0.2 | dense_dp | source0 | mean4 | 3 | 3 | 0 |
| 0.2 | dense_dp | all4 | task0 | 4 | 4 | 0 |
| 0.2 | dense_dp | all4 | mean4 | 4 | 4 | 0 |
| 0.2 | lutq | source0 | task0 | 1 | 1 | 0 |
| 0.2 | lutq | source0 | mean4 | 1 | 1 | 0 |
| 0.2 | lutq | all4 | task0 | 2 | 2 | 0 |
| 0.2 | lutq | all4 | mean4 | 2 | 2 | 0 |
| 0.2 | yeh | source0 | task0 | 1 | 1 | 0 |
| 0.2 | yeh | source0 | mean4 | 1 | 1 | 0 |
| 0.2 | yeh | all4 | task0 | 2 | 2 | 0 |
| 0.2 | yeh | all4 | mean4 | 2 | 2 | 0 |
| 2.0 | prefix | source0 | task0 | 4 | 4 | 0 |
| 2.0 | prefix | source0 | mean4 | 4 | 4 | 0 |
| 2.0 | prefix | all4 | task0 | 5 | 5 | 0 |
| 2.0 | prefix | all4 | mean4 | 5 | 5 | 0 |
| 2.0 | lc | source0 | task0 | 1 | 1 | 0 |
| 2.0 | lc | source0 | mean4 | 1 | 1 | 0 |
| 2.0 | lc | all4 | task0 | 3 | 3 | 0 |
| 2.0 | lc | all4 | mean4 | 3 | 3 | 0 |
| 2.0 | dense_dp | source0 | task0 | 3 | 3 | 0 |
| 2.0 | dense_dp | source0 | mean4 | 3 | 3 | 0 |
| 2.0 | dense_dp | all4 | task0 | 4 | 4 | 0 |
| 2.0 | dense_dp | all4 | mean4 | 4 | 4 | 0 |
| 2.0 | lutq | source0 | task0 | 1 | 1 | 0 |
| 2.0 | lutq | source0 | mean4 | 1 | 1 | 0 |
| 2.0 | lutq | all4 | task0 | 3 | 3 | 0 |
| 2.0 | lutq | all4 | mean4 | 3 | 3 | 0 |
| 2.0 | yeh | source0 | task0 | 1 | 1 | 0 |
| 2.0 | yeh | source0 | mean4 | 1 | 1 | 0 |
| 2.0 | yeh | all4 | task0 | 3 | 3 | 0 |
| 2.0 | yeh | all4 | mean4 | 3 | 3 | 0 |

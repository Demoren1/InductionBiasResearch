# Paired masked-source-refit audit

All90 main conditions repeat extraction after the existing masked_weights source fit. The raw96-condition audit remains separate. The same masks, source observations/noise, budgets and observed-weight selection score are used. This isolates the refit step; it still does not reproduce full functional-bank selection or target transfer. The table is the explicit oracle-budget3/204/204 slice, not blind m selection. Unknown descriptor-gauge cases are excluded. Source_relative_weight_error in the CSV continues to describe the original raw dense observation, not the local masked refit.

| Rank | n | Source noise | G34 exact | Pair exact | Fixed-scale profile exact |
|---:|---:|---|---:|---:|---:|
| 1 | 20 | clean:0 | 2/2 | 2/2 | 2/2 |
| 1 | 20 | observations:0.03 | 2/2 | 1/2 | 2/2 |
| 1 | 20 | observations:0.1 | 1/2 | 0/2 | 1/2 |
| 1 | 20 | weights:0.03 | 2/2 | 2/2 | 2/2 |
| 1 | 20 | weights:0.1 | 1/2 | 1/2 | 1/2 |
| 1 | 80 | clean:0 | 2/2 | 2/2 | 2/2 |
| 1 | 80 | observations:0.03 | 2/2 | 2/2 | 2/2 |
| 1 | 80 | observations:0.1 | 2/2 | 1/2 | 2/2 |
| 1 | 80 | weights:0.03 | 2/2 | 2/2 | 2/2 |
| 1 | 80 | weights:0.1 | 2/2 | 1/2 | 2/2 |
| 1 | 200 | clean:0 | 2/2 | 2/2 | 2/2 |
| 1 | 200 | observations:0.03 | 2/2 | 2/2 | 2/2 |
| 1 | 200 | observations:0.1 | 2/2 | 1/2 | 2/2 |
| 1 | 200 | weights:0.03 | 2/2 | 2/2 | 2/2 |
| 1 | 200 | weights:0.1 | 2/2 | 1/2 | 2/2 |
| 2 | 20 | clean:0 | 0/2 | 1/2 | 0/2 |
| 2 | 20 | observations:0.03 | 0/2 | 0/2 | 0/2 |
| 2 | 20 | observations:0.1 | 1/2 | 1/2 | 1/2 |
| 2 | 20 | weights:0.03 | 1/2 | 0/2 | 0/2 |
| 2 | 20 | weights:0.1 | 0/2 | 1/2 | 0/2 |
| 2 | 80 | clean:0 | 2/2 | 2/2 | 2/2 |
| 2 | 80 | observations:0.03 | 2/2 | 2/2 | 0/2 |
| 2 | 80 | observations:0.1 | 2/2 | 2/2 | 0/2 |
| 2 | 80 | weights:0.03 | 2/2 | 2/2 | 0/2 |
| 2 | 80 | weights:0.1 | 2/2 | 2/2 | 0/2 |
| 2 | 200 | clean:0 | 2/2 | 2/2 | 2/2 |
| 2 | 200 | observations:0.03 | 2/2 | 2/2 | 0/2 |
| 2 | 200 | observations:0.1 | 2/2 | 2/2 | 0/2 |
| 2 | 200 | weights:0.03 | 2/2 | 2/2 | 0/2 |
| 2 | 200 | weights:0.1 | 2/2 | 2/2 | 0/2 |
| 5 | 20 | clean:0 | 2/2 | 2/2 | 0/2 |
| 5 | 20 | observations:0.03 | 2/2 | 2/2 | 0/2 |
| 5 | 20 | observations:0.1 | 2/2 | 2/2 | 0/2 |
| 5 | 20 | weights:0.03 | 2/2 | 2/2 | 0/2 |
| 5 | 20 | weights:0.1 | 2/2 | 2/2 | 0/2 |
| 5 | 80 | clean:0 | 2/2 | 2/2 | 2/2 |
| 5 | 80 | observations:0.03 | 2/2 | 2/2 | 2/2 |
| 5 | 80 | observations:0.1 | 2/2 | 2/2 | 1/2 |
| 5 | 80 | weights:0.03 | 2/2 | 2/2 | 2/2 |
| 5 | 80 | weights:0.1 | 2/2 | 2/2 | 0/2 |
| 5 | 200 | clean:0 | 2/2 | 2/2 | 2/2 |
| 5 | 200 | observations:0.03 | 2/2 | 2/2 | 2/2 |
| 5 | 200 | observations:0.1 | 2/2 | 2/2 | 1/2 |
| 5 | 200 | weights:0.03 | 2/2 | 2/2 | 2/2 |
| 5 | 200 | weights:0.1 | 2/2 | 2/2 | 0/2 |

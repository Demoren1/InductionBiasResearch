# Fresh replication of the previously frozen aSRR procedures

512 new conditional episodes per dataset. No tuning on these query answers.

| Dataset | aSRR | Dense24 | Practical G34 | aSRR minus Dense (pp, 95% CI half-width) |
|---|---:|---:|---:|---:|
|mnist|96.27|96.02|96.09|0.254 ± 0.105|
|omniglot|81.75|81.16|79.95|0.594 ± 0.521|

This replication is separate from the ongoing broader development search.
MNIST uses 100 fitting + 50 within-task validation + 100 query examples/class.
Omniglot uses 10/5/5 in a five-way task. Source/class environments were studied previously.

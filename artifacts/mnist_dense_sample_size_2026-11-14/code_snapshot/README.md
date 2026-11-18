# MNIST sample-size diagnostic

See [PROTOCOL.md](PROTOCOL.md). This changes the source dataset to isolate
sample-size limitations; it does not rerun any Omniglot target score.

Use `/home/udeneev-av/myvenv/bin/python` and
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. Module entry points:

1. `prepare --root LOCAL_STUDY --raw LOCAL_RAW --output COMPACT_OUTPUT`
   reads only MNIST's official training split and freezes registration.
2. `fit --root LOCAL_STUDY --device cuda:0` fits all four sample sizes.
3. `analyze --root LOCAL_STUDY --output COMPACT_OUTPUT` reads retained
   source weights/histories and writes the report, criteria and figures.

Default experiment paths:
`artifacts/local/mnist_dense_sample_size_2026-11-14/{study,raw}` and
`artifacts/mnist_dense_sample_size_2026-11-14`. Full data and numerical
trajectories remain local. No software tests are created or run.

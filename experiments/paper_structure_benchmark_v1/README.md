# Transferable structure benchmark

The current comparison studies transfer of a fixed first-layer structure.
It includes G34 joint calibrated projection, native LUT-Q assignments,
Yeh MAP and continuous assignments, MSR continuous reparameterization,
and dense fitting. The [paper report](../../paper/BENCHMARK_REPORT.md)
describes results, failed controls and limitations.

## Source layout

| Package | Responsibility |
|---|---|
| `protocol` | Synthetic/real observation preparation, data-access boundaries, structure export |
| `methods` | Projection, LUT-Q, Yeh, MSR and the rejected functional refinement |
| `fitting` | Common Adam/Gauss--Newton numerical adaptation and dense controls |
| `execution` | Dependency-aware GPU fleet and export readiness watcher |
| `reporting` | Scoring, source diagnostics, figures/tables and compact artifact export |

`current_method.py` and the registered source files in
`../sharing_structure_research_v1/` retain their original paths and bytes.
They form the immutable G34 reference. Other modules were moved into the
packages above; [module_moves.json](../../archive/2026-11-07/module_moves.json)
records old and new names. The fleet normalizes historical module names in
execution plans. Original frozen records and executed source snapshots are
not rewritten.

## Audit and report reconstruction

The [published artifact bundle](../../artifacts/benchmark_2026-11-07/README.md)
is sufficient to reproduce all reported metrics and manuscript figures on
CPU. From the repository root:

```bash
python -m experiments.paper_structure_benchmark_v1.reporting.evaluate \
  --base artifacts/benchmark_2026-11-07 --phase published-score
MPLCONFIGDIR=/tmp/ibr-mpl-cache \
  python -m experiments.paper_structure_benchmark_v1.reporting.render_report \
  --base artifacts/benchmark_2026-11-07 --paper paper
```

The export contains exact query predictions for every task. Numerical
states are limited to the representative first task, while source display
curves and selected structures are retained. Audit answers were exported
only after the original prediction freeze. They must not be used for new
model selection or tuning.

## New runs

The completed local series remains in `results/20261107/`, excluded from
Git. New runs must use new output roots. Each component is a module with
`--help`; for example:

```bash
python -m experiments.paper_structure_benchmark_v1.protocol.prepare --help
python -m experiments.paper_structure_benchmark_v1.current_method --help
python -m experiments.paper_structure_benchmark_v1.methods.lutq --help
python -m experiments.paper_structure_benchmark_v1.fitting.target_fit --help
python -m experiments.paper_structure_benchmark_v1.execution.fleet --help
```

Historical fleet plans in the artifact bundle describe exact executed
commands, configurations, dependencies and measurements. Their absolute
paths refer to the original checkout; fresh experiments need fresh paths.
The fleet uses GPU 0--5 only, leaving 6 and 7 untouched. CUDA on the original
mt-v100-dev host is available outside its sandbox. No GPU work is launched
by reporting commands.

Training used Python 3.12, Torch 2.6.0+cu118 and torchvision 0.21.0+cu118 on
V100 GPUs. `requirements.txt` records the package versions; the CUDA wheel
build is an environment choice. Reporting dependencies are separated into
`requirements-reporting.txt`.

## Scientific contract

- Source support observations fit numbers. Source numerical-selection
  observations choose numerical states and construct proposals. A separate
  source panel compares refitted candidate structures.
- Only architecture dimensions and the prescribed synthetic code budget are
  exposed to learners. True weights, biases, masks, groups and coordinate
  roles are excluded from extraction and selection.
- Primary transfer carries only I or a native continuous basis. Target
  codebooks, hidden biases, output matrices and offsets start afresh.
- Real development tasks select a global code budget and numerical rule;
  final real tasks fit their support observations only. Synthetic tasks
  receive the common multi-start Adam/Gauss--Newton budget.
- Native MSR initialization transfer and larger-budget mask controls have
  separate rows. Continuous competitors are not post-clustered into I.
- Every predictor was frozen before final query scoring or reference-I
  inspection. `reporting.evaluate --phase freeze` is one-time and refuses
  to overwrite an original registration.

The original extractor was retained on synthetic data and Omniglot.
MNIST development loss triggered source-only hard-assignment refinement;
it failed to improve development transfer and was not adopted. Releasing
weights inside the learned mask is a separately labelled larger-budget
control, with a degree-matched random-mask comparison.

## Measurement scope

Parameter counts cover all numerical model coordinates per target task.
Masked models distinguish active parameters from allocated optimizer
coordinates. Target time includes packed adaptation and query-input
prediction, measured separately without the FLOP profiler. Offline seconds
include profiling overhead and shared scheduling. Torch operation counts
omit NumPy/SciPy and unsupported operators. Synthetic target counts cover
all three support sizes, while target timing covers 1024 fitting examples;
they cannot be divided into one throughput rate.

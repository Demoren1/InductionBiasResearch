# Hidden convolution and new indexators

The learner is a plain 64-36-1 ReLU MLP with zero biases and fixed readout 1/6.
Its teacher uses 36 overlapping 3x3 windows on a hidden 8x8 arrangement, one of
four shared filters per window, and a fixed unknown permutation of the 64 inputs.
The 36 numeric filter coefficients change across functions, as do the filter
assignments to windows. A matrix indexator has 2304 entries: zero disables a
connection, and indices 1-36 select a numeric codebook coefficient. The true
teacher has 324 active connections. The solver knows the MLP shape, codebook
budget 36 and fixed zero code; it is not given windows, geometry, four filters,
nine coefficients per filter or the teacher indexators.

The central question is whether source observations can yield useful connectivity
and weight-sharing assignments for new combinations. VAE is one optional
generator, alongside column prototypes and directly optimized categorical indices.

## Data and source collection

`benchmark.py` writes separate observation and teacher-audit files. Inputs are
independent standard normals; input correlation does not reveal the hidden layout.
There are 32 source functions, each with 16384 support and 2048 query examples.
Source values are permutations of a separated 36-point grid in [-1,1] with small
uniform jitter .003. These values are not given to the solver. Calibration and
target values are new continuous uniform draws in [-1,1].

`source.py` fits four ordinary dense models per source function for 8192 Adam
steps, batch 256. Two starts are unregularized and two have L1-sum coefficient
.0001; source query error selects checkpoints. `extract.py` quantizes fitted
weights, preserving a fixed zero code. Seven zero-treatment candidates include
ordinary KMeans, a log-magnitude split and magnitude cutoffs at [.5,1,2,4,8]
times the median absolute fitted weight. No active-connection count is prescribed.
Each candidate refits only 36 numeric values for 1024 source-data steps; source
query error chooses the quantizer, then the best two models per source function.

The source bank contains 64 selected indexators. Entire source functions are
separated: 48 indexators from 24 functions train priors; 16 from eight other
functions validate reconstruction. Selection is relative quality, not a claim
that the selected indexators are correct. Functional fit and structural accuracy
are evaluated separately in the report.

## Source-only alignment and support refinement

Column permutations are initially aligned by inferred support overlap. Code
labels are aligned by support overlap and groups inferred from column-occurrence
patterns (complete-linkage Jaccard cut .2). Neither number nor size of groups is
specified. Early banks remained almost dense despite small function errors.

The best source training indexator by source query error has 324 connections.
`bootstrap.py --alignment energy` uses its inferred support for source-only
refinement. Hungarian assignment matches each fitted column by the fraction of
its squared weight energy inside each inferred support column. This uses fitted
weights and inferred connectivity, not the teacher layout. Numeric source weights
are then refitted on these connections for 4096 steps, Adam .003 then .001.
Quantization and source-query selection are repeated. A majority support from
the resulting source-training bank is then frozen for generative controls.
The source refinement is exploratory and did not improve mean source function
fit; it improved consistency of inferred connectivity. The report includes both.

## Generators and controls

- `column_search`: nonparametric bank of observed, aligned column indexators.
  Learned categorical logits select a source prototype independently per neuron.
- `entry_search`: categorical logits select one of 36 codes independently at
  each active connection, initialized from source code frequencies.
- `entry_uniform`: same optimization with uniform logits, retaining only the
  source-derived support. This separates connectivity transfer from a code prior.
- Column VAE: local latent8 per column (288 total), shared neural encoder/decoder
  with learned column-position embeddings. Global VAE: one latent64. Both see
  only source indexators and use source-function-disjoint reconstruction selection.
- Whole-bank retrieval, the best whole source indexator, random column mixing,
  an untrained column VAE, random codes on learned support and shuffled support.
- `best_support`: the best source support with independent numeric weights,
  isolating connectivity from weight sharing (324 numeric values in this run).
- Dense Adam, SGDM and unrestricted adaptation from the **original dense** source
  models. `original_dense_source_models.npz` preserves this control after refinement.
- An explicitly named oracle fits 36 values with the true target indexator.

Numeric values are learned anew. Frozen neural generators can optimize their z;
non-neural generators optimize categorical logits with a straight-through
surrogate. Structural updates start after one quarter of numeric training and
occur every four steps. Untrained variables do not receive generator gradients.
All shared-support controls infer that support from source indexators; it is not
a teacher mask. A sparse support does not by itself imply recovered weight sharing.

## Target evaluation and direct structural metrics

Two separate calibration functions choose numeric learning rates. Each target
condition uses eight functions, four starts, 2048 full-support steps, 256 gradient
training examples, 512 checkpoint/restart selection examples and 4096 test examples.
The target label budget therefore exceeds 256. Steps/starts/training presentations
are matched; source computation, FLOPs and structural variable counts are not.
For example, categorical per-entry logits have more variables than a VAE latent.

`run.py --stage refit` fixes each query-selected indexator and independently
refits its numeric values with four fresh starts. This distinguishes bad
indexators from poor joint numeric optimization. `diagnostics.py` and `results.py`
alone access teacher layouts for comparisons (apart from the explicit oracle).
Support F1 measures recovered connections; sharing F1 measures pairs of active
connections assigned the same numeric value, penalizing both incorrect and missed
bindings. Hidden columns and code names are matched for diagnostic display only.
Dense methods are displayed as actual numeric weights, not invented indexators.

`pilot_v2` was developmental: support extraction failed and its test was opened.
After changing source alignment and adding the non-VAE generators, `new_target.py`
generated fresh target functions, codebooks and observations with seed283100.
`pilot_v4` uses that fresh target split. Source and calibration data are unchanged.
The diagnostic reports distance from every source-training teacher arrangement,
allowing a global permutation of the four filter names. The formal space 4^36
does not establish hardness for structured or factorized search algorithms.

## Reproduction

The saved run reuses initial dense source models. For a fresh run, use a new
output path and execute stages in this order from the repository root:

```bash
python experiments/hidden_convolution_v1/benchmark.py
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/source.py
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/extract.py
# Source-only refinement; the parent needs the seven-quantizer extraction.
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/bootstrap.py --parent experiments/hidden_convolution_v1/runs/pilot_v1 --out experiments/hidden_convolution_v1/runs/fresh --alignment energy
python experiments/hidden_convolution_v1/new_target.py --out experiments/hidden_convolution_v1/runs/fresh --seed 283100
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/extract.py --out experiments/hidden_convolution_v1/runs/fresh
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/prior.py --out experiments/hidden_convolution_v1/runs/fresh --kind column
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/prior.py --out experiments/hidden_convolution_v1/runs/fresh --kind global
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/run.py --out experiments/hidden_convolution_v1/runs/fresh --stage calibrate
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/run.py --out experiments/hidden_convolution_v1/runs/fresh --stage evaluate
CUDA_VISIBLE_DEVICES=5 python experiments/hidden_convolution_v1/run.py --out experiments/hidden_convolution_v1/runs/fresh --stage refit
MPLCONFIGDIR=/tmp/induction-mpl python experiments/hidden_convolution_v1/results.py
```

The saved `pilot_v1` used only two quantizers; `pilot_v2` added source-query
magnitude selection while preserving the original raw models/data. `pilot_v3`
tested support-based alignment and was rejected using source quality alone.
`pilot_v4` is the final energy-alignment run. Their source-stage summaries and
audit hashes preserve these distinctions. `results.py` builds final plots and
the developmental v2 comparison from the saved directories; it does not rerun fits.
Datasets, source states, all restart states and best C/I/z, calibration histories,
source snapshots and frozen hashes are retained. No tests were written or run.
Unused preliminary VAE checkpoints and unused global-untrained checkpoints were
removed during the requested disk cleanup; `runs/unused_checkpoint_cleanup.json`
lists them. Checkpoints used in the evaluated methods remain available.

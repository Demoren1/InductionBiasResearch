# Three structure-transfer checks, 2026-10-28

Readable report with mathematics, figures and conclusions:
[STRUCTURE_TRANSFER_2026-10-28.md](../../mds/STRUCTURE_TRANSFER_2026-10-28.md).
The original Pattern result is [reported separately](../../mds/PATTERN_CODEBOOK_2026-10-28.md).

The study contains one synthetic regression family and **modified** MNIST and
Fashion-MNIST binary tasks. The image experiments add seven distractor images
and hide all pixel locations with a shared permutation. They are not standard
unmodified dataset benchmarks. Source functions/classes and held-out target
functions/classes are separated. The input layout stays shared across tasks.

The transferred object is a discrete indexator. Its numeric codebook is fitted
anew on each target task. In NF+VAE variants, NF means neural functional encoder;
the decoder is frozen and only its latent input can be optimized. The synthetic
proposer uses source Gaussian moments; the image proposer uses source-input
dependence plus source-network ablation utility. Neither discovery is attributed
solely to VAE. Direct use of each proposer is an essential control.

## Environment

Run from the repository root. Recorded environment: Python 3.12, PyTorch 2.6.0,
torchvision 0.21.0, NumPy, SciPy, scikit-learn and matplotlib; NVIDIA V100.
The local interpreter was `/home/udeneev-av/myvenv/bin/python`.
The numeric scripts use CUDA. Choose a free GPU with `CUDA_VISIBLE_DEVICES`.
No test suite was written or run. These commands are the actual experiment
pipeline and artifact summarization, not tests.

Download official train/test data with torchvision integrity checks:

```bash
python experiments/structure_search_v2/download.py
```

Data are excluded from Git; their image/label SHA256 hashes are in
`runs/dataset_manifest.json`. Image pixels are divided by 255 and averaged in
nonoverlapping 2x2 cells to obtain 14x14 images.

## Synthetic experiment

The source teacher uses 64 inputs, eight groups of eight, four shared ReLU
filters with 32 numeric coefficients. A dense 64-32-1 network has 2113
parameters; the structured network has 41 numeric parameters, plus the discrete
input permutation. The source teacher permutation is stored only for generating
data and auditing the frozen proposal. The proposer never reads it.

```bash
python experiments/structure_search_v2/scrambled_pattern.py
python experiments/structure_search_v2/scrambled_controls.py
python experiments/structure_search_v2/moment_proposal.py
python experiments/structure_search_v2/scrambled_transfer.py --stage bank
python experiments/structure_search_v2/scrambled_transfer.py --stage prior
python experiments/structure_search_v2/scrambled_evaluate.py --stage calibrate
python experiments/structure_search_v2/scrambled_evaluate.py --stage evaluate
```

The first three steps record source-only exploration and the moment proposal in
`runs/scrambled_pattern_source_pilot_v1`. The directed bank, frozen prior,
source-calibrated baseline models and eight held-out target tasks are in
`runs/scrambled_transfer_v1`. All task/restart metrics and numeric states are
retained. NMSE divides MSE by support-label variance; target test is an independent
finite sample, not an exact population-risk computation. Success means NMSE <0.01.

## Balanced image experiment

Run the following sequence for each dataset (`MNIST` and `FashionMNIST`).
The scripts deliberately have explicit stages; there is no hidden automatic
selection on target test accuracy.

```bash
python experiments/structure_search_v2/distractor_bank.py --dataset MNIST --version v3 --balanced
python experiments/structure_search_v2/distractor_fit.py --dataset MNIST --version v3 --stage pretrain
python experiments/structure_search_v2/distractor_fit.py --dataset MNIST --version v3 --stage tune
python experiments/structure_search_v2/functional_ablation.py --dataset MNIST --parent-version v3 --version v4
python experiments/structure_search_v2/component_proposal.py --dataset MNIST
python experiments/structure_search_v2/distractor_bank.py --dataset MNIST --version v5 --balanced --pool-from experiments/structure_search_v2/runs/distractor_MNIST_v4/component_proposal.npz
python experiments/structure_search_v2/distractor_prior.py --dataset MNIST --version v5 --plain
python experiments/structure_search_v2/distractor_fit.py --dataset MNIST --version v5 --stage pretrain
```

Before v5 calibration, copy the unchanged **fresh dense** source calibration
rows from v3. Both stages use identical source observations and calibration
splits; the prior and masked-model calibrations must be fitted in v5.

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path('experiments/structure_search_v2/runs')
for dataset in ['MNIST', 'FashionMNIST']:
    parent = root / f'distractor_{dataset}_v3'
    out = root / f'distractor_{dataset}_v5'
    rows = json.loads((parent / 'tuning.json').read_text())
    rows = [r for r in rows if r['method'] in ['dense_adam', 'dense_sgd']]
    (out / 'tuning.json').write_text(json.dumps(rows, indent=2))
PY
python experiments/structure_search_v2/distractor_fit.py --dataset MNIST --version v5 --stage calibrate_prior
python experiments/structure_search_v2/distractor_fit.py --dataset MNIST --version v5 --stage evaluate
```

Repeat the last two commands with `FashionMNIST`. On a fresh reproduction use
new output folders or move existing run artifacts first: evaluation skips
method files already present, calibration skips methods already in tuning.json,
and bank/ablation collection protects existing output folders.

The initial balanced source search (v3) contains 6656 mask/numeric states;
1664 relatively good partial solutions are kept. Source feature ablation and
covariance grouping (v4) infer a useful pool of 165/195 coordinates. A second
source search within that pool (v5) contains 1664 states, retaining 416 networks
for the NF+VAE bank. Source optimization inherits numeric states within each
lineage; states are not independent trainings. VAE reconstruction splits use
disjoint lineages; identical mask sets may still recur across lineages.

There are 13 source pairs from classes 0-5; pairs (0,5) and (2,3) are reserved for
calibration. Each source search task uses 24 support and 128 query images per
class. Main and distractor sampling are balanced, without replacement within
each block, using disjoint support/query/calibration image pools. The useful
block is never identified through a repeated-image-rate difference.

Target classes 6-9 produce six pairs x eight support/query draws = 48 cases.
Each case uses 16 support and 64 query images per class and four starts x1024
steps. Each step processes all 32 support examples. The selected restart and
checkpoint minimize query BCE, not test error. Official test evaluation uses
512 images per class, new balanced distractors and independent main/donor RNG
seeds. Repeated cases share classes and may share test images; they are not
48 independent dataset families. No hyperparameters were tuned after opening
this final test evaluation.

Dense controls have the same hidden width 16. Gradient pretraining sees all
3952 source observations and original six-class labels, a nine-option
AdamW grid, then target adaptation or a frozen-body binary head. Target
optimization options are selected on reserved source pairs. Additional sweep
controls allocate four target starts to four different learning rates, with the
same four-fit target budget. Source-search costs are not equalized with
pretraining costs, and FLOPs are not equalized by matching step counts.

`evaluation_frozen.json` records prior, bank, pretrained body, settings and
calibration hashes before the first target evaluation. Effective Gumbel sampling
noise scale is 1.0; it is held fixed per restart during latent optimization.
The no-noise VAE control is separate. Top-k projection has exactly 64 distinct
selected inputs. Its backward pass is an explicitly biased straight-through
relaxation. Source-bank code stores masked dense tensors internally, but target
structural fits contain exactly 64x16 numeric codebook entries, not a hidden
trainable full input-weight matrix.

## Artifacts and plots

```bash
MPLCONFIGDIR=/tmp/induction-mpl python experiments/structure_search_v2/study_results.py
```

This reads frozen results, writes `runs/study_summary.json` and
`runs/study_metrics.csv`, audits stored hashes/mask novelty/support-query image
IDs, and renders six figures under `mds/assets/2026-10-28/structure_transfer`.
Reported summary metrics select each restart by query, then average test
metrics. All-start averages and per-pair values are also available.

The numerical states for all final target starts are saved in each method NPZ,
including selected indices, numeric parameters, best z and checkpoint step.
`prior_samples.npz` stores latent-only projected masks without Gumbel noise;
actual target proposals, including proposal noise, are in method NPZ files.
Do not treat a finite source mask bank as an exhaustive indexator catalogue.

Earlier source explorations are retained with explicit version labels:

- `feature_bank_*`: ordinary single-image sparse-mask pilot; did not beat dense
  source-calibrated training, so it was not evaluated on final target test.
- `distractor_*_v1/v2`: deprecated bootstrap-distractor source exploration.
  Main and distractor image repetition rates could differ; these versions do
  not enter final tables. Correct balanced sampling starts at v3.
- `pointset_pilot_*`: manual DeepSets feasibility control, not learned structure.
  It used source train/validation only and transferred no weights to final runs.

Protocols, histories and frozen source snapshots are versioned. Regenerable
full source hyperparameter weight grids and early failed-pilot checkpoints are
kept locally and excluded from Git; their tuning metrics and histories remain.
Final result NPZ files and selected source banks are versioned. Re-running a
script may produce small GPU-dependent numerical differences.

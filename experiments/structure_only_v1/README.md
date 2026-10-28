# Structure-only transfer control

This is a paired follow-up to the three-case study. The extractor receives only
source mask indices, source task IDs and source lineage split IDs. It never reads
observations, numeric weights, gradients, functional feature utility, the hidden
layout, correlations or a number of latent groups. Source banks were previously
collected by random mask search / mutation with numeric fitting and filtered to
the best quarter per source task/lineage. This filtering uses source query quality;
quality is subsequently equalized for extraction.

Each mask chooses 64 inputs from a vector. This is a general feature-selection
family, not an unrestricted arbitrary-architecture search. Target numeric codebooks
have 64x16 entries, plus biases/readout; they are fitted anew. The prior is a VAE
of **masks**, not the prior neural-functional encoder of numeric networks.

Run from repository root with the existing Python/PyTorch environment and CUDA.
For each dataset (`MNIST` and `FashionMNIST`), train the raw-mask VAE and shuffled
control:

```bash
python experiments/structure_only_v1/prior.py --bank experiments/structure_search_v2/runs/distractor_MNIST_v3/good_bank.npz --out experiments/structure_only_v1/runs/MNIST --inputs 1568
python experiments/structure_only_v1/prior.py --bank experiments/structure_search_v2/runs/distractor_MNIST_v3/good_bank.npz --out experiments/structure_only_v1/runs/MNIST_shuffled --inputs 1568 --shuffle
```

The extractor reads only the `indices`, `task`, `role` arrays in that bank. The
input dimension is a declared model shape. It writes mask-only tokens; the two
unused channels in the reused encoder interface are zero. No image information
is encoded in them. A second source-only regularization setting (`--beta 10`) was
checked in sibling `_beta10` directories; source held-lineage reconstruction
selected beta0.5 for all four cases, recorded in `prior_selection.json`.

```bash
python experiments/structure_only_v1/run.py --dataset MNIST --stage calibrate
python experiments/structure_only_v1/run.py --dataset MNIST --stage evaluate
```

Repeat for FashionMNIST. Numeric learning rate / weight decay are selected on
reserved source class pairs. Source mask reconstruction and target-task numeric
fitting are distinct stages. Source-mask VAE architectures/checkpoints never use
target task labels. The numeric solver necessarily uses each task's own support
labels and query labels for checkpoint/restart selection.

The fixed methods include whole-source-mask retrieval, arithmetic feature votes,
geometric consensus across source tasks, VAE mask proposals, frozen VAE with
optimized z, an untrained decoder, and independently shuffled-source-mask controls.
The consensus score is the mean over source tasks of log(feature frequency +
K/D), with K the mask size and D the input dimension. It uses no grouping and no
input correlations. Independent mask shuffling preserves mask size but destroys
shared coordinate identities. Shuffled masks are artificial negative controls,
not claimed to remain good solutions of their original source tasks.

Evaluation uses the same48 target cases x4 starts x1024 steps as the earlier
study, with exact equality of saved support/query image IDs checked before reuse
of dense controls. The official test generator is unchanged. This is a
**retrospective paired ablation**, not a new blind test or new dataset family.
Frozen prior/bank/tuning hashes are recorded before evaluating these methods.
Regenerable full source calibration weight grids stay local; metrics, histories,
selected priors, source masks and all final numeric states are versioned. Prior
collection/calibration scripts protect or skip existing completed artifacts.

# Agreement between two frozen VAE decoders

Question: does minimizing disagreement between independently trained decoders
recover the shared local-window support, without using that support or task
labels during latent optimization?

The optimization objective is agreement, not task loss. Both decoders remain
frozen. Two independent latent vectors are optimized for every start. Columns
of one generated mask are matched to columns of the other with exact Hungarian
assignment; input coordinates are never permuted. Gold masks, window templates,
target labels, and downstream performance do not participate in this matching,
latent updates, or selection of the best iterate.

## Fixed protocol

- Existing split seed 42: 12 meta-train patterns, 4 held-out patterns.
- Two unconditional VAEs, model/training seeds 42 and 43; identical map-level
  train/validation partition and loader order, seed 42.
- Raw top-10% importance maps, BCE-sum, beta 0.1, 80 epochs, hidden 256,
  latent 32. Checkpoints selected on internal meta-train map validation only.
- 64 independent latent-pair starts, seed 20260906; 1000 Adam updates,
  learning rate 0.03. Each latent starts from a standard Gaussian and is
  projected into the radius-8 ball, both at initialization and after updates.
- Fixed-temperature (0.5) soft top-32 scores, with sigmoid threshold chosen
  to make their sum 32. Implicit differentiation accounts for the threshold's
  dependence on the decoder output. The objective is the mean squared
  difference after column assignment.
- Keep the lowest-agreement-loss iterate separately for every start, including
  initialization. Evaluate all 64 resulting masks. No gold-based start selection.
- Actual final masks are hard global top-32, always 32 active edges.

The soft relaxation can still become diffuse: fixed cardinality alone does not
guarantee binary agreement. Consequently, actual hard-mask Hamming distance and
IoU are primary checks alongside the soft optimization loss. The radius bound
limits latent excursions but does not prove samples remain on the learned data
distribution.

## Controls and evaluation

Both initial decoder samples are retained as paired before/after controls.
A random-search control evaluates 1001 independent latent pairs per start,
with the same Gaussian proposal, radius bound, soft mask mapping, and matching
objective. It keeps one best pair per start. This matches the nominal number
of decoder-pair proposals in gradient search, not wall-clock cost.

The six generated-mask groups (two initial, two optimized, two random-search)
are evaluated along with exact-32 random masks and the ideal support. For each
of the four held-out tasks, all methods use matched initial MLP weights and
the same training batches, 2000 Adam updates, batch size 128. Evaluation data
use the existing OOD evaluation seed rule, distinct from candidate selection
data. Synthetic examples can repeat; this is task-level OOD, not a disjoint
partition of all possible input bit strings.

Only after the generated-mask artifact is saved do we compute best-permutation
gold IoU, exact local-window coverage, diversity, and fresh-MLP accuracy/BCE.
Figures show the first four starts in fixed order, without quality-based
selection. Gold IoU describes closeness to the repository's ideal support,
including its duplicated windows; it is not a test of literal matrix Toeplitz
equality in the decoder's original column order.

Agreement need not imply correct structure: the decoders can share systematic
errors or recover the same nonideal masks. The decisive comparison is whether
optimized masks improve structure and downstream performance over initial
samples and random-search controls.

## Reproduction and artifacts

Activate the project's `ras` conda environment and run with GPU access:

```bash
GPU_ID=2 bash pattern/scripts/10_decoder_agreement.sh
```

The fixed numerical protocol is saved before execution in
`outputs/decoder_agreement/seed_20260906/protocol.json`. The output directory
contains two VAE checkpoints and training diagnostics, `optimization.pt`,
`random_search.pt`, frozen `masks.pt`, hashed provenance, per-task evaluation
records, `summary.json`, and figures. Existing mask artifacts are not silently
overwritten. Evaluation can be resumed with:

```bash
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python pattern/evaluation/run_decoder_agreement.py --stage evaluate --device cuda
```

This experiment uses one pair of VAE seeds and one task split. The 64 starts
measure variation within those fixed decoders, not robustness over independently
trained decoder pairs. Accuracy differences should not be directly combined
with the earlier five-split aggregate.

The follow-up [oracle reachability test](ORACLE_IDEAL.md) explicitly optimizes
latents against ideal support while keeping these same decoders frozen. Its
gold-guided results are recorded separately from this agreement-only experiment.

[The same-VAE task-loss comparison](SINGLE_Z_COMPARISON.md) holds the VAE42
checkpoint and initial latents fixed and compares its agreement-generated
masks against task-adapted single-z masks under a common final evaluation.

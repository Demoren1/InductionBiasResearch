# MSR reproduction

This is a separate experiment series requested on 14 November 2026. It does
not modify the manuscript. The target is the complete experimental scope of
Zhou, Knowles and Finn, *Meta-Learning Symmetries by Reparameterization*,
ICLR 2021, arXiv:2007.02933v3. Read `PROTOCOL.md` before interpreting results.

Runtime data and full checkpoints:
`artifacts/local/msr_reproduction_2026-11-14`.
Compact registrations, logs and scientific figures:
`artifacts/msr_reproduction_2026-11-14`.

The native synthetic comparison follows the paper's task counts and numerical
adaptation. The paper and public source disagree; both versions are preserved
in the references. Synthetic MSR is never post-clustered. Additional fresh-ridge
rows explicitly discard numerical source initializations and learned rates.

The first discrete bank implementation had an inadequate mask grid, and its
global Pair threshold bisection could retain a non-Pair initialization when no
partition was found. These issues were discovered on validation before test
access. The initial evidence remains in `initial_bank_validation`; invalid
fallbacks are excluded from all primary comparisons. The expanded bank adds
all-source energy masks, masked OLS, full Pair threshold search and functional
source-refit ranking. Original valid candidates remain in the selection pool.
The amendment is recorded in `bank_amendment_before_test.json`.

The public MSR image code is marked WIP and is unavailable. Image experiments
are reconstructions from the paper and cited MAML/TorchMeta. They use the
official Omniglot Vinyals split and the checksum-verified TorchMeta MiniImageNet
archive. Primary source runs have 60000 outer steps. The pilot image runs were
archived after correcting tensor crop-boundary interpolation; primary runs
started again from scratch. GPU bilinear transforms are not bitwise PIL.

CNN G34/Pair banks compress a completed MAML meta-initialization and are
explicitly labelled G34/Pair + MAML. Numeric initialization transfer in this
regime matches the native MSR experiment and must not be confused with
structure-only transfer. Convolutional kernels have separate codebooks;
heads, biases and batch normalization remain free. The full-budget endpoint
is the unchanged MAML model. No accuracy at a full endpoint is attributed to
learned sharing.

One-source G34 uses the same Schur calibration and scalar assignment objective
without constructing an entries-by-codes array. Pair forests use 32-bit labels
when intermediate partitions exceed 32767 groups; final compact budgets remain
small. These are scalability changes, not supplied oracle groups.

No software tests are added or run. Real training/evaluation and the minimal
static checks needed to execute these experiments are the validation workflow.

The paper-protocol synthetic study is complete. Read
`artifacts/msr_reproduction_2026-11-14/SYNTHETIC_REPORT.md`; do not interpret its
20-example source setting as an identified unconstrained dense fit. A new
`dense_sample_diagnostic` measures this directly, while the separate
`source_sample_ablation` repeats the complete comparison at 20/80/200 source
examples on a new paired cohort. Original results and target arrays remain
preserved. The full image suite is still in progress; D8 architecture pilots
use validation only and no image target-test comparison is available yet.
Omniglot selects the wider trivial-readout D8 variant (83.32% versus 67.96%
after 2000 source steps); its full run continues to 60000 total steps.
MiniImageNet has separate architecture pilots for each shot count. Rejected
checkpoints, startup compatibility failures and pilot costs are preserved.

`synthetic_diagnostics.py` recreates the roles of the paper's appendix Table 4
and Figures 6–7 from saved native models. The output is in
`artifacts/msr_reproduction_2026-11-14/appendix_diagnostics`: signed Joint inner
rates across three seeds, raw translation U blocks, and all rotation filters
after native adaptation on one validation support example. The m=3 translation
view is labelled as a diagnostic; the main fixed performance row keeps m=70.
No indexer is constructed from MSR and no extra test evaluation is needed.

The source sample ablation is complete. `--report-only` regenerates its
tables and figures from the already-saved per-task results. The report includes
all original baselines, selected code counts, the full m=N endpoint, task
confidence intervals and training-seed dispersion in a numerical export.

That study selects MTSR's outer rate at m=70. The additional
`mtsr_dimension_followup.py` gives MTSR the same m=3/6/15/30/70/204 search as
MSR. It reuses source/validation data and saved baseline checkpoints, freezes
native and fresh choices, then generates a new shared target panel using a
seed registered before the sweep. Previous test results remain separate.
Its outputs go to `artifacts/msr_mtsr_dimensions_2026-11-14`, including the
old m=70 reference evaluated on the same new target tasks.

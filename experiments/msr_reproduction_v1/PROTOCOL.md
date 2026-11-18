# MSR reproduction and comparison

This study is separate from the manuscript. It follows Zhou, Knowles and Finn,
*Meta-Learning Symmetries by Reparameterization*, ICLR 2021, arXiv 2007.02933v3.
The public source revision is b9907f067bdf4ffc20ee83bb466364367cb981fe.

## Scope

1. Table 1: ranks 1, 2 and 5, each with small and large source datasets;
   MAML-FC, MAML-LC, MAML-Conv, MTSR-FC, MSR-Joint-FC, MSR-FC.
2. Table 2: 45 degree rotations and rotations plus flips, with MAML-Conv and
   MSR-Conv. The appendix also shows a 90 degree rotation example.
3. Table 3: augmented Omniglot, 5/20 way and 1/5 shot; augmented MiniImageNet,
   5 way and 1/5 shot. The baselines are MAML, MAML-Big, ANIL, ProtoNet and
   MAML-D8. Native MSR is evaluated without post hoc clustering.
4. Appendix diagnostics: learned filters, symmetry matrices and joint-update
   learning rates. They describe the fitted models, rather than independent
   accuracy benchmarks.

The public repository contains synthetic code only; its image section is WIP.
Reconstructing the augmented experiments from the paper must be labelled as
such. The implementation is not an unchanged execution of unavailable code.

## Protocol discrepancies and decisions

The released synthetic loader uses 8000/2000 tasks rather than Table 5's
400/100 for rank 1 and 800/200 for ranks 2 and 5. The paper specifies 3 train
and 9 test inner updates, outer Adam 0.0005 and initial inner rate 0.02.
The released defaults instead use 1 update, Adam 0.001 and inner rate 0.1.
The paper's main text specifies latent length 70, while the source uses
3/6/30 for ranks 1/2/5. Our primary fixed-parameter reproduction follows the
paper. Tuned variants are additional, clearly identified results.

The low-rank generator is the author's fixed softmax mixture of independent
width-3 standard-normal filters. Rank 2/5 need not admit a small hard indexer.
There is no hidden output permutation and no oracle sparsity mask supplied to
our fully connected models. Locally connected and convolutional author
baselines explicitly have their published architectural priors.

Independent validation coefficient-tasks are added to the original train/test
counts. Training workers can read train and validation panels only. All choices
are frozen before any test targets are read. Synthetic outer rates are selected
from 0.0003/0.0005/0.001 (MTSR: 0.0003/0.001/0.003). MSR additionally compares
latent dimensions 3/6/15/30/70/204. Test adaptation follows the paper, retaining
the learned initialization and rates; a fresh least-squares fit is a separate
structure-only diagnostic, never called the native MSR result.

Our discrete indexers use their own registered budget grid, including the full
4760-weight identity control for the linear experiment. Generalizing the
recovery rule to an observed linear layer requires respecting fixed output
coordinates: there is no hidden-neuron permutation alignment, and independent
output scales are not added to the recovered linear hypothesis class.

Image runs retain query-only augmentation in both training and testing:
random resized crop (scale 0.8 to 1), vertical/horizontal flips independently
with probability 0.5, and bilinear rotation between -30 and +30 degrees.
Support images are not augmented by these transformations. Omniglot class
augmentation by 90/180/270 degrees is distinguished from query augmentation.
Use TorchMeta's Vinyals class split (1028/172/423) and MiniImageNet's Ravi
class split (64/16/20), not the previously used alphabet-separated Omniglot
or digit-pair benchmark. Source training has 60000 outer steps, one inner
step, meta-batch 32/10, and test adaptation has three steps. Evaluate on
1000 common held-out tasks only after validation selection.

## Reporting

The paper's 20 examples per source function leave a 70-input dense matrix
underdetermined. On 14 November the user requested a careful check of this
limitation. `dense_sample_diagnostic.py` uses a new diagnostic cohort to measure
global-optimum source fits, weight error and independent query error for
2/4/10/20/40/70/80/100/200/500 examples. `source_sample_ablation.py` separately
repeats the native and fresh comparisons at 20/80/200 examples with the same
source functions, nested source observations and common new target tasks.
It preserves the full original m/rate grids and three selected-config seeds.
Its choices must be frozen independently before opening its target outcomes.
These studies do not replace the original published-protocol results.

The completed source-count comparison tunes MTSR's source outer rate with its
paper-protocol latent dimension 70. A separate registered follow-up gives MTSR
the MSR dimension grid 3/6/15/30/70/204. It reuses source/validation data and
baseline checkpoints, retains the m=70 rate-only reference, and evaluates all
models on new common target tasks. The registration acknowledges that earlier
test results have been seen and fixes the new panel seed before the sweep.
New target panels are generated only after both native and fresh choices are
frozen. Original results and target arrays remain separate.

CNN banks include matched Lloyd and binary quantization controls, with common
code-value calibration and validation adaptation selection. These are our
controls, not implementations of native LUT-Q. G34/Pair compress MAML numerical
initialization in this comparison. Their full-budget identity control is MAML.

The D8 image architecture is underspecified in the paper and not released.
The first activation-matched reconstruction has four pooled readout features
on Omniglot and low validation accuracy. Before any image-test access, two
2000-step validation-only pilots compare trivial-representation readouts at
the original and fourfold hidden physical widths. On the same 100 validation
episodes, these reach 67.96% and 83.32%, respectively. The wider variant is
selected for Omniglot. Its full run resumes the selected 2000-step checkpoint,
including optimizer and RNG states, to a total of 60000 steps. The rejected
activation-matched run, its checkpoints and log are archived separately.
MiniImageNet has separate 2000-step pilots for both widths and both shot counts;
its layouts are not inferred from Omniglot results. Record the extra pilot
training cost. No architecture is selected using published or held-out target
accuracies. Dataset-specific layouts are stored in `d8_architecture.json`.
The isolated E2CNN 0.2.3 dependency has recorded compatibility patches for
NumPy 2 scalar conversion and PyTorch boolean mask indexing. The latter was
required by MiniImageNet's 1x1 kernels before training could start; failed
startup logs and the exact patch are retained separately.

Keep published numbers, new fixed-protocol reproductions, tuned native models,
and adaptations distinct. Report task-level confidence intervals conditional on
the generated environment, training seed variation, wall time and parameter
counts. Mark failed and incomplete experiments explicitly. Save scientific
figures here; do not change the manuscript or its compiled PDF.

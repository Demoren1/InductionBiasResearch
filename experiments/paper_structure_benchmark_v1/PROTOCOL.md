# Paper benchmark: registered comparison

Status: pilot preparation, 7 November 2026. No new quality claims yet.

The user approved four datasets in total: Pattern 0110, synthetic DeepSets,
Omniglot and one additional real-data problem selected from primary literature.
Candidate fourth problem: the MNIST image digit-sum benchmark of Deep Sets.
The final real-data choice must have an explicit feasible protocol; a pilot
with changed dimensions must not be compared directly to published full-scale
numbers.

## Primary questions

- Does the selected indexator transfer to previously unseen numerical tasks?
- What predictive quality and structural recovery follow from fresh numerical
  fitting, compared with a fully connected layer and native competitors?
- What are the offline extraction and target adaptation costs, measured
  separately in time, memory and counted operations?

## Mandatory comparisons

Fully connected Adam; fully connected Adam plus the numerical refinement used
by our extractor; our unchanged current G34 procedure; random indexators with
matched group/connectivity budgets; Yeh's categorical A-phi with actual
train/validation hypergradient (soft and native MAP reported separately);
continuous MSR; native LUT-Q assignments. Additional LC control is optional.
Continuous MSR is not converted by a post-hoc clustering step.

The original papers and author implementations determine the objectives.
Shared-task or signed-code adaptations must be labelled explicitly. Published
CNN/weight-initialization results are context, not matched fully connected,
structure-only measurements.

## Data and tuning

The synthetic pilot has three independent families per dataset and sixteen
source tasks per family. Each source has 4096 support, 512 numerical selection,
1024 structure selection and 2048 sealed query observations. Eight additional
numerical tasks per family are generated but remain private until extraction
and method/hyperparameter choices are frozen.

Only observations and prescribed network/codebook dimensions enter training.
No true mask, groups, weights, bias, noise, coordinate roles or target labels
can enter proposal construction, numerical fitting or model selection.
Separate learner processes enforce this file-access boundary.

The original G34 procedure is evaluated first. If it is competitive, it is not
modified. If it loses materially, use pilot source observations to diagnose
bank coverage, matching/calibration, discretization and fitting; register a
specific modification and retain the original result. Confirmation uses new
families and noise conditions. Teacher diagnostics are late descriptive audits
and cannot determine a revision's hyperparameters.

## Resources and outputs

GPU 0–5 only; never GPU 6/7. Pack independent tasks, restarts and candidates.
Record synchronized wall time, peak CUDA allocated/reserved bytes, CPU peak
RSS, and registered PyTorch-operator FLOPs. The latter exclude NumPy/SciPy
and unsupported operators and must be labelled accordingly.

Retain per-step train/validation loss, applicable predictive/structural recall,
selected indexators and numerical states, heatmaps, protocol/code hashes,
access receipts and summarized metrics. Report extraction/HPO cost separately
from fresh target fitting. Remove only unneeded heavy artifacts after the
paper, figures and reproducibility records have been preserved.

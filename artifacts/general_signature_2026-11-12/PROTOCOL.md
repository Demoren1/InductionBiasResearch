# Weighted signatures: fresh transfer on general synthetic tasks

12 November 2026. Evaluate the first geometrically ranked signed candidate
from the already frozen uncertainty-weighted signature bank, compared with
the G34 structure selected on the same retained source models. No source
functional reranking of signature candidates, new extraction, or retuning.

Use the six structural families from the 10 November general cohort:
local/quartet_mix/random, each with the original one-output/common-bias and
two-output/independent-bias strata. Source conditions are clean and noise
standard deviation 0.2. The architecture is 16-input, 16-hidden-unit ReLU,
with four independent first-layer codes and a free output head.

Source structures and their recovery outcomes are already known. This is
an exploratory transfer extension on those structures, not an independent
source-cohort confirmation. Freeze the first candidate and the source-only
G34 choice before generating any new target observations. Canonical code
names/signs and hidden-unit order are shared across methods. Canonicalizing
uses no reference structure and preserves the represented function family.

One of twelve conditions (case_04_c1, noisy random) has no complete signature
candidate. Report an extraction failure with no replacement. Its shuffled
control is also absent. Comparisons average only matched available strata
and disclose both the numerator and the full denominator. Also report G34
and Dense on all six families, so missingness cannot hide a hard case.

Generate eight fresh numerical target tasks per structural family, with
seed 26111286017 + 100003 * case number. Use the original distribution of
codebooks, biases, free output matrices and offsets. Inputs are standard
Gaussian. Target responses are clean. For each task generate 1024 support,
256 separate numerical-selection and 2048 sealed query observations.
The same tasks and observations evaluate both source-noise conditions.

Transfer only a signed indexator. Every target codebook, hidden bias,
readout and offset is initialized afresh. Use the unchanged benchmark
target_fit adapter: support N=64/256/1024, numerical selection N/4, 18 Adam
starts (three learning rates, three bias quantiles, two restarts), 2000
updates, four retained starts and 48 Gauss--Newton updates. Dense receives
the same numerical budget; save both its Adam and Adam+GN predictions.
No learned source numerical values initialize a target task.

Shuffle the geometrically selected indexator's entries using seed
26111299117 + 100003 * case number. Preserve total density, signed code
frequencies and code budget; do not claim to isolate mask, signs or
sharing separately. Identical canonical structures within one case share
one target fit, including across noise conditions and method labels.

Primary endpoint: query MSE at N=1024 by geometry and source-noise
condition. N=64/256 describe adaptation with less data. Report every
stratum and task, availability, signed/unsigned recovery and masks.
Do not average raw MSE across different geometries into an overall score.
Two strata per geometry do not support a general source-level confidence
claim; eight numerical functions sharing a structure are not eight
independent structural discoveries. The two strata jointly change output
dimension and bias distribution, so their causal effects are not isolated.

Target workers may read only exported structures, support, numerical
selection and query inputs. Reuse the existing TargetBoundary access audit.
Freeze all predictions and verify registered inputs before a separate
scorer opens query answers or structural references. Read reference
structures only for the late audit. Keep full numerical states locally
and compact predictions, answers, selections, receipts and summaries in
the result bundle. Do not overwrite earlier experiments or the manuscript.
Use free GPUs 0--7, one process per GPU; do not disturb others' jobs.
No software tests.

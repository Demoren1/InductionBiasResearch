# One projection call per anchor

Registered 10 November 2026 before new source fitting or outcome scoring.

Compare three source-search variants on identical newly fitted source models:

- `full`: unchanged G34, four anchors, original support proposals and starts.
- `anchor_fixed`: one call of the original fixed-support projector per anchor,
  initialized by that anchor and using its own support.
- `anchor_free`: one call per anchor with the same initialization; assignments
  include zero, as in the general Algorithm 1 in the manuscript.

The single-start variants remove the additional source/random starts and
alternative support proposals. Consequently `anchor_fixed` is a direct test of
the proposed simplification, not an isolated estimate of the effect of starts
while keeping all four support proposals. `anchor_free` also releases support.
Original signed-orientation handling, retention of the initial bank, source
numerical fitting and independent structure selection are preserved.

Pattern and DeepSets: three structures each, source noise 0/0.1. General:
local/quartet_mix/random, two strata each (one/two outputs and common/independent
bias), source noise 0/0.2. Fresh generation seeds 27111054013 and 27111064019.
Sixteen source tasks per structure, support 4096, numerical validation 512,
structure selection 1024. Eight fresh target tasks per structure.

Source numerical settings remain 12 Adam starts / 2500 steps, four dense GN
starts / 32 steps, two revival rounds, and source-derived initialization.
Initial bank and four anchors are shared across all three variants. Each variant
retains the same maximum of 32 new candidates for four-start / 48-step numerical
refitting, then selects by independent source structure-validation loss.

Primary endpoint: recovery of the signed first-layer indexator, up to the
established code/sign and hidden-unit permutations. Report mask/sharing errors,
whether the selected structure was already in the initial bank, and projection
call/candidate counts. Secondary: clean target MSE at support N=1024 with the
unchanged 18-start / 2000-step Adam and four-start / 48-step GN adapter. All
target numbers are initialized afresh, with free hidden biases and output heads.
Only identical selected structures share a target fit; no source numbers are
transferred. Smaller target budgets are omitted for this bounded quick control.

No true indexators, codebooks, biases, readouts, noise levels or query responses
enter fitting or model selection. All structures and predictions are frozen
before a separate scoring process opens sealed answers. No hyperparameter
tuning on the new results. Historical scores are context, not the paired baseline.

All eight local GPUs are authorized by the user. Other users' processes are
untouched. Use up to two processes per GPU where free memory permits and pack
tasks/restarts inside each process. Limit independent GN batch size for available
VRAM; this changes batching, not the independent numerical update equations.
No software tests. Scratch data/checkpoints live outside the repository; retain
compact results, selected structures, registrations, receipts and source state.

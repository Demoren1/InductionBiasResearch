# One projection run per anchor

Completed on 10 November 2026 and archived after publication. Final results
and the compact CPU audit are in
[artifacts/anchor_start_2026-11-10](../../../artifacts/anchor_start_2026-11-10/README.md).
The `audit` module runs directly from this archived location. The training
commands below record the original layout: they require restoring this
directory under `experiments/`, preparing new observations and retaining the
run registration files. The original scratch directory has been removed.

Paired control of the simplification discussed while editing the Method section.
The original G34 learner is imported unchanged. This package controls the source
candidate starts, bounds independent numerical batches to available VRAM, and
compares source-selected structures on fresh tasks.

The precise scope, seeds, numerical budgets and limits of interpretation are in
[PROTOCOL.md](PROTOCOL.md). The main result is structural recovery; target fitting
uses only support N=1024 to keep the control bounded. `anchor_fixed` and
`anchor_free` both make exactly one projector call per selected anchor. The
original initial bank and signed-orientation handling remain available to the
functional selector; a selected initial structure is marked explicitly.

From the repository root, with CUDA available outside the sandbox:

```bash
/home/udeneev-av/myvenv/bin/python -m experiments.anchor_start_ablation_v1.run \
  --base /tmp/ibr-anchor-start-new --phase prepare
/home/udeneev-av/myvenv/bin/python -m experiments.anchor_start_ablation_v1.run \
  --base /tmp/ibr-anchor-start-new --phase run
/home/udeneev-av/myvenv/bin/python -m experiments.anchor_start_ablation_v1.run \
  --base /tmp/ibr-anchor-start-new --phase report
```

The coordinator writes `state.json`, individual logs, access receipts and a
target-cache map. Its final `frozen.json` is required before scoring opens private
answers. Foreign GPU processes are left running. CPU proposal jobs and GPU
numerical jobs are scheduled separately. Registration records execution-only
formatting before training; no model or source parameters were tuned on outcomes.

The compact report is written to `artifacts/anchor_start_2026-11-10`. Scratch
observations and unselected checkpoints belong outside the repository. No
software tests are created or run by this package.

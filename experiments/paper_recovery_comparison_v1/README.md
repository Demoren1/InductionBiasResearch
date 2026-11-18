# Focused manuscript comparison

The manuscript uses one synthetic recovery benchmark from
`single_task_recovery_v1` and two new, class-disjoint image recognition
comparisons. See [PROTOCOL.md](PROTOCOL.md) for data and selection rules.

Use `/home/udeneev-av/myvenv/bin/python`, `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and a writable `MPLCONFIGDIR`. Module entry points:

1. `prepare --root STUDY --raw RAW --output OUTPUT --dataset mnist|fashion`
   creates fresh source/development/private observations and registration.
2. `workflow --root STUDY --output OUTPUT --device cuda:0` runs the
   dependent training, projection, direct recovery, bank calibration,
   development selection and prediction freeze/scoring stages.
3. `report --base COMPARISON_OUTPUT --paper paper` renders saved results
   and source training curves without fitting or model selection.
4. Once per completed fresh output, `finalize --root STUDY --output OUTPUT
   --raw RAW --dataset mnist|fashion` verifies original source images,
   selection curves and frozen scores and curates the compact evidence.
   Existing completed bundles retain `audit.json` and exact scoring arrays;
   finalization does not refit their models.

The workflow records completed stages in `OUTPUT/workflow.json` and
skips them on a later invocation. Verify the original process is terminal
before resuming; a journal alone is not evidence that a job has stopped.
Never run two workflows concurrently on the same study root. Separate
datasets use separate roots and GPUs. The source module can reuse a
completed, hash-verified Adam archive when resuming its L-BFGS stage.

Current runtime roots are
`artifacts/local/paper_recovery_comparison_2026-11-14/{mnist,fashion}`;
compact results are in
`artifacts/paper_recovery_comparison_2026-11-14/{mnist,fashion}`.
MNIST raw observations reuse the earlier sample-size cache. Fashion-MNIST
is downloaded from its official repository through torchvision's
checksum-verified loader.

G34 and direct pair recovery call their existing implementations. They
share the exact Dense archive and the common numerical adapter. Target
query labels stay private until all methods' numerical rules, structures
and predictions are frozen. Reported intervals describe episodes under
the chosen class split, including possible image/class-pair reuse.

The initial L-BFGS call exposed a noncontiguous-gradient layout error
before its first update. The corrected continuation uses contiguous
parameters and retains the same successful Adam fits. Registration and
the original code snapshot are retained; `source_layout_fix.json` and
`code_snapshot/source_contiguous.py` record the correction.

These are disclosed 2-way 16-shot adaptations on MNIST and Fashion-MNIST;
published standard classification scores are not matched baselines.
No software tests are created or run.

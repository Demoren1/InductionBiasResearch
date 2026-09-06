# Numerical diagnosis: phase-1 SGD failure

## Scope

This document diagnoses the `clip_grad_norm_` failures in the stopped phase-1
SGD-3 capacity sweep.  It does not change the experiment code or its
configuration.

The directly reproduced subject is `xlarge_seed42`; the matching reference is
the recorded trajectory through its last successful outer step.  The direct
failure records for the other affected variants are:

- [`xlarge_seed42.log`](../phase1_dispatch/xlarge_seed42.log): fails at outer step 10.
- [`wide_rank_seed42.log`](../phase1_dispatch/wide_rank_seed42.log): fails after step 11.
- [`unconditional_seed42.log`](../phase1_dispatch/unconditional_seed42.log): fails after step 151.

The exact configuration and successful-step values for the reproduced run are
in [`xlarge_seed42/protocol.json`](../phase1/xlarge_seed42/protocol.json) and
[`xlarge_seed42/training.jsonl`](../phase1/xlarge_seed42/training.jsonl).

## Reproduction

The throwaway replay script `/tmp/meta_u_numeric_replay.py` loaded
`xlarge_seed42/latest.pt` (the step-0 checkpoint), recreated the same model,
Adam outer optimizer, task split, and all deterministic episode seeds, then
replayed on CUDA with `CUDA_VISIBLE_DEVICES=7`.

Its recorded outer steps 1--9 reproduce `training.jsonl` exactly.  In
particular, the first pathological but still accepted outer step is step 7:

| Outer step / slot | Length, pattern | Query BCE | Outer gradient norm |
| --- | --- | ---: | ---: |
| 7 / 0 | 3, `001` | `4.0077e14` | `8.9105e17` |
| 9 / 1 | 4, `1110` | `1.2532e6` | `6.9106e8` |
| 10 / 0 | 3, `100` | `6.4419e18` | float32 `Inf` |

At outer step 10, every individual gradient entry was finite.  Their largest
absolute value was `2.2986e23`, and the aggregate L2 norm computed in float64
was `5.4004e24`.  PyTorch's float32 norm reduction overflowed while squaring
those finite values, producing `Inf`; consequently
`clip_grad_norm_(..., error_if_nonfinite=True)` raised the recorded exception.

## First divergent stage and mechanism

The first divergent stage is the **inner adaptation**, before the outer norm
calculation.  In the exact step-7 / slot-0 episode (`k=3`, pattern `001`),
functional SGD with 50 steps and `lr=3` becomes unstable:

| Inner step | Support loss | max abs logit | max abs (v_1) | max abs (v_2) |
| ---: | ---: | ---: | ---: | ---: |
| 9 | `8.44e-1` | `1.34e1` | `4.03e-1` | `1.84e0` |
| 20 | `3.00e3` | `1.00e4` | `5.50e1` | `2.64e1` |
| 30 | `7.38e6` | `2.33e7` | `3.10e3` | `1.26e3` |
| 40 | `1.05e11` | `3.29e11` | `2.57e5` | `8.93e5` |
| 50 | `1.14e14` | `1.58e15` | `1.12e7` | `3.21e7` |

The full MAML meta-gradient through this divergent path is extremely large;
the float32 norm overflow is therefore a secondary symptom, not the root
cause.

The alternative explanation that column normalization is singular was tested
and refuted for this trajectory.  The minimum raw generator-column norm of
`U2` was `1.16` at step 7 and `2.01` at step 10 (and `U1` was larger), rather
than approaching the `1e-8` normalization floor.

## Fixed-U optimizer control

The throwaway script `/tmp/meta_u_inner_probe.py` reconstructed the exact
model state immediately before step 7, fixed the generated (U), and reran
only the same support/query episode.  The only changed setting was the inner
optimizer.

| Inner optimizer | 50-step query BCE | Result |
| --- | ---: | --- |
| SGD, `lr=3` | `4.0077e14` | diverges while remaining finite in float32 |
| SGD, `lr=1` | `0.2743` | stable |
| Adam, `lr=0.1` | `0.2399` | stable |

This isolates the immediate cause as the SGD-3 inner dynamics for particular
learned (U), rather than generator parameter count alone.  The capacity may
change how often such a (U) is visited, but this diagnosis does not establish
it as the cause.

## Consequence for the experiment

The SGD-3 phase-1 architecture comparison is not a valid capacity comparison:
some variants crash and others have already experienced strongly clipped,
catastrophic episodes.  Do not resume it or disable `error_if_nonfinite` as a
workaround.

The matched 50-step known-length calibration selected inner Adam `lr=0.1`,
init scale `0.1`.  A fresh Adam-0.1 run is the appropriate continuation.  A
float64 norm reduction can preserve clipping of finite, large gradients while
retaining a separate rejection for actual NaN or Inf entries.

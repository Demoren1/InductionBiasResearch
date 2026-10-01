# Independent validation: permutation dual-loss run

**Result: PASS** — all four arms completed and passed the checks below. This is a source-only wiring and replay validation, not a target generalization result.

Focused CPU tests passed in the requested environment:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/udeneev-av/miniconda3/envs/ras/bin/python -m unittest deepsets_vaae.permutation_bank_encoder_tests -v
exit 0 — 6 tests passed
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/udeneev-av/miniconda3/envs/ras/bin/python -m unittest deepsets_vaae.permutation_utility_loss_tests -v
exit 0 — 4 tests passed
```

The full saved-data replay used an inline read-only Python command (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/udeneev-av/miniconda3/envs/ras/bin/python - <<'PY' ...`), exited 0, and wrote the JSON and Markdown reports in this output directory.

| Arm | Updates | Best checkpoint update | Best source objective | Terminal encoder consistency MSE | Same-noise hard-mask agreement | Child plateau records |
|---|---:|---:|---:|---:|---:|---:|
| `set_joint` | 48 | 24 | 0.814565182 | 2.81e-15 | 100.0% | 3520/3520 |
| `set_quality` | 48 | 24 | 0.814565182 | 3.27e-15 | 100.0% | 3520/3520 |
| `position_joint` | 48 | 0 | 0.815244794 | 4.72e-10 | 100.0% | 3520/3520 |
| `position_quality` | 48 | 0 | 0.815244794 | 4.72e-10 | 100.0% | 3520/3520 |

NumPy independently replayed all 64 saved fresh-child query NMSEs per arm from the complete saved child parameters and that arm’s recorded source query tensors. Maximum absolute error against the saved child metrics was **3.58e-07**; the diagonal `[task, policy task, 2 draws, 2 paired initializations]` credits match the saved `own_query` values within **1.79e-07**. The child-state masks reproduce the task/draw/replica ordering. All policy masks contain exactly K=7,526 edges. Replaying both views with the same saved Gumbel noise reproduced the original monitor masks; original/permuted sampled hard masks agreed 100% in these fixed draws. This is a separate check from deterministic `exact_topk` on the unperturbed monitor logits: `position_joint` differs at 2 of 100,352 edge positions (logged agreement 0.999980032; independent count fraction 0.999980070), while `set_joint`, `set_quality`, and `position_quality` are identical. Each deterministic view still has exactly K=7,526 edges.

The 24 dense/random/functional paired control children also replayed: maximum absolute NMSE delta **2.38e-07**; replayed per-task values match the saved control JSON within **1.79e-07**. All 24 controls report plateau flags.

Terminal encoder predictions reproduce the saved monitor outputs within 1e-5. The default set arms are invariant up to floating-point roundoff. Position-biased controls have nonzero order-consistency error at the best checkpoint (minimum MSE **9.62e-09**, versus set-arm terminal MSE at most **3.27e-15**), so the consistency comparison is non-vacuous; the recorded paired Gumbel draws still select the same hard masks. Backpropagation through the saved monitor objective produced finite policy gradients, while the query reward received no gradient.

The source snapshot and live code match all 14 protocol hashes. The functional context and all four referenced bank hashes match. Saved train-teacher references match `train_rows`, are disjoint from held-out rows, and the fixed quality channel independently recomputes from train-row `queryNMSE` only. Old and new source train and validation split hashes are exactly equal. An AST audit evaluates the worker's source loader loop as `source_train` on block 0 (1,000 examples per digit) and `source_validation` on block 1 (300 per digit); its `_read_split` call receives the loop's `block` variable, and the worker passes only those two splits to `task_sets` with no test split. The run uses four previously used source tasks and reuses the fixed 51-query set for policy rewards and monitoring; therefore this monitor is not independent validation. The fresh child sets are separate draws from the same source image pool. No target/test quality claim is supported.

Both the meta-model and child-fit plateau fields were checked against actual saved records: each arm stopped at update 48 with `plateau=true`, `capped=false`, and 3,520/3,520 child plateau flags; each final monitor contains 64/64 plateau flags. These are empirical plateau diagnostics, not a general convergence guarantee. The quality consistency coefficient pairs have identical best source objectives within architecture (set: 0.814565182; position: 0.815244794); no quality-regularizer effect is claimed.

The validation wrote only `independent_validation.json` and `independent_validation.md` under the new dual-loss output root.

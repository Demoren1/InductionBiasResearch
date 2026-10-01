# Independent utility-pipeline review

**Verdict: PASS.** The frozen run matches its recorded protocol and independently replayed test metrics.

The audit covered all eight seeds and eight new target tasks per seed. A separate NumPy evaluator recomputed all 2,304 final NMSE rows from the saved fresh-child parameters and saved 512-set test batches; the largest absolute difference from `results.json` was 1.79e-7.

## Freeze and provenance

All 10/10 source snapshot hashes and live-code hashes match the protocol. Bank protocol and child-selection hashes match their frozen references. All 8/8 bank-context hashes and 16/16 field-model hashes match `source_frozen.json`.

For all 64 target-task folders, external hashes validate the frozen mask file, sparse-child file, dense-child file, and aggregate child manifest. Train/query split hashes match on the post-freeze reload; train/query/test row-ID sets are disjoint. The test split is not present in pre-freeze provenance and is loaded only after the aggregate child-freeze manifest is written. No parameter or mask update follows test loading.

All 512 chosen sparse masks and 1,536 search candidates have exactly K=7,526 edges in their 784×32 maps.

## Convergence and symmetry

All 8192/8192 source, feedback, query-selection, and final child fits used exactly 2,000 updates and had a true descriptive plateau flag. All 16 outer fields plateaued (GNN: 450 updates each; flow: 475–800). The child plateau statistic’s nominal 50-update range is only approximated by two checkpoints 100 updates apart; it does not control the fixed horizon.

On the frozen seed-4100 fields and all eight support-derived task contexts, jointly permuting the hidden axis in state and context (and also the flow initial noise) preserved all 16 exact-K masks. Maximum score residual was 4.82e-5; no top-K membership changed. This is consistent with the field architecture, while leaving the documented tied-cutoff limitation.

## Final test results

| Method | Mean NMSE | SD | n |
|---|---:|---:|---:|
| functional | 0.677501 | 0.115789 | 256 |
| pixel_prior | 0.683885 | 0.114730 | 256 |
| random | 0.661540 | 0.113843 | 256 |
| gnn_single | 0.675272 | 0.116915 | 256 |
| flow_single | 0.663131 | 0.115748 | 256 |
| gnn_search8 | 0.673230 | 0.117340 | 256 |
| flow_search8 | 0.662044 | 0.114862 | 256 |
| functional_search8 | 0.676550 | 0.116237 | 256 |
| dense_tuned | 0.656344 | 0.123319 | 256 |

The standard deviations are descriptive over task×seed×child-initialization records, not confidence intervals over independent task families. Dense-tuned and random are strongest in the aggregate; GNN/flow search do not beat the dense control. Search8 query metrics are post-selection because the same query split chooses the mask; the reported test scores remain untouched by that selection.

The separate source-bank exact audit averages 0.231809 for sparse candidates versus 0.204134 for paired dense controls, with 448/10,240 sparse candidates winning. Its wins vary with density and recipe; it is source-only evidence, not target-test performance.

Limitations to state with the result: only four source task vectors train the utility fields; teacher coordinates use train-only Hungarian alignment; and the known synthetic source generator provides privileged per-image source labels. The outer field monitor is archive-based, not held-out-task validation.

Structured audit details: [`independent_review.json`](/home/udeneev-av/ResearchProject/outputs/deepsets_vaae/20261001_rebuilt_utility_graph_flow/independent_review.json).

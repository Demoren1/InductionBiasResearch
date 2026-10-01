# Figure captions and scope

`per_task_balanced_bce.png`: each panel is one held-out pattern and support budget. The x-axis names the six mask methods; the y-axis is test balanced BCE (equal positive/negative class weight, lower is better). Points average four child initializations within each of outer seeds 8100–8103. Error bars are 95% Student-t intervals over the four seed means (df=3). Each panel uses all 414 test IDs.

`per_task_accuracy.png`: each panel is one held-out pattern and support budget. The x-axis names the six mask methods; the y-axis is natural test accuracy at the strict logit threshold >0. Points average child initializations within each seed; error bars are descriptive 95% t intervals across four seed means (df=3). Accuracy is secondary to balanced BCE.

`paired_deltas.png`: balanced-BCE improvement from dense on the same task, seed, budget and child initialization (dense balanced BCE minus method balanced BCE). The x-axis is held-out pattern; positive y values favor the named mask method. Error bars are descriptive 95% t intervals over four paired seed means (df=3).

`paired_accuracy_deltas.png`: secondary natural-accuracy difference from dense on the same task, seed, budget and child initialization. Positive values favor the named method. Error bars are descriptive 95% t intervals over four paired seed means (df=3).

`toeplitz_quality.png`: left x-axis is exact-mask support IoU after Hungarian matching to gold windows; right x-axis is the fraction of signed `W×mask` squared norm explained by projection onto every constant-offset diagonal. The y-axis is held-out balanced BCE (lower is better). This plot shows the raw, coordinate- and positive-ReLU-gauge-dependent weight score; readout-scaled and affine-column-normalized scores are retained in `structure_metrics.npz`. Gold is used only after all checkpoints and masks are frozen. These associations do not establish that structure caused quality.

`selected_masks_and_weights.png`: columns are methods, top row is the binary 11×8 mask and bottom row is the signed trained `W×mask` matrix. All panels use seed 8100, test task index 0, budget 128 and child initialization 0. Mask colors share [0,1]; signed weights share one symmetric color range across all methods. Rows are input coordinate i and columns are hidden unit h.

`source_bank_quality.png`: overlapping histograms of each seed's frozen source-bank teacher quality, measured as minimum query balanced BCE across saved source-training snapshots. The x-axis is balanced BCE and the y-axis is number of source maps. The source query labels may curate the bank; no target or held-out test labels enter this plot.

`source_bank_loss.png`: x-axis is source-teacher training step and y-axis is balanced BCE averaged over saved teachers within each seed and task. Solid lines show source-train scores and dashed lines show source-query scores. Source-query labels curate the frozen bank; these curves are source diagnostics, not target-task validation or evidence of causal bank use.

`meta_curves.png`: x-axis is meta-training step and y-axis is utility BCE. Thin lines show the fixed train-monitor query curves; opaque lines show meta-validation query curves for each method and outer seed. Each seed has its own trace; values from different fits are never connected. Labels record the initial outer learning rate, which may decay later. The checkpoints were selected on validation data; held-out test tasks are not included.

`child_convergence.png`: columns are support budgets 32 and 128; top-row y-axis is support/query balanced BCE, and bottom-row y-axis is number of runs with a saved observation at that step. Curves average available runs by method; lower panels disclose changing run counts after individual stopping. The vertical dotted line marks the 1,000-step minimum. Query loss selects checkpoints; test loss is absent.

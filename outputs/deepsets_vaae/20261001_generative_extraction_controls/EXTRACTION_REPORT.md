# Source-only extraction controls for generated functional maps

This follow-up isolates extraction from source functional maps. It uses the immutable 4×205 aligned source-train maps, exact top-7526 output masks, and does not access target labels while constructing masks. Target evaluation is the existing paired 800-update protocol at budget 256. Both target populations remain exploratory.

`functional_realign_mean` and `functional_realign_logit_mean` aggregate 820 maps after a fresh source-only Hungarian match to the original pooled functional-train mean. `functional_pixel_marginal` keeps only a pixel prior. `gnn_sample_agreement` applies the same discrete multi-task agreement operator to the saved 4×32 GNN samples; `empirical_sample_agreement` applies it to matched empirical source samples. These are extraction controls, not retrained generators.

## Fresh target results

| Method | NMSE [95% t-CI] | Δ vs functional_mean_large [95% t-CI] |
|---|---:|---:|
| functional_realign_mean | 0.7041 [0.6916; 0.7166] | -0.0025 [-0.0113; +0.0062] |
| functional_realign_logit_mean | 0.7233 [0.7107; 0.7360] | +0.0167 [+0.0073; +0.0262] |
| functional_pixel_marginal | 0.7036 [0.6891; 0.7180] | -0.0030 [-0.0085; +0.0024] |
| gnn_sample_agreement | 0.7368 [0.7229; 0.7506] | +0.0301 [+0.0149; +0.0454] |
| empirical_sample_agreement | 0.7331 [0.7185; 0.7477] | +0.0265 [+0.0084; +0.0445] |

A negative Δ is better for the first method. Intervals are paired across the eight seeds (df=7), conditional on fixed, previously viewed task vectors and without multiplicity correction.

## Figures

- [All extraction methods](extraction_target_methods.png): paired seed means and 95% t-CI for old/fresh populations.
- [Fresh paired contrasts](extraction_target_contrasts_fresh.png): extractions versus direct functional mean, plus saved GNN-sample agreement versus the primary GNN sample-mean mask.
- [Fixed W×M heatmap](extraction_fresh_task0_budget256_effective_weights.png): seed 4100, fresh task 0, budget 256, init 0; the signed scale is shared across the five extraction controls and dense.

[summary.json](summary.json) and [extraction_figure_data.npz](extraction_figure_data.npz) preserve numerical values. The source-only construction, hashes and replay audit are in [protocol.json](protocol.json), [source_controls.json](seed_4100/source_controls.json) and per-seed artifacts.

The bank is itself fixed at 20% retained source edges (80% sparsity) and every extraction uses 30% target top-K; this control cannot select an optimal density. Variable density requires a source bank/evaluation spanning multiple K, with source-only or nested held-out selection.

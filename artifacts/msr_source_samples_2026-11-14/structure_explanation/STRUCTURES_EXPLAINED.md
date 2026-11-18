# Position-dependent filters and selected structures

The generator uses a fixed convex mixing matrix shared by all source and new tasks. Each task draws new width-3 basis filters. One basis filter gives ordinary convolution; two or five yield position-dependent filters. Rank here counts basis filters, not the rank of the full 68-by-70 weight matrix.

## Rank-2 construction

$$
f_i = \alpha_i f_1 + (1-\alpha_i)f_2
$$

The first image uses the first actual rank-2 source function. Its ordinary-convolution reference repeats the first basis filter; its position-dependent panel uses the generator's actual mixing matrix. The first ten output positions are shown.

## Actual selected structures

These are exactly the fresh-validation choices shown in the source-sample comparison. The hard-indexer images preserve the signed code IDs. Equal absolute IDs denote one shared code; a minus sign reverses its contribution. All entries outside the three-tap band are zero in both hard methods. Only twelve output positions are shown.

MSR has no discrete indexer. Its continuous representation is:

$$
\operatorname{vec}(W) = Uc
$$

The MSR images show actual unmodified learned coefficients: 36 band-weight rows and up to 24 original latent columns. No column alignment, clustering, or truncation is used by the evaluated model. Full matrices are in the NPZ files. Column appearance depends on the chosen latent coordinates and should not be interpreted as discrete groups. Color limits clip the upper two percent of absolute displayed coefficients.

| Generator | True task coefficients | G34 groups | Pair groups | MSR columns |
|---|---:|---:|---:|---:|
| 1 basis filters | 3 | 3 | 3 | 6 |
| 2 basis filters | 6 | 70 | 204 | 204 |
| 5 basis filters | 15 | 204 | 204 | 30 |

Pair at rank 2 and both hard methods at rank 5 have 204 singleton groups: they recover the sparse band but do not tie any two nonzero weights. G34 at rank 2 has 70 groups (13 singletons, maximum size six). MSR's nominal latent dimension is not a count of equality groups or necessarily its effective dimension.

The true continuous generator has 3 distinct nonzero equality groups at rank 1 and 204 at ranks 2 and 5. Thus the singleton indexer correctly describes its exact equalities while omitting its additional linear dependencies. The generated family has only 3, 6 or 15 task coefficients despite those 204 distinct entries. This comparison measures the usefulness of the representation as well as how well a discrete indexer is recovered.

## Full exported matrices

| Generator | G34 indexer | Pair indexer | Indexers and continuous matrices |
|---|---|---|---|
| Rank 1 | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank1_g34_index.csv) | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank1_pair_index.csv) | [NPZ](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank1_selected_structures.npz) |
| Rank 2 | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank2_g34_index.csv) | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank2_pair_index.csv) | [NPZ](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank2_selected_structures.npz) |
| Rank 5 | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank5_g34_index.csv) | [CSV](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank5_pair_index.csv) | [NPZ](/home/udeneev-av/InductionBiasResearch/artifacts/msr_source_samples_2026-11-14/structure_explanation/rank5_selected_structures.npz) |

This is a read-only explanation of saved structures, not a new selection or a performance evaluation. No target-test data or source training code is modified.

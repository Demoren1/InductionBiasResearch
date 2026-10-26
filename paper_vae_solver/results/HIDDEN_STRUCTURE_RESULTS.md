# Hidden-structure transfer: completed follow-up experiments

26 October 2026. All runs use the frozen v1 protocol, new target functions, and source solutions from the previous low-frequency dense-SGD banks.

The study contains 132 task contexts and 20,988 returned solutions across all methods. Each stochastic method has twelve searches per context; each specialized non-gradient method has one. These are paired contexts, not independent samples of that many tasks. There are three source-bank replications.

## Shared hidden permutation, full source banks

| Method | 16 weights | 32 weights | 48 weights |
|---|---:|---:|---:|
| Full-vector VAE | 0.0% (0/144) | 38.5% (111/288) | 15.3% (44/288) |
| Inferred components | 0.0% (0/144) | 52.8% (152/288) | 17.0% (49/288) |
| Known source blocks (privileged) | 93.8% (135/144) | 54.9% (158/288) | 18.4% (53/288) |
| Wrong block boundaries | 0.0% (0/144) | 0.7% (2/288) | 0.0% (0/288) |
| Independent source marginals | 7.6% (11/144) | 0.0% (0/288) | 0.0% (0/288) |
| Whole-assignment replay | 0.0% (0/144) | 0.0% (0/288) | 0.0% (0/288) |
| Uniform assignments | 0.0% (0/144) | 0.0% (0/288) | 0.0% (0/288) |
| Enumerate inferred support | 0.0% (0/144) | 55.6% (160/288) | 13.2% (38/288) |
| CEM over inferred components | 0.0% (0/144) | 44.8% (129/288) | 17.7% (51/288) |
| Coordinatewise CEM | 3.5% (5/144) | 0.0% (0/288) | 0.0% (0/288) |
| Exact assignments (privileged) | 100.0% (144/144) | 100.0% (288/288) | 100.0% (288/288) |
| Dense SGD | 8.3% (12/144) | 0.0% (0/288) | 0.0% (0/288) |
| Dense Adam | 38.2% (55/144) | 1.0% (3/288) | 0.0% (0/288) |
| Spectral / inferred support | 0.0% (0/12) | 100.0% (24/24) | 100.0% (24/24) |
| Spectral / uniform assignments | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/24) |
| First moment + radial search | 0.0% (0/12) | 0.0% (0/24) | 0.0% (0/24) |

## Source-bank size and broken transfer

| Method | 32 weights, 8 rows | 48 weights, 8 rows | 48 weights, full bank, new target permutation |
|---|---:|---:|---:|
| Full-vector VAE | 27.4% (79/288) | 1.7% (5/288) | 0.0% (0/288) |
| Inferred components | 51.7% (149/288) | 12.2% (35/288) | 0.0% (0/288) |
| Known source blocks (privileged) | 48.3% (139/288) | 9.4% (27/288) | 0.0% (0/288) |
| Wrong block boundaries | 0.7% (2/288) | 0.0% (0/288) | 0.0% (0/288) |
| Independent source marginals | 0.0% (0/288) | 0.0% (0/288) | 0.0% (0/288) |
| Whole-assignment replay | 0.0% (0/288) | 0.0% (0/288) | 0.0% (0/288) |
| Uniform assignments | 0.0% (0/288) | 0.0% (0/288) | 0.0% (0/288) |
| Enumerate inferred support | 55.6% (160/288) | 9.4% (27/288) | 0.0% (0/288) |
| CEM over inferred components | 44.1% (127/288) | 9.4% (27/288) | 0.0% (0/288) |
| Coordinatewise CEM | 0.0% (0/288) | 0.0% (0/288) | 0.0% (0/288) |
| Exact assignments (privileged) | 100.0% (288/288) | 100.0% (288/288) | 100.0% (288/288) |
| Dense SGD | 0.0% (0/288) | 0.0% (0/288) | 0.0% (0/288) |
| Dense Adam | 1.0% (3/288) | 0.0% (0/288) | 0.0% (0/288) |
| Spectral / inferred support | 100.0% (24/24) | 75.0% (18/24) | 0.0% (0/24) |
| Spectral / uniform assignments | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |
| First moment + radial search | 0.0% (0/24) | 0.0% (0/24) | 0.0% (0/24) |

## Coverage and bank-level variation

The following rows use full source banks and a shared hidden permutation. Coverage is the fraction of searches proposing an exact assignment (up to the simultaneous swap of the two codebook values). Bank rates are descriptive; three banks do not establish precise uncertainty across arbitrary source distributions.

| Blocks | Method | Exact assignment seen | Success in each bank |
|---:|---|---:|---|
| 2 | CEM over inferred components | 0.0% | 9101: 0.0%, 9102: 0.0%, 9103: 0.0% |
| 2 | Inferred components | 0.0% | 9101: 0.0%, 9102: 0.0%, 9103: 0.0% |
| 2 | Enumerate inferred support | 0.0% | 9101: 0.0%, 9102: 0.0%, 9103: 0.0% |
| 2 | Full-vector VAE | 0.0% | 9101: 0.0%, 9102: 0.0%, 9103: 0.0% |
| 2 | Spectral / inferred support | 0.0% | 9101: 0.0%, 9102: 0.0%, 9103: 0.0% |
| 4 | CEM over inferred components | 78.5% | 9101: 43.8%, 9102: 39.6%, 9103: 51.0% |
| 4 | Inferred components | 89.6% | 9101: 57.3%, 9102: 49.0%, 9103: 52.1% |
| 4 | Enumerate inferred support | 100.0% | 9101: 57.3%, 9102: 47.9%, 9103: 61.5% |
| 4 | Full-vector VAE | 79.2% | 9101: 38.5%, 9102: 37.5%, 9103: 39.6% |
| 4 | Spectral / inferred support | 100.0% | 9101: 100.0%, 9102: 100.0%, 9103: 100.0% |
| 6 | CEM over inferred components | 41.3% | 9101: 18.8%, 9102: 16.7%, 9103: 17.7% |
| 6 | Inferred components | 41.3% | 9101: 19.8%, 9102: 14.6%, 9103: 16.7% |
| 6 | Enumerate inferred support | 47.6% | 9101: 11.5%, 9102: 8.3%, 9103: 19.8% |
| 6 | Full-vector VAE | 40.3% | 9101: 11.5%, 9102: 19.8%, 9103: 14.6% |
| 6 | Spectral / inferred support | 100.0% | 9101: 100.0%, 9102: 100.0%, 9103: 100.0% |

## Learned decomposition

Components are recovered statistical dependencies, not estimates of the original geometric block boundaries. Constant coordinates form one fixed component.

| Blocks | Bank | Source rows | Components | Product support | Prior training seconds |
|---:|---:|---:|---:|---:|---:|
| 2 | 9101 | 4 | 2 | 2 | 11.25 |
| 2 | 9102 | 4 | 2 | 2 | 11.06 |
| 2 | 9103 | 4 | 2 | 2 | 11.04 |
| 4 | 9101 | 8 | 5 | 16 | 11.07 |
| 4 | 9101 | 16 | 5 | 16 | 11.21 |
| 4 | 9102 | 8 | 5 | 16 | 11.18 |
| 4 | 9102 | 16 | 5 | 16 | 11.56 |
| 4 | 9103 | 8 | 5 | 16 | 11.77 |
| 4 | 9103 | 16 | 5 | 16 | 11.17 |
| 6 | 9101 | 8 | 7 | 64 | 11.00 |
| 6 | 9101 | 64 | 7 | 64 | 10.79 |
| 6 | 9102 | 8 | 7 | 64 | 10.81 |
| 6 | 9102 | 64 | 7 | 64 | 11.09 |
| 6 | 9103 | 8 | 6 | 32 | 10.93 |
| 6 | 9103 | 64 | 7 | 64 | 11.72 |

## Budget and provenance

Each stochastic search uses 64 fits of 1,024 updates with batch size 64: 4,194,304 gradient-example presentations. Exploration uses independent continuous initializations, unlike the four-repeated-start exploration in the earlier codebook study. Consequently, differences from that study cannot be attributed solely to hidden boundaries.

The structural spectral methods evaluate 64 structures at 64 radial frequencies on 1,024 examples: 4,194,304 prediction-example evaluations. Their antipodal single-sine parameterization uses extra model knowledge; this is not a FLOP- or latency-matched comparison. The first-moment method uses the same 65,536 unique training examples as the stochastic fits, then performs 64 radial evaluations. All selection uses a separate 4,096-example sample; analytic population risk is used only for reporting.

Source acquisition costs remain those of the original full banks. The eight-row conditions consume subsets of these banks and do not demonstrate cheaper acquisition. A full bank costs 0.5, 2, or 8 target gradient-search budgets at 2, 4, or 6 blocks, respectively, excluding VAE training. These costs are paid once per bank, not per reported target context. Timing for jointly batched gradient methods is aggregate and must not be interpreted as per-method latency.

Aggregate batched-search time: 906.2 seconds across workers; specialized-solver time: 2.3 seconds. These sums are not elapsed wall time.

All raw selection indices, selected population losses, and success counts were checked once against the saved candidate arrays during report generation. Software tests were not added or run. Artifacts, immutable plans, source snapshots, and VAE checkpoints are exported to `runs/hidden_structure_v1/`.

## Scope

This is a controlled synthetic study with binary codebooks, recovered noiseless source assignments, shared coordinate semantics, and at most six varying factors. It does not establish generic structure discovery, VAE superiority, end-to-end efficiency, or a new general separation between weight sharing and dense optimization. The two-block case diagnoses source non-identifiability; the fresh-target-permutation case deliberately violates transfer correspondence.

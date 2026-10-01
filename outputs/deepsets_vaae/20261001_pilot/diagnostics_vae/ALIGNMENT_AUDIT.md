# Alignment audit

The VAE was **not** trained on raw unaligned maps. `extract_masks` splits each bank, performs four consensus-alignment rounds on training maps, then aligns validation maps to that training-only consensus before VAE fitting. This audit recreates those same splits.

## Pairwise distance plot

Each bar summarizes four deterministic partners for every one of 26 training maps, over four source tasks and eight seeds. `raw columns` compares original hidden-column order. `consensus fixed` compares the actual consensus-aligned column order. `consensus pairwise opt.` permits an additional pair-specific Hungarian permutation. The pixel-shuffled null independently permutes pixels within every map-column, preserving every column's support density and values while destroying shared pixel geometry.

| Measurement | mean ± SD |
| --- | ---: |
| raw identity MSE | 0.01624 ± 0.00369 |
| consensus fixed-column MSE | 0.01573 ± 0.00368 |
| consensus pairwise-optimal MSE | 0.01442 ± 0.00353 |
| null fixed-column MSE | 0.01692 ± 0.00363 |
| null pairwise-optimal MSE | 0.01570 ± 0.00338 |
| fraction of columns moved by residual pairwise match | 90.8% ± 2.4% |

The raw-to-consensus reduction establishes that the pipeline did apply nontrivial column alignment. The residual consensus-to-pairwise reduction measures remaining permutation mismatch. Compare it to the null reduction: only the excess over the pixel-shuffled null is evidence that matching follows real shared pixel geometry rather than the combinatorial benefit of selecting among 32 sparse columns.

## Mean-map BCE check

The raw original-column mean has BCE 3705.02940 ± 409.25116; the same calculation after the actual consensus alignment has BCE 3678.94741 ± 410.56159. Thus alignment materially changes the averaged target. This does not establish semantic structure: it only prevents the VAE from averaging arbitrarily permuted hidden units.

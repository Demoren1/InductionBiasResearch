# WSR transfer on Omniglot

18 November 2026. On 1,024 new episodes, WSR reached 83.30% accuracy, aSRR
83.08%, and Dense 80.97%. Both fixed grouping methods outperformed Dense.
The paired WSR minus aSRR difference was +0.22 percentage points with a
95% interval of [−0.20, +0.64].

| Method | First-layer codes | Test accuracy | Test cross-entropy |
| --- | ---: | ---: | ---: |
| WSR transfer | 1,024 | 83.30% | 0.5272 |
| aSRR input | 1,024 | 83.08% | 0.5288 |
| Dense | 12,544 | 80.97% | 0.5815 |

| Episodes | Count | WSR | aSRR | Dense | WSR − aSRR, pp (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development | 96 | 83.58% | 84.42% | 82.00% | |
| Test A | 512 | 82.99% | 82.91% | 80.41% | +0.09 [−0.53, +0.70] |
| Test B | 512 | 83.61% | 83.25% | 81.54% | +0.36 [−0.22, +0.94] |

The pooled gains over Dense were +2.33 pp [1.92, 2.74] for WSR and +2.11 pp
[1.74, 2.47] for aSRR. The paired cross-entropy difference between WSR and
aSRR was −0.0016 [−0.0111, +0.0079]. Metrics average episode means. Intervals
are paired and conditional on the fixed source cohort and held-out characters.

## Protocol

WSR uses the same 16 source tasks as the frozen `asrr_input_m1024` bank. For
each source task, we start from its saved Dense fit and run 500 Adam updates
at learning rate 0.003 on that task. Each update applies the scalar WSR
proximal operator to the first layer by sorting and isotonic regression.
Bisection selects the proximal strength using the group count alone, and
rewinding with coefficient 0.98 preserves most of the original block means.
The counts include every first-layer connection. We use the minimum source
validation cross-entropy checkpoint among steps 0, 25, …, 500 for each task.
[WSR, Shakerinava et al.](https://proceedings.mlr.press/v238/shakerinava24a.html)

We treated the 16 WSR groupings as candidates and selected task 105 by
development accuracy, with cross-entropy and task ID as tie-breaks. The
selected source checkpoint was at step 100. Its largest group contains
11,498 of 12,544 weights. The largest aSRR group contains 218 weights.

Both groupings were frozen before drawing the two test cohorts. Each episode
has five classes, with 10 training, five numerical validation and five query
images per class. WSR and aSRR transfer only the integer assignment of weights
to codes. All three methods initialize numerical parameters afresh and run the
same 24 Adam starts through 500 steps, averaging the probabilities from the
four solutions selected by within-episode validation. The scoring program
validated all six prediction files and their access records before reading
query labels.

[Registration](registration.json) · [Source fits](source_fits.json) ·
[Frozen choices](choices_frozen.json) · [Indexators](frozen_indexators) ·
[Exact results](results.json) · [Prediction records](predictions_frozen.json)

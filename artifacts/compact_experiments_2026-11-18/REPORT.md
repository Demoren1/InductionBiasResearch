# Compact manuscript experiments

18 November 2026. The three comparisons use fixed task-specific budgets
of 4, 1,024 and 128 for the synthetic case, Omniglot and UCI HAR,
respectively. Dense uses one independent coefficient per first-layer
weight. Each comparison has eight rows covering Dense, G34, aSRR, LUT-Q,
Yeh MAP, Yeh soft, MSR-FC and WSR.

- **Synthetic.** Retained geometry A (case 04) and the same four numerical
  target functions. Added WSR using the shared numerical solver. G34
  recovers the exact signed structure and has mean query MSE
  $1.91\times10^{-11}$. WSR gives MSE 4.02. [Exact scores](synthetic/results.json)
- **Omniglot.** Evaluated all eight methods on the same 1,024 episodes
  from the initial WSR comparison. The added candidates and their rules
  were fixed on source and development data. Yeh soft leads at 83.79%,
  with a paired advantage of 0.71 pp [0.21, 1.22] over aSRR. WSR and aSRR
  give 83.30% and 83.08%. The separate 4,096-episode confirmation of that
  pair gives WSR minus aSRR +0.11 pp [−0.11, 0.32].
  [Exact scores](omniglot10/results.json) · [Confirmation](../wsr_omniglot_confirmation_2026-11-18/REPORT.md)
- **HAR.** Evaluated on 180 new episodes covering nine test subjects. At
  128 codes, aSRR gives 91.50%, Yeh soft 90.87%, WSR 85.02%, and Dense
  86.43%. The paired aSRR gains over WSR and Dense are 6.48 pp [4.00, 8.96]
  and 5.07 pp [2.90, 7.24]. Intervals use the nine subject means.
  [Exact scores](har3/results.json)

The tables include the wall time to generate source candidates. On
Omniglot, the fixed aSRR construction from saved Dense fits takes 0.41 s,
and the matched WSR source procedure takes 23.76 s. On HAR these times
are 7.08 s and 6.82 s. The time records combine verified replays and
archived runs with their original construction scopes. Yeh MAP and Yeh
soft share one source run. The 1,024-code Yeh run on Omniglot uses MAGMA
for the spectral penalty, following a cuSOLVER error. Both runs are
recorded in the logs. [Source timing and scope](source_times.json)

[Registration](registration.json) · [Synthetic table](synthetic_table.tex) ·
[Omniglot table](omniglot10_table.tex) · [HAR table](har3_table.tex)

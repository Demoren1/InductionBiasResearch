# Transfer after source-only budget selection

Twenty-four cases, each with independently selected1/5-support rules. All choices froze before generating256 new target functions per rank. Numbers average two descriptor-subset banks on the same target panel. The explicit source observation noise is10% output RMS; target support/query are clean. These are structure-only closed-form fits, not native MSR adaptation.

| r | Source n | Source labels | Support | G34 | Pair | G34+SVD | Pair+SVD | Oracle equality I |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 20 | clean | 1 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 1 | 20 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 1 | 20 | observations | 1 | 0.00729209 | 1.55425 | 9.53532e-05 | 0.000706105 | <1e-14 |
| 1 | 20 | observations | 5 | <1e-14 | <1e-14 | 6.31966e-05 | 0.000638783 | <1e-14 |
| 1 | 200 | clean | 1 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 1 | 200 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 1 | 200 | observations | 1 | <1e-14 | 0.671858 | <1e-14 | 3.22869e-05 | <1e-14 |
| 1 | 200 | observations | 5 | <1e-14 | <1e-14 | <1e-14 | 2.93386e-05 | <1e-14 |
| 2 | 20 | clean | 1 | 0.271497 | 1.19304 | <1e-14 | <1e-14 | 1.19217 |
| 2 | 20 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 2 | 20 | observations | 1 | 0.230367 | 1.19398 | 0.000837312 | 0.000837805 | 1.19217 |
| 2 | 20 | observations | 5 | <1e-14 | <1e-14 | 0.000713855 | 0.000714697 | <1e-14 |
| 2 | 200 | clean | 1 | 0.317365 | 1.19217 | <1e-14 | <1e-14 | 1.19217 |
| 2 | 200 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 2 | 200 | observations | 1 | 0.300644 | 1.23837 | 7.47887e-05 | 7.47887e-05 | 1.19217 |
| 2 | 200 | observations | 5 | <1e-14 | <1e-14 | 6.45942e-05 | 6.45942e-05 | <1e-14 |
| 5 | 20 | clean | 1 | 0.654205 | 0.654205 | <1e-14 | <1e-14 | 0.654205 |
| 5 | 20 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 5 | 20 | observations | 1 | 0.654205 | 0.654205 | 0.00190562 | 0.00190562 | 0.654205 |
| 5 | 20 | observations | 5 | <1e-14 | <1e-14 | 0.00142122 | 0.00142122 | <1e-14 |
| 5 | 200 | clean | 1 | 0.654205 | 0.654205 | <1e-14 | <1e-14 | 0.654205 |
| 5 | 200 | clean | 5 | <1e-14 | <1e-14 | <1e-14 | <1e-14 | <1e-14 |
| 5 | 200 | observations | 1 | 0.654205 | 0.654205 | 0.000162459 | 0.000162459 | 0.654205 |
| 5 | 200 | observations | 5 | <1e-14 | <1e-14 | 0.000124322 | 0.000124322 | <1e-14 |

Full m/d/ridge selections and conditional per-case intervals: transfer_metrics.csv. grouped_results.json computes intervals across the256 target functions after averaging the two fixed banks; it does not count512 independent targets. All target calculations are float64, so exact fits can reach approximately1e-29. Source labels retain the original float32 precision; noiseless SVD floors can be1e-17. Oracle controls were never selectable. Full dense N4760 and with_full aliases are in the CSV.

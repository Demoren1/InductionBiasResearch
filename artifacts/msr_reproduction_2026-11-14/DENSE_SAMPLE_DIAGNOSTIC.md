# Source dense sample sufficiency

This is a new diagnostic cohort, not a replacement for the MSR benchmark. The dense least-squares solution is a global optimum of the source training objective. Its training error can vanish even when its weights and predictions on new inputs are wrong. Each row averages 32 source functions over three independently generated seeds.

| Examples per source | Rank 1: relative weight error | Rank 2 | Rank 5 |
|---|---:|---:|---:|
| 2 | 0.9716 | 0.9711 | 0.9709 |
| 4 | 0.9431 | 0.9424 | 0.9424 |
| 10 | 0.8576 | 0.8566 | 0.856 |
| 20 | 0.7149 | 0.7139 | 0.7135 |
| 40 | 0.4296 | 0.4282 | 0.4278 |
| 70 | 1.318e-10 | 5.747e-09 | 7.102e-12 |
| 80 | 4.997e-15 | 4.842e-15 | 5.015e-15 |
| 100 | 1.528e-15 | 1.537e-15 | 1.497e-15 |
| 200 | 3.496e-16 | 3.503e-16 | 3.493e-16 |
| 500 | 1.069e-16 | 1.068e-16 | 1.056e-16 |

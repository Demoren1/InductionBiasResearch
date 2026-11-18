# Provisional structure-only validation

All numerical source initialization and learned adaptation rates are discarded. Ridge and representation choices use the same 200 validation tasks shown here. These are selection scores from source seed 0, not held-out test results or native MSR adaptation.

| Case | MSR fresh | G34 fresh | Pair fresh | Oracle Conv | Dense full |
|---|---:|---:|---:|---:|---:|
| rank1_n20 | 0.00057074 | 2.0577e-15 | 2.0577e-15 | 2.0577e-15 | 2.9358 |
| rank1_n200 | 0.00054149 | 2.0577e-15 | 2.0577e-15 | 2.0577e-15 | 2.9358 |
| rank1_n80 | 0.00065303 | 2.0577e-15 | 2.0577e-15 | 2.0577e-15 | 2.9358 |
| rank2_n20 | 0.011495 | 0.64205 | 1.2264 | 0.34108 | 1.8028 |
| rank2_n200 | 0.014053 | 0.64205 | 1.2264 | 0.34108 | 1.8028 |
| rank2_n80 | 0.00902 | 0.49548 | 1.2264 | 0.34108 | 1.8028 |
| rank5_n20 | 0.052138 | 0.58676 | 0.58676 | 0.35353 | 0.86324 |
| rank5_n200 | 0.041306 | 0.58676 | 0.58676 | 0.35353 | 0.86324 |
| rank5_n80 | 0.032244 | 0.58676 | 0.58676 | 0.35353 | 0.86324 |

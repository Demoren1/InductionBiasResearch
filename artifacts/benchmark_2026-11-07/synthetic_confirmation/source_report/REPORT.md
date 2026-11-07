# Development source comparison

These are source structure-selection losses, not independent transfer results. Target data and true indexators remain unopened. All numbers include required dense preprocessing when it is used for initialization. Continuous Yeh/MSR need DenseAdam only, while hard common refits need the full descriptor preparation. Dense operation/memory counts come from a one-time unchanged measurement replay per shape; its numerical outputs never enter selection. Torch FLOPs exclude NumPy/SciPy and unsupported operators.

| Dataset | Method | Cases | Source MSE | Offline seconds | Torch TFLOPs | GPU peak MiB |
|---|---|---:|---:|---:|---:|---:|
| Pattern 0110 | Dense Adam | 6 | 1.69e-02 | 34.17 | 0.11 | 106.60 |
| Pattern 0110 | Dense + GN/revival | 6 | 6.45e-03 | 53.72 | 1.15 | 732.77 |
| Pattern 0110 | Current | 6 | 5.40e-03 | 136.03 | 2.25 | 893.75 |
| Pattern 0110 | LUT-Q + refit | 6 | 4.99e-03 | 198.15 | 2.87 | 925.94 |
| Pattern 0110 | Yeh MAP + refit | 6 | 1.45e-02 | 445.36 | 3.44 | 925.94 |
| Pattern 0110 | Yeh soft | 6 | 5.65e-03 | 365.27 | 1.43 | 832.86 |
| Pattern 0110 | MSR-FC | 6 | 1.68e-02 | 182.13 | 0.95 | 832.86 |
| DeepSets | Dense Adam | 6 | 1.27e-02 | 34.60 | 0.32 | 175.10 |
| DeepSets | Dense + GN/revival | 6 | 6.56e-03 | 58.52 | 10.71 | 3876.94 |
| DeepSets | Current | 6 | 7.24e-03 | 539.89 | 20.26 | 3876.94 |
| DeepSets | LUT-Q + refit | 6 | 8.34e-03 | 298.03 | 20.39 | 3876.94 |
| DeepSets | Yeh MAP + refit | 6 | 2.71e-02 | 559.96 | 21.97 | 3876.94 |
| DeepSets | Yeh soft | 6 | 2.31e-02 | 397.47 | 6.09 | 3611.36 |
| DeepSets | MSR-FC | 6 | 8.08e-02 | 196.49 | 4.50 | 3611.36 |

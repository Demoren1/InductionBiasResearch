# Development source comparison

These are source structure-selection losses, not independent transfer results. Target data and true indexators remain unopened. All numbers include common dense preprocessing when it is used for initialization. Torch FLOPs exclude NumPy/SciPy and unsupported operators.

| Dataset | Method | Cases | Source MSE | Offline seconds | Torch TFLOPs | GPU peak MiB |
|---|---|---:|---:|---:|---:|---:|
| Pattern 0110 | Dense Adam | 3 | 9.58e-03 | 34.22 | - | - |
| Pattern 0110 | Dense + GN/revival | 3 | 1.77e-03 | 53.63 | 1.15 | 732.77 |
| Pattern 0110 | Current | 3 | 3.68e-24 | 145.48 | 2.57 | 925.94 |
| Pattern 0110 | LUT-Q + refit | 3 | 3.69e-24 | 197.83 | 2.87 | 925.94 |
| Pattern 0110 | Yeh MAP + refit | 3 | 1.71e-02 | 441.35 | 3.42 | 925.94 |
| Pattern 0110 | Yeh soft | 3 | 8.61e-04 | 382.63 | 2.47 | 832.86 |
| Pattern 0110 | MSR-FC | 3 | 1.38e-02 | 203.16 | 2.00 | 832.86 |
| DeepSets | Dense Adam | 3 | 1.64e-02 | 34.38 | - | - |
| DeepSets | Dense + GN/revival | 3 | 2.81e-03 | 58.46 | 10.71 | 3876.94 |
| DeepSets | Current | 3 | 3.75e-03 | 540.27 | 20.29 | 3876.94 |
| DeepSets | LUT-Q + refit | 3 | 9.75e-03 | 286.75 | 20.48 | 3876.94 |
| DeepSets | Yeh MAP + refit | 2 | 5.65e-02 | 525.48 | 21.95 | 3876.94 |
| DeepSets | Yeh soft | 3 | 5.38e-02 | 394.83 | 16.46 | 3876.94 |
| DeepSets | MSR-FC | 3 | 1.73e-01 | 209.56 | 14.88 | 3876.94 |

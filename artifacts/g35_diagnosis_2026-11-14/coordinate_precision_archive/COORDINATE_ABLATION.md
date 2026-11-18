# Initialization versus optimization coordinates

New nonlinear development episodes only, seed35111601. No target labels are read. Every cell uses the fixed m512 bank and the original positive-residual d2/r1 transform. Rows specify the initial decoded weight distribution; columns specify Adam coordinates. Within each row the same random numbers, decoded initial weights, hidden biases and heads are used. An analytical inverse initializes transformed coordinates. All cells select the same learning-rate/activity grid and checkpoint through2000 updates, averaging two starts. Matching completed endpoint fits are reused. Settings are selected separately in each cell, so conclusions concern the registered numerical grid and development tasks.

| Dataset | Bank | Initial weights | Adam coordinates | Development CE | Updates |
|---|---|---|---|---:|---:|
| har | g34 | raw | raw | 0.319704 | 500 |
| har | g34 | raw | map | 0.385023 | 400 |
| har | g34 | map | raw | 0.324941 | 475 |
| har | g34 | map | map | 0.380334 | 400 |
| har | pair | raw | raw | 0.355698 | 950 |
| har | pair | raw | map | 0.465905 | 325 |
| har | pair | map | raw | 0.349117 | 750 |
| har | pair | map | map | 0.438900 | 325 |
| fashion | g34 | raw | raw | 0.049144 | 525 |
| fashion | g34 | raw | map | 0.051814 | 575 |
| fashion | g34 | map | raw | 0.049094 | 450 |
| fashion | g34 | map | map | 0.050920 | 500 |
| fashion | pair | raw | raw | 0.075472 | 500 |
| fashion | pair | raw | map | 0.084860 | 350 |
| fashion | pair | map | raw | 0.070886 | 475 |
| fashion | pair | map | map | 0.077538 | 275 |
| mnist | g34 | raw | raw | 0.182598 | 25 |
| mnist | g34 | raw | map | 0.193258 | 125 |
| mnist | g34 | map | raw | 0.182542 | 25 |
| mnist | g34 | map | map | 0.198033 | 125 |
| mnist | pair | raw | raw | 0.188740 | 75 |
| mnist | pair | raw | map | 0.241980 | 150 |
| mnist | pair | map | raw | 0.195361 | 75 |
| mnist | pair | map | map | 0.225372 | 50 |

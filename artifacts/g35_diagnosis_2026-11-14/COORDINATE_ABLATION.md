# Initialization versus optimization coordinates

New nonlinear development episodes only, seed35111601. No target labels are read. Every cell uses the fixed m512 bank and the original positive-residual d2/r1 transform. Rows specify the initial decoded weight distribution; columns specify Adam coordinates. Within each row the same random numbers, decoded initial weights, hidden biases and heads are used. An analytical inverse initializes transformed coordinates. All cells select the same learning-rate/activity grid and checkpoint through2000 updates, averaging two starts. Every cell is recomputed in float64 to control inverse-map cancellation. Settings are selected separately in each cell, so conclusions concern the registered numerical grid and development tasks.

| Dataset | Bank | Initial weights | Adam coordinates | Development CE | Updates |
|---|---|---|---|---:|---:|
| har | g34 | raw | raw | 0.338671 | 425 |
| har | g34 | raw | map | 0.373655 | 375 |
| har | g34 | map | raw | 0.334132 | 425 |
| har | g34 | map | map | 0.375671 | 400 |
| har | pair | raw | raw | 0.352194 | 675 |
| har | pair | raw | map | 0.438099 | 450 |
| har | pair | map | raw | 0.346485 | 775 |
| har | pair | map | map | 0.418580 | 375 |
| fashion | g34 | raw | raw | 0.046867 | 175 |
| fashion | g34 | raw | map | 0.050054 | 175 |
| fashion | g34 | map | raw | 0.047061 | 150 |
| fashion | g34 | map | map | 0.050255 | 175 |
| fashion | pair | raw | raw | 0.072812 | 625 |
| fashion | pair | raw | map | 0.089662 | 100 |
| fashion | pair | map | raw | 0.072636 | 175 |
| fashion | pair | map | map | 0.088271 | 225 |
| mnist | g34 | raw | raw | 0.184227 | 125 |
| mnist | g34 | raw | map | 0.189929 | 50 |
| mnist | g34 | map | raw | 0.179899 | 25 |
| mnist | g34 | map | map | 0.189568 | 50 |
| mnist | pair | raw | raw | 0.187138 | 75 |
| mnist | pair | raw | map | 0.234184 | 125 |
| mnist | pair | map | raw | 0.185867 | 75 |
| mnist | pair | map | map | 0.218670 | 150 |

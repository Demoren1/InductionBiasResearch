# Source-only diagnosis

No private files, target tasks or true numerical/structural parameters were opened.

Each source has 25 labelled examples, at most 100 independent output constraints, versus 12544 first-layer weights before biases/readout. These counts show why individually fitted dense directions can be underidentified; they do not prove that the shared-task problem is unidentifiable.

## Frozen dense model

| Panel | Mean loss |
|---|---:|
| support | 0.264585 |
| numerical | 1.2898 |
| structure | 1.39649 |

## Common-refit structures

| Method/budget | Source M loss |
|---|---:|
| current_k16 | 1.43109 |
| current_k32 | 1.3418 |
| current_k8 | 1.44744 |
| lutq_k8 | 1.49314 |

## Native continuous controls before common refitting

| Method/budget | Best source M loss |
|---|---:|
| msr_k8 | 1.85953 |
| msr_k16 | 2.56363 |
| msr_k32 | 2.88837 |
| msr_k64 | 2.96519 |
| yeh_k8 | 1.85166 |
| yeh_k16 | 1.87043 |

## Hypotheses and conditional next step

H1: few-shot dense interpolation leaves individual directions poorly identified. H2: matching independent source neurons therefore aligns arbitrary fitting choices. H3: once a wrong discrete bank is frozen, a better codebook/readout solver cannot recover absent assignments. These hypotheses must be assessed from observable losses, direction dispersion and bank diversity, without a true mask.

If the common-refit current method materially loses to the native/common-refit competitors, retain it and add a source-pooled functional discrete refinement. Evaluate candidate group changes through the task loss gradient and generalized Gauss-Newton diagonal, accept changes only after a true functional-loss decrease, and alternate with codebook/bias/readout optimization. Include label0 so connectivity can change. Select checkpoints on source M, then freeze before held-out task development/evaluation. If the current method remains competitive, do not activate this revision.

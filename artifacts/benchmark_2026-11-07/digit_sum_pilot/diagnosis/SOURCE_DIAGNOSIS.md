# Source-only diagnosis

No private files, target tasks or true numerical/structural parameters were opened.

Each source has 4096 labelled examples, at most 4096 independent output constraints, versus 3136 first-layer weights before biases/readout. These counts show why individually fitted dense directions can be underidentified; they do not prove that the shared-task problem is unidentifiable.

## Frozen dense model

| Panel | Mean loss |
|---|---:|
| support | 0.245948 |
| numerical | 0.423042 |
| structure | 0.467372 |

## Common-refit structures

| Method/budget | Source M loss |
|---|---:|
| current_k128 | 0.628525 |
| current_k16 | 0.777705 |
| current_k32 | 0.720988 |
| current_k64 | 0.672356 |
| lutq_k128 | 0.643677 |
| lutq_k16 | 0.758174 |
| lutq_k32 | 0.731522 |
| lutq_k64 | 0.681666 |
| msr_k128 | 0.605288 |
| msr_k16 | 0.666078 |
| msr_k32 | 0.636732 |
| msr_k64 | 0.632615 |
| yeh_map_k128 | 0.650821 |
| yeh_map_k16 | 0.741946 |
| yeh_map_k32 | 0.718606 |
| yeh_map_k64 | 0.695535 |
| yeh_soft_k128 | 0.621984 |
| yeh_soft_k16 | 0.698333 |
| yeh_soft_k32 | 0.674841 |
| yeh_soft_k64 | 0.647423 |

## Native continuous controls before common refitting

| Method/budget | Best source M loss |
|---|---:|
| msr_k16 | 0.751793 |
| msr_k32 | 0.705408 |
| msr_k64 | 0.727607 |
| msr_k128 | 0.739809 |
| yeh_k16 | 0.670371 |
| yeh_k32 | 0.667313 |
| yeh_k64 | 0.657734 |
| yeh_k128 | 0.653998 |

## Hypotheses and conditional next step

H1: few-shot dense interpolation leaves individual directions poorly identified. H2: matching independent source neurons therefore aligns arbitrary fitting choices. H3: once a wrong discrete bank is frozen, a better codebook/readout solver cannot recover absent assignments. These hypotheses must be assessed from observable losses, direction dispersion and bank diversity, without a true mask.

If the common-refit current method materially loses to the native/common-refit competitors, retain it and add a source-pooled functional discrete refinement. Evaluate candidate group changes through the task loss gradient and generalized Gauss-Newton diagonal, accept changes only after a true functional-loss decrease, and alternate with codebook/bias/readout optimization. Include label0 so connectivity can change. Select checkpoints on source M, then freeze before held-out task development/evaluation. If the current method remains competitive, do not activate this revision.

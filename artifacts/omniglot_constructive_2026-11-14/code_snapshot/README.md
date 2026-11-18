# Constructive recovery on Omniglot

This applies the saturated-pair codebook construction and the fixed-index
minimax LP from `constructive_dense_recovery_v1` to the saved Omniglot
source weights. See [PROTOCOL.md](PROTOCOL.md) and the separate
[source-only follow-up](INTERVAL_FOLLOWUP.md).

Use the existing `/home/udeneev-av/myvenv/bin/python`,
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. No GPU is required.

1. `run --source-root STUDY --output NEW_OUTPUT` freezes registration,
   exhausts the scalar saturation intervals for budgets2…64 on14 inputs,
   calls the unchanged public recoverer and evaluates the32 old proposals
   with the existing minimax LP.
2. `complete_intervals` with the same arguments checks the monotone
   arithmetic bound and exports a concrete old-group cycle witness.
3. `report --output OUTPUT` writes the report and figures from saved
   results without repeating recognition, optimization or training.
4. Optional `dense_convergence --source-root STUDY --output OUTPUT`
   reads the original Dense fitting histories and saved source models;
   it writes convergence diagnostics and plots without fitting. Run
   `report --output OUTPUT` afterwards to include the diagnostic section.

Module names are `experiments.omniglot_constructive_v1.<entry point>`.
The original source root is
`artifacts/local/omniglot_single_task_2026-11-14/study`;
compact results are `artifacts/omniglot_constructive_2026-11-14`.

The construction applies to a regular saturated class. Failure to obtain
a candidate does not exclude approximate sharing outside that class,
another equivalent dense representation, or good functional transfer.
No software tests are added or run. The prior synthetic illustrations
are not repeated. The general graphical-matroid recognizer is not
implemented by the referenced package and is not silently replaced.

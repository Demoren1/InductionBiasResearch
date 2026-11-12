# Weighted signature top-one on Aug-Omniglot

The user specified the first candidate under the frozen geometric criterion,
not the source-functional choice. This experiment transfers only its signed
indexator. The weighted SVD, threshold grid, magnitude score and sign score
come from `theorem_pair_recovery_v1`; no target labels tune their settings.

The source and development episodes reproduce the existing 128/128 episode
comparison. The dense source learner is the unchanged original implementation:
12 numerical scenarios, 2500 Adam steps, checkpoint selection on numerical
source observations. The final 1000 episodes use seed 26111272013 and the same
historical test class split. Their responses stay sealed until every model's
predictions have been frozen. This is a new episode sample, not a new class
split or an independent source cohort.

The architecture is the existing fully connected 784-input, 16-hidden-unit,
5-output ReLU model. Tasks are 5-way, 5-shot. Queries receive the historical
crop/flip/rotation augmentation; supports do not. These are the existing
Aug-Omniglot conditions, not a convolutional few-shot benchmark.

For each code budget 8, 16, 32, 64, keep the first geometrically ranked
signature candidate. Do not replace absent complete partitions with k-means
or forcibly merge components. Development tasks may select the code budget
and the common adaptation rule, averaging both fresh restarts. They may not
replace a budget's first signature candidate with another structure.

Compare with Dense and the unchanged G34 extraction/functional selection on
the same source models, source panels, development sample and new targets.
Target codebook, biases, readout and offsets are fresh. Use the existing
development grid and final support-only numerical adapter. Report query
accuracy, cross-entropy and paired episode confidence intervals. There is no
ground-truth neural indexator for these real images; accuracy is not an exact
structure recovery metric.

A shuffled control takes the development-selected signature structure and
permutes its entries with fixed seed 261112991. It preserves the code budget,
global density, code frequencies and sign frequencies. It receives its own
development-selected numerical rule. This controls whether the extracted
arrangement transfers better than a random arrangement at the same budget.

The large pair clouds use float64 matrix lower bounds only for screening;
returned neighbours are checked by direct coordinate differences. Every
retained partition exhausts all relevant radius matches. Early rejection uses
only monotone facts: already fewer than m position components, already fewer
than m-r calibration components, or more than m certified isolated vertices.
Completed CPU rank jobs are retained; only unfinished jobs move to CUDA.

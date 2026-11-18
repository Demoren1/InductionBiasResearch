# Predictive follow-up: frozen shared indices

Saved before new predictive development runs and before drawing test tasks.

Omniglot uses the original protocol unchanged: 48 retained development
tasks, fresh 128 test tasks, five classes, 10 train + 5 numerical validation
+ 5 query drawings per class, 784–16–5 ReLU. Reuse the already completed
G34 and Dense development fits. Refit fresh task-specific numerical
parameters for every test method.

The second predictive dataset is MNIST: source classes 0–4 (the ten
retained binary source tasks, 8000 training observations each), development
classes 5–6, and target pairs among 7–9. Development has 24 episodes;
test has 128 fresh episodes. Each class supplies 100 training and 50
numerical-validation images from official training data, and 100 query
images from the official test set. This is a larger-support transfer
diagnostic, not the old 16-shot MNIST protocol. Disjoint character/digit
classes separate source, development and test environments. Repeated
episodes overlap in their underlying image populations; confidence
intervals describe conditional episode variation.

Use the retained G34 banks at m=8/16/32/64/128/512, then nested source-only
splits of the m512 bank to m=1024/2048/4096/8192, inheriting signs and
support. Split the group with largest calibrated source-trajectory SSE
along its largest-variance coordinate at the median. This is a capacity
extension of the practical G34 pipeline, not a fresh native G34 search at
each enlarged budget. Include Dense and execute the unique-code endpoint
m=12544 as a common no-sharing control.

On both datasets, evaluate every SRR candidate selected by the source-only
procedure with pooled score <=9 and m>=2. Choose the budget on development
query CE only. When the radius-2 filter rejects it, any numerical result
is explicitly labelled **SRR without margin filter**, a diagnostic
ablation. Strict SRR keeps its abstention; the ablation must not replace
its coverage in the main conclusion. Budget 1 stays a collapse control.

For Omniglot use the existing six-start Adam numerical solver unchanged.
For MNIST use the existing generic adapter: six starts at .003/.01/.03,
activation quantile .5, 200 Adam steps; select checkpoints and starts on
within-task numerical validation only. All methods within a dataset
receive the same solver. No target task reuses source numerical weights.

Include a retained MSR-FC basis-only control, with fresh numerical fits:
Omniglot m64; MNIST m8/16/32/64/128/512, budget chosen on development.
These are existing FC adaptations and do not represent the paper's CNN
benchmark or a new MSR source-budget search. Do not restart stopped source
or CNN training.

MNIST development RNG:3511158600. Fresh test RNGs: Omniglot3511158601,
MNIST3511158602. Freeze every method/budget choice before generating these
test panels. Fitters read only train labels, numerical-validation labels
and query inputs. Freeze predictions for every method before scoring any
test query answers. Save development curves, paired episode differences,
coverage and source diagnostics. GPU concurrency is permitted; wall time
is operational information and not a speed comparison.

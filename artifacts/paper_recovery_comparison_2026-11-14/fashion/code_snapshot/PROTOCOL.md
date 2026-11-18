# Matched applied recovery comparison

14 November 2026. Replaces the manuscript's exploratory experimental
collection with one synthetic recovery benchmark and two image tasks.

Use MNIST and Fashion-MNIST, original28×28 pixels/255,784–16–2 ReLU.
Class-disjoint split: source0–4, development5–6, target7–9. Source has
all ten unordered class pairs. Per class,4000 original training images,
500 numerical-selection H images and500 independent structure-selection
M images are fixed; each pair therefore has8000/1000/1000 examples.
The same images may occur in different source pair tasks. No augmentation.

Development:16 episodes on classes5/6,16 support images per class and
100 other query images per class, from the official training split.
Target:96 episodes cycling the three pairs of classes7–9,16 support
images per class from official training images,100 query images per
class from the official test split. Episodes may reuse images; reported
intervals are conditional episode variability, not independent class
environments. No target query labels enter any fitting or selection.

Source Dense uses the existing12 Adam scenarios (.001/.003/.01,
weight decay0/1e−4,two starts),2500 updates and H checkpoints every25.
Retain four H-best fits per pair. Continue these by full-gradient
float64 L-BFGS for200 iterations; keep each fit's H-best state, including
its original Adam state. Record training/H CE and accuracy and gradients;
no ground-truth neural structure is available or used. Low prediction
error is distinguished from identification or exact parameter convergence.
Both extractors use exactly this shared Dense preparation.

G34: unchanged real_project/scalable_projection with its two H-ranked
anchors, four support proposals, two initializations,40 projection steps.
Pair recovery: unchanged single-task direct log-ratio forests, two
H-ranked anchors, seven masks, threshold grid plus24 bisection steps;
keep naturally complete partitions only. This is the same pair rule
tested on the synthetic recovery benchmark, not a newly invented
approximate clustering method or the stricter saturated-codebook recognizer.
Both budgets8/16/32/64, at most8 candidates per budget. Missing pair
budgets are reported, not completed with another algorithm.

For each candidate use the same existing numerical adapter:18 Adam
starts (three rates,three activity quantiles,two restarts),2000 updates,
including one shared-rule Dense projection and17 fresh starts. H selects
per-task numerical states; independent source M chooses one indexator
within each budget. Separate development selects the budget, global
rate/activity quantile/update count, averaging both restarts. Dense uses
the same development grid. Target uses one fresh restart, support only,
with the fixed rule selected on development. Only I transfers; every
target codebook,bias,head and offset starts afresh.

Freeze all three models and predictions before opening target query
labels. Preserve registered code snapshots,selected structures,curves,
per-episode scores and access receipts. Original G34 code is unchanged.
No software tests are created or run. These are disclosed2-way16-shot
fully connected adaptations, not matched published benchmark scores.

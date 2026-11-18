# aSRR under the manuscript's evaluation protocol

17 November 2026.

This adds aSRR to the previously evaluated benchmark, with the same
source/development observations, budget grid, numerical solver and target
episode draws. It is an added-method follow-up, not a new independent
test environment. All aSRR choices were frozen on development before
its query responses were scored. Baseline artifacts are unchanged.

See `experiments/paper_asrr_v1/PROTOCOL.md` and the compact records in
`artifacts/paper_asrr_2026-11-17` for candidate families and provenance.

## Applied tasks

Accuracy intervals: 96 conditional image episodes; nine HAR subject means.
Constrained rows use 8/16/32/64/128/512 codes, selected on development.

| Dataset | Method | Codes | Parameters | Updates | Accuracy (%) | CE | Full endpoint preferred |
|---|---|---:|---:|---:|---:|---:|---|
|MNIST|Dense|12544|12594|50|93.26 ± 0.73|0.189|No|
|MNIST|G34|512|562|150|92.46 ± 0.81|0.198|Yes|
|MNIST|aSRR|128|178|150|92.57 ± 0.71|0.222|Yes|
|Fashion-MNIST|Dense|12544|12594|100|95.80 ± 0.89|0.143|No|
|Fashion-MNIST|G34|512|562|125|95.78 ± 0.91|0.151|No|
|Fashion-MNIST|aSRR|128|178|125|95.34 ± 0.84|0.149|Yes|
|UCI HAR|Dense|8976|9094|325|86.15 ± 2.82|0.398|No|
|UCI HAR|G34|512|630|475|84.78 ± 2.70|0.389|No|
|UCI HAR|aSRR|512|630|575|88.19 ± 3.05|0.352|No|

Paired accuracy differences use each method's development-selected
constrained budget; intervals are in percentage points.

| Dataset | aSRR minus | Difference (pp) | 95% half interval |
|---|---|---:|---:|
|MNIST|Dense|-0.69|0.46|
|MNIST|G34|0.10|0.58|
|Fashion-MNIST|Dense|-0.46|0.37|
|Fashion-MNIST|G34|-0.44|0.31|
|UCI HAR|Dense|2.04|1.35|
|UCI HAR|G34|3.41|1.15|

## Two fixed synthetic geometries

| Geometry | aSRR query MSE | Signed exact recovery | Support errors |
|---|---:|---|---:|
|4|1.42635|False|96|
|5|1.11998|False|121|

## Scope

The image results apply to the 16-example-per-class setting and
the compressed grid 8--512. They do not reproduce or invalidate
the earlier 100-example-per-class MNIST result with 1024 codes
in APPROXIMATE_SRR_RESULTS_2026-11-15.md. That study also selected
families on development and used within-task numerical validation
and multiple fresh restarts. Here source M first selects one
candidate per budget, before development chooses the numeric rule
and budget. Insufficient code budget is a hypothesis requiring
a separate extension; these measurements do not isolate its effect.

The Ward generator is approximate and is not covered by an exact
single-task recovery guarantee. Real-data indexators are evaluated
by predictive transfer, without claiming a known ground-truth grouping.
The synthetic teachers have sparse supports; the full-support aSRR
generator does not propose their zero connections and does not recover
either signed indexator. Its MSE is higher than Dense and G34 in both cases.
No software tests were created or run.

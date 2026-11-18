# Recovery and native competitors: expanded comparison

14 November 2026. Source and development choices are fixed before drawing the new target cohort.
The original three-method results remain in artifacts/paper_recovery_comparison_2026-11-14.

## Applied results

Constrained choices use the common grid 8/16/32/64/128/512. The full-code endpoint is executed separately and equals Dense exactly.

| Dataset | Method | Constrained budget | Accuracy (%) | Full budget preferred on development |
|---|---|---:|---:|---|
|MNIST|Dense|12544|93.26 ± 0.73|No|
|MNIST|G34|512|92.46 ± 0.81|Yes|
|MNIST|Pair recovery|512|91.57 ± 0.96|Yes|
|MNIST|LUT-Q|512|91.97 ± 0.82|Yes|
|MNIST|Yeh (soft)|128|89.19 ± 1.05|Yes|
|MNIST|Yeh (MAP)|512|92.25 ± 0.84|Yes|
|MNIST|MSR-FC|512|92.36 ± 0.76|Yes|
|Fashion-MNIST|Dense|12544|95.80 ± 0.89|No|
|Fashion-MNIST|G34|512|95.78 ± 0.91|No|
|Fashion-MNIST|Pair recovery|512|95.27 ± 0.83|Yes|
|Fashion-MNIST|LUT-Q|512|95.96 ± 0.87|No|
|Fashion-MNIST|Yeh (soft)|64|94.12 ± 0.84|Yes|
|Fashion-MNIST|Yeh (MAP)|512|95.74 ± 0.85|Yes|
|Fashion-MNIST|MSR-FC|512|95.68 ± 0.83|Yes|
|UCI HAR|Dense|8976|86.15 ± 2.82|No|
|UCI HAR|G34|512|84.78 ± 2.70|No|
|UCI HAR|Pair recovery|32|79.41 ± 4.52|Yes|
|UCI HAR|LUT-Q|512|83.93 ± 3.36|No|
|UCI HAR|Yeh (soft)|512|79.96 ± 4.41|Yes|
|UCI HAR|Yeh (MAP)|512|82.52 ± 4.09|Yes|
|UCI HAR|MSR-FC|64|80.63 ± 2.27|Yes|

## Synthetic transfer

| Method | Geometry A MSE | Geometry B MSE |
|---|---:|---:|
|Dense|4.726e-01|2.883e-01|
|G34|1.910e-11|6.211e-15|
|Pair recovery|2.110e-12|6.159e-15|
|LUT-Q|2.086e+00|1.127e+00|
|Yeh (soft)|1.526e+00|1.042e+00|
|Yeh (MAP)|1.485e+00|1.027e+00|
|MSR-FC|1.403e+00|1.043e+00|

## Main findings

On the same frozen image episodes, Pair accuracy rises from 87.00% to 91.57% on MNIST and from 92.33% to 95.27% on Fashion-MNIST when moving from 64 to 512 codes. The two image extractors use 562 allocated numerical coordinates at 512 codes, compared with 12594 for Dense.

A full positive unique-code indexator (12544 image weights;8976 HAR weights) produces bit-identical Dense predictions. Matching Dense through this endpoint provides no sharing or compression.

HAR development selects 32 Pair codes (79.41% test accuracy). The prespecified 512-code evaluation scores 84.89%; it is shown in the curve and does not replace the development choice. Budget selection across people is a limitation of this split. All reported choices remain frozen.

G34 and Pair recover both synthetic indexators exactly and transfer with nearly zero MSE. These fixed four-code cases with one source function do not imply a general advantage on naturally occurring tasks or against published implementations.

Yeh uses a 20-term Neumann inverse-Hessian approximation. At budget 512, the final recorded relative residual ranges across the 12 scenarios are 0.90–0.99 (MNIST), 0.70–0.98 (Fashion-MNIST), and 0.27–0.96 (HAR). These diagnostics do not certify an optimized implicit solution; the rows describe this finite-budget adaptation and do not establish limits of the published method.

## Interpretation and scope

These are matched fully connected adaptations of LUT-Q, Yeh and MSR, not unchanged author scripts or published benchmark scores. MSR transfers only its basis; learned initialization and rates are discarded. Continuous methods are never post-clustered. All source and target fitting choices are saved.

Image confidence intervals describe 96 episodes conditional on the fixed class split and source cohort. HAR intervals use nine subject means, each averaging ten episodes. Neither constitutes evidence across all possible dataset environments. The public image class environments were already examined in the original comparison; the new episodes are a conditional follow-up.

Budget denotes allocated code/latent coordinates. Native discrete methods may leave codes unused. Counts of zero connections and group sizes are retained separately; a larger budget can coexist with a dominant shared group. Numerical parameter counts include biases and the full output layer, but exclude the offline structure/basis storage.

The finite grid does not establish a globally optimal sharing budget. Native algorithms have different source objectives and costs; shared observations and the common downstream solver do not imply equal total computation. Concurrent GPU runs do not support a wall-time speed claim.

Source Dense predictive quality is checked on separate numerical-selection examples and curves; global parameter convergence is not claimed. No software tests were created or run.

Reproduction: experiments/paper_recovery_extension_v1. Compact scores, freezes and development choices: artifacts/paper_recovery_extension_2026-11-14. Raw registered native states and source fits: artifacts/local/paper_recovery_extension_2026-11-14.

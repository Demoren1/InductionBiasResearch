# VAE checkpoint diagnostic

This is a read-only audit of saved checkpoints from the DeepSets pilot. It recreates the actual VAE train/validation splits from each bank using the original CUDA RNG sequence, aligns held-out maps to the reconstructed training consensus, and evaluates the checkpoint selected during the original run.

## What the BCE plot means

`validation_bce_baselines` reports BCE summed over 25,088 entries in a map. Lower is better. The task train-mean and global-consensus bars use no information from an individual held-out map. Target entropy is the theoretical per-map lower bound for BCE if a predictor could emit the held-out soft target itself. The posterior-mean reconstruction does encode each held-out map, so its gap from the constant baselines measures reconstruction ability rather than unconditional generation.

Across 32 task/checkpoint pairs, posterior-mean validation BCE is 3683.2 ± 404.1, compared with 3678.9 ± 410.6 for the task-mean baseline and 3611.2 ± 394.3 for global consensus. MC posterior prediction is 3683.2 ± 404.0; target entropy is 1916.6 ± 146.6.

The raw single standard-normal prior output scores 12543.1 ± 842.0 BCE on those same maps; its 16-draw prior predictive mean scores 12505.8 ± 392.0. These are calibration probes, not a requirement that an unconditional prior draw reconstruct a particular held-out map. Their mean decoded intensities are respectively target=0.0334, posterior=0.0364, and normal-prior=0.3815. The agreement search projects latent codes into radius 12, whereas 100.0% of held-out posterior means lie outside that radius. This is posterior/prior support mismatch evidence, not a claim of posterior collapse or proof that all useful decoder outputs are unreachable.

## What the IoU plot means

Each held-out importance map and each decoded output is reduced to the original fixed cardinality K=5,018, then decoder columns are matched by Hungarian assignment. This asks whether topology is recovered after allowing hidden-unit permutation. Posterior reconstructions reach 0.140 ± 0.002; a single standard-normal prior draw reaches 0.145 ± 0.001. The latter tests unconditional samples and should not be conflated with the former.

## Reproducibility check

For each saved best checkpoint, the recreated posterior-mean validation objective agrees with `mask_diagnostics.json` within a maximum absolute difference of 0.00000. This confirms the audit uses the same held-out maps and alignment convention, to ordinary floating-point rounding.

## Scope and limits

The VAE saw only 26 aligned maps per source task and has 16 latent coordinates, so this audit can establish reconstruction of six held-out bank maps, not a broad claim about the distribution of all successful solutions. The train/validation banks themselves were selected winners of 128 candidate models; their representativeness remains an open empirical question.

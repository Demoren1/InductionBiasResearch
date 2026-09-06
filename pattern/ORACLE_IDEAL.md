# Oracle reachability of the ideal support

This follows the two-decoder agreement experiment. The question is whether
the same frozen decoders can generate the ideal pattern support at all, even
when gold is supplied explicitly to the latent optimizer.

Both VAE checkpoints remain unchanged. For each VAE separately, optimize 64
latent vectors against the ideal binary mask, with Hungarian matching of hidden
columns. The target is the repository's 8x8 ideal support: all five length-four
windows, with windows 0, 1, 2 duplicated. No input-coordinate permutation is
allowed.

The primary arm matches the agreement search: same saved initial latents,
1000 Adam updates, learning rate 0.03, soft top-32 temperature 0.5, radius 8.
The secondary arm repeats from exactly the same initial points with radius 16.
These are independent searches, not continuation runs.

The completed primary radius-8 arm finds exact ideal masks in 59/64 starts for
VAE seed 42 and 62/64 for seed 43. Mean IoU is 0.9953 and 0.9981, respectively.
Thus the earlier agreement result near 0.705 is not a representational ceiling
for these decoders: ideal support is already reachable inside the original
radius bound. Agreement alone did not select it. This does not establish that
ideal masks are common under ordinary Gaussian sampling.

For every start, retain both the best soft-loss iterate and the best actual
binary-mask IoU seen over all 1001 evaluated points, including initialization.
For equal hard IoU, prefer the lower soft loss. This separates the optimizer's
surrogate objective from the actual question of exact binary reachability.

Gold is deliberately used for optimization and selection. These results must
not be combined with zero-shot transfer results. An exact result is a concrete
witness that the ideal support is reachable modulo column permutations. A
failed finite search is not proof of impossibility. A solution with large z
norm need not be typical under the VAE's Gaussian prior.

The implementation verifies that decoder tensors are unchanged, and records
the original checkpoint hashes, initial-latent artifact hash, source hashes,
the protocol, complete optimization traces, masks and witness latents.

## Run

Use the `ras` conda environment and GPU access outside the sandbox. Example:

```bash
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python pattern/evaluation/oracle_ideal.py --model_seed 42 --radius 8 --device cuda
```

Repeat for seeds 42/43 and radii 8/16. Existing result files are not overwritten.
After all four runs:

```bash
python pattern/evaluation/oracle_ideal.py --stage report
```

The saved protocol and results are under
`outputs/decoder_agreement/seed_20260906/oracle_ideal/`.
See [the numerical report](outputs/decoder_agreement/seed_20260906/oracle_ideal/RESULTS.md).

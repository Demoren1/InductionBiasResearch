# Generated U on Aug-Omniglot

The tuned comparison and its final test table are in [TUNING.md](TUNING.md).
The earlier untuned comparison remains in [RESULTS.md](RESULTS.md).

The experiment tests whether a coordinate MLP can produce a useful shared
reparameterization U across few-shot classification tasks. Each task samples
five previously unseen Omniglot characters. It provides one unmodified image
of each character for adaptation and five transformed query images per class
for evaluation. The class order changes between tasks.

The model is a four-block 3×3 Conv32 network with batch normalization, ReLU,
2×2 max-pooling, and a task-specific linear classifier. Every convolutional
kernel is reparameterized as

    W_l = V_l ×₁ U_l,out ×₂ U_l,in ×₃ U_l,spatial.

All factors of U are shared between tasks. The 1-shot support set adapts the
convolutional V, batch-normalization affine parameters, and classifier weights.
The outer loop updates their initialization and, where applicable, U. There is
no router and no mixture of experts.

Arms (same episodes, augmentation seeds, architecture, optimizer, and budget):

* `generated`: one coordinate MLP generates all U factors. Inputs are layer and
  factor identifiers plus row/column coordinates and their difference; output
  is a residual added to identity. No latent z is used. Width 64 has 4,993
  generator parameters; width 80 has 7,521, nearly matching direct U. This is
  a new coordinate-MLP implementation of Gψ for square convolution factors,
  not a byte-for-byte reuse of the earlier meta-pattern or MNIST8m model.
* `direct`: the same U factors are freely learned matrices, initialized to
  identity, as in the Kronecker MSR parameterization.
* `random`: fixed orthogonal U factors, with task-specific V still adapted.
* `identity`: fixed identity U, equivalent to a plain Conv4 parameterization.

`torchvision.datasets.Omniglot` provides the 964 background and 659 evaluation
characters. Five background alphabets selected with seed 2020 form the
validation set; the remaining background alphabets form training. The official
evaluation alphabet split is held out for final testing. Support and query
images never share an underlying source image within an episode.

Query transformations approximate the paper's random resized crop (scale
0.8–1), horizontal and vertical flips, and ±30° rotation with a batched affine
grid on GPU. Support images are unchanged. The full second-order MAML outer
gradient is used. Each outer step samples four tasks, has one support update,
and uses Adam at 0.001. Each convolutional block and the classifier has its
own learned inner-loop step size, initialized at 0.4. Validation and test use
three support updates. Training stops after validation loss fails to improve
for eight checks, at least 1,600 steps. The initial cap is 6,000 steps; a run
whose validation loss has not plateaued near the cap is resumed until it does.
Checkpoints support resuming the same command with a larger `--max-steps`.

Run one arm:

    python -m aug_omniglot.run --kind generated --device cuda:0 --seed 42
    python -m aug_omniglot.run --kind generated --generator-width 80 --device cuda:4 --seed 42

After all arms finish, generate a paired held-out comparison and plot:

    python -m aug_omniglot.analyze --device cuda:5 --seed 42

Results go to `aug_omniglot/outputs/<arm>_seed42/`; generated data and
checkpoints are ignored by git. `config.json`, `history.jsonl`, and
`result.json` are human-readable. This is an internal matched ablation, **not
an exact reproduction** of the paper's published numbers: the authors trained
60,000 steps with meta-batch 32, used TorchMeta's particular split and image
pipeline, and their repository does not include finished Aug-Omniglot code.
The classifier head is adapted directly here rather than reparameterized with
another U factor. Thus the published accuracy is context, not a directly
comparable target.

Reference: Zhou, Knowles, Finn, *Meta-Learning Symmetries by
Reparameterization*, ICLR 2021, section 6 and appendix D.3,
<https://arxiv.org/pdf/2007.02933>.

## Support-conditioned U experiment

`conditional_run.py` tests a different question: can the **five support images
of the current episode** help choose its convolutional U matrices? The
generator sees only those unmodified images, with no query images or labels.
It produces one U for the entire episode. The existing Conv4 model then adapts
V, batch-normalization affine parameters, and the classifier on the five
labelled support examples and predicts the 25 transformed query examples.
The query error trains both the generator and the shared V initialization.

Three matched arms use the same episodes, training protocol, initial V and
test tasks for each seed:

* `static`: directly learned U, shared across every episode;
* `mean`: a small CNN encodes each support image, and their average determines U;
* `transformer`: the same CNN plus one set transformer layer determines U. The
  transformer has no sequence-position embeddings, so changing the order of
  support images leaves U unchanged.

The conditional arms start with U equal to identity. They learn a common base U
and four rank-four basis corrections, combined using coefficients predicted
from the support set. This keeps the task-specific change smaller than emitting
all 7,493 matrix elements independently. At test time, they are evaluated both
with the correct support images and with unrelated support images to test
whether the image-conditioned U matters. The latter changes only the
generator input; V still adapts to the correct labelled support examples.
As in the earlier Conv4 runs, batch normalization in the classifier uses the
statistics of each query batch. The generator itself sees no query images.

Run all three arms on three seeds, using every GPU whose *compute utilization*
is zero when the command starts:

    conda run --no-capture-output -n ras python -m aug_omniglot.conditional_sweep --devices auto

The command shows a live progress bar, starts one run per selected GPU, and
prints per-seed and across-seed accuracy when finished. Nine runs are queued,
so all eight GPUs can be occupied simultaneously if idle. Detailed logs are in
`aug_omniglot/outputs/conditional_u/logs/`; each run also has `progress.json`,
`history.jsonl`, `best.pt`, and `result.json`. The aggregate is `summary.json`.
Re-running skips finished runs and resumes interrupted runs from validation
checkpoints. Training stops when validation loss has failed to improve at
ten consecutive checks after 4,000 steps, or at the 30,000-step safety cap.
If any run hits the cap, increase it with `--max-steps 60000` and repeat the
command to continue those runs. A seed-specific run can be launched through
`python -m aug_omniglot.conditional_run --arm mean --seed 42 --device cuda:0`.

This is an internal comparison using our augmentation and class split, not an
exact reproduction of the published Aug-Omniglot benchmark. Conditional arms
have more parameters than the static control, so the matched versus unrelated
support comparison is necessary for interpreting any accuracy difference.

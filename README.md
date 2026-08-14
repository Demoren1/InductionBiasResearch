# Masked MLP + CVAE: inductive bias discovery on MA(k, s)

## Overview

Train 10,000 sparse MLPs (20% active first-layer entries) per (kernel, offset)
pair on the **shifted moving average** MA(k, s) task, extract importance maps
from trained weights, then learn a CVAE that generates masks conditioned on
(k, s). The CVAE is trained on a **diagonal subset** of the kernel×offset grid
to test whether it discovers the underlying inductive bias.

**Task:** predict y = mean(x[t-s-k ... t-s]) from x[t-31 ... t] (L=32 window).
Kernel k controls the window width, offset s shifts which positions matter.
Theoretically optimal mask: active entries on rows [L-s-k : L-s].

**Key finding:** CVAE interpolates well along the kernel dimension (held-out
k=4,6 succeed) but fails on offset extrapolation (k=8,9 tested at s=0 while
trained at s=4,6) — the model does not infer the shift rule from diagonal-only
training data.

## Pipeline

### 1. Data generation
```bash
bash scripts/01_generate_data.sh
```
Generates fixed validation sets (10k windows per kernel) and example plots.
Each window: 32 i.i.d. N(0, 1). Target: mean of the last k values *shifted by s*.

### 2. Train 10,000 masked MLPs per (kernel, offset)
```bash
GPU_IDS="0 1 2 3" bash scripts/02_train.sh
```
**Architecture:** y = w2·ReLU((W1 ⊙ M)·x + b1) + b2
- x ∈ R³², H = 16 hidden, output = 1
- M ∈ {0,1}³²ˣ¹⁶ — fixed Bernoulli(p=0.2) mask, frozen during training
- Only masked entries trained

**Training:** 5,000 Adam steps / LR=1e-3 / batch 128. 4,096 MLPs batched in one
model via einsum. 4 GPUs, ~20s per (kernel, offset) pair.

**Grid:** kernels {3,5,7,9} × offsets {0,2,4,6} = 16 configurations.
Saved: `outputs/checkpoints/kernel_{k}/offset_{s}/gpu*_round*.pt`

### 3. Selection + importance maps
```bash
bash scripts/03_select.sh
```
- **Selection:** top-10% (1,000) MLPs per (k,s) by val MSE → `best10pct.pt`
- **Importance maps:** for ALL 10,000 MLPs: `importance = |W1| / max(|W1|)`
  where mask=1, else 0. Normalised to [0,1] per MLP → `importance.pt`
- **Plot:** selected masks overlaid with trained weights

The best MLPs reach near-zero val MSE with seemingly random masks — the
**weight magnitudes** encode which inputs matter (×4-8 larger on the
bottom rows), not the binary mask.

### 4. Train CVAE on importance maps
```bash
bash scripts/04_train_cvae.sh
```
**CVAE architecture:**
```
Encoder: concat(x, [k/10, s/10]) → FC(256) → FC(256) → μ, logvar (latent=32)
Decoder: concat(z, [k/10, s/10]) → FC(256) → FC(256) → sigmoid → p-map ∈ [0,1]⁵¹²
```
**Loss:** MSE(p_map, importance) + β·KL  (β=1.0, KL collapses → deterministic)
**Training data:** diagonal only: (3,0), (5,2), (7,4), (9,6) — 40,000 maps.
80 epochs / Adam / LR=1e-3 / batch 128.
Saved: `outputs/cvae/cvae_best.pt`, `outputs/plots/cvae/cvae_*.png`

**Mask generation:** z=0 (prior mode) → decoder → top-K (102 active entries
out of 512, matching 20% sparsity).

Also trains two baselines:
- **MeanImportance:** average importance map per training (k,s); nearest-
  neighbour fallback for held-out kernels/offsets
- **DetRegressor:** small 2→256→256→512 MLP mapping (k/10, s/10) → importance
  map, trained on all training data

### 5. Evaluate masks on the MA task
```bash
GPU_IDS="0 1 2 3" bash scripts/05_eval.sh
```
Tests k ∈ {3,4,5,6,7,8,9,10} at offset s=0 with 6 methods:

| Method | Description |
|--------|-------------|
| random | Bernoulli(p=0.2) masks |
| ideal | Exactly rows [L-s-k : L-s) active |
| cvae | CVAE-generated masks (sample_det + top-102) |
| mean_imp | Mean importance from nearest trained (k,s) |
| det_reg | Deterministic MLP regressor (k,s) → importance → top-K |
| top10% | Actual best-10% masks (only k∈{3,5,7,9}) |

Each configuration trains 128 BatchedMaskedMLPs for 5,000 steps.
Saved: `outputs/eval/eval_results.json`, `outputs/plots/eval/eval_mse.png`

## Results

Mean validation MSE after 5,000 steps (offset=0):

| k | ideal | random | cvae | mean_imp | det_reg | top10% |
|---|-------|--------|------|----------|---------|--------|
| 3 | 1.4e-6 | 1.1e-2 | 1.4e-6 | 1.5e-6 | 1.5e-6 | 1.3e-5 |
| 4 | 6.9e-7 | 8.9e-3 | 8.9e-7 | 5.2e-6 | 7.8e-7 | — |
| 5 | 4.4e-7 | 7.7e-3 | 7.3e-7 | 5.0e-7 | 5.9e-7 | 7.4e-6 |
| 6 | 3.2e-7 | 3.5e-3 | 4.6e-7 | 3.1e-6 | 5.5e-7 | — |
| 7 | 2.0e-7 | 3.6e-3 | 4.1e-7 | 5.1e-7 | 3.7e-7 | 6.4e-6 |
| 8 | 1.6e-7 | 3.5e-3 | 0.016 | 0.016 | 0.016 | — |
| 9 | 1.3e-7 | 2.9e-3 | 0.025 | 4.4e-7 | 0.025 | 4.2e-6 |
| 10 | 9.9e-8 | 2.9e-3 | 0.041 | 0.010 | 0.031 | — |

**Interpretation:**

- **k=3-7 (interpolation):** CVAE matches ideal within ×1.5-3, beats random
  by ×1,000-10,000. Kernel interpolation works — the model learned that
  the active region width ∝ k.
- **k=8,9,10 (offset extrapolation):** CVAE fails catastrophically
  (MSE ~0.016-0.041 vs ideal ~1e-7). Trained at offset 0/2/4/6 but tested
  at offset 0 — the model didn't learn the shift rule because each offset
  value appeared with only one kernel during training.
- **mean_imp at k=9:** uses real importance maps from (9,6)
  → 4.4e-7 (only ×3.4 worse than ideal). The importance structure
  transfers across offsets but the CVAE's conditioning doesn't capture it.
- **det_reg:** follows CVAE closely — also fails on offset extrapolation.

CVAE successfully discovers the kernel-width inductive bias but requires
multiple offset examples to learn the shift rule. Diagonal-only training
is insufficient for full (k,s) generalisation.

## Project structure

```
ResearchProject/
├── config.py
├── README.md
├── scripts/
│   ├── run_all.sh                 # full pipeline
│   ├── 01_generate_data.sh
│   ├── 02_train.sh                # GPU_IDS="0 1 2 3"
│   ├── 03_select.sh               # selection + importance + plots
│   ├── 04_train_cvae.sh           # CVAE + baselines
│   └── 05_eval.sh                 # GPU_IDS="0 1 2 3"
├── data/
│   ├── generate.py
│   └── plot.py
├── models/
│   ├── mlp.py                     # BatchedMaskedMLP
│   ├── cvae.py                    # CVAE model + loss + data loaders
│   ├── train.py                   # MLP training
│   └── train_cvae.py              # CVAE training / sampling
├── selection/
│   ├── select_best.py             # top-10% per (k,s)
│   └── plot_selected.py           # weight + mask overlay plots
├── evaluation/
│   ├── importance.py              # |W1| → importance maps
│   ├── baselines.py               # MeanImportance, DetRegressor
│   ├── eval_generated_masks.py    # main evaluation
│   ├── merge_eval.py              # merge parallel GPU results
│   └── plot_importance.py
└── outputs/
    ├── data/                       # val datasets
    ├── plots/
    │   ├── data/                   # data example plots
    │   ├── selected/               # weight overlay plots
    │   ├── cvae/                   # loss, reconstructions, samples
    │   ├── importance/             # importance profiles
    │   └── eval/                   # eval_mse.png bar chart
    ├── checkpoints/
    │   └── kernel_{k}/
    │       └── offset_{s}/
    │           ├── gpu*_round*.pt  # weights + masks (10k MLPs)
    │           ├── best10pct.pt    # top-10% weights + masks
    │           └── importance.pt   # importance maps (10k MLPs)
    ├── cvae/                       # cvae_best.pt, cvae_meta.pt, cvae_samples.pt
    └── eval/                       # eval_results.json, eval_results.pt, det_reg.pt
```

## Quick start

```bash
cd ~/ResearchProject

# Full pipeline on 4 GPUs:
GPU_IDS="0 1 2 3" bash scripts/run_all.sh

# Or step by step:
bash scripts/01_generate_data.sh
GPU_IDS="0 1 2 3" bash scripts/02_train.sh
bash scripts/03_select.sh
bash scripts/04_train_cvae.sh
GPU_IDS="0 1 2 3" bash scripts/05_eval.sh

# Single GPU:
GPU_IDS="0" bash scripts/02_train.sh
GPU_IDS="0" bash scripts/04_train_cvae.sh
GPU_IDS="0" bash scripts/05_eval.sh
```

## Key configuration (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| L | 32 | Input window length |
| H | 16 | Hidden dimension |
| P | 0.2 | Mask sparsity |
| KERNELS | [3,5,7,9] | Training kernel sizes |
| OFFSETS | [0,2,4,6] | Training offsets |
| N_MLPS_PER_KERNEL | 10,000 | MLPs per (kernel, offset) |
| TRAIN_STEPS | 5,000 | Gradient steps per MLP |
| LATENT_DIM | 32 | CVAE latent size |
| CVAE_HIDDEN | 256 | CVAE hidden width |
| CVAE_COND_SCALE_K | 10.0 | Kernel normalisation |
| CVAE_COND_SCALE_S | 10.0 | Offset normalisation |
| CVAE_EPOCHS | 80 | |
| CVAE_BETA | 1.0 | KL weight |
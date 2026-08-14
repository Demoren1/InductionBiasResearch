"""Global configuration and directory layout for the masked-MLP MA experiment."""

from pathlib import Path

# --- Data -----------------------------------------------------------------
L = 32            # input window length
KERNELS = [3, 5, 7, 9]          # MA kernel sizes (one "family" per kernel)
OFFSETS = [0, 2, 4, 6]             # target offset for shifted MA
N_VAL_SAMPLES = 10_000          # fixed validation set per kernel
VAL_BATCH_SIZE = 512

# --- Sparsity mask ---------------------------------------------------------
P = 0.2           # fraction of active (non-zero) entries in the first layer mask

# --- Architecture ----------------------------------------------------------
H = 16            # hidden dimension

# --- Training --------------------------------------------------------------
N_MLPS_PER_KERNEL = 10_000     # total MLPs trained per kernel family
N_MASKS_PER_KERNEL = 10_000    # distinct binary masks per kernel == # of MLPs
MLPS_PER_MASK = N_MLPS_PER_KERNEL // N_MASKS_PER_KERNEL   # inits per mask = 1
NUM_GPUS = 4                    # cards used in parallel
MLPS_PER_GPU = 16_384            # MLPs trained simultaneously on one GPU
                                # (one GPU's whole load fits in a single round)
TRAIN_STEPS = 5_000             # gradient steps per MLP (converges ~3000-5000)
TRAIN_BATCH_SIZE = 128
LR = 1e-3
EVAL_EVERY = 500                # validation loss reporting interval

# --- Selection -------------------------------------------------------------
TOP_FRACTION = 0.1              # keep best 10% per kernel family

# --- CVAE ------------------------------------------------------------------
# Trained on the selected (best 10%) masks, conditioned on kernel size.
# The condition is the *continuous* kernel value normalized to [0, 1]
# (k / CVAE_COND_SCALE), so any kernel can be queried at inference time,
# including held-out ones (4, 6, 8, 10), via interpolation.
MASK_DIM = L * H                 # 32 * 16 = 512 flattened binary mask
CVAE_COND_SCALE_K = 10.0         # divides kernel value to reach ~[0, 1]
CVAE_COND_SCALE_S = 10.0           # offset dimension
CVAE_HIDDEN = 256
LATENT_DIM = 32
CVAE_EPOCHS = 80
CVAE_BATCH_SIZE = 128
CVAE_LR = 1e-3
CVAE_BETA = 1.0                 # KL weight
CVAE_SEED = 42
CVAE_VAL_FRACTION = 0.15

# --- CVAE training / evaluation pairs ---------------------------------------
# 8 random (kernel, offset) pairs for training; 8 for testing.
# Each kernel appears twice in each set; offsets are balanced.
CVAE_TRAIN_PAIRS = [
    (3, 0), (3, 6), (5, 2), (5, 4), (7, 0), (7, 2), (9, 4), (9, 6),
]
CVAE_TEST_PAIRS = [
    (3, 2), (3, 4), (5, 0), (5, 6), (7, 4), (7, 6), (9, 0), (9, 2),
]

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA_DIR = OUTPUTS / "data"
PLOT_DIR = OUTPUTS / "plots"
PLOT_DATA_DIR = PLOT_DIR / "data"
PLOT_SELECTED_DIR = PLOT_DIR / "selected"
PLOT_CVAE_DIR = PLOT_DIR / "cvae"
PLOT_EVAL_DIR = PLOT_DIR / "eval"
PLOT_IMPORTANCE_DIR = PLOT_DIR / "importance"
MASK_DIR = OUTPUTS / "masks"
CKPT_DIR = OUTPUTS / "checkpoints"
CVAE_DIR = OUTPUTS / "cvae"


def kernel_dir(kernel: int, offset: int = None) -> Path:
    d = CKPT_DIR / f"kernel_{kernel}"
    if offset is not None:
        d = d / f"offset_{offset}"
    return d


def ensure_plot_dirs() -> None:
    for d in [PLOT_DATA_DIR, PLOT_SELECTED_DIR, PLOT_CVAE_DIR, PLOT_EVAL_DIR, PLOT_IMPORTANCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
"""Config for pattern-in-sequence inductive-bias experiment."""
from pathlib import Path

SEQ_LEN = 8
PATTERN_LEN = 4
N_WINDOWS = SEQ_LEN - PATTERN_LEN + 1   # 5
H = 8
P = 0.5                                 # match ideal Toeplitz density (32/64)
SIGNED_IMPORTANCE = False               # unsigned |W1| importance in [0,1]
MASK_DIM = SEQ_LEN * H                  # 64

def all_pattern_strings() -> list[str]:
    return [format(i, f"0{PATTERN_LEN}b") for i in range(2 ** PATTERN_LEN)]

PATTERNS = all_pattern_strings()  # '0000'..'1111'

CVAE_TRAIN_PATTERNS = ["0000", "0011", "0101", "0110", "1001", "1010", "1100", "1111"]
CVAE_TEST_PATTERNS  = ["0001", "0010", "0100", "0111", "1000", "1011", "1101", "1110"]

N_VAL_SAMPLES = 2048
VAL_BATCH_SIZE = 256
POS_FRACTION = 0.5

N_MLPS_PER_PATTERN = 2_000
N_MASKS_PER_PATTERN = 2_000
MLPS_PER_MASK = 1
NUM_GPUS = 4
MLPS_PER_GPU = 2_000
TRAIN_STEPS = 2_000
TRAIN_BATCH_SIZE = 128
LR = 1e-3
EVAL_EVERY = 200
TOP_FRACTION = 0.1

CVAE_COND_DIM = 0                       # unconditional VAE; mask shared across patterns
CVAE_HIDDEN = 256
LATENT_DIM = 32
CVAE_EPOCHS = 80
CVAE_BATCH_SIZE = 128
CVAE_LR = 1e-3
CVAE_BETA = 1.0
CVAE_SEED = 42
CVAE_VAL_FRACTION = 0.15
K_ACTIVE = round(P * MASK_DIM)

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA_DIR = OUTPUTS / "data"
PLOT_DIR = OUTPUTS / "plots"
PLOT_DATA_DIR = PLOT_DIR / "data"
PLOT_SELECTED_DIR = PLOT_DIR / "selected"
PLOT_CVAE_DIR = PLOT_DIR / "cvae"
PLOT_EVAL_DIR = PLOT_DIR / "eval"
PLOT_IMPORTANCE_DIR = PLOT_DIR / "importance"
CKPT_DIR = OUTPUTS / "checkpoints"
CVAE_DIR = OUTPUTS / "cvae"
EVAL_DIR = OUTPUTS / "eval"

def pattern_dir(pat: str) -> Path:
    return CKPT_DIR / f"pattern_{pat}"

def val_path(pat: str) -> Path:
    return DATA_DIR / f"val_pattern_{pat}.pt"

def ensure_plot_dirs() -> None:
    for d in [PLOT_DATA_DIR, PLOT_SELECTED_DIR, PLOT_CVAE_DIR,
              PLOT_EVAL_DIR, PLOT_IMPORTANCE_DIR, DATA_DIR, CKPT_DIR,
              CVAE_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def pattern_to_pm1(pat: str):
    import torch
    bits = torch.tensor([float(c) for c in pat])
    return 2.0 * bits - 1.0

def pattern_to_bits(pat: str):
    import torch
    return torch.tensor([int(c) for c in pat], dtype=torch.long)
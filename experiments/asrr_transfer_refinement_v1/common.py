"""Settings for validation-only selection of refinements to input sharing."""

from pathlib import Path

SEED = 2026112050
OUT = Path("artifacts/asrr_transfer_refinement_2026-11-20")
DATASETS = ("mnist_torus", "aug_omniglot", "fc100", "wisdm")
BLOCK_COUNTS = (1, 2, 4, 8, 16, 32)
SCREEN_RATES = (0.003, 0.01)
SCREEN_STEPS = (75, 150)
SCREEN_EPISODES = 48
GRADIENT_DRAWS = 8

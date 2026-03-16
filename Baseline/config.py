# Baseline/config.py
"""Shared constants for the ADNI survival pipeline."""
from pathlib import Path

RANDOM_SEED = 42
N_FOLDS     = 5
HORIZONS    = [3, 5]

BASE_DIR       = Path(__file__).parent
FIG_DIR        = BASE_DIR / 'figures'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
OUT_DIR        = BASE_DIR / 'outputs'

MRI_HARMONIZE_COLS = [
    'Hippocampus', 'Entorhinal', 'Ventricles',
    'Fusiform', 'MidTemp', 'WholeBrain'
]

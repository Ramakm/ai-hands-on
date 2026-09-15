"""The sklearn `digits` dataset read as sequences: one image row per timestep."""

from functools import lru_cache

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

TIMESTEPS = 8
FEATURES = 8
NUM_CLASSES = 10


@lru_cache(maxsize=4)
def load_digit_sequences(test_size=0.2, random_state=0):
    """Return X_tr, X_te, y_tr, y_te with X shaped (N, T=8, D=8) in [0, 1]."""
    digits = load_digits()
    X = digits.images / 16.0          # (N, 8, 8) -> (N, T=8, D=8), scaled to [0, 1]
    y = digits.target
    splits = train_test_split(X, y, test_size=test_size,
                              random_state=random_state, stratify=y)
    for arr in splits:                # cached: make sure nobody mutates the copies
        arr.flags.writeable = False
    return splits


def as_batch(pixels):
    """8x8 nested list / array -> (1, T, D) float array."""
    X = np.asarray(pixels, dtype=np.float64)
    if X.shape != (TIMESTEPS, FEATURES):
        raise ValueError(f"expected shape ({TIMESTEPS}, {FEATURES}), got {X.shape}")
    return X[None]

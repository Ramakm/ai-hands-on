"""All runtime configuration, read once from the environment."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- paths ---
ARTIFACT_DIR = Path(os.getenv("SEQNET_ARTIFACT_DIR", ROOT_DIR / "artifacts"))
WEB_DIR = Path(os.getenv("SEQNET_WEB_DIR", ROOT_DIR / "web"))

# --- data ---
TEST_SIZE = float(os.getenv("SEQNET_TEST_SIZE", "0.2"))
SPLIT_SEED = int(os.getenv("SEQNET_SPLIT_SEED", "0"))

# --- training (identical for both models so the comparison is fair) ---
HIDDEN_SIZE = int(os.getenv("SEQNET_HIDDEN_SIZE", "64"))
EPOCHS = int(os.getenv("SEQNET_EPOCHS", "30"))
BATCH_SIZE = int(os.getenv("SEQNET_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("SEQNET_LR", "2e-3"))
GRAD_CLIP = float(os.getenv("SEQNET_GRAD_CLIP", "5.0"))
MODEL_SEED = int(os.getenv("SEQNET_MODEL_SEED", "42"))
SHUFFLE_SEED = int(os.getenv("SEQNET_SHUFFLE_SEED", "0"))

# --- server ---
HOST = os.getenv("SEQNET_HOST", "127.0.0.1")
PORT = int(os.getenv("SEQNET_PORT", "8000"))
LOG_LEVEL = os.getenv("SEQNET_LOG_LEVEL", "INFO")
CORS_ORIGINS = [o.strip() for o in os.getenv("SEQNET_CORS_ORIGINS", "").split(",") if o.strip()]
MAX_BATCH = int(os.getenv("SEQNET_MAX_BATCH", "256"))

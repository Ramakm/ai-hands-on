import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

ANN_MODEL_PATH = MODEL_DIR / "ann.keras"
CNN_MODEL_PATH = MODEL_DIR / "cnn.keras"
METRICS_PATH = MODEL_DIR / "metrics.json"

ANN_EPOCHS = int(os.getenv("ANN_EPOCHS", 10))
ANN_BATCH_SIZE = int(os.getenv("ANN_BATCH_SIZE", 32))
CNN_EPOCHS = int(os.getenv("CNN_EPOCHS", 10))
CNN_BATCH_SIZE = int(os.getenv("CNN_BATCH_SIZE", 64))
VALIDATION_SPLIT = 0.1

IMG_SIZE = 28
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

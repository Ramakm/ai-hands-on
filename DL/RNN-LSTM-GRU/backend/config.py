import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "Data")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "results.json")

MODEL_PATHS = {
    "Simple RNN": os.path.join(ARTIFACTS_DIR, "rnn_model.keras"),
    "LSTM": os.path.join(ARTIFACTS_DIR, "lstm_model.keras"),
    "GRU": os.path.join(ARTIFACTS_DIR, "gru_model.keras"),
}

TIMESTEPS = 33
FEATURES = 17

EPOCHS = int(os.getenv("EPOCHS", 30))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 64))

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{PORT}")

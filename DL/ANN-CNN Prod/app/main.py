"""FastAPI service serving the ANN and CNN Fashion-MNIST classifiers.

Run:  uvicorn app.main:app --reload
"""
import base64
import json
import math
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import UnidentifiedImageError

import config
from app.preprocessing import preprocess, to_png

MODELS: dict[str, tf.keras.Model] = {}
METRICS: dict[str, dict] = {}
ModelName = Literal["ann", "cnn"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    for name, path in (("ann", config.ANN_MODEL_PATH), ("cnn", config.CNN_MODEL_PATH)):
        if path.exists():
            MODELS[name] = tf.keras.models.load_model(path)
            print(f"Loaded {name.upper()} from {path}")
        else:
            print(f"WARNING: {path} not found. Run `python train.py` first.")
    if config.METRICS_PATH.exists():
        METRICS.update(json.loads(config.METRICS_PATH.read_text()))
    yield
    MODELS.clear()


app = FastAPI(title="Fashion-MNIST ANN vs CNN", version="1.0.0", lifespan=lifespan)


def _run_model(name: str, x: np.ndarray, true_idx: int | None) -> dict:
    if name not in MODELS:
        raise HTTPException(503, f"{name.upper()} model not loaded. Run `python train.py`.")

    batch = x.reshape(1, 28, 28) if name == "ann" else x.reshape(1, 28, 28, 1)
    probs = MODELS[name].predict(batch, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    top3 = np.argsort(probs)[::-1][:3]
    m = METRICS.get(name, {})

    result = {
        "model": name.upper(),
        "predicted_class": config.CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": round(confidence, 4),
        # Probability mass the model put on other classes (per-image uncertainty).
        "prediction_error": round(1 - confidence, 4),
        "top_3": [{"class": config.CLASS_NAMES[i], "probability": round(float(probs[i]), 4)} for i in top3],
        "probabilities": {c: round(float(p), 4) for c, p in zip(config.CLASS_NAMES, probs)},
        # Accuracy / error of the model on the 10,000-image Fashion-MNIST test set.
        "model_test_accuracy": m.get("test_accuracy"),
        "model_test_error_rate": m.get("test_error_rate"),
        "model_test_loss": m.get("test_loss"),
        "class_accuracy_on_test_set": m.get("per_class_accuracy", {}).get(config.CLASS_NAMES[pred_idx]),
    }
    if true_idx is not None:
        p_true = max(float(probs[true_idx]), 1e-7)
        result.update({
            "true_class": config.CLASS_NAMES[true_idx],
            "correct": pred_idx == true_idx,
            "cross_entropy_loss": round(-math.log(p_true), 4),
        })
    return result


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (max 10 MB).")
    try:
        return preprocess(data)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Could not read image. Upload a PNG/JPG/BMP/WebP file.")


def _parse_true_label(true_label: str | None) -> int | None:
    if true_label is None or true_label.strip() == "":
        return None
    label = true_label.strip()
    if label.isdigit() and 0 <= int(label) < len(config.CLASS_NAMES):
        return int(label)
    lowered = [c.lower() for c in config.CLASS_NAMES]
    if label.lower() in lowered:
        return lowered.index(label.lower())
    raise HTTPException(422, f"true_label must be 0-9 or one of {config.CLASS_NAMES}")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(config.BASE_DIR / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": sorted(MODELS)}


@app.get("/classes")
def classes():
    return {i: c for i, c in enumerate(config.CLASS_NAMES)}


@app.get("/metrics")
def metrics():
    if not METRICS:
        raise HTTPException(404, "No metrics found. Run `python train.py`.")
    return METRICS


@app.post("/predict/{model_name}")
async def predict_one(
    model_name: ModelName,
    file: UploadFile = File(...),
    true_label: str | None = Form(None, description="Optional: 0-9 or class name, to compute correctness and loss"),
):
    x = await _read_image(file)
    return _run_model(model_name, x, _parse_true_label(true_label))


@app.post("/predict")
async def predict_both(
    file: UploadFile = File(...),
    true_label: str | None = Form(None, description="Optional: 0-9 or class name, to compute correctness and loss"),
):
    x = await _read_image(file)
    true_idx = _parse_true_label(true_label)
    ann, cnn = _run_model("ann", x, true_idx), _run_model("cnn", x, true_idx)
    return {
        "filename": file.filename,
        "preprocessed_image": "data:image/png;base64," + base64.b64encode(to_png(x)).decode(),
        "ann": ann,
        "cnn": cnn,
        "models_agree": ann["predicted_index"] == cnn["predicted_index"],
    }

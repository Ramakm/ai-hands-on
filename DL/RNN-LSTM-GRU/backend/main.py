import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model

from backend import config, data

STATE = {}


def _load_state():
    if not os.path.exists(config.RESULTS_PATH):
        raise RuntimeError(
            "No trained artifacts found. Run `python -m backend.train` first."
        )

    with open(config.RESULTS_PATH) as f:
        results = json.load(f)

    models = {name: load_model(path) for name, path in config.MODEL_PATHS.items()}

    (
        X_train_seq,
        y_train_enc,
        X_test_seq,
        y_test_enc,
        encoder,
        test_df,
    ) = data.load_prepared_data()

    STATE["results"] = results
    STATE["models"] = models
    STATE["X_test"] = X_test_seq
    STATE["y_test"] = y_test_enc
    STATE["encoder"] = encoder
    STATE["class_names"] = list(encoder.classes_)
    STATE["test_df"] = test_df


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_state()
    yield


app = FastAPI(title="HAR RNN vs LSTM vs GRU API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    index: int
    model: str = "all"  # "Simple RNN" | "LSTM" | "GRU" | "all"


@app.get("/health")
def health():
    return {"status": "ok", "num_test_samples": int(STATE["X_test"].shape[0])}


@app.get("/results")
def get_results():
    """Model comparison metrics computed during training."""
    return STATE["results"]


@app.get("/samples")
def list_samples(limit: int = 50, offset: int = 0):
    """A page of test-set samples the user can pick from, with ground-truth activity."""
    y_test = STATE["y_test"]
    class_names = STATE["class_names"]
    subjects = STATE["test_df"]["subject"].values

    n = len(y_test)
    end = min(offset + limit, n)
    items = [
        {
            "index": i,
            "subject": int(subjects[i]),
            "actual_activity": class_names[y_test[i]],
        }
        for i in range(offset, end)
    ]
    return {"total": n, "items": items}


@app.post("/predict")
def predict(req: PredictRequest):
    X_test = STATE["X_test"]
    y_test = STATE["y_test"]
    class_names = STATE["class_names"]
    models = STATE["models"]

    if req.index < 0 or req.index >= len(X_test):
        raise HTTPException(status_code=400, detail=f"index must be in [0, {len(X_test) - 1}]")

    if req.model != "all" and req.model not in models:
        raise HTTPException(status_code=400, detail=f"model must be one of {list(models)} or 'all'")

    sample = X_test[req.index : req.index + 1]
    actual = class_names[y_test[req.index]]

    model_names = list(models) if req.model == "all" else [req.model]

    predictions = {}
    for name in model_names:
        probs = models[name].predict(sample, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        predictions[name] = {
            "predicted_activity": class_names[pred_idx],
            "correct": class_names[pred_idx] == actual,
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(class_names, probs)},
        }

    return {
        "index": req.index,
        "actual_activity": actual,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=config.HOST, port=config.PORT, reload=False)

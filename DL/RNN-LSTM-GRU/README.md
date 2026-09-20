# HAR: RNN vs LSTM vs GRU

Production version of `HAR_RNN_LSTM_GRU.ipynb` — trains all 3 recurrent models on the
UCI HAR (smartphone sensor) dataset, serves predictions via FastAPI, and lets you query
any test sample from a Streamlit UI.

## Structure
- `backend/config.py` — paths and hyperparameters
- `backend/data.py` — load, encode, scale, reshape data (same steps as the notebook)
- `backend/model.py` — shared model builder (RNN/LSTM/GRU, 64→32 units)
- `backend/train.py` — trains all 3 models, saves weights + scaler/encoder + metrics to `artifacts/`
- `backend/main.py` — FastAPI serving layer: `/health`, `/results`, `/samples`, `/predict`
- `frontend/app.py` — Streamlit UI: pick a test sample and get all 3 models' predictions,
  compare accuracy/training curves, view confusion matrix

## Setup (one time)
```bash
python3.11 -m venv .venv          # TensorFlow needs Python <=3.12
.venv/bin/pip install -r requirements.txt
```

## Train the models
```bash
.venv/bin/python -m backend.train
```
Saves `artifacts/{rnn,lstm,gru}_model.keras`, `scaler.pkl`, `encoder.pkl`, `results.json`.

## Run
```bash
./start.sh
```
This trains (if not already trained), starts the API on `:8000`, and the UI on `:8501`.

Or manually:
```bash
.venv/bin/python -m backend.main        # API on :8000
.venv/bin/streamlit run frontend/app.py # UI on :8501
```

Open http://localhost:8501, pick a test-set sample (optionally filter by activity), click
**Predict**, and see what each of the 3 models predicts vs. the actual activity, with
confidence bars. The other tabs show accuracy/training-curve comparison and the confusion
matrix + classification report for the best model (GRU, ~90.4% test accuracy in the last run).

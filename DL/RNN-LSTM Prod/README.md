# RNN vs LSTM, from scratch

A vanilla RNN and an LSTM written by hand in NumPy (forward pass, BPTT, Adam),
trained on the scikit-learn `digits` dataset read one row per timestep, and
served side by side through a FastAPI app with a web UI.

```bash
./start.sh                # venv + train (if needed) + serve -> http://127.0.0.1:8000
```

or step by step:

```bash
make install              # .venv with runtime + dev dependencies
make train                # artifacts/{rnn,lstm}.{npz,json}
make serve                # http://127.0.0.1:8000   (API docs at /docs)
make test                 # gradient checks, persistence, API, notebook sync
make notebook             # regenerate + execute rnn_from_scratch.ipynb
make docker               # container with weights baked in
```

## Layout

| Path | What |
|---|---|
| `seqnet/models/rnn.py`, `seqnet/models/lstm.py` | the models: `forward`, `backward` (BPTT), `predict`, `predict_proba`, `trace` |
| `seqnet/functional.py`, `seqnet/optim.py` | softmax / cross-entropy / sigmoid, Adam |
| `seqnet/training.py` | `fit` loop and `gradient_check` |
| `seqnet/train.py` | CLI; both models get identical data, seeds, batches and learning rate |
| `seqnet/persistence.py` | `.npz` weights + `.json` metadata, written atomically, validated on load |
| `seqnet/service.py`, `seqnet/api.py` | inference service and HTTP API |
| `web/` | comparison UI (plain HTML/CSS/JS, no build step) |
| `scripts/build_notebook.py` | builds the notebook, pulling model code from the package |
| `rnn_from_scratch.ipynb` | walkthrough: RNN, LSTM, and the comparison |

The notebook's model, optimiser and training cells are generated from the
package source; `tests/test_notebook_sync.py` fails if they drift apart.

## API

| Method | Path | |
|---|---|---|
| GET | `/api/health` | `ok` / `degraded` / `unavailable` + loaded models |
| GET | `/api/models` | metadata, metrics, training history, confusion matrix |
| POST | `/api/predict` | `{"pixels": 8x8 in [0,1], "models"?: ["rnn","lstm"]}` → per-model probabilities, per-row predictions, hidden states, LSTM gates |
| POST | `/api/predict/batch` | `{"inputs": [8x8, ...]}` (≤ `SEQNET_MAX_BATCH`) |
| GET | `/api/compare` | test-set accuracy, agreement breakdown, digits either model gets wrong |
| GET | `/api/samples/random?digit=` · `/api/samples/{index}` | held-out test digits |

## Configuration

Environment variables, read in `seqnet/config.py`: `SEQNET_ARTIFACT_DIR`,
`SEQNET_HIDDEN_SIZE` (64), `SEQNET_EPOCHS` (30), `SEQNET_BATCH_SIZE` (64),
`SEQNET_LR` (2e-3), `SEQNET_GRAD_CLIP` (5.0), `SEQNET_MODEL_SEED` (42),
`SEQNET_SHUFFLE_SEED` (0), `SEQNET_TEST_SIZE` (0.2), `SEQNET_SPLIT_SEED` (0),
`SEQNET_HOST`, `SEQNET_PORT`, `SEQNET_LOG_LEVEL`, `SEQNET_CORS_ORIGINS`, `SEQNET_MAX_BATCH`.

## Results (default settings)

| | params | test acc | train time |
|---|---|---|---|
| RNN | 5,322 | 98.3% | ~0.4 s |
| LSTM | 19,338 | 95.8% | ~2 s |

With only 8 timesteps, vanishing gradients never become a problem, so the
simpler RNN wins here. The notebook's "Things to try" section has experiments
with longer sequences, where the LSTM should do better.

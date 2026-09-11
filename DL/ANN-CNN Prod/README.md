# Fashion-MNIST: ANN vs CNN (FastAPI)

Production version of `ANN_&_CNN.ipynb`. The same ANN and CNN architectures are trained on Fashion-MNIST and served through FastAPI, so you can upload an image and get predictions from both models.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py            # trains both models -> models/ann.keras, models/cnn.keras, models/metrics.json
.venv/bin/uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 for the upload page, or http://127.0.0.1:8000/docs for Swagger.

`python train.py --model ann` (or `cnn`) retrains a single model. Epochs and batch sizes can be overridden with `ANN_EPOCHS`, `CNN_EPOCHS`, `ANN_BATCH_SIZE`, `CNN_BATCH_SIZE`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/predict` | Upload an image; returns ANN and CNN results side by side |
| POST | `/predict/ann` | ANN only |
| POST | `/predict/cnn` | CNN only |
| GET | `/metrics` | Test-set accuracy, error rate, loss, and per-class accuracy for each model |
| GET | `/classes` | Class index → name |
| GET | `/health` | Which models are loaded |

Form fields: `file` (image, required) and `true_label` (optional, `0-9` or a class name).

```bash
curl -F "file=@shoe.jpg" -F "true_label=Sneaker" http://127.0.0.1:8000/predict
```

## What "accuracy" and "error" mean in the response

A single image has no accuracy of its own, so the response reports three kinds of numbers:

- **Per image:** `confidence` (softmax probability of the predicted class) and `prediction_error` = 1 − confidence.
- **Per image, if you pass `true_label`:** `correct` (true/false) and `cross_entropy_loss` = −log p(true class).
- **Per model:** `model_test_accuracy`, `model_test_error_rate`, and `model_test_loss` on the 10,000 test images, plus `class_accuracy_on_test_set` for the predicted class.

## Preprocessing uploaded images

Fashion-MNIST images are light items on a black background, 28×28 grayscale. Uploads go through the same conversion: grayscale, invert if the background is light, pad to a square, resize to 28×28, and scale to [0, 1]. The `/predict` response includes the resulting 28×28 image so you can see what the model received. Photos with busy backgrounds or several items will classify poorly, because the models were only trained on centered single items.

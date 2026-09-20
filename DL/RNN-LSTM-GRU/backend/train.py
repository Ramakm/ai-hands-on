"""
Production training script for the HAR RNN vs LSTM vs GRU comparison.
Trains all 3 models on the UCI HAR dataset and saves everything the API
needs to serve predictions: scaler, label encoder, model weights,
training curves and evaluation metrics.

Run: .venv/bin/python -m backend.train
"""
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, SimpleRNN

from backend import config, data
from backend.model import build_model

tf.random.set_seed(42)
np.random.seed(42)

LAYER_TYPES = {
    "Simple RNN": SimpleRNN,
    "LSTM": LSTM,
    "GRU": GRU,
}


def train_and_evaluate(model, name, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    start = time.time()
    history = model.fit(
        X_train,
        y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2,
    )
    train_seconds = time.time() - start
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{name} Test Accuracy: {acc * 100:.2f}%  ({train_seconds:.1f}s)")
    return history, acc, train_seconds


def main():
    print("Loading data...")
    train_df, test_df = data.load_raw()
    X_train, y_train, X_test, y_test = data.split_features_labels(train_df, test_df)
    (
        X_train_seq,
        y_train_enc,
        X_test_seq,
        y_test_enc,
        scaler,
        encoder,
    ) = data.fit_transform(X_train, y_train, X_test, y_test)

    class_names = list(encoder.classes_)
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("X_train:", X_train_seq.shape, "X_test:", X_test_seq.shape)

    results = {}
    histories = {}
    models = {}

    for name, layer_type in LAYER_TYPES.items():
        print(f"\n=== Training {name} ===")
        model = build_model(layer_type, num_classes)
        model.summary()
        history, acc, train_seconds = train_and_evaluate(
            model, name, X_train_seq, y_train_enc, X_test_seq, y_test_enc
        )
        model.save(config.MODEL_PATHS[name])

        models[name] = model
        histories[name] = history
        results[name] = {
            "test_accuracy": float(acc),
            "train_seconds": train_seconds,
            "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
            "val_loss": [float(v) for v in history.history["val_loss"]],
            "train_accuracy": [float(v) for v in history.history["accuracy"]],
            "train_loss": [float(v) for v in history.history["loss"]],
        }

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_model = models[best_name]
    print("\nBest model:", best_name)

    y_pred = np.argmax(best_model.predict(X_test_seq, verbose=0), axis=1)
    report = classification_report(
        y_test_enc, y_pred, target_names=class_names, output_dict=True
    )
    cm = confusion_matrix(y_test_enc, y_pred).tolist()

    summary = {
        "class_names": class_names,
        "results": results,
        "best_model": best_name,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(config.RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to {config.RESULTS_PATH}")
    print(f"Saved models to {config.ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()

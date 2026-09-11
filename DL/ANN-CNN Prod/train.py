"""Train the ANN and CNN from ANN_&_CNN.ipynb on Fashion-MNIST and save them.

Usage:  python train.py            # trains both
        python train.py --model cnn
"""
import argparse
import json

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import config


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    return x_train, y_train, x_test, y_test


def build_ann():
    model = models.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_cnn():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def evaluate(model, x_test, y_test, history):
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)
    per_class = {
        name: float(np.mean(preds[y_test == i] == i))
        for i, name in enumerate(config.CLASS_NAMES)
    }
    return {
        "test_accuracy": float(acc),
        "test_error_rate": float(1 - acc),
        "test_loss": float(loss),
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "per_class_accuracy": per_class,
        "params": int(model.count_params()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ann", "cnn", "both"], default="both")
    args = parser.parse_args()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_data()
    print("Train shape:", x_train.shape, "Test shape:", x_test.shape)

    metrics = json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else {}

    if args.model in ("ann", "both"):
        ann = build_ann()
        ann.summary()
        hist = ann.fit(x_train, y_train, epochs=config.ANN_EPOCHS, batch_size=config.ANN_BATCH_SIZE,
                       validation_split=config.VALIDATION_SPLIT, verbose=1)
        metrics["ann"] = evaluate(ann, x_test, y_test, hist)
        ann.save(config.ANN_MODEL_PATH)
        print(f"\nANN Test Accuracy: {metrics['ann']['test_accuracy'] * 100:.2f}%")

    if args.model in ("cnn", "both"):
        x_train_cnn = x_train.reshape(-1, 28, 28, 1)
        x_test_cnn = x_test.reshape(-1, 28, 28, 1)
        cnn = build_cnn()
        cnn.summary()
        hist = cnn.fit(x_train_cnn, y_train, epochs=config.CNN_EPOCHS, batch_size=config.CNN_BATCH_SIZE,
                       validation_split=config.VALIDATION_SPLIT, verbose=1)
        metrics["cnn"] = evaluate(cnn, x_test_cnn, y_test, hist)
        cnn.save(config.CNN_MODEL_PATH)
        print(f"\nCNN Test Accuracy: {metrics['cnn']['test_accuracy'] * 100:.2f}%")

    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved models + metrics to {config.MODEL_DIR}")


if __name__ == "__main__":
    main()

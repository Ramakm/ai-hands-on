from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

from backend import config


def build_model(layer_type, num_classes: int):
    model = Sequential([
        Input(shape=(config.TIMESTEPS, config.FEATURES)),
        layer_type(64, return_sequences=True),
        Dropout(0.2),
        layer_type(32),
        Dropout(0.2),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

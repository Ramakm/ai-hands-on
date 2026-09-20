import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from backend import config


def load_raw():
    train = pd.read_csv(config.TRAIN_CSV)
    test = pd.read_csv(config.TEST_CSV)
    return train, test


def split_features_labels(train: pd.DataFrame, test: pd.DataFrame):
    train_cleaned = train.dropna(subset=["Activity"])

    X_train = train_cleaned.drop(["subject", "Activity"], axis=1).values
    y_train = train_cleaned["Activity"].values

    X_test = test.drop(["subject", "Activity"], axis=1).values
    y_test = test["Activity"].values

    return X_train, y_train, X_test, y_test


def fit_transform(X_train, y_train, X_test, y_test):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq = X_train_scaled.reshape(-1, config.TIMESTEPS, config.FEATURES)
    X_test_seq = X_test_scaled.reshape(-1, config.TIMESTEPS, config.FEATURES)

    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(encoder, config.ENCODER_PATH)

    return X_train_seq, y_train_enc, X_test_seq, y_test_enc, scaler, encoder


def load_preprocessors():
    scaler = joblib.load(config.SCALER_PATH)
    encoder = joblib.load(config.ENCODER_PATH)
    return scaler, encoder


def load_prepared_data():
    """Load train/test data already scaled + reshaped, using saved preprocessors."""
    train, test = load_raw()
    X_train, y_train, X_test, y_test = split_features_labels(train, test)
    scaler, encoder = load_preprocessors()

    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_seq = X_train_scaled.reshape(-1, config.TIMESTEPS, config.FEATURES)
    X_test_seq = X_test_scaled.reshape(-1, config.TIMESTEPS, config.FEATURES)

    return X_train_seq, y_train_enc, X_test_seq, y_test_enc, encoder, test

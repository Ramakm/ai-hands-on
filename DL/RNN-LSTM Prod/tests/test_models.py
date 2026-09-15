import json

import numpy as np
import pytest

from seqnet.models import LSTM, MODEL_REGISTRY, RNN
from seqnet.persistence import load_model, save_model
from seqnet.training import gradient_check

MODELS = [RNN, LSTM]


@pytest.mark.parametrize("cls", MODELS)
def test_gradients_match_finite_differences(cls):
    rng = np.random.default_rng(1)
    net = cls(4, 6, 3, seed=1)
    X = rng.normal(size=(5, 7, 4))
    y = rng.integers(0, 3, size=5)
    for _ in range(5):                                   # several random entries per param
        for name, (analytic, numeric, rel) in gradient_check(net, X, y, rng).items():
            assert rel < 1e-6, f"{cls.name}.{name}: analytic={analytic} numeric={numeric}"


@pytest.mark.parametrize("cls", MODELS)
def test_shapes_and_probabilities(cls):
    net = cls(8, 16, 10)
    X = np.random.default_rng(0).random((3, 8, 8))
    probs = net.predict_proba(X)
    assert probs.shape == (3, 10)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert net.predict(X).shape == (3,)

    tr = net.trace(X)
    assert tr["hidden"].shape == (3, 8, 16)
    assert tr["step_probs"].shape == (3, 8, 10)
    # the last step's readout is exactly the model's output
    np.testing.assert_allclose(tr["step_probs"][:, -1], probs)


def test_lstm_gates_in_unit_interval():
    tr = LSTM(8, 16, 10).trace(np.random.default_rng(0).random((2, 8, 8)))
    for g in tr["gates"].values():
        assert g.shape == (2, 8, 16) and g.min() >= 0 and g.max() <= 1


@pytest.mark.parametrize("cls", MODELS)
def test_save_load_roundtrip(cls, tmp_path):
    net = cls(8, 12, 10, seed=3)
    save_model(net, tmp_path, {"metrics": {"test_acc": 0.5}})
    loaded, meta = load_model(cls.name, tmp_path)
    X = np.random.default_rng(0).random((4, 8, 8))
    np.testing.assert_array_equal(net.predict_proba(X), loaded.predict_proba(X))
    assert meta["metrics"]["test_acc"] == 0.5


def test_load_rejects_mismatched_weights(tmp_path):
    save_model(RNN(8, 12, 10), tmp_path, {})
    meta = json.loads((tmp_path / "rnn.json").read_text())
    meta["config"]["hidden_size"] = 13
    (tmp_path / "rnn.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="shape"):
        load_model("rnn", tmp_path)


def test_load_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("lstm", tmp_path)


def test_registry():
    assert set(MODEL_REGISTRY) == {"rnn", "lstm"}

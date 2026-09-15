import pytest
from fastapi.testclient import TestClient

from seqnet.api import create_app
from seqnet.config import WEB_DIR


@pytest.fixture(scope="module")
def client(tiny_artifacts):
    with TestClient(create_app(artifact_dir=tiny_artifacts, web_dir=WEB_DIR)) as c:
        yield c


@pytest.fixture(scope="module")
def empty_client(tmp_path_factory):
    with TestClient(create_app(artifact_dir=tmp_path_factory.mktemp("empty"), web_dir=None)) as c:
        yield c


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["models"] == ["lstm", "rnn"]


def test_models_metadata(client):
    body = client.get("/api/models").json()
    for name in ("rnn", "lstm"):
        assert body[name]["parameters"] > 0
        assert len(body[name]["history"]["test_acc"]) == 1


def test_predict_side_by_side(client):
    sample = client.get("/api/samples/0").json()
    body = client.post("/api/predict", json={"pixels": sample["pixels"]}).json()
    assert set(body["results"]) == {"rnn", "lstm"}
    for name, r in body["results"].items():
        assert 0 <= r["prediction"] <= 9
        assert abs(sum(r["probabilities"]) - 1) < 1e-3
        assert len(r["steps"]) == 8 and len(r["hidden"]) == 8
    assert body["results"]["rnn"]["gates"] is None
    assert set(body["results"]["lstm"]["gates"]) == {"input", "forget", "output"}
    assert isinstance(body["agree"], bool)


def test_predict_single_model(client):
    body = client.post("/api/predict", json={"pixels": [[0.0] * 8] * 8, "models": ["lstm"]}).json()
    assert list(body["results"]) == ["lstm"]


@pytest.mark.parametrize("pixels", [
    [[0.0] * 8] * 7,                 # too few rows
    [[0.0] * 9] * 8,                 # too many columns
    [[1.5] * 8] * 8,                 # out of range
])
def test_predict_rejects_bad_input(client, pixels):
    assert client.post("/api/predict", json={"pixels": pixels}).status_code == 422


def test_batch(client):
    body = client.post("/api/predict/batch", json={"inputs": [[[0.5] * 8] * 8] * 3}).json()
    assert len(body["rnn"]["predictions"]) == 3 and len(body["lstm"]["confidences"]) == 3


def test_compare(client):
    body = client.get("/api/compare").json()
    agreement = body["agreement"]
    assert sum(agreement[k] for k in ("both_correct", "only_rnn_correct",
                                      "only_lstm_correct", "both_wrong")) == body["test_samples"]


def test_samples(client):
    assert client.get("/api/samples/random?digit=7").json()["label"] == 7
    assert client.get("/api/samples/99999").status_code == 404


def test_ui_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "RNN" in r.text


def test_unavailable_without_artifacts(empty_client):
    assert empty_client.get("/api/health").json()["status"] == "unavailable"
    assert empty_client.post("/api/predict", json={"pixels": [[0.0] * 8] * 8}).status_code == 503

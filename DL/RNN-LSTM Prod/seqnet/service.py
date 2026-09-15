"""Inference service: holds the loaded models and turns arrays into API payloads."""

import logging
import time

import numpy as np

from seqnet.data import load_digit_sequences
from seqnet.models import MODEL_REGISTRY
from seqnet.persistence import load_model

log = logging.getLogger("seqnet.service")


def _round(arr, decimals=4):
    return np.round(np.asarray(arr, dtype=float), decimals).tolist()


class ModelService:
    def __init__(self, artifact_dir):
        self.artifact_dir = artifact_dir
        self.models, self.metadata, self.errors = {}, {}, {}
        self.X_te = self.y_te = None
        self.comparison = None

    # ---------------- lifecycle ----------------
    def load(self):
        for name in MODEL_REGISTRY:
            try:
                self.models[name], self.metadata[name] = load_model(name, self.artifact_dir)
                log.info("loaded %s from %s", name, self.artifact_dir)
            except (FileNotFoundError, ValueError, KeyError) as exc:
                self.errors[name] = str(exc)
                log.error("could not load %s: %s", name, exc)

        # Serve samples from the exact test split the models were evaluated on.
        split = next(iter(self.metadata.values()), {}).get("data", {})
        _, self.X_te, _, self.y_te = load_digit_sequences(
            split.get("test_size", 0.2), split.get("split_seed", 0))
        self.comparison = self._compare_on_test_set()

    @property
    def ready(self):
        return bool(self.models)

    # ---------------- inference ----------------
    def predict(self, X, names=None):
        """X: (1, T, D). Full per-model breakdown for the side-by-side view."""
        out = {}
        for name in names or self.models:
            net = self.models[name]
            started = time.perf_counter()
            probs = net.predict_proba(X)[0]
            latency_ms = (time.perf_counter() - started) * 1000
            tr = net.trace(X)

            step_probs = tr["step_probs"][0]                          # (T, C)
            hidden = tr["hidden"][0]                                  # (T, H)
            result = {
                "model": name,
                "prediction": int(probs.argmax()),
                "confidence": float(probs.max()),
                "probabilities": _round(probs),
                "latency_ms": round(latency_ms, 3),
                "steps": [{"row": t + 1, "prediction": int(p.argmax()),
                           "confidence": round(float(p.max()), 4)}
                          for t, p in enumerate(step_probs)],
                "step_probabilities": _round(step_probs),
                "hidden": _round(hidden, 3),
                "hidden_norm": _round(np.linalg.norm(hidden, axis=1)),
                "gates": None,
                "cell_norm": None,
            }
            if "gates" in tr:
                result["gates"] = {k: _round(v[0].mean(axis=1)) for k, v in tr["gates"].items()}
                result["cell_norm"] = _round(np.linalg.norm(tr["cell"][0], axis=1))
            out[name] = result
        return out

    def predict_batch(self, X, names=None):
        out = {}
        for name in names or self.models:
            probs = self.models[name].predict_proba(X)
            out[name] = {"predictions": probs.argmax(axis=1).tolist(),
                         "confidences": _round(probs.max(axis=1))}
        return out

    # ---------------- test-set comparison ----------------
    def _compare_on_test_set(self):
        if not self.models:
            return None
        preds = {name: net.predict(self.X_te) for name, net in self.models.items()}
        y = self.y_te
        summary = {"test_samples": int(len(y)),
                   "accuracy": {n: float((p == y).mean()) for n, p in preds.items()}}

        if {"rnn", "lstm"} <= preds.keys():
            r_ok, l_ok = preds["rnn"] == y, preds["lstm"] == y
            summary["agreement"] = {
                "both_correct": int((r_ok & l_ok).sum()),
                "only_rnn_correct": int((r_ok & ~l_ok).sum()),
                "only_lstm_correct": int((~r_ok & l_ok).sum()),
                "both_wrong": int((~r_ok & ~l_ok).sum()),
                "same_prediction": int((preds["rnn"] == preds["lstm"]).sum()),
            }
            interesting = np.where(~r_ok | ~l_ok)[0]
            summary["hard_cases"] = [
                {"index": int(i), "label": int(y[i]),
                 "rnn": int(preds["rnn"][i]), "lstm": int(preds["lstm"][i]),
                 "pixels": _round(self.X_te[i], 3)}
                for i in interesting
            ]
        return summary

    def sample(self, index):
        return {"index": int(index), "label": int(self.y_te[index]),
                "pixels": _round(self.X_te[index], 4)}

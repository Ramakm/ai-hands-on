"""Stateless numerical building blocks shared by every model."""

import numpy as np


def sigmoid(x):
    # Algebraically identical to 1 / (1 + exp(-x)) but never overflows.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def softmax(logits):
    z = logits - logits.max(axis=-1, keepdims=True)       # stability
    exp = np.exp(z)
    return exp / exp.sum(axis=-1, keepdims=True)


def softmax_cross_entropy(logits, y):
    """Mean cross-entropy over the batch and its gradient w.r.t. the logits."""
    probs = softmax(logits)

    B = logits.shape[0]
    loss = -np.log(probs[np.arange(B), y] + 1e-12).mean()

    dlogits = probs.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    return loss, dlogits

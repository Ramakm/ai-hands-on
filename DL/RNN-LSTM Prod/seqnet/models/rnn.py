"""Vanilla (Elman) RNN classifier - forward pass and BPTT by hand."""

import numpy as np

from seqnet.functional import softmax


class RNN:
    name = "rnn"

    def __init__(self, input_size, hidden_size, output_size, clip=5.0, seed=0): #Default constructor 
        r = np.random.default_rng(seed)
        self.H = hidden_size
        self.clip = clip
        self.config = {"input_size": input_size, "hidden_size": hidden_size,
                       "output_size": output_size, "clip": clip, "seed": seed}
        # Xavier-style scaling keeps initial activations in a sane range.
        self.params = {
            "Wxh": r.normal(0, 1 / np.sqrt(input_size),  (input_size,  hidden_size)),
            "Whh": r.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, hidden_size)),
            "bh":  np.zeros(hidden_size),
            "Why": r.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, output_size)),
            "by":  np.zeros(output_size),
        }

    # ---------------- forward ----------------
    def forward(self, X):
        """X: (B, T, D) -> logits (B, C), plus the cache needed for BPTT."""
        p = self.params
        B, T, _ = X.shape

        h = np.zeros((B, self.H))
        hs = [h]                                  # hs[t] == h_t, hs[0] == h_0 == 0
        for t in range(T):
            a = X[:, t] @ p["Wxh"] + h @ p["Whh"] + p["bh"]
            h = np.tanh(a)
            hs.append(h)

        logits = h @ p["Why"] + p["by"]           # only the last state is read
        return logits, (X, hs)

    # ---------------- backward (BPTT) ----------------
    def backward(self, dlogits, cache):
        p = self.params
        X, hs = cache
        T = X.shape[1]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        grads["Why"] = hs[-1].T @ dlogits
        grads["by"]  = dlogits.sum(axis=0)
        dh = dlogits @ p["Why"].T                 # gradient flowing into h_T

        for t in reversed(range(T)):
            h_t, h_prev = hs[t + 1], hs[t]
            da = dh * (1.0 - h_t ** 2)            # tanh'(a) = 1 - tanh(a)^2

            grads["Wxh"] += X[:, t].T @ da        # += : weights are shared
            grads["Whh"] += h_prev.T  @ da
            grads["bh"]  += da.sum(axis=0)

            dh = da @ p["Whh"].T                  # hand the gradient to h_{t-1}

        for g in grads.values():                  # tame exploding gradients
            np.clip(g, -self.clip, self.clip, out=g)
        return grads

    # ---------------- inference ----------------
    def predict_proba(self, X):
        return softmax(self.forward(X)[0])

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    def trace(self, X):
        """Per-timestep internals: hidden states and what the output layer
        would predict if the sequence stopped after row t."""
        _, (_, hs) = self.forward(X)
        hidden = np.stack(hs[1:], axis=1)                     # (B, T, H)
        step_logits = hidden @ self.params["Why"] + self.params["by"]
        return {"hidden": hidden, "step_probs": softmax(step_logits)}

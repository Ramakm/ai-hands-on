"""LSTM classifier - forward pass and BPTT by hand."""

import numpy as np

from seqnet.functional import sigmoid, softmax


class LSTM:
    name = "lstm"

    def __init__(self, input_size, hidden_size, output_size, clip=5.0, seed=0):
        r = np.random.default_rng(seed)
        H = self.H = hidden_size
        self.clip = clip
        self.config = {"input_size": input_size, "hidden_size": hidden_size,
                       "output_size": output_size, "clip": clip, "seed": seed}
        # The four gates share one matrix each, stacked in the order [i, f, o, g].
        b = np.zeros(4 * H)
        b[H:2 * H] = 1.0                          # forget-gate bias 1: remember by default
        self.params = {
            "Wx":  r.normal(0, 1 / np.sqrt(input_size),  (input_size,  4 * H)),
            "Wh":  r.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, 4 * H)),
            "b":   b,
            "Why": r.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, output_size)),
            "by":  np.zeros(output_size),
        }

    # ---------------- forward ----------------
    def forward(self, X):
        """X: (B, T, D) -> logits (B, C), plus the cache needed for BPTT."""
        p, H = self.params, self.H
        B, T, _ = X.shape

        h = np.zeros((B, H))
        c = np.zeros((B, H))
        steps = []
        for t in range(T):
            a = X[:, t] @ p["Wx"] + h @ p["Wh"] + p["b"]
            i = sigmoid(a[:, :H])                 # input gate:  how much new info to write
            f = sigmoid(a[:, H:2 * H])            # forget gate: how much old memory to keep
            o = sigmoid(a[:, 2 * H:3 * H])        # output gate: how much memory to expose
            g = np.tanh(a[:, 3 * H:])             # candidate values to write

            h_prev, c_prev = h, c
            c = f * c_prev + i * g                # additive memory update
            tanh_c = np.tanh(c)
            h = o * tanh_c
            steps.append((h_prev, c_prev, i, f, o, g, tanh_c))

        logits = h @ p["Why"] + p["by"]           # only the last state is read
        return logits, (X, steps, h)

    # ---------------- backward (BPTT) ----------------
    def backward(self, dlogits, cache):
        p = self.params
        X, steps, h_T = cache
        T = X.shape[1]
        grads = {k: np.zeros_like(v) for k, v in p.items()}

        grads["Why"] = h_T.T @ dlogits
        grads["by"]  = dlogits.sum(axis=0)
        dh = dlogits @ p["Why"].T                 # gradient flowing into h_T
        dc = np.zeros_like(dh)                    # nothing reads c_T except h_T

        for t in reversed(range(T)):
            h_prev, c_prev, i, f, o, g, tanh_c = steps[t]

            do = dh * tanh_c
            dc = dc + dh * o * (1.0 - tanh_c ** 2)    # two paths into c_t
            di = dc * g
            df = dc * c_prev
            dg = dc * i

            da = np.concatenate([di * i * (1.0 - i),  # sigmoid' = s(1 - s)
                                 df * f * (1.0 - f),
                                 do * o * (1.0 - o),
                                 dg * (1.0 - g ** 2)], axis=1)

            grads["Wx"] += X[:, t].T @ da         # += : weights are shared
            grads["Wh"] += h_prev.T  @ da
            grads["b"]  += da.sum(axis=0)

            dh = da @ p["Wh"].T                   # hand the gradient to h_{t-1}
            dc = dc * f                           # ... and to c_{t-1}, scaled only by f

        for g_ in grads.values():                 # tame exploding gradients
            np.clip(g_, -self.clip, self.clip, out=g_)
        return grads

    # ---------------- inference ----------------
    def predict_proba(self, X):
        return softmax(self.forward(X)[0])

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    def trace(self, X):
        """Per-timestep internals: hidden and cell states, gate activations,
        and what the output layer would predict if the sequence stopped at row t."""
        _, (_, steps, _) = self.forward(X)
        h_prev, c_prev, i, f, o, g, tanh_c = (np.stack(s, axis=1) for s in zip(*steps))
        hidden = o * tanh_c                                   # (B, T, H)
        step_logits = hidden @ self.params["Why"] + self.params["by"]
        return {"hidden": hidden, "cell": f * c_prev + i * g,
                "gates": {"input": i, "forget": f, "output": o},
                "step_probs": softmax(step_logits)}

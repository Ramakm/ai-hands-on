"""Mini-batch training loop and gradient check, shared by the CLI, tests and notebook."""

import numpy as np

from seqnet.functional import softmax_cross_entropy
from seqnet.optim import Adam


def fit(net, X_tr, y_tr, X_te, y_te, epochs=30, batch_size=64, lr=2e-3,
        rng=None, log_every=5, log=print):
    rng = np.random.default_rng(0) if rng is None else rng
    opt = Adam(net.params, lr=lr)
    n = len(X_tr)
    history = {"loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        running = 0.0

        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            Xb, yb = X_tr[idx], y_tr[idx]

            logits, cache = net.forward(Xb)             # forward
            loss, dlogits = softmax_cross_entropy(logits, yb)
            grads = net.backward(dlogits, cache)        # BPTT
            opt.step(net.params, grads)                 # update

            running += loss * len(idx)

        tr_acc = float((net.predict(X_tr) == y_tr).mean())
        te_acc = float((net.predict(X_te) == y_te).mean())
        history["loss"].append(running / n)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if log and (epoch == 1 or epoch % log_every == 0):
            log(f"  [{net.name}] epoch {epoch:3d} | loss {running / n:.4f} "
                f"| train acc {tr_acc:.3f} | test acc {te_acc:.3f}")
    return history


def gradient_check(net, X, y, rng, eps=1e-5):
    """Compare one random entry of every analytic gradient with a central
    finite difference. Returns {param: (analytic, numeric, relative_error)}."""
    logits, cache = net.forward(X)
    _, dlogits = softmax_cross_entropy(logits, y)
    grads = net.backward(dlogits, cache)

    report = {}
    for name, W in net.params.items():
        idx = tuple(rng.integers(0, s) for s in W.shape)   # one random entry
        orig = W[idx]

        W[idx] = orig + eps
        loss_p, _ = softmax_cross_entropy(net.forward(X)[0], y)
        W[idx] = orig - eps
        loss_m, _ = softmax_cross_entropy(net.forward(X)[0], y)
        W[idx] = orig

        numeric  = (loss_p - loss_m) / (2 * eps)
        analytic = grads[name][idx]
        rel = abs(numeric - analytic) / max(1e-12, abs(numeric) + abs(analytic))
        report[name] = (float(analytic), float(numeric), float(rel))
    return report

"""Train the RNN and LSTM with identical settings and write artifacts.

    python -m seqnet.train                      # both models, settings from config.py
    python -m seqnet.train --models lstm --epochs 50
"""

import argparse
import logging
import time
from datetime import datetime, timezone

import numpy as np

from seqnet import __version__, config
from seqnet.data import FEATURES, NUM_CLASSES, load_digit_sequences
from seqnet.models import MODEL_REGISTRY
from seqnet.persistence import parameter_count, save_model
from seqnet.training import fit

log = logging.getLogger("seqnet.train")


def confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def train_models(names=tuple(MODEL_REGISTRY), artifact_dir=config.ARTIFACT_DIR,
                 hidden_size=config.HIDDEN_SIZE, epochs=config.EPOCHS,
                 batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE,
                 clip=config.GRAD_CLIP, model_seed=config.MODEL_SEED,
                 shuffle_seed=config.SHUFFLE_SEED, test_size=config.TEST_SIZE,
                 split_seed=config.SPLIT_SEED):
    X_tr, X_te, y_tr, y_te = load_digit_sequences(test_size, split_seed)
    results = {}

    for name in names:
        net = MODEL_REGISTRY[name](FEATURES, hidden_size, NUM_CLASSES, clip=clip, seed=model_seed)
        log.info("training %s (%d parameters)", name, parameter_count(net))

        # A fresh RNG per model: both see exactly the same mini-batches in the same order.
        rng = np.random.default_rng(shuffle_seed)
        started = time.perf_counter()
        history = fit(net, X_tr, y_tr, X_te, y_te, epochs=epochs, batch_size=batch_size,
                      lr=lr, rng=rng, log=log.info)
        seconds = time.perf_counter() - started

        y_pred = net.predict(X_te)
        metadata = {
            "version": __version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "parameters": parameter_count(net),
            "training": {"epochs": epochs, "batch_size": batch_size, "lr": lr,
                         "shuffle_seed": shuffle_seed, "seconds": round(seconds, 2)},
            "data": {"dataset": "sklearn.digits", "test_size": test_size,
                     "split_seed": split_seed, "train_samples": len(X_tr),
                     "test_samples": len(X_te)},
            "metrics": {"train_acc": history["train_acc"][-1],
                        "test_acc": float((y_pred == y_te).mean()),
                        "final_loss": history["loss"][-1]},
            "history": history,
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        }
        weights, _ = save_model(net, artifact_dir, metadata)
        log.info("%s: test acc %.4f in %.1fs -> %s",
                 name, metadata["metrics"]["test_acc"], seconds, weights)
        results[name] = metadata
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_REGISTRY),
                        default=list(MODEL_REGISTRY))
    parser.add_argument("--artifact-dir", default=config.ARTIFACT_DIR)
    parser.add_argument("--hidden-size", type=int, default=config.HIDDEN_SIZE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--clip", type=float, default=config.GRAD_CLIP)
    parser.add_argument("--seed", type=int, default=config.MODEL_SEED)
    args = parser.parse_args(argv)

    logging.basicConfig(level=config.LOG_LEVEL,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    results = train_models(args.models, args.artifact_dir, args.hidden_size, args.epochs,
                           args.batch_size, args.lr, args.clip, args.seed)
    for name, meta in results.items():
        m = meta["metrics"]
        print(f"{name:>5}: test acc {m['test_acc']:.4f} | train acc {m['train_acc']:.4f} "
              f"| {meta['parameters']:,} params | {meta['training']['seconds']}s")


if __name__ == "__main__":
    main()

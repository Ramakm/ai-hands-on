"""Save / load trained models as `<name>.npz` (weights) + `<name>.json` (metadata)."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from seqnet.models import MODEL_REGISTRY


def _atomic_write(path: Path, write_fn, suffix):
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=suffix)
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_model(net, directory, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    weights = directory / f"{net.name}.npz"
    meta = directory / f"{net.name}.json"

    _atomic_write(weights, lambda p: np.savez(p, **net.params), ".npz")
    doc = {"model": net.name, "config": net.config, **metadata}
    _atomic_write(meta, lambda p: Path(p).write_text(json.dumps(doc, indent=2)), ".json")
    return weights, meta


def load_model(name, directory):
    """Rebuild a model from disk. Raises FileNotFoundError / ValueError on bad artifacts."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'")
    directory = Path(directory)
    meta_path, weights_path = directory / f"{name}.json", directory / f"{name}.npz"
    if not meta_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"missing artifacts for '{name}' in {directory}")

    metadata = json.loads(meta_path.read_text())
    if metadata.get("model") != name:
        raise ValueError(f"{meta_path} describes '{metadata.get('model')}', not '{name}'")

    net = MODEL_REGISTRY[name](**metadata["config"])
    with np.load(weights_path) as stored:
        missing = set(net.params) - set(stored.files)
        if missing:
            raise ValueError(f"{weights_path} is missing parameters {sorted(missing)}")
        for key, expected in net.params.items():
            if stored[key].shape != expected.shape:
                raise ValueError(f"{weights_path}:{key} has shape {stored[key].shape}, "
                                 f"expected {expected.shape}")
            net.params[key] = stored[key].astype(np.float64)
    return net, metadata


def parameter_count(net):
    return int(sum(v.size for v in net.params.values()))

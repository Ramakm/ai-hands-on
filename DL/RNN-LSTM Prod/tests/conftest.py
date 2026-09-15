import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory):
    """Both models trained for one quick epoch - enough to exercise the full stack."""
    from seqnet.train import train_models

    out = tmp_path_factory.mktemp("artifacts")
    train_models(artifact_dir=out, hidden_size=8, epochs=1)
    return out

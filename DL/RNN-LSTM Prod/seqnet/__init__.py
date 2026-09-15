"""seqnet - RNN and LSTM sequence classifiers written from scratch in NumPy."""

from seqnet.models import LSTM, MODEL_REGISTRY, RNN
from seqnet.optim import Adam

__all__ = ["RNN", "LSTM", "Adam", "MODEL_REGISTRY"]
__version__ = "1.0.0"

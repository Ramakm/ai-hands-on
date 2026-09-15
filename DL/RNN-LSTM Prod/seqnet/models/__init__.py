from seqnet.models.lstm import LSTM
from seqnet.models.rnn import RNN

MODEL_REGISTRY = {RNN.name: RNN, LSTM.name: LSTM}

__all__ = ["RNN", "LSTM", "MODEL_REGISTRY"]

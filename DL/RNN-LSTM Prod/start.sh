#!/usr/bin/env bash
# One command to go from a fresh checkout to the comparison UI on localhost.
set -euo pipefail
cd "$(dirname "$0")"

HOST="${SEQNET_HOST:-127.0.0.1}"
PORT="${SEQNET_PORT:-8000}"
PY=.venv/bin/python

if [ ! -x "$PY" ]; then
  echo "==> creating virtualenv"
  python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r requirements.txt
fi

ARTIFACTS="${SEQNET_ARTIFACT_DIR:-artifacts}"
if [ ! -f "$ARTIFACTS/rnn.npz" ] || [ ! -f "$ARTIFACTS/lstm.npz" ]; then
  echo "==> no trained models found, training RNN and LSTM"
  "$PY" -m seqnet.train
fi

echo "==> serving on http://$HOST:$PORT"
exec "$PY" -m uvicorn seqnet.api:app --host "$HOST" --port "$PORT"

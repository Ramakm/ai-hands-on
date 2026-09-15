PY     := .venv/bin/python
HOST   ?= 127.0.0.1
PORT   ?= 8000

.PHONY: install train serve run test notebook docker clean

install:
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements-dev.txt

train:
	$(PY) -m seqnet.train

serve:
	$(PY) -m uvicorn seqnet.api:app --host $(HOST) --port $(PORT)

run:
	./start.sh

test:
	$(PY) -m pytest -q

notebook:
	$(PY) -m ipykernel install --user --name rnn-lstm --display-name "Python 3 (rnn-lstm)" > /dev/null
	$(PY) scripts/build_notebook.py --execute --kernel rnn-lstm

docker:
	docker build -t rnn-vs-lstm .
	docker run --rm -p $(PORT):8000 rnn-vs-lstm

clean:
	rm -rf artifacts .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

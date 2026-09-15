"""The notebook's model / training cells must be the package source, verbatim."""

import importlib
import inspect
import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "rnn_from_scratch.ipynb"


def _resolve(ref):
    module, qualname = ref.split(":")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def test_notebook_cells_match_package_source():
    nb = json.loads(NOTEBOOK.read_text())
    synced = [c for c in nb["cells"] if "seqnet_source" in c.get("metadata", {})]
    assert synced, "notebook has no synced cells - rebuild with scripts/build_notebook.py"

    covered = set()
    for cell in synced:
        refs = cell["metadata"]["seqnet_source"]
        expected = "\n\n\n".join(inspect.getsource(_resolve(r)).rstrip() for r in refs)
        actual = "".join(cell["source"])
        assert actual == expected, (
            f"notebook cell for {refs} is out of date - run `make notebook`")
        covered.update(refs)

    for required in ("seqnet.models.rnn:RNN", "seqnet.models.lstm:LSTM",
                     "seqnet.optim:Adam", "seqnet.training:fit"):
        assert required in covered

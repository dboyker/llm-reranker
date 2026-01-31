import json
from pathlib import Path


def load_dataset(path: Path) -> list[dict]:
    """Load and return a dataset stored in a given path. @TODO: delete and replace by HF version."""
    with open(path) as f:
        dataset = [json.loads(line) for line in f]
    return dataset
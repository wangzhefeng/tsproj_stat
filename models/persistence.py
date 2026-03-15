from __future__ import annotations

import pickle
from pathlib import Path


def save_model(model, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str):
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)

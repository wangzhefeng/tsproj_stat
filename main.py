from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from task.app import ModelApp
from task.config import AppConfig


def main() -> None:
    cfg = AppConfig()
    app = ModelApp(cfg)
    result = app.run()
    print("ModelApp done:", result)


if __name__ == "__main__":
    main()

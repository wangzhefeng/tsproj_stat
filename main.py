from __future__ import annotations

from runtime_env import ensure_mpl_config_dir

ensure_mpl_config_dir()

from app import ModelApp
from app.runtime import configure_warnings
from config import AppConfig


def main() -> None:
    configure_warnings()
    cfg = AppConfig()
    result = ModelApp(cfg).run()
    print("ModelApp done:", result)


if __name__ == "__main__":
    main()

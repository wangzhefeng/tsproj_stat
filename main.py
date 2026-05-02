from __future__ import annotations

from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()
from config import AppConfig
from app import ModelApp




def main() -> None:
    cfg = AppConfig()
    result = ModelApp(cfg).run()
    print("ModelApp done:", result)

if __name__ == "__main__":
    main()

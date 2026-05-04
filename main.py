from __future__ import annotations

from utils.runtime_env import ensure_mpl_config_dir
ensure_mpl_config_dir()
from config import AppConfig
from app import ModelApp

from utils.log_util import logger



def main() -> None:
    """最小示例入口：使用默认 AppConfig 直接运行完整 ModelApp。"""
    cfg = AppConfig()
    result = ModelApp(cfg).run()
    logger.info("ModelApp done:", result)

if __name__ == "__main__":
    main()

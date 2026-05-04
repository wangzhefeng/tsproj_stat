"""
日志工具模块

该模块提供了日志记录功能，包括：
- 控制台日志输出
- 按天轮转的文件日志
- 日志级别通过环境变量SERVICE_LOG_LEVEL配置
- JSON结构化格式（通过 configure_logging 启用）
- run_id 注入与阶段耗时追踪
"""

import contextlib
import datetime
import json
import os
import re
import sys
import time
import logging
from logging import handlers
from pathlib import Path


# 项目根路径
ROOT_PATH = Path.cwd()

# 日志路径
LOG_DIR = Path(f"{ROOT_PATH}/logs/{os.environ.get('LOG_NAME')}")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR.joinpath("service")

# 日志级别，默认为INFO
LOG_LEVEL = os.environ.get("SERVICE_LOG_LEVEL", "INFO")

# 当前 run_id（由 configure_logging 或 set_run_id 设置）
_current_run_id: str = ""


class _RunIdFilter(logging.Filter):
    """Inject current run_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _current_run_id
        return True


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "run_id": getattr(record, "run_id", _current_run_id),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "stage"):
            payload["stage"] = record.stage
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = record.duration_ms
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


# 默认文本格式
_text_formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"
)

# 控制台日志处理器
stream_handler = logging.StreamHandler(stream=sys.stderr)
stream_handler.setLevel(LOG_LEVEL)
stream_handler.setFormatter(_text_formatter)

# 按天轮转文件日志处理器
time_rotating_file_handler = handlers.TimedRotatingFileHandler(
    filename=LOG_PATH,
    when="MIDNIGHT",
    interval=1,
    backupCount=10,
    encoding="utf-8",
)
time_rotating_file_handler.suffix = "%Y-%m-%d.log"
time_rotating_file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
time_rotating_file_handler.setLevel(LOG_LEVEL)
time_rotating_file_handler.setFormatter(_text_formatter)

# 主日志记录器
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addHandler(time_rotating_file_handler)
logger.addFilter(_RunIdFilter())
logger.setLevel(LOG_LEVEL)
logger.propagate = False


def set_run_id(run_id: str) -> None:
    """Update the run_id injected into all subsequent log records."""
    global _current_run_id
    _current_run_id = run_id


def configure_logging(log_format: str = "text", run_id: str = "") -> None:
    """Reconfigure the module logger format and optionally set run_id.

    Args:
        log_format: "text" (default) or "json"
        run_id: identifier for the current run; injected into every record
    """
    global _current_run_id
    _current_run_id = run_id

    if log_format == "json":
        fmt = JsonFormatter()
    else:
        fmt = _text_formatter

    for handler in logger.handlers:
        handler.setFormatter(fmt)


@contextlib.contextmanager
def timed_stage(stage_name: str, _logger=None):
    """Context manager that logs elapsed time for a named pipeline stage.

    Usage:
        with timed_stage("train"):
            model.fit(...)
    """
    _log = _logger or logger
    t0 = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        _log.info(
            f"[{stage_name}] completed in {elapsed_ms}ms",
            extra={"stage": stage_name, "duration_ms": elapsed_ms},
        )


def main():
    """日志功能演示"""
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")


if __name__ == "__main__":
    main()
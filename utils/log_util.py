"""
日志工具模块

该模块提供了日志记录功能，包括：
- 控制台日志输出
- 按天轮转的文件日志（懒挂载：LOG_NAME 环境变量存在时，在 import 或
  configure_logging 时创建 logs/{LOG_NAME}/service；未设置时只输出控制台）
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


# 日志级别，默认为INFO
LOG_LEVEL = os.environ.get("SERVICE_LOG_LEVEL", "INFO")

# 当前 run_id（由 configure_logging 或 set_run_id 设置）
_current_run_id: str = ""


class _RunIdFilter(logging.Filter):
    """把当前 run_id 注入每条日志记录。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _current_run_id
        return True


class JsonFormatter(logging.Formatter):
    """把日志记录格式化为单行 JSON 对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "run_id": getattr(record, "run_id", _current_run_id),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if hasattr(record, "stage"):
            payload["stage"] = getattr(record, "stage")
        if hasattr(record, "duration_ms"):
            payload["duration_ms"] = getattr(record, "duration_ms")
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

# 主日志记录器
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.addFilter(_RunIdFilter())
logger.setLevel(LOG_LEVEL)
logger.propagate = False

# 按天轮转文件日志处理器（懒挂载）
_file_handler: handlers.TimedRotatingFileHandler | None = None


def _build_file_handler(log_name: str) -> handlers.TimedRotatingFileHandler:
    """在 cwd/logs/{log_name}/ 下创建按天轮转的文件日志处理器。"""
    log_dir = Path.cwd() / "logs" / log_name
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = handlers.TimedRotatingFileHandler(
        filename=log_dir / "service",
        when="MIDNIGHT",
        interval=1,
        backupCount=10,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d.log"
    handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    handler.setLevel(LOG_LEVEL)
    handler.setFormatter(_text_formatter)
    return handler


def _ensure_file_handler() -> None:
    """LOG_NAME 已设置且尚未挂载文件 handler 时挂载（幂等）。

    文件日志目录在首次需要时才创建，避免 import 顺序竞态决定日志落点，
    也避免未命名运行产生 logs/None 目录。
    """
    global _file_handler
    if _file_handler is not None:
        return
    log_name = os.environ.get("LOG_NAME")
    if not log_name:
        return
    _file_handler = _build_file_handler(log_name)
    logger.addHandler(_file_handler)


# import 时尝试挂载（脚本经 export LOG_NAME 运行时保持旧行为）
_ensure_file_handler()


def set_run_id(run_id: str) -> None:
    """更新注入后续所有日志记录的 run_id。"""
    global _current_run_id
    _current_run_id = run_id


def configure_logging(log_format: str = "text", run_id: str = "") -> None:
    """重配置模块 logger 的格式，并可选设置 run_id。

    Args:
        log_format: "text"（默认）或 "json"
        run_id: 当前运行标识，注入每条记录
    """
    global _current_run_id
    _current_run_id = run_id

    # 运行边界处兜底挂载文件日志（LOG_NAME 在 import 后才设置的场合）
    _ensure_file_handler()

    if log_format == "json":
        fmt = JsonFormatter()
    else:
        fmt = _text_formatter

    for handler in logger.handlers:
        handler.setFormatter(fmt)


@contextlib.contextmanager
def timed_stage(stage_name: str, _logger=None):
    """记录具名流水线阶段耗时的上下文管理器。

    用法：
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
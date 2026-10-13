"""结构化日志。

统一走 structlog：开发时是彩色可读文本，需要时把 processor 换成 JSONRenderer
就能直接喂给日志采集系统（无需改业务代码）。
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.core.config import get_settings

_configured = False


def setup_logging() -> None:
    """初始化日志，幂等。应在应用启动时调用一次。"""
    global _configured
    if _configured:
        return

    level_name = get_settings().log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None):
    """获取 logger。`setup_logging()` 之前调用也安全。"""
    return structlog.get_logger(name)

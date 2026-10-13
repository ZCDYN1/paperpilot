"""FastAPI 应用入口。

装配顺序：配置 → 日志 → 数据库 → 路由。
检索索引与 BGE 模型**不在启动时加载**：它们体积大、且 mock 模式下根本用不到，
放进 lifespan 会让"打开程序"变成"等 3 分钟"。改为按需懒加载（见 HybridRetriever）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app import __version__
from app.api.v1 import health, research
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import dispose_engine, init_db

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭钩子。"""
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    logger.info("paperpilot.starting", version=__version__, **settings.summary())

    # 建表（幂等）。生产环境应换 Alembic 迁移，这里保证 clone 下来就能跑。
    try:
        await init_db()
        logger.info("paperpilot.db_ready", sqlite_url=settings.sqlite_url)
    except Exception as exc:  # noqa: BLE001 - 建表失败不该让整个服务起不来（/healthz 仍要可用）
        logger.error("paperpilot.db_init_failed", error=str(exc)[:300])

    # 配置自检：把问题在启动时就喊出来，而不是等第一次提问才炸
    if settings.llm_provider == "openai" and not settings.openai_api_key:
        logger.warning(
            "paperpilot.llm_not_configured",
            hint="OPENAI_API_KEY 为空。要么填上它，要么把 LLM_PROVIDER 设为 mock 以便离线跑通。",
        )
    if not settings.ingest_enabled:
        logger.info(
            "paperpilot.ingest_disabled",
            hint="INGEST_ENABLED=false：只检索已有索引，不联网抓取（离线演示 / CI 用）。",
        )

    yield

    await dispose_engine()
    logger.info("paperpilot.stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()

    app = FastAPI(
        title="PaperPilot",
        description="基于 LangGraph 多 Agent 的 arXiv 论文深度研究工具",
        version=__version__,
        lifespan=lifespan,
    )

    # /healthz 按约定挂在根路径（容器 healthcheck 用的就是它）
    app.include_router(health.router)
    # 业务路由统一挂 /api/v1（架构 §7）
    app.include_router(research.router, prefix="/api/v1")

    return app


app = create_app()

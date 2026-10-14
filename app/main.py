"""FastAPI 应用入口。

装配顺序：配置 → 日志 → **追踪** → 数据库 → 路由。
检索索引与 BGE 模型**不在启动时加载**：它们体积大、且 mock 模式下根本用不到，
放进 lifespan 会让"打开程序"变成"等 3 分钟"。改为按需懒加载（见 HybridRetriever）。

★ 为什么 lifespan 里要专门调一次 `configure_tracing()`：
`Settings` 从 `.env` 读到的 `LANGCHAIN_*` 只活在 Pydantic 实例里，**不会自动进 `os.environ`**，
而 langsmith 判断"要不要上报"恰恰只看进程环境。少了这一步，
"我在 `.env` 里填了 key"与"真的有 trace"之间就断着一层静默无效（见该模块 docstring）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app import __version__
from app.api.v1 import health, papers, research, terms
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.resilience import BREAKER_HTTP, BREAKER_LLM, get_breaker
from app.db.session import dispose_engine, init_db
from app.observability.tracing import configure_tracing

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动 / 关闭钩子。"""
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    # 追踪：把 LANGCHAIN_* 同步进进程环境。返回的是脱敏字典，可以安全打日志。
    # 放在建表之前：它只是写几个环境变量，没有 I/O，也不该被 DB 故障挡住。
    tracing = configure_tracing(settings)

    logger.info("paperpilot.starting", version=__version__, **settings.summary())
    # 追踪状态单独一条：新用户第一眼就能看到"到底有没有在上报 trace"，
    # 而不是等半天在 LangSmith 控制台什么都没等到（那正是以前最费时间的"静默无效"）。
    logger.info("paperpilot.tracing", **tracing)

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
    # 容错层在启动时就把熔断器实例建好（阈值此刻从 settings 读定），
    # 并让 /healthz 第一次就能报出 llm / http 两个熔断器的状态 ——
    # "熔断是死代码"的另一个特征是运维根本看不到它存在过。
    logger.info(
        "paperpilot.resilience_ready",
        enabled=settings.resilience_enabled,
        llm_breaker=get_breaker(BREAKER_LLM).name,
        http_breaker=get_breaker(BREAKER_HTTP).name,
        failure_threshold=settings.breaker_failure_threshold,
        recovery_timeout=settings.breaker_recovery_timeout,
        timeout_budget=settings.resilience_timeout,
        attempts=settings.resilience_attempts,
    )
    logger.info("paperpilot.input_guard", enabled=settings.input_guard_enabled, max_chars=settings.input_max_chars, injection_action=settings.input_injection_action)

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
    app.include_router(papers.router, prefix="/api/v1")
    app.include_router(terms.router, prefix="/api/v1")

    return app


app = create_app()

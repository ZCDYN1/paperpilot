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
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__

# 别名导入而不是 `from app.api.v1 import eval`：`eval` 是内置函数名，直接引进来会遮蔽它
from app.api.v1 import eval as eval_routes
from app.api.v1 import health, papers, research, stats, terms
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


def _mount_web_ui(app: FastAPI) -> bool:
    """把 `web/dist` 挂成单端口前端；没构建过就安静地跳过。

    ★ 为什么要挂在这里而不是让前端单独跑一个 Node 服务：
    同源之后**不需要 CORS、不需要第二个端口、不需要第二个容器**，
    别人 clone 下来 `npm run build` 一次，之后 `启动PaperPilot.bat` 就是完整产品。
    没构建时也不报错 —— 后端 API 与 Streamlit 面板照常可用。

    路由顺序是关键：本函数在 `include_router` **之后**调用，所以 `/api/v1/*`、
    `/healthz`、`/docs` 都会先匹配到；兜底路由再把其余 GET 交给 SPA。
    """
    dist = Path(__file__).resolve().parents[1] / "web" / "dist"
    index = dist / "index.html"
    if not index.is_file():
        return False

    assets = dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="web-assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str) -> FileResponse:
        # 兜底不能吞掉 API：否则拼错的接口路径会返回 200 + 一坨 HTML，
        # 前端 `resp.json()` 报"解析失败"，真实原因（404）被彻底盖住。
        if full_path.startswith(("api/", "docs", "redoc")) or full_path == "healthz":
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(index)

    return True


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
    app.include_router(stats.router, prefix="/api/v1")
    app.include_router(eval_routes.router, prefix="/api/v1")

    # 前端挂载必须排在所有 API 路由之后（见 _mount_web_ui 的路由顺序说明）
    if _mount_web_ui(app):
        logger.info("paperpilot.web_ui_mounted", hint="打开 http://127.0.0.1:8000 即为 React 控制台")
    else:
        logger.info(
            "paperpilot.web_ui_missing",
            hint="未找到 web/dist。构建： cd web && npm install && npm run build",
        )

    return app


app = create_app()

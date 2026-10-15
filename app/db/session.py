"""异步引擎与会话管理。

⚠️ **SQLite 必须开 WAL**：多 Agent 分支会并行写 `citations` / `tool_call_logs` /
checkpoint，默认 journal 模式下会直接抛 `database is locked`。
这个错误单线程测试永远碰不到，一开并行就随机出现、且极难复现 ——
所以它属于"必须最先落地的基础设施"，而不是事后调优项。

`busy_timeout` 同样重要：它让 SQLite 在遇到锁时等待而不是立即失败。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db.base import Base

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _install_sqlite_pragmas(engine: AsyncEngine) -> None:
    """每条新连接都设置一遍 PRAGMA（连接池会复用连接，所以挂在 connect 事件上）。"""

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")     # 并发读 + 单写，避免 database is locked
            cursor.execute("PRAGMA synchronous=NORMAL")   # WAL 下兼顾安全与吞吐
            cursor.execute("PRAGMA busy_timeout=5000")    # 遇锁等待 5s 再报错
            cursor.execute("PRAGMA foreign_keys=ON")      # SQLite 默认不启用外键约束
        finally:
            cursor.close()


def get_engine() -> AsyncEngine:
    """进程内单例 engine。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        _engine = create_async_engine(
            settings.sqlite_url,
            echo=False,
            future=True,
            # ★ 为什么是 NullPool 而不是默认的队列池（这条是实测撞出来的，不是预防性调优）：
            # 一条 SSE 流被客户端掐断时，uvicorn 取消该请求的任务，而它当时正持着一条
            # **池里共享的** aiosqlite 连接 —— 日志里表现为紧接着
            # `research.client_disconnected` 出现 `Exception terminating connection <AdaptedConnection ...>`。
            # 被 terminate 的连接不会干净地回到池里，下一个任务拿到它就是**永挂**：
            # 表现为 `planner.done` 之后再也没有任何节点日志、SSE 一条事件都不发，
            # 直到客户端自己的 300 秒 bodyTimeout 才断开。踩不踩中是随机的，
            # 所以本地看起来"重跑一遍就绿"，最容易被判成测试不稳定。
            #
            # aiosqlite 的连接本身就绑在创建它的任务上，跨任务复用没有收益只有风险；
            # NullPool 每次用完即关，僵尸连接无从传染。代价是每次签出一条新连接，
            # 对 SQLite 而言这个开销可以忽略（打开的是本地文件，且 WAL 下并发读不打架），
            # 换来的是"取消一个请求不会危害其它请求"。
            poolclass=NullPool,
            # `pool_pre_ping` 在 NullPool 下没有对象可校验（每次都是新连接），去掉。
        )
        _install_sqlite_pragmas(_engine)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """事务作用域：正常提交、异常回滚、结束关闭。"""
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """建表（幂等）+ 最小列迁移。生产环境应改用 Alembic，这里保证开箱即用。

    ⚠️ `create_all` **不会给已存在的表加列**（它只建缺失的表）：dev 库里
    `data/sqlite/paperpilot.db` 已经带着真实数据，所以每次加列都必须配一段
    "PRAGMA 查缺 → ALTER TABLE 补缺"的幂等迁移（见 `_ensure_citation_columns`）。
    新列一律可空，历史行保持 NULL —— 不丢数据、可重复执行、不引入迁移框架。
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_ensure_citation_columns)


# citations 表的最小列迁移清单：`列名 → 与 ORM 模型一致的 DDL 类型`。
# 与 `app/db/models.py` 的 Citation 手工保持同步（这正是"没有 Alembic"的代价，
# 刻意只收留这一处，避免它长成第二套 schema 定义）。
_CITATION_ADDED_COLUMNS: tuple[tuple[str, str], ...] = (
    ("resolution_how", "VARCHAR(16)"),
    ("verify_reanchored", "BOOLEAN"),
)


def _ensure_citation_columns(sync_conn) -> None:  # noqa: ANN001 - run_sync 传入的同步 Connection
    """给已存在的 `citations` 表补齐缺失的列（幂等：跑多少遍结果都一样）。

    只允许 `ADD COLUMN`（可空、无默认值回填），绝不 DROP/重建 ——
    SQLite 的 ALTER 能力有限，而这张表里是真实运行数据。
    """
    existing = {row[1] for row in sync_conn.execute(text("PRAGMA table_info(citations)"))}
    for name, ddl_type in _CITATION_ADDED_COLUMNS:
        if name not in existing:
            sync_conn.execute(text(f"ALTER TABLE citations ADD COLUMN {name} {ddl_type}"))


async def dispose_engine() -> None:
    """关闭连接池。应用 shutdown 时调用。"""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None

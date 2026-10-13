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

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

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
            pool_pre_ping=True,
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
    """建表（幂等）。生产环境应改用 Alembic 迁移，这里保证开箱即用。"""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """关闭连接池。应用 shutdown 时调用。"""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None

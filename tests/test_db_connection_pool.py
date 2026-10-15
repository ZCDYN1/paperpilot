"""连接池形态这条不变量（`app/db/session`）。

为什么值得单独钉一条：这不是性能调优，而是一条**取消安全性**的边界。

真实经过（本地实测，非假想）：前端在 HITL 中断点会关掉 SSE 连接，`contract.spec.ts`
里也有两条故意 `reader.cancel()` 的用例。默认的 `AsyncAdaptedQueuePool` 会把 aiosqlite
连接跨请求复用，被取消的那个任务在清理时把**池里的共享连接** terminate 掉 —— 后端日志里
表现为紧跟 `research.client_disconnected` 出现 `Exception terminating connection`。
中毒的连接不会干净地回到池里，下一个任务拿到它就是**永挂**：`planner.done` 之后
再没有任何节点日志，SSE 一条都不发，直到客户端自己的 300 秒 bodyTimeout 才断。
踩中与否是随机的（实测 3 轮里中 2 轮），所以最容易被误判成"测试不稳定、重跑就好"。

改成 NullPool 后同样的压测（18 次中途取消 × 3 轮）：`Exception terminating connection`
0 次，每轮随后的完整研究都在 1 秒内正常跑完。

这条断言防的是"有人为了少开连接把池加回来" —— 那会把上面整个故障模式原样带回来。

⚠️ 后两条用例刻意**自己造一个 tmp_path 上的引擎**，不碰 `get_engine()` 的进程单例：
那会连到开发者机器上 `data/sqlite/paperpilot.db` 里的真库（本仓库的约定是测试绝不写 dev 库）。
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.db.session import _install_sqlite_pragmas, get_engine


def test_sqlite_engine_does_not_pool_connections_across_tasks():
    """只看引擎配的池型，不建任何连接。"""
    engine = get_engine()
    assert isinstance(engine.pool, NullPool), (
        "aiosqlite 的连接绑在创建它的任务上，跨请求复用会让被取消的 SSE 请求"
        "terminate 掉共享连接并毒化池；详见本文件模块说明。"
    )


def _tmp_engine(tmp_path: Path):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'pool-probe.db').as_posix()}"
    engine = create_async_engine(url, future=True, poolclass=NullPool)
    _install_sqlite_pragmas(engine)
    return engine


async def test_every_new_connection_still_gets_wal_and_busy_timeout(tmp_path):
    """换池型不能把并发前提弄丢：WAL 与 busy_timeout 是并行写 citations 的底线。

    NullPool 表示每条连接都是新开的，PRAGMA 必须仍然挂在 connect 事件上逐条生效 ——
    这条比上一条更容易被漏掉，因为"连接不复用"看上去像与并发无关的改动。
    """
    engine = _tmp_engine(tmp_path)
    try:
        for _ in range(2):  # 两次签出 = 两条真实的新连接
            async with engine.connect() as conn:
                mode = (await conn.execute(text("PRAGMA journal_mode"))).scalar_one()
                timeout = (await conn.execute(text("PRAGMA busy_timeout"))).scalar_one()
            assert str(mode).lower() == "wal", "WAL 没生效：并行分支会直接 database is locked"
            assert int(timeout) >= 5000, "busy_timeout 没生效：遇锁立即失败而不是等待"
    finally:
        await engine.dispose()


async def test_committed_write_is_visible_to_the_next_fresh_connection(tmp_path):
    """用完即关不能把提交一起关掉：下一条新连接必须看得见上一条已写入的数据。

    这条盯的是 NullPool 的失败模式 —— 如果连接在提交前就被 disposed，症状是**写丢**而不是报错，
    而那正是本项目最不愿再见到的一类静默失效。
    """
    engine = _tmp_engine(tmp_path)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("CREATE TABLE probe (v INTEGER)"))
            await conn.execute(text("INSERT INTO probe VALUES (1)"))
            await conn.commit()

        async with engine.connect() as conn:
            count = (await conn.execute(text("SELECT COUNT(*) FROM probe"))).scalar_one()
        assert count == 1, "新连接看不见上一条已提交的写入：连接生命周期出了问题"
    finally:
        await engine.dispose()

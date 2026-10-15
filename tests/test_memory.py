"""记忆层测试（架构 §9）。

短期记忆验证"窗口 + 滚动摘要压缩"；长期记忆验证"写入 / 召回 / 淘汰"。
测试用**临时 SQLite 库** + mock embedder 隔离，不碰真实的 `data/sqlite/paperpilot.db`。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 导入以把 12 张表注册进 metadata
from app.db.base import Base
from app.db.models import LongTermMemory
from app.memory.long_term import (
    TYPE_DIRECTION,
    TYPE_READ_PAPER,
    LongTermMemoryStore,
)
from app.memory.short_term import SUMMARY_NODE, ShortTermMemory
from app.retrieval.embedder import MockEmbedder


@pytest.fixture
async def scope(tmp_path) -> AsyncIterator:
    """临时库的事务作用域工厂（签名与 `app.db.session.session_scope` 一致）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'memory.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    yield _scope
    await engine.dispose()


@pytest.fixture
def embedder() -> MockEmbedder:
    return MockEmbedder(dim=32)


@pytest.fixture
def long_term(scope, embedder) -> LongTermMemoryStore:
    return LongTermMemoryStore(scope=scope, embedder=embedder, max_items=3)


# =============================================================================
# 短期记忆
# =============================================================================
async def test_short_term_append_and_load_in_order(scope):
    mem = ShortTermMemory(max_turns=6, scope=scope)
    await mem.append_message("s1", "user", "第一个问题", run_id="r1")
    await mem.append_message("s1", "assistant", "第一个回答", node_name="writer")

    context = await mem.load_context("s1")
    assert [m["role"] for m in context] == ["user", "assistant"]
    assert context[0]["content"] == "第一个问题"
    assert context[1]["node_name"] == "writer"


async def test_short_term_window_and_rolling_summary(scope):
    """超过窗口的旧消息应被压缩成摘要，且摘要里保留早先内容的痕迹。"""
    mem = ShortTermMemory(max_turns=2, scope=scope)  # 窗口 = 4 条消息
    for i in range(1, 5):  # 4 轮 = 8 条消息
        await mem.append_message("s1", "user", f"历史问题{i}")
        await mem.append_message("s1", "assistant", f"历史回答{i}")

    context = await mem.load_context("s1", max_turns=2)
    # 首条是摘要，其后最多 4 条原文
    assert context[0]["node_name"] == SUMMARY_NODE
    assert len(context) == 1 + 4
    assert context[-1]["content"] == "历史回答4"
    # 早先内容（历史问题1）被折进摘要
    assert "历史问题1" in context[0]["content"]


async def test_short_term_uses_injected_summarizer(scope):
    """压缩必须把"更早的旧消息"交给摘要器，而不是模型自己瞎编。"""
    seen: list[list[str]] = []

    async def summarizer(older, prior):
        seen.append([m.content for m in older])
        return f"[SUMMARY]{prior}[cut]"

    mem = ShortTermMemory(max_turns=1, scope=scope, summarizer=summarizer)  # 窗口 = 2
    for i in range(1, 6):
        await mem.append_message("s1", "user", f"msg{i}")

    assert seen, "摘要器应被调用"
    # 第一次压缩时应把最早的 msg1 交给摘要器
    assert seen[0] == ["msg1"]
    context = await mem.load_context("s1", max_turns=1)
    assert context[0]["node_name"] == SUMMARY_NODE
    assert context[0]["content"].startswith("[SUMMARY]")
    assert len(context) == 1 + 2


async def test_short_term_does_not_compact_within_window(scope):
    mem = ShortTermMemory(max_turns=3, scope=scope)  # 窗口 = 6
    for i in range(1, 4):
        await mem.append_message("s1", "user", f"q{i}")
    context = await mem.load_context("s1")
    assert all(m["node_name"] != SUMMARY_NODE for m in context)
    assert [m["content"] for m in context] == ["q1", "q2", "q3"]


# =============================================================================
# 长期记忆
# =============================================================================
async def test_long_term_remember_and_recall(long_term):
    await long_term.remember("用户长期关注 RAG 与 Agent 相关论文", session_id="s1", importance=0.8)
    await long_term.remember("用户偏好用表格对比各方法", session_id="s1", importance=0.7)

    # 用与某条记忆完全相同的文本去召回（mock embedder 对同文本给同一向量 → 余弦 1.0）
    hits = await long_term.recall("用户偏好用表格对比各方法", top_k=1, session_id="s1")
    assert hits == ["用户偏好用表格对比各方法"]


async def test_long_term_recall_updates_last_used_at(long_term, scope):
    await long_term.remember("关注长序列建模", session_id="s1")
    await long_term.recall("关注长序列建模", top_k=1, session_id="s1")

    async with scope() as session:
        rows = (await session.execute(select(LongTermMemory))).scalars().all()
    assert rows and rows[0].last_used_at is not None


async def test_long_term_eviction_drops_lowest_retention(scope, embedder):
    """★ 淘汰：容量超限时，按 importance+新鲜度 删掉最差的那批（防"越用越慢、越用越吵"）。"""
    store = LongTermMemoryStore(scope=scope, embedder=embedder, max_items=3)
    for text, imp in [
        ("高价值方向", 0.9),
        ("较重要偏好", 0.8),
        ("一次性问题", 0.05),
        ("边角记忆", 0.1),
        ("中等记忆", 0.2),
    ]:
        await store.remember(text, session_id="s1", importance=imp)

    assert await store.count(session_id="s1") == 3
    kept = set(await store.recall("高价值方向", top_k=10, session_id="s1"))
    # 高重要度的留下，低重要度的被淘汰
    assert {"高价值方向", "较重要偏好", "中等记忆"} <= kept
    assert "一次性问题" not in kept
    assert "边角记忆" not in kept


async def test_long_term_evict_is_noop_below_capacity(scope, embedder):
    store = LongTermMemoryStore(scope=scope, embedder=embedder, max_items=10)
    await store.remember("a", session_id="s1")
    await store.remember("b", session_id="s1")
    assert await store.evict(session_id="s1") == 0
    assert await store.count(session_id="s1") == 2


async def test_long_term_remember_run_writes_direction_and_papers(scope, embedder):
    """`finalize` 节点通过 `remember_run` 接入：方向 + 已读论文都要写入。

    ## ★ 为什么这里刻意不断言"被淘汰的是哪一条"
    两篇已读论文的重要度都是 0.4（`remember_run` 写死的），而保留分是
    `importance * 0.7 + 新鲜度 * 0.3`。`created_at` 来自 SQLite 的
    `CURRENT_TIMESTAMP`，**精度只有 1 秒**：
    - 两次写入落在同一秒内 → 分数完全相等 → 淘汰谁由排序稳定性决定；
    - 两次写入恰好跨过秒边界 → 早写进去的那条新鲜度更低 → 换成它被淘汰。
    也就是说"哪一篇活下来"取决于墙钟读数，而不是被测逻辑。以前这个用例断言
    `2401.00001` 一定活下来，机器负载一高（写入间隔变大）就会偶发红：
    `assert any("2401.00001" in h for h in hits)` —— 单跑必过、全量偶发红。
    所以下面只断言**与墙钟无关**的量：写进去的条数 / 类型 / 已读清单，
    淘汰部分断言"少了一条已读论文、方向与偏好仍在"。
    """
    # ① 写入语义：容量给足，先看清 remember_run 到底写了什么（不发生淘汰）
    store = LongTermMemoryStore(scope=scope, embedder=embedder, max_items=10)
    ids = await store.remember_run(
        session_id="s1",
        run_id="r1",
        question="RAG 重排技术发展脉络",
        paper_ids=["2401.00001", "2401.00002"],
        summary="重排显著提升召回质量",
    )
    assert len(ids) == 4  # direction + 结论摘要(preference) + 2 篇已读论文
    assert await store.count(session_id="s1") == 4  # 未超容量 → 一条不剩地都在

    # 两篇都已登记为"已读"（这是 F10 排除重复精读的唯一依据）
    assert await store.read_paper_ids(session_id="s1") == {"2401.00001", "2401.00002"}

    hits = await store.recall("已读论文 arxiv_id=2401.00001", top_k=10, session_id="s1")
    assert any("2401.00001" in h for h in hits)
    assert any("2401.00002" in h for h in hits)

    # 数据类型标记正确
    from app.db.models import LongTermMemory  # noqa: PLC0415

    async with store._scope() as session:  # noqa: SLF001 - 测试内直查校验
        types = {r.memory_type for r in (await session.execute(select(LongTermMemory))).scalars().all()}
    assert types == {TYPE_DIRECTION, TYPE_READ_PAPER, "preference"}

    # ② 淘汰语义：把容量压到 3（4 条 → 必须删 1 条），只断言与墙钟无关的性质
    store.max_items = 3
    assert await store.evict(session_id="s1") == 1
    assert await store.count(session_id="s1") == 3

    # 被淘汰的只会是两条并列 0.4 的已读论文之一 —— 具体是哪一条不作断言（见 docstring）
    survivors = await store.read_paper_ids(session_id="s1")
    assert len(survivors) == 1
    assert survivors <= {"2401.00001", "2401.00002"}

    # 重要度更高的方向 / 偏好记忆不该被挤掉
    remaining = set(await store.recall("RAG 重排技术发展脉络", top_k=10, session_id="s1"))
    assert "RAG 重排技术发展脉络" in remaining
    assert any("重排显著提升召回质量" in r for r in remaining)


async def test_eviction_breaks_score_ties_by_insertion_order(scope, embedder):
    """并列分时**更早写入的那条先淘汰**，不能取决于数据库返回行的顺序。

    这条钉的是 `evict` 的次级排序键。SQLite 的 `CURRENT_TIMESTAMP` 只有秒级精度，
    同一秒写入、重要度又相同的两条记忆保留分完全相等 —— 没有次级键时"谁被淘汰"
    就是掷硬币，表现正是"单跑必过、全量偶发红"那类最难查的失败
    （`test_long_term_remember_run_writes_direction_and_papers` 的注释里记着现场）。
    """
    store = LongTermMemoryStore(scope=scope, embedder=embedder, max_items=2)
    for i in (1, 2, 3):
        await store.remember(f"记忆{i}", session_id="s1", importance=0.5)

    # `remember` 末尾自己会调一次全局 evict，所以这里断言的是"写完三条之后的稳定状态"
    hits = await store.recall("记忆", top_k=10, session_id="s1")
    assert not any("记忆1" == h for h in hits), "最早写入的那条必须被淘汰"
    assert any("记忆3" == h for h in hits)
    assert await store.count(session_id="s1") == 2
    assert await store.evict(session_id="s1") == 0, "已在容量内却还在删（说明淘汰没收敛）"

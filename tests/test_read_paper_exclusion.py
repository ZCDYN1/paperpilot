"""长期记忆「已读论文清单」的读取语义测试（产品定义 F10）。

这里守的是一条很容易在重构中被"顺手优化掉"的性质：
**排除已读论文要的是集合的完整性，不是与当前问题的相关性**。
一旦有人把 `read_paper_ids` 改成复用向量召回 `recall`，重复精读就会悄悄回来，
而且只在"上次读过的论文与这次问题语义不近"时才复现 —— 靠跑一遍主流程发现不了。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 导入以把表注册进 metadata
from app.agents.nodes import reader as reader_node
from app.agents.nodes._common import prioritize_unread
from app.agents.schemas import ReadOutput
from app.core.config import get_settings
from app.db.base import Base
from app.memory import long_term as lt
from app.memory.long_term import LongTermMemoryStore
from app.retrieval.embedder import MockEmbedder


@pytest.fixture
async def scope(tmp_path) -> AsyncIterator:
    """临时库的事务作用域工厂（签名与 `app.db.session.session_scope` 一致），不碰真实库。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'readpapers.db'}")
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
def store(scope) -> LongTermMemoryStore:
    return LongTermMemoryStore(scope=scope, embedder=MockEmbedder(dim=32), max_items=50)


_READ = ["2503.16581", "2410.05076", "2407.01219"]


async def _seed_read_papers(store: LongTermMemoryStore, session_id: str) -> None:
    await store.remember_run(
        session_id=session_id,
        run_id="run-1",
        question="RAG 重排策略有哪些",
        paper_ids=_READ,
    )


async def test_read_paper_ids_returns_every_read_paper_not_just_similar_ones(store):
    """★ 核心性质：与当前问题语义不近的已读论文也必须在集合里。

    `recall` 只返回 top-k 最相关的，用它做排除会漏掉"上次为别的子问题读过的那篇"，
    于是那篇被再精读一遍 —— 这正是 F10 要消灭的行为。
    """
    await _seed_read_papers(store, "s1")

    ids = await store.read_paper_ids(session_id="s1")
    assert ids == set(_READ)

    # 同一个库上，向量召回拿不全：top_k=1 只剩一条。
    # 两者都"能返回已读论文"，但只有全集才能用来做排除。
    recalled = await store.recall("完全不同的话题：图神经网络的可解释性", top_k=1, session_id="s1")
    assert len(recalled) == 1


async def test_read_paper_ids_ignores_other_memory_types(store):
    """方向 / 偏好类记忆的正文里也可能出现 arxiv_id 字样，不能被当成已读论文。"""
    await store.remember("关注 RAG 方向，常引用 arxiv_id=9999.99999 这篇综述",
                         session_id="s1", memory_type=lt.TYPE_DIRECTION)
    await store.remember(lt._read_paper_content("2503.16581"),
                         session_id="s1", memory_type=lt.TYPE_READ_PAPER)

    assert await store.read_paper_ids(session_id="s1") == {"2503.16581"}


async def test_read_paper_ids_is_scoped_by_session(store):
    await _seed_read_papers(store, "s1")
    await store.remember_run(session_id="s2", run_id="run-2", paper_ids=["2401.00001"])

    assert await store.read_paper_ids(session_id="s1") == set(_READ)
    assert await store.read_paper_ids(session_id="s2") == {"2401.00001"}
    # 不传 session_id = 跨会话全量（与 `recall` 的过滤行为一致）
    assert await store.read_paper_ids() == set(_READ) | {"2401.00001"}


async def test_write_and_parse_format_cannot_drift_apart(store):
    """写入格式与解析正则共用一个常量；这条测试锁住这个耦合。

    如果哪天有人改了 `remember_run` 里的正文措辞却忘了改正则，
    "已读"会静默失效（不报错，只是不再排除），只有这条测试会红。
    """
    await _seed_read_papers(store, "s1")
    assert await store.read_paper_ids(session_id="s1") == set(_READ)


async def test_read_paper_ids_does_not_refresh_last_used_at(store):
    """查询已读清单是索引查询，不该给这些记忆"续命"。

    若在这里刷新 `last_used_at`，已读记忆会因为被反复查询而永不被淘汰，
    把容量挤满、顶掉真正该长期保留的用户偏好。
    """
    await _seed_read_papers(store, "s1")
    await store.read_paper_ids(session_id="s1")

    async with store._scope() as session:  # noqa: SLF001 - 测试需要直接查证副作用
        from sqlalchemy import select

        from app.db.models import LongTermMemory

        rows = (await session.execute(select(LongTermMemory))).scalars().all()
    read_rows = [r for r in rows if r.memory_type == lt.TYPE_READ_PAPER]
    assert read_rows and all(r.last_used_at is None for r in read_rows)


async def test_safe_read_paper_ids_degrades_to_empty_set(store, monkeypatch):
    """记忆查询失败必须降级成"当作没读过"，而不是让精读节点抛异常打断整次研究。"""

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("数据库锁住了")

    monkeypatch.setattr(store, "read_paper_ids", _boom)
    monkeypatch.setattr(lt, "_default", store)

    assert await lt.safe_read_paper_ids() == set()


# =============================================================================
# ★ 跨会话：这一组守的是"长期记忆按 session_id 过滤 = 恒为空"这个真实的坑
#   （research.py:136 `session_id = run_id`，而记忆要等 finalize 才写入）
# =============================================================================
async def test_read_history_written_by_an_earlier_run_is_visible(store, monkeypatch):
    """上一次运行写下的已读清单，下一次运行（全新 session_id）必须看得到。

    若把读取按当前会话过滤，这里会拿到空集：排除机制静默失效，
    而打桩了记忆读取的单测照样全绿 —— 只有这条用例能抓住。
    """
    await store.remember_run(session_id="run-1", run_id="run-1", paper_ids=[_P1])
    monkeypatch.setattr(lt, "_default", store)

    # 模块级入口刻意不接受 session_id：没有这个参数，就没法把按会话过滤写回来
    assert await lt.safe_read_paper_ids() == {_P1}


async def test_eviction_caps_the_whole_table_not_each_session(scope):
    """淘汰必须全局计数：按会话计数的话每个新会话都从 0 开始，上限永远不触发。

    实测库里已积到 157 行而从未淘汰过一次 —— 架构 §9 警告的"越用越慢、越用越吵"
    正是这样发生的。
    """
    store = LongTermMemoryStore(scope=scope, embedder=MockEmbedder(dim=32), max_items=3)
    for i, sid in enumerate(("run-1", "run-2", "run-3", "run-4")):
        await store.remember(f"第 {i} 次运行记住的东西", session_id=sid)

    assert await store.count() == 3  # 四个会话各写一条，全局上限仍然生效


async def test_plan_recalls_memory_written_by_a_previous_run(monkeypatch, scope):
    """端到端：`plan` 节点在全新 session_id 下仍能召回上次运行写的记忆。"""
    from app.agents.nodes import planner
    from app.agents.schemas import PlanOutput
    from app.schemas.research import SubQuestion

    store = LongTermMemoryStore(scope=scope, embedder=MockEmbedder(dim=32), max_items=50)
    await store.remember_run(
        session_id="run-1", run_id="run-1", question="RAG 重排的代价", paper_ids=[_P1]
    )
    monkeypatch.setattr(lt, "_default", store)

    seen: dict[str, str] = {}

    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        seen["prompt"] = "\n".join(str(m.content) for m in messages)
        return (
            PlanOutput(
                rationale="按方法/代价两个角度拆解。",
                sub_questions=[
                    SubQuestion(
                        id=f"sq-{i}", text=f"子问题{i}", keywords=["reranking"], category=None
                    )
                    for i in range(1, 4)  # PlanOutput 有"至少 3 个子问题"的下限
                ],
            ),
            {},
        )

    monkeypatch.setattr(planner, "structured_invoke", _invoke)

    out = await planner.plan_node({"question": "RAG 重排有哪些新做法", "session_id": "run-2"})

    # 召回为空 == 长期记忆又退回"只写不读"，F10 等于没上线
    assert out["memory_context"]
    assert out["events"][0]["memory_count"] == len(out["memory_context"])
    assert "历史长期记忆" in seen["prompt"]


# =============================================================================
# 选篇侧：read 节点真的把已读论文让出候选位了吗
# =============================================================================
_P1, _P2, _P3 = "2401.00001", "2401.00002", "2401.00003"


def _chunk(chunk_id: int, text: str) -> dict:
    return {"chunk_id": chunk_id, "section": "abstract", "source": "abstract", "text": text}


def _retrieved() -> list[dict]:
    """三篇论文，重排分依次递减 —— `dedup_papers` 会按分数排序，P1 在最前。"""
    return [
        {"arxiv_id": _P1, "title": "Paper One", "score": 0.9, "chunks": [_chunk(101, "one")]},
        {"arxiv_id": _P2, "title": "Paper Two", "score": 0.8, "chunks": [_chunk(102, "two")]},
        {"arxiv_id": _P3, "title": "Paper Three", "score": 0.7, "chunks": [_chunk(103, "three")]},
    ]


def _state() -> dict:
    return {
        "question": "混合检索怎么做融合？",
        "session_id": "s1",
        "retrieved": _retrieved(),
        "stage": "corpus",  # 不是 STAGE_READ，否则命中幂等哨兵直接返回 {}
    }


@pytest.fixture(autouse=True)
def _fast_settings(monkeypatch):
    """精读数压到 2，让"谁被挤出去"可观察；同时强制 mock，避免测试里出现网络调用。"""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    monkeypatch.setenv("READ_TOP_K", "2")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def captured_prompt(monkeypatch):
    """截住送去抽取的 prompt —— 断言"哪几篇进了精读"最直接的证据就是它。"""
    box: dict = {}

    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        box["prompt"] = "\n".join(str(m.content) for m in messages)
        return ReadOutput(claims=[]), {}

    monkeypatch.setattr(reader_node, "structured_invoke", _invoke)
    return box


@pytest.fixture
def stub_read_ids(monkeypatch):
    def _set(ids: set[str]) -> None:
        async def _fake(session_id=None):  # noqa: ANN001
            return set(ids)

        monkeypatch.setattr(reader_node, "safe_read_paper_ids", _fake)

    return _set


async def test_already_read_paper_gives_up_its_slot(captured_prompt, stub_read_ids):
    """分数最高的 P1 已读过 → 让位给 P3，prompt 里不该再出现 P1。"""
    stub_read_ids({_P1})

    await reader_node.read_node(_state())

    prompt = captured_prompt["prompt"]
    assert _P1 not in prompt
    assert _P2 in prompt and _P3 in prompt


async def test_no_read_history_keeps_the_old_behaviour(captured_prompt, stub_read_ids):
    """没有已读记录时必须与改动前完全一致：按重排分取头部 P1、P2。

    这条是"新功能不许改变默认路径"的回归闸门。
    """
    stub_read_ids(set())

    await reader_node.read_node(_state())

    prompt = captured_prompt["prompt"]
    assert _P1 in prompt and _P2 in prompt
    assert _P3 not in prompt


async def test_read_paper_fills_back_when_unread_runs_out(captured_prompt, stub_read_ids, monkeypatch):
    """未读不够时已读论文补位，但排在未读之后（降级而非硬剔除）。"""
    monkeypatch.setenv("READ_TOP_K", "3")
    get_settings.cache_clear()
    stub_read_ids({_P1})

    await reader_node.read_node(_state())

    prompt = captured_prompt["prompt"]
    assert all(p in prompt for p in (_P1, _P2, _P3))
    # 渲染顺序 == 选篇顺序：未读的 P2/P3 在前，已读的 P1 补位在最后
    assert prompt.index(_P2) < prompt.index(_P1)
    assert prompt.index(_P3) < prompt.index(_P1)


async def test_retrieved_is_not_shrunk_by_the_exclusion(captured_prompt, stub_read_ids):
    """★ 排除只动选篇，绝不动 `retrieved`。

    `retrieved` 另有五个消费者（verify 的合法 chunk 全集、compare、write、
    finalize 的"本次覆盖 N 篇"、revise）。在选篇这一步删它，等于顺带改掉
    校验口径和报告里的数字 —— 那才是真正难查的回归。
    """
    stub_read_ids({_P1})
    state = _state()

    out = await reader_node.read_node(state)

    assert len(state["retrieved"]) == 3
    assert "retrieved" not in out  # 节点没有把裁剪后的候选写回 state


def test_prioritize_unread_is_a_stable_partition():
    """组内保持原有（重排后）顺序：不引入任何新的打分口径。"""
    papers = [{"arxiv_id": a, "score": s} for a, s in ((_P1, 0.9), (_P2, 0.8), (_P3, 0.7))]

    ordered = prioritize_unread(papers, {_P2})

    assert [p["arxiv_id"] for p in ordered] == [_P1, _P3, _P2]
    assert prioritize_unread(papers, set()) == papers  # 空集合 = 原样返回


def test_degraded_run_does_not_mark_the_whole_corpus_as_read():
    """★ 抽取失败（`readings` 为空）时，绝不能把全部检索命中记成"已读"。

    以前这里会退回 `retrieved`。在已读清单还没人消费时那只是不精确，
    但 `read` 节点现在会拿它排除候选 —— 一次降级运行就能把下一次的候选压光。
    """
    from app.agents.nodes.finalize import _read_paper_ids

    retrieved = [{"arxiv_id": _P1}, {"arxiv_id": _P2}, {"arxiv_id": _P3}]
    assert _read_paper_ids({"readings": [], "retrieved": retrieved}) == []
    assert _read_paper_ids({"retrieved": retrieved}) == []  # 没有 readings 键同样如实为空
    assert _read_paper_ids({"readings": [{"paper_id": _P2}], "retrieved": retrieved}) == [_P2]

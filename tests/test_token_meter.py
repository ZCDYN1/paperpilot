"""按节点细分 token 计量（架构 §12）的测试。

覆盖三类性质，缺一不可：
1. **分桶正确**：不同节点的调用分别计数，不会全糊在一个桶里；
2. **并发不串桶**：并行/交错调用时，各任务只算到自己头上（contextvars 的核心承诺）；
3. **端到端非空**：mock 模式跑完整张图后 `state["token_by_node"]` 真的有各节点的分列。

新增文件，不改动任何已有测试。
"""

from __future__ import annotations

import asyncio
from typing import Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.llm import get_llm
from app.observability.token_meter import (
    UNKNOWN_NODE,
    TokenMeter,
    get_token_meter,
    node_scope,
    reset_token_meter,
)


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    """切成 mock 三件套并清空账本。

    `Settings` 是 lru_cache 单例，改环境变量必须清缓存；账本是进程内单例，
    不清的话用例之间会互相污染（前一个用例的 plan 用量会算进后一个）。
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    reset_token_meter()
    yield
    reset_token_meter()
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 1) 分桶正确：显式 node_scope（contextvars 路径）
# ---------------------------------------------------------------------------
async def test_bucket_by_node_scope():
    with node_scope("plan"):
        await get_llm().ainvoke([HumanMessage(content="a" * 20)])
    with node_scope("write"):
        await get_llm().ainvoke([HumanMessage(content="b" * 20)])
        await get_llm().ainvoke([HumanMessage(content="c" * 20)])

    snap = get_token_meter().snapshot()
    assert snap["plan"]["calls"] == 1
    assert snap["write"]["calls"] == 2
    # 三个口径都要有值：只记 total 会让"按节点细分"失去意义
    assert snap["plan"]["prompt_tokens"] > 0
    assert snap["plan"]["completion_tokens"] > 0
    assert snap["plan"]["total_tokens"] == snap["plan"]["prompt_tokens"] + snap["plan"]["completion_tokens"]


# ---------------------------------------------------------------------------
# 2) 分桶正确：LangGraph 注入的 langgraph_node（metadata 路径，不需节点配合）
# ---------------------------------------------------------------------------
async def test_bucket_from_langgraph_node_metadata():
    from langgraph.graph import END, START, StateGraph

    class S(TypedDict, total=False):
        x: int

    async def node_plan(state: S) -> dict:
        await get_llm().ainvoke([HumanMessage(content="plan " * 5)])
        return {}

    async def node_write(state: S) -> dict:
        await get_llm().ainvoke([HumanMessage(content="write " * 5)])
        return {}

    g = StateGraph(S)
    g.add_node("plan", node_plan)
    g.add_node("write", node_write)
    g.add_edge(START, "plan")
    g.add_edge("plan", "write")
    g.add_edge("write", END)

    await g.compile().ainvoke({"x": 1}, {"configurable": {"thread_id": "t-meta"}})

    snap = get_token_meter().snapshot()
    # 两个节点各一次调用，且**没有**用 node_scope —— 完全靠 LangGraph metadata 归位
    assert set(snap) == {"plan", "write"}
    assert snap["plan"]["calls"] == 1
    assert snap["write"]["calls"] == 1


# ---------------------------------------------------------------------------
# 3) ★ 并发/交错调用不串桶（contextvars 的核心保证）
# ---------------------------------------------------------------------------
async def test_concurrent_calls_do_not_cross_buckets():
    async def worker(name: str, calls: int) -> None:
        # 每个任务在自己的 contextvars 副本里设定节点名
        with node_scope(name):
            for _ in range(calls):
                await get_llm().ainvoke([HumanMessage(content="x" * 12)])
                await asyncio.sleep(0)  # 强制交错，暴露"实例属性串桶"的实现

    await asyncio.gather(worker("read", 3), worker("verify", 5))

    snap = get_token_meter().snapshot()
    assert snap["read"]["calls"] == 3
    assert snap["verify"]["calls"] == 5
    assert "unknown" not in snap  # 每一次调用都应被正确归位
    # 两条流各自独立累加，没有被对方的用量污染（3 次 vs 5 次的 total 不会相等）
    assert snap["read"]["total_tokens"] != snap["verify"]["total_tokens"]


# ---------------------------------------------------------------------------
# 4) 用量解析：llm_output.token_usage 优先
# ---------------------------------------------------------------------------
def test_on_llm_end_extracts_usage_from_llm_output():
    meter = TokenMeter()
    meter.on_chat_model_start({}, [[]], run_id="r1", metadata={"langgraph_node": "plan"})
    response = LLMResult(
        generations=[],
        llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
    )
    meter.on_llm_end(response, run_id="r1")

    bucket = meter.snapshot()["plan"]
    assert bucket == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
        "calls": 1,
    }


# ---------------------------------------------------------------------------
# 5) 用量解析：退回 message.usage_metadata；且无 metadata 时用 contextvars 兜底
# ---------------------------------------------------------------------------
def test_on_llm_end_falls_back_to_usage_metadata():
    message = AIMessage(
        content="x",
        usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
    )
    response = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output=None)

    meter = TokenMeter()
    with node_scope("read"):  # metadata 缺席 → 走 contextvars
        meter.on_chat_model_start({}, [[]], run_id="r2", metadata=None)
        meter.on_llm_end(response, run_id="r2")

    assert meter.snapshot()["read"]["prompt_tokens"] == 5
    assert meter.snapshot()["read"]["completion_tokens"] == 2


def test_unknown_node_when_no_metadata_and_no_context():
    meter = TokenMeter()
    meter.on_chat_model_start({}, [[]], run_id="r3", metadata=None)
    meter.on_llm_end(
        LLMResult(generations=[], llm_output={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}}),
        run_id="r3",
    )
    # 拿不到节点名时归入 unknown，而不是丢弃这笔账（丢弃会让总量对不上）
    assert meter.snapshot()[UNKNOWN_NODE]["calls"] == 1


# ---------------------------------------------------------------------------
# 6) mock 的结构化输出路径也要被计量（它不经过 `_generate`，靠手动补记）
# ---------------------------------------------------------------------------
async def test_mock_structured_output_is_metered():
    class PlanLike(BaseModel):
        name: str = ""

    with node_scope("plan"):
        await get_llm().with_structured_output(PlanLike).ainvoke(
            [HumanMessage(content="研究问题：RAG 重排的代价")]
        )

    snap = get_token_meter().snapshot()
    assert snap["plan"]["calls"] == 1
    assert snap["plan"]["total_tokens"] > 0


def test_reset_clears_all_buckets():
    meter = TokenMeter()
    meter.record("plan", {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})
    assert meter.snapshot()
    meter.reset()
    assert meter.snapshot() == {}


# ---------------------------------------------------------------------------
# 7) ★ 端到端：mock 模式跑完整图，token_by_node 必须非空且含各节点
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_shared_retriever():
    """用例之间丢弃进程内共享检索器，避免跨用例用到别的索引。"""
    from app.agents.tools.hybrid_search import reset_shared_retriever

    yield
    reset_shared_retriever()


async def test_end_to_end_graph_populates_token_by_node(tmp_path, monkeypatch):
    from app.agents import graph as graph_mod
    from app.agents.nodes import finalize as finalize_mod
    from app.agents.tools import hybrid_search as hs
    from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever

    # 受控语料：既不让 ensure_corpus 去联网，也不污染真实的 DB/索引目录
    spec = [
        ("2501.00001", 1, "abstract", "We study reranking for retrieval with a cross encoder baseline."),
        ("2501.00001", 2, "method", "The reranker scores retrieval candidates and improves ranking."),
        ("2501.00002", 11, "abstract", "A survey of retrieval reranking costs and latency tradeoffs."),
        ("2501.00002", 12, "results", "Reranking improves recall at the cost of extra latency."),
        ("2501.00003", 21, "abstract", "Dense retrieval combined with reranking on benchmark datasets."),
        ("2501.00003", 22, "method", "Cross encoder reranking is expensive but effective."),
    ]
    records = [
        ChunkRecord(chunk_id=cid, arxiv_id=aid, section=sec, text=text, source="html", year=2025)
        for aid, cid, sec, text in spec
    ]
    titles = {aid: f"Paper {aid}" for aid, *_ in spec}

    retriever = HybridRetriever(index_dir=tmp_path / "faiss")
    await retriever.build_index(records, titles)
    monkeypatch.setattr(hs, "_shared", retriever)

    # 语料已被受控替换 → 用 stub 顶掉 ensure_corpus，避免读真实库/写真实索引
    async def fake_corpus(state: dict) -> dict:
        return {"corpus_ready": True, "corpus_stats": {"total_papers": len(titles)}, "stage": "corpus"}

    monkeypatch.setattr(graph_mod, "ensure_corpus_node", fake_corpus)

    # finalize 的落库/记忆是"副作用"，本用例只验证 token 计量 → 打桩掉，保持用例隔离
    async def noop_persist(*args: Any, **kwargs: Any) -> None:
        return None

    async def noop_memory(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(finalize_mod, "_persist", noop_persist)
    monkeypatch.setattr(finalize_mod, "_write_memory", noop_memory)

    reset_token_meter()
    async with graph_mod.open_checkpointer(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(saver)
        config = {"configurable": {"thread_id": "t-e2e"}}
        # 第一次跑到中断点 A（plan 之后），再用 ainvoke(None, config) 续跑到底
        await graph.ainvoke(
            {"session_id": "s-e2e", "run_id": "r-e2e", "question": "reranking retrieval 的代价是什么？"},
            config,
        )
        final = await graph.ainvoke(None, config)

    token_by_node = final.get("token_by_node") or {}
    assert token_by_node, "端到端跑完后 state['token_by_node'] 不能为空"
    # plan（结构化）与 write（自由文本）两条不同的调用路径都要被计上
    assert "plan" in token_by_node
    assert "write" in token_by_node
    for bucket in token_by_node.values():
        assert bucket["calls"] >= 1
        assert bucket["total_tokens"] > 0

    # finalize 也要把同一份分列挂到 token_usage 事件上（前端成本面板靠它）
    usage_events = [e for e in (final.get("events") or []) if e.get("type") == "token_usage"]
    assert usage_events and usage_events[-1]["by_node"] == token_by_node

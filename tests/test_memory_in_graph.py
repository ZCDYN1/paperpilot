"""长期记忆「真正接进图」的测试（架构 §9 / 产品定义 F10）。

要证明的不是"函数能跑"，而是**它确实被图调用了、且结果真的影响/来自 state**：
- plan：召回结果进了 `state["memory_context"]`，并被注入 prompt；
- plan：召回失败必须降级，不能拖垮整次研究；
- finalize：本次产生的记忆（方向 / 已读论文 / 偏好）与短期对话都落了库；
- finalize：把 token_meter 的分列结果写进 `state["token_by_node"]`。

新增文件，不改动任何已有测试。全部用 monkeypatch 打桩外部副作用，保持用例隔离、
不碰真实的 `data/sqlite/paperpilot.db`。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import get_settings
from app.observability.token_meter import get_token_meter, reset_token_meter


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    get_settings.cache_clear()
    reset_token_meter()
    yield
    reset_token_meter()
    get_settings.cache_clear()


def _plan_output(n: int = 3):
    from app.agents.schemas import PlanOutput
    from app.schemas.research import SubQuestion

    return PlanOutput(
        rationale="按方法/代价/评测三个角度拆解。",
        sub_questions=[
            SubQuestion(
                id=f"sq-{i}",
                text=f"子问题{i}：该方向的核心方法与代价是什么？",
                keywords=["reranking", "retrieval"],
                category=None,
            )
            for i in range(1, n + 1)
        ],
    )


# ---------------------------------------------------------------------------
# 1) ★ plan 召回长期记忆：写进 state，并注入 prompt
# ---------------------------------------------------------------------------
async def test_plan_recalls_memory_into_state_and_prompt(monkeypatch):
    from app.agents.nodes import planner
    from app.memory import long_term

    memories = ["已读论文 arxiv_id=2401.00001", "用户长期关注 RAG 重排"]

    async def fake_recall(query: str, top_k: int = 5, **kwargs: Any):
        # 顺带验证 planner 传下来的确实是当前问题、且带了 top_k
        assert "重排" in query
        assert top_k >= 1
        return memories

    captured: dict[str, str] = {}

    async def fake_structured_invoke(schema, messages, **kwargs):  # noqa: ANN001
        captured["prompt"] = "\n".join(str(m.content) for m in messages)
        return _plan_output(), {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}

    monkeypatch.setattr(long_term, "recall", fake_recall)
    monkeypatch.setattr(planner, "structured_invoke", fake_structured_invoke)

    out = await planner.plan_node({"question": "RAG 重排的代价", "session_id": "s1"})

    assert out["memory_context"] == memories
    assert "2401.00001" in captured["prompt"]  # 记忆确实进了 prompt
    assert "研究问题：RAG 重排的代价" in captured["prompt"]
    # 记忆在 `研究问题：` 之前（否则会污染 mock 的英文关键词抽取）
    assert captured["prompt"].index("2401.00001") < captured["prompt"].index("研究问题：")
    assert out["events"][0]["memory_count"] == len(memories)


# ---------------------------------------------------------------------------
# 2) ★ 召回失败降级：plan 不能挂
# ---------------------------------------------------------------------------
async def test_plan_degrades_when_recall_fails(monkeypatch):
    from app.agents.nodes import planner
    from app.memory import long_term

    async def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("vector store down")

    async def fake_structured_invoke(schema, messages, **kwargs):  # noqa: ANN001
        return _plan_output(), {}

    monkeypatch.setattr(long_term, "recall", boom)
    monkeypatch.setattr(planner, "structured_invoke", fake_structured_invoke)

    out = await planner.plan_node({"question": "RAG 重排的代价", "session_id": "s1"})

    assert out["memory_context"] == []          # 降级为空，而不是抛栈
    assert len(out["sub_questions"]) == 3        # 拆解照常完成
    assert out["stage"] == "planned"


async def test_plan_handles_missing_memory_module(monkeypatch):
    """记忆模块整体不可用（import 失败）时也要照常拆解。"""
    from app.agents.nodes import planner

    async def fake_structured_invoke(schema, messages, **kwargs):  # noqa: ANN001
        return _plan_output(), {}

    async def no_memory(*args: Any, **kwargs: Any) -> list[str]:
        return []

    monkeypatch.setattr(planner, "structured_invoke", fake_structured_invoke)
    monkeypatch.setattr(planner, "_recall_memory", no_memory)

    out = await planner.plan_node({"question": "RAG 重排", "session_id": "s1"})
    assert out["memory_context"] == []
    assert len(out["sub_questions"]) == 3


# ---------------------------------------------------------------------------
# 3) ★ finalize 写长期记忆（方向 / 已读论文 / 偏好）+ 短期对话
# ---------------------------------------------------------------------------
async def test_finalize_writes_long_and_short_term_memory(monkeypatch):
    from app.agents.nodes import finalize as finalize_mod
    from app.memory import long_term, short_term

    calls: dict[str, Any] = {"remember_run": None, "remember": None, "appended": []}

    async def fake_remember_run(**kwargs: Any):
        calls["remember_run"] = kwargs
        return [1, 2, 3]

    async def fake_remember(content: str, **kwargs: Any):
        calls["remember"] = {"content": content, **kwargs}
        return 9

    async def fake_append(session_id: str, role: str, content: str, **kwargs: Any):
        calls["appended"].append((session_id, role, content, kwargs.get("node_name")))

    async def noop_persist(*args: Any, **kwargs: Any):
        return None

    monkeypatch.setattr(long_term, "remember_run", fake_remember_run)
    monkeypatch.setattr(long_term, "remember", fake_remember)
    monkeypatch.setattr(short_term, "append_message", fake_append)
    monkeypatch.setattr(finalize_mod, "_persist", noop_persist)

    state = {
        "session_id": "s1",
        "run_id": "r1",
        "question": "RAG 重排的代价",
        "retrieved": [{"arxiv_id": "a1"}, {"arxiv_id": "a2"}, {"arxiv_id": "a3"}],
        # 只有 a1/a2 真的进了 Reader → 已读论文只应记这两篇
        "readings": [{"paper_id": "a1"}, {"paper_id": "a2"}],
        "comparison": [{"arxiv_id": "a1"}],  # 产出了对比表 → 应记一条输出偏好
        "draft": "本次研究结论：重排显著提升召回，但带来额外延迟。",
        "citations": [],
    }

    out = await finalize_mod.finalize_node(state)

    assert out["stage"] == "done"
    assert calls["remember_run"]["paper_ids"] == ["a1", "a2"]
    assert calls["remember_run"]["question"] == "RAG 重排的代价"
    assert calls["remember_run"]["summary"].startswith("本次研究结论")
    assert calls["remember"]["memory_type"] == long_term.TYPE_PREFERENCE
    # 短期记忆：一问一答两条
    assert [c[1] for c in calls["appended"]] == ["user", "assistant"]
    assert calls["appended"][0][2] == "RAG 重排的代价"
    assert calls["appended"][1][2].startswith("本次研究结论")


async def test_finalize_memory_failure_is_swallowed(monkeypatch):
    """记忆写入失败不能改变最终结论（收尾失败不毁掉整次研究）。"""
    from app.agents.nodes import finalize as finalize_mod
    from app.memory import long_term, short_term

    async def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("db locked")

    async def noop_persist(*args: Any, **kwargs: Any):
        return None

    monkeypatch.setattr(long_term, "remember_run", boom)
    monkeypatch.setattr(short_term, "append_message", boom)
    monkeypatch.setattr(finalize_mod, "_persist", noop_persist)

    out = await finalize_mod.finalize_node({"question": "q", "session_id": "s1", "draft": "d"})
    assert out["stage"] == "done"


# ---------------------------------------------------------------------------
# 4) ★ finalize 把 token_meter 的分列写进 state["token_by_node"]
# ---------------------------------------------------------------------------
async def test_finalize_exposes_token_by_node(monkeypatch):
    from app.agents.nodes import finalize as finalize_mod

    async def noop(*args: Any, **kwargs: Any):
        return None

    monkeypatch.setattr(finalize_mod, "_persist", noop)
    monkeypatch.setattr(finalize_mod, "_write_memory", noop)

    get_token_meter().record("plan", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    get_token_meter().record("write", {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28})

    out = await finalize_mod.finalize_node({"question": "q", "session_id": "s1"})

    by_node = out["token_by_node"]
    assert by_node["plan"]["total_tokens"] == 15
    assert by_node["write"]["total_tokens"] == 28
    # 同一份分列也要出现在 token_usage 事件里（前端成本面板消费它）
    event = next(e for e in out["events"] if e["type"] == "token_usage")
    assert event["by_node"] == by_node


async def test_finalize_persists_per_node_token_rows(monkeypatch, tmp_path):
    """落库时 `token_usage` 表按节点各落一行（+1 行 all 总量）——不再只有一条 all。

    用临时库 + 替换 `session_scope` 的方式隔离，不碰真实 DB。
    """
    from contextlib import asynccontextmanager

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.db.models  # noqa: F401 - 注册全部 ORM 表
    from app.agents.nodes import finalize as finalize_mod
    from app.db import session as session_mod
    from app.db.base import Base
    from app.db.models import TokenUsage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 't.db'}")
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

    async def _init_db() -> None:
        return None

    async def noop(*args: Any, **kwargs: Any) -> None:
        return None

    # `_persist` 在函数体内 import 这两个名字 → 打桩模块属性即可生效
    monkeypatch.setattr(session_mod, "session_scope", _scope)
    monkeypatch.setattr(session_mod, "init_db", _init_db)
    monkeypatch.setattr(finalize_mod, "_write_memory", noop)

    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    by_node = {"plan": usage, "write": {"prompt_tokens": 4, "completion_tokens": 2}}
    await finalize_mod._persist(  # noqa: SLF001 - 测试内直调内部入口
        {"run_id": "r-db", "session_id": "s-db", "question": "q", "retrieved": [], "citations": []},
        usage,
        by_node,
    )

    async with _scope() as session:
        names = {r.node_name for r in (await session.execute(select(TokenUsage))).scalars().all()}
    assert names == {"all", "plan", "write"}
    await engine.dispose()

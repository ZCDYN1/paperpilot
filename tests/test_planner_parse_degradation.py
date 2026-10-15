"""`plan` 节点面对"模型没产出合规结构化输出"时的行为。

背景（真实故障，不是假想敌）：`with_structured_output(..., include_raw=True)` 解析失败
**不抛异常**，只把 `parsed` 置成 None。节点里 `parsed.sub_questions` 于是抛
AttributeError，一路冒到 SSE 变成整次研究失败 —— 用户看到的是"研究挂了"，
而实际只是"这一轮的拆解没拿到，本可以退回单路检索"。
"""

from __future__ import annotations

from typing import Any

import pytest


async def _no_session(monkeypatch, planner):
    """隔离掉记忆/会话召回：本文件只关心拆解失败时的降级，不测上下文。"""
    from app.memory import long_term

    async def _empty(*args: Any, **kwargs: Any):
        return []

    monkeypatch.setattr(long_term, "recall", _empty)
    monkeypatch.setattr(planner, "_recall_session_history", _empty)


async def test_parsed_none_degrades_instead_of_crashing(monkeypatch):
    from app.agents.nodes import planner

    await _no_session(monkeypatch, planner)

    async def fake_invoke(schema, messages, **kwargs):  # noqa: ANN001
        return None, {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}

    monkeypatch.setattr(planner, "structured_invoke", fake_invoke)

    out = await planner.plan_node({"question": "RAG 重排降低幻觉的方法"})

    subs = out["sub_questions"]
    assert len(subs) == 1, "降级产物必须至少有一个子问题，否则扇出为 0、后面全是空报告"
    # 关键词只能来自问题本身：这里没有模型给的关键词，兜底要从中文里挑出英文词
    assert subs[0]["keywords"] == ["rag"], subs[0]["keywords"]
    assert subs[0]["text"] == "RAG 重排降低幻觉的方法"

    names = [e["type"] for e in out["events"]]
    assert names == ["plan_created", "error"], names
    err = out["events"][1]
    assert err["node"] == "plan" and err["code"] == "plan_degraded"
    assert err["recoverable"] is True
    assert out["plan_rationale"], "降级理由要写进 state，别只留在日志里"


async def test_llm_transport_error_is_not_swallowed(monkeypatch):
    """传输层失败（超时/熔断/连不上）**保持上抛**，不跟着解析失败一起降级。

    这条与上一条是一对：降级只针对"模型跑完了但输出不合规"。
    下游真挂了的时候，容错层要往流里送 `internal_error` / `llm_circuit_open`，
    症状必须还是"研究失败"，不能变成一份看起来正常的单路报告。
    """
    from app.agents.nodes import planner

    await _no_session(monkeypatch, planner)

    async def boom(schema, messages, **kwargs):  # noqa: ANN001
        raise RuntimeError("LLM 调用超时")

    monkeypatch.setattr(planner, "structured_invoke", boom)

    with pytest.raises(RuntimeError, match="超时"):
        await planner.plan_node({"question": "RAG 重排降低幻觉的方法"})


async def test_pure_chinese_question_still_gets_searchable_keywords(monkeypatch):
    """问题里一个英文词都没有时，兜底关键词不能是空的（BM25 拿空串等于不发检索）。"""
    from app.agents.nodes import planner

    await _no_session(monkeypatch, planner)

    async def fake_invoke(schema, messages, **kwargs):  # noqa: ANN001
        return None, {}

    monkeypatch.setattr(planner, "structured_invoke", fake_invoke)

    out = await planner.plan_node({"question": "大模型如何做检索增强"})

    assert out["sub_questions"][0]["keywords"] == ["retrieval"]
    assert [e["type"] for e in out["events"]] == ["plan_created", "error"]

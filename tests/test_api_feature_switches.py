"""一次真实请求带着 F2 / F3 / F14 / F17 四个开关跑完，并导出报告。

单测能证明"函数认得这些参数"，证明不了"从 HTTP 进来一路到节点还在"。
本项目已经吃过三次这个亏：`top_k` / `rerank` 在请求体里躺了很久，
`_initial_state` 却没把它们放进 state，开关于是纯装饰 —— 而且从接口响应到日志都看不出差别；
补上转发之后本文件仍然红，因为还有第三个坑在下面这条路上：

    API → state → `Send` 扇出 → retrieve

`Send` 的分支**只看得到 payload 里的那些键**，全局 state 不会自动带过去，
所以"请求级过滤条件在检索那一步凭空消失"（`app/agents/graph.py::_worker_payload` 就是为此存在）。
这个文件同时是那条隐式契约的回归钉子。

夹具是 `tests/api_support.py` 那一套（mock provider + tmp_path 临时库，`MIN_CORPUS=0`
让 `ensure_corpus` 不联网；索引为空，所以检索会返回 0 篇 —— 这不影响本文件要证的事：
retrieve 节点仍然会执行一次，工具入参照旧进 `tool_call` 事件）。
"""

from __future__ import annotations

import pytest

from tests.api_support import STREAM_URL, api_client, first_data, parse_sse


@pytest.fixture
async def client(monkeypatch, tmp_path):
    async with api_client(monkeypatch, tmp_path) as c:
        yield c


async def _run_to_completion(client, body: dict) -> tuple[str, str, str]:
    """stream → 拿到中断点 A 的 run_id → resume 跑完。

    返回 `(run_id, 首段 SSE, 恢复后的 SSE)`。**别把两段拼一起断言 `done`** ——
    第一段结尾的 `done` 是 `interrupted`（中断点 A 的正常产物），
    只有恢复之后那段的 `done` 才是 `ok`。
    """
    first = (await client.post(STREAM_URL, json=body)).text
    run_id = first_data(first, "run_started")["run_id"]
    tail = (await client.post(f"/api/v1/research/{run_id}/resume", json={})).text
    return run_id, first, tail


async def test_request_switches_reach_the_retrieve_tool(client):
    """`top_k` / `rerank` / `has_code` 必须出现在 retrieve 节点真正调工具用的入参里。"""
    _, first, tail = await _run_to_completion(
        client,
        {
            "question": "近两年 RAG 的重排方法有哪些代价",
            "top_k": 3,
            "rerank": False,
            "filters": {"has_code": True, "year_gte": 2024},
        },
    )

    tool_args = [
        data["args"]
        for _, data in parse_sse(f"{first}\n{tail}")
        if _ == "tool_call" and data.get("tool") == "hybrid_search" and "args" in data
    ]
    assert tool_args, "SSE 里没有 hybrid_search 的 tool_call 事件，开关无处可查"
    for args in tool_args:
        assert args["top_k"] == 3, "请求级 top_k 没落到工具入参（又变成装饰字段）"
        assert args["rerank"] is False, "请求级 rerank 开关没被转发"
        assert args["has_code"] is True, "F2 的 has_code 过滤没传到检索侧"
        assert args["year_gte"] == 2024


async def test_directed_compare_request_completes_without_network_fetch(client):
    """F14：点名论文时不抓新语料，但整条链路照常跑完并给出如实的空结果。"""
    run_id, _, tail = await _run_to_completion(
        client,
        {
            "question": "只对比这两篇重排方法",
            "arxiv_ids": ["2305.15334", "2401.00001"],
        },
    )

    assert first_data(tail, "done")["status"] == "ok"
    # 抓取被跳过（不是"抓了但没用"）：语料统计里的 fetched 必须是 0
    corpus = first_data(tail, "corpus")
    assert corpus["fetched"] == 0
    # 点名的 id 必须出现在检索的过滤条件里，否则"定向对比"只是请求体上的装饰
    limited = [
        data.get("extra_filters", {}).get("arxiv_ids")
        for _, data in parse_sse(tail)
        if _ == "tool_call" and data.get("tool") == "hybrid_search"
    ]
    assert limited and all(ids == ["2305.15334", "2401.00001"] for ids in limited)
    detail = (await client.get(f"/api/v1/research/{run_id}")).json()
    assert detail["status"] == "completed"


async def test_language_switch_and_export_round_trip(client):
    """F17 + F8：`language=en` 跑完，导出的报告仍是一份结构完整的 .md（正文语言由节点负责）。"""
    run_id, _, _ = await _run_to_completion(
        client, {"question": "reranking methods and their costs", "language": "en"}
    )

    resp = await client.get(f"/api/v1/research/{run_id}/export.md")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    body = resp.text
    assert "# 文献研究报告：reranking methods and their costs" in body
    # 固定声明与 run_id 是导出的两条硬内容，不依赖模型是否配合
    assert "> 本文由 AI 生成，引用请以原文为准。" in body
    assert run_id in body


async def test_unknown_request_field_is_still_rejected(client):
    """加字段不许放松入口校验：`extra="forbid"` 是挡住"前端拼错参数名以为生效了"的那道闸。"""
    resp = await client.post(STREAM_URL, json={"question": "q", "has_code": True})
    assert resp.status_code == 422

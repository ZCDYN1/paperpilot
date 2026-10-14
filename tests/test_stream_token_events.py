"""`token` 事件的**真实发送点**（架构 §8「LLM 输出增量 · 前端打字机」）。

## 这个文件存在的理由
`SSEEmitter.token` 与枚举里的 `token` 一直在，但**全仓没有一处调用**：
真实模式下一条 `token` 都收不到，前端的打字机效果只有 mock replay 里活着。
所以这里测的不是"事件类型存不存在"，而是三件会各自失败的事：

1. **增量是真的**：来自 `llm.astream` 的生成过程，不是把成稿切片后逐段"演"一遍；
2. **实时**：增量在 write 节点**执行期间**就到用户面前（走 `state["events"]` 会等节点跑完）；
3. **与重试共存**：一次尝试已经吐过字又失败时，同一段内容**不会**被第二次吐出去。

## 全部离线
LLM 用 `mock` provider（真实多图执行）或自造 ChatModel（单元用例），
不发任何网络请求；容错退避归零，避免用例真的睡够指数退避。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.agents.nodes import _common, writer
from app.core.config import get_settings
from app.core.resilience import TransientError, reset_breakers
from app.observability.events import (
    SCHEMA_VERSION,
    SSEEmitter,
    _split_stream_chunk,
)
from tests.api_support import STREAM_URL, api_client, first_data, parse_sse

NO_WAIT_ENV = {
    "LLM_PROVIDER": "mock",
    "EMBEDDING_PROVIDER": "mock",
    "RESILIENCE_ENABLED": "true",
    "RESILIENCE_ATTEMPTS": "2",
    "RESILIENCE_BACKOFF": "0",
    "RESILIENCE_MAX_BACKOFF": "0",
    "RESILIENCE_JITTER": "0",
    "RESILIENCE_TIMEOUT": "30",
    "BREAKER_FAILURE_THRESHOLD": "5",
}


@pytest.fixture(autouse=True)
def _offline_settings(monkeypatch):
    for k, v in NO_WAIT_ENV.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    reset_breakers()
    yield
    reset_breakers()
    get_settings.cache_clear()


class _StreamingChatModel:
    """按脚本产出增量的假 ChatModel。

    脚本 = `[(增量列表, 结束时抛的异常或 None), …]`，一条对应**一次尝试**。

    两个刻意之处：
    - 增量用真的 `AIMessageChunk`：`_text_call` 靠 `chunk + chunk` 把 `usage_metadata`
      并进同一条消息，换成自造对象就等于"用量记账"只是测试自己断言出来的；
    - 同时提供 `ainvoke`：这样"没给 on_delta 时不许偷偷走流式"是可断言的。
    """

    def __init__(self, script: list[tuple[list[str], BaseException | None]]) -> None:
        self.script = script
        self.stream_calls = 0
        self.ainvoke_calls = 0

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.ainvoke_calls += 1
        return AIMessage(
            content="整段返回",
            usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        )

    async def astream(self, messages: Any):
        attempt = self.stream_calls
        self.stream_calls += 1
        deltas, exc = self.script[min(attempt, len(self.script) - 1)]
        for delta in deltas:
            yield AIMessageChunk(
                content=delta,
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            )
        if exc is not None:
            raise exc


@pytest.fixture
def sink_events():
    """一个"把 custom 通道上的 payload 全留下来"的假图 writer + 对应的 `TokenSink`。"""
    captured: list[dict[str, Any]] = []
    return captured, _common.TokenSink("write", captured.append)


# ---------------------------------------------------------------------------
# 1) 增量提取：provider 的三种 content 形态都要能拿到纯文本
# ---------------------------------------------------------------------------
def test_chunk_text_never_leaks_python_repr():
    """`[{'type': 'text', 'text': '你好'}]` 必须被读成 `你好`。

    直接用 `str(content)` 的写法会把这串 Python 字面量原样吐进用户的报告预览 ——
    真实 provider（langchain-core 1.x 的内容块表示）迟早会走到这条分支。
    """
    assert _common.chunk_text(AIMessageChunk(content="abc")) == "abc"
    assert _common.chunk_text(AIMessageChunk(content=[{"type": "text", "text": "你好"}])) == "你好"
    assert _common.chunk_text(AIMessage(content="整段")) == "整段"
    assert _common.chunk_text(None) == ""
    assert _common.chunk_text(object()) == ""  # 没有 content 的对象不该抛错


# ---------------------------------------------------------------------------
# 2) ★ 流式路径：每个真实增量都按序送出去，返回值与用量口径不变
# ---------------------------------------------------------------------------
async def test_text_invoke_streams_each_generation_delta(monkeypatch, sink_events):
    captured, sink = sink_events
    model = _StreamingChatModel([(["综述", "第一", "句"], None)])
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: model)

    text, usage = await _common.text_invoke(
        [HumanMessage(content="写综述")], on_delta=sink.feed, on_abort=sink.abort
    )

    assert text == "综述第一句", "返回值必须仍是完整正文（流式只是顺带把增量送出去）"
    assert [p["type"] for p in captured] == ["token", "token", "token"]
    assert "".join(p["delta"] for p in captured) == text
    assert all(p["node"] == "write" for p in captured)
    # usage 从归并后的 chunk 上取：流式不许把 token 记账弄丢（产出物 E 的成本口径靠它）
    assert usage["completion_tokens"] == 3
    assert sink.chunks_sent == 3 and not sink.closed


async def test_text_invoke_without_sink_stays_on_ainvoke(monkeypatch, sink_events):
    """没给 `on_delta` 时**一行都不许多走**：其余节点与 followup 的行为逐字不变。"""
    captured, _ = sink_events
    model = _StreamingChatModel([(["不该出现"], None)])
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: model)

    text, usage = await _common.text_invoke([HumanMessage(content="写综述")])

    assert text == "整段返回"
    assert model.ainvoke_calls == 1 and model.stream_calls == 0
    assert captured == []
    assert usage["total_tokens"] == 5


# ---------------------------------------------------------------------------
# 3) ★ 流式 × 重试：已经吐过字的那次失败不许让内容重复第二遍
# ---------------------------------------------------------------------------
async def test_mid_stream_failure_marks_stream_discarded_and_stops_restreaming(monkeypatch, sink_events):
    captured, sink = sink_events
    model = _StreamingChatModel(
        [
            (["半", "截"], TransientError("connection reset")),  # 第一次：吐了两个字才挂
            (["正", "文", "完整"], None),  # 重试：又生成一遍
        ]
    )
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: model)

    text, _ = await _common.text_invoke(
        [HumanMessage(content="写综述")], on_delta=sink.feed, on_abort=sink.abort
    )

    assert model.stream_calls == 2, "重试语义没变：该重试的还是一次不少地重试"
    assert text == "正文完整", "正文必须是成功那次的完整内容"
    # ★ 核心断言：只流出了失败那次的 2 段，重试那次的 3 段**没有**再往外吐
    assert [p["delta"] for p in captured if p["type"] == "token"] == ["半", "截"]
    assert sink.chars_sent == 2 and sink.closed is True

    notices = [p for p in captured if p["type"] == "error"]
    assert len(notices) == 1, "作废必须明确告知前端，而不是悄悄停在半截"
    assert notices[0]["code"] == _common.STREAM_ABORT_CODE
    assert notices[0]["recoverable"] is True, "内容是可恢复的：最终报告随后就到"
    assert notices[0]["node"] == "write"


async def test_failure_before_first_delta_keeps_stream_live_for_retry(monkeypatch, sink_events):
    """一个字都没吐过就失败 → 不作废、不关流，重试那次照样能实时流。

    这是更常见的一类抖动（连接都没建起来就 5xx），
    如果这里也关流，打字机效果会在"下游稍抖一下"时整体失效。
    """
    captured, sink = sink_events
    model = _StreamingChatModel([([], TransientError("provider down")), (["重", "试"], None)])
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: model)

    text, _ = await _common.text_invoke(
        [HumanMessage(content="写综述")], on_delta=sink.feed, on_abort=sink.abort
    )

    assert text == "重试"
    assert sink.closed is False
    assert [p["type"] for p in captured] == ["token", "token"]
    assert not [p for p in captured if p["type"] == "error"]


async def test_total_timeout_during_stream_also_announces_discard(monkeypatch, sink_events):
    """`RESILIENCE_TIMEOUT` 掐掉正在生成的流时，同样要给作废通知。

    超时预算包在重试外面，取消点恰好落在 `async for` 中间 ——
    没有这条通知的话，前端只会看到一个停在半截的预览 + 一句含糊的失败。
    """
    captured, sink = sink_events

    class _SlowModel:
        async def astream(self, messages: Any):
            yield AIMessageChunk(content="前半", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
            await asyncio.sleep(0.5)
            yield AIMessageChunk(content="不会到", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})

    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _SlowModel())
    monkeypatch.setenv("RESILIENCE_TIMEOUT", "0.05")
    monkeypatch.setenv("RESILIENCE_ATTEMPTS", "1")
    get_settings.cache_clear()

    with pytest.raises(asyncio.TimeoutError):
        await _common.text_invoke([HumanMessage(content="写综述")], on_delta=sink.feed, on_abort=sink.abort)

    assert [p["type"] for p in captured] == ["token", "error"]
    assert captured[1]["code"] == _common.STREAM_ABORT_CODE


# ---------------------------------------------------------------------------
# 4) 没有流的时候不许出事（单测直调节点 / 脚本跑 ainvoke）
# ---------------------------------------------------------------------------
async def test_make_token_sink_is_silent_outside_a_graph_stream():
    sink = _common.make_token_sink("write")
    assert sink.streaming is False, "拿不到 custom 通道时就是个 no-op，不该抛错"
    sink.feed("增量")
    sink.abort(TransientError("x"))
    assert sink.chunks_sent == 0


def test_sink_push_failure_never_breaks_generation():
    """通道写坏了（例如图已被取消）也不能把一次正在进行的生成带崩。"""

    def _boom(payload: dict[str, Any]) -> None:
        raise RuntimeError("channel closed")

    sink = _common.TokenSink("write", _boom)
    sink.feed("增量")  # 不抛
    sink.abort(TransientError("x"))  # 已吐过字 → 走作废通知，同样不许抛
    assert sink.chunks_sent == 1


# ---------------------------------------------------------------------------
# 5) 协议层：custom 通道上的 payload → 与便捷方法**逐字节同形**的帧
# ---------------------------------------------------------------------------
def test_live_event_matches_token_convenience_method_frame():
    """字段形状只有一份定义：实时路径与 `SSEEmitter.token` 必须产出同一帧。

    前端只解析 `data.delta`（`web/src/hooks/useResearchStream.ts`），
    两边各写一遍就是将来漂移的起点，所以这里直接逐字比字符串。
    """
    emitter = SSEEmitter("run-shape")
    via_method = emitter.token("write", "增量").to_sse()
    emitter2 = SSEEmitter("run-shape")
    via_channel = emitter2.live_event({"type": "token", "node": "write", "delta": "增量"}).to_sse()

    assert via_channel == via_method == f"event: token\ndata: {_token_json('增量')}\n\n"
    # 便捷方法会把事件留在 `_buffer`，实时路径取走即清（一条报告上千个增量不该都攒着）
    assert emitter2.drain() == []


def _token_json(delta: str) -> str:
    import json

    return json.dumps({"schema_version": SCHEMA_VERSION, "node": "write", "delta": delta}, ensure_ascii=False)


def test_live_event_drops_anything_outside_the_contract():
    """未知类型一律丢弃：`custom` 是节点直接往对外协议上写的口子，不能放行后门。"""
    emitter = SSEEmitter("run-drop")
    assert emitter.live_event({"type": "circuit_open", "delta": "x"}) is None
    assert emitter.live_event({"delta": "没有 type 也没有 node"}) is None
    assert emitter.live_event("不是 dict") is None
    assert emitter.drain() == []

    # 但合法的类型要照 `error` 的契约构造出字段（前端按 code 分支）
    ev = emitter.live_event(
        {"type": "error", "node": "write", "code": "token_stream_aborted", "message": "作废", "recoverable": True}
    )
    assert ev is not None and ev.type == "error"
    assert ev.data == {"node": "write", "code": "token_stream_aborted", "message": "作废", "recoverable": True}


def test_split_stream_chunk_accepts_both_shapes():
    """列表模式给元组、单字符串模式（以及假图）给裸 dict —— 两种都得收。

    只按元组解包的话，退化症状是整条 SSE 流在第一个 chunk 上抛 ValueError，
    一次已经跑成功的运行被一个协议适配函数变成 500。
    """
    assert _split_stream_chunk(("custom", {"type": "token"})) == ("custom", {"type": "token"})
    bare = {"plan": {"events": []}}
    assert _split_stream_chunk(bare) == ("updates", bare)


async def test_stream_yields_custom_frames_before_the_node_chunk():
    """实时落地：`custom` 帧必须在**该节点的 updates chunk 之前**出现。

    这是"真流式"与"成稿切片"唯一能在协议层区分开的性质 ——
    切片做法只能把增量塞进 `state["events"]`，而那里要等节点跑完才随 chunk 落地。
    """

    class _FakeGraph:
        async def astream(self, inputs, config, *, stream_mode=None):  # noqa: ANN001
            assert stream_mode == ["updates", "custom"], "没同时开两种模式的话增量根本到不了"
            yield ("custom", {"type": "token", "node": "write", "delta": "第一"})
            yield ("updates", {"write": {"events": [{"type": "final_report", "markdown": "第一段第二段", "citations": []}]}})

    emitter = SSEEmitter("run-live")
    frames = [f async for f in emitter.stream(_FakeGraph(), {"question": "q"}, {"configurable": {"thread_id": "t"}})]
    parsed = parse_sse("".join(frames))
    types = [t for t, _ in parsed]

    assert types.index("token") < types.index("node_started") < types.index("final_report")
    assert "token" in types and "done" in types
    assert [d for t, d in parsed if t == "token"][0]["delta"] == "第一"


async def test_stream_still_carries_out_of_band_and_state_events():
    """改 stream_mode 不许动既有链路：带外事件与 state 事件照常落地。"""

    class _FakeGraph:
        async def astream(self, inputs, config, *, stream_mode=None):  # noqa: ANN001
            from app.observability.events import emit_out_of_band

            assert emit_out_of_band("error", node="resilience", code="circuit_open", message="下游挂了")
            yield ("custom", {"type": "token", "node": "write", "delta": "x"})
            yield {"plan": {"events": [{"type": "plan_created", "count": 3}]}}  # 裸 dict 形状

    emitter = SSEEmitter("run-mixed")
    body = "".join([f async for f in emitter.stream(_FakeGraph(), {"question": "q"}, {"configurable": {"thread_id": "t"}})])
    assert "event: token" in body
    assert "circuit_open" in body
    assert "event: plan_created" in body
    assert emitter.last_status == "ok"


# ---------------------------------------------------------------------------
# 6) write 节点：流的是**生成正文**，程序追加的那几节绝不许混进流里
# ---------------------------------------------------------------------------
async def test_write_node_streams_generated_body_only(monkeypatch):
    streamed: list[dict[str, Any]] = []
    monkeypatch.setattr(
        writer, "make_token_sink", lambda node: _common.TokenSink(node, streamed.append)
    )

    async def _fake_invoke(messages, *args, **kwargs):  # noqa: ANN001
        # 模拟 provider 逐段生成（真实增量由 on_delta 送出，不是事后切片）
        for delta in ["模型写的", "正文。[1]"]:
            kwargs["on_delta"](delta)
        return "模型写的正文。[1]", {"total_tokens": 9}

    monkeypatch.setattr(writer, "text_invoke", _fake_invoke)

    state = {
        "question": "重排技术对比",
        "retrieved": [],
        "citations": [
            {"paper_id": "2999.00001", "chunk_id": 1, "claim": "结论乙", "verified": False, "source": "html"}
        ],
        "verification": {"passed": False, "unsupported": ["结论乙"]},
        "corpus_ready": True,
    }
    out = await writer.write_node(state)

    assert [p["delta"] for p in streamed] == ["模型写的", "正文。[1]"]
    assert all(p["node"] == "write" and p["type"] == "token" for p in streamed)
    # 程序追加的部分进了成稿，但**不在流里** —— 这正是"不许把报告切片假装流"的执行点
    markdown = out["draft"]
    assert "## ⚠️ 未找到原文支撑的结论" in markdown
    assert "本文由 AI 生成" in markdown
    assert "⚠️" not in "".join(p["delta"] for p in streamed)
    assert "".join(p["delta"] for p in streamed) in markdown


async def test_write_node_without_a_stream_still_produces_the_report(monkeypatch):
    """没有流（单测直调节点）时，报告照常产出 —— 流式只是显示层，不参与正确性。"""

    async def _fake_invoke(messages, *args, **kwargs):  # noqa: ANN001
        kwargs["on_delta"]("正文")
        return "正文", {}

    monkeypatch.setattr(writer, "text_invoke", _fake_invoke)
    out = await writer.write_node({"question": "q", "retrieved": [], "citations": [], "corpus_ready": True})
    assert "正文" in out["draft"]


# ---------------------------------------------------------------------------
# 7) ★ 端到端：真实图 + mock provider，用户收得到 token
# ---------------------------------------------------------------------------
async def test_real_graph_stream_delivers_token_events_to_the_wire(monkeypatch, tmp_path):
    """把"前端打字机在真实模式下是死的"这个缺陷直接钉成回归用例。

    三段断言各司其职：
    - 收得到 `token`，且字段就是前端已在消费的 `{node, delta}`（没有新键要前端改）；
    - 增量之和是 `final_report` 的**一部分**（不是整份报告被切片）；
    - `token` 早于 write 的 `node_started` —— 后者来自节点的 updates chunk，
      只有节点跑完才有，所以这条顺序就是"实时"与"事后切片"的分界线。
    """
    async with api_client(monkeypatch, tmp_path) as client:
        body = (await client.post(STREAM_URL, json={"question": "hybrid retrieval rerank tradeoffs"})).text
        run_id = first_data(body, "run_started")["run_id"]
        assert "token" not in [t for t, _ in parse_sse(body)], "中断前的那半条流没有正文可流"

        resumed = (await client.post(f"/api/v1/research/{run_id}/resume", json={})).text
        events = parse_sse(resumed)
        types = [t for t, _ in events]

        assert "token" in types, "真实模式下必须能收到 token 事件（此前全仓零发送点）"
        tokens = [d for t, d in events if t == "token"]
        assert tokens and all(d["node"] == "write" for d in tokens)
        assert all(d["delta"] for d in tokens), "delta 必须是真文本，不是空串凑数"
        assert {k for d in tokens for k in d} == {"schema_version", "node", "delta"}

        streamed = "".join(d["delta"] for d in tokens)
        markdown = first_data(resumed, "final_report")["markdown"]
        assert streamed in markdown and streamed != markdown, "流里只该有生成的正文，不该是整份报告"

        token_idx = types.index("token")
        write_start_idx = next(
            i for i, (t, d) in enumerate(events) if t == "node_started" and d.get("node") == "write"
        )
        assert token_idx < write_start_idx < types.index("final_report")

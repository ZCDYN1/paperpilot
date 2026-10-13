"""容错层「真的接进主流程」的测试（交付目标 A）。

`tests/test_resilience.py` 测的是**机制本身**（重试/超时/熔断三个零件各自的行为、
以及 `retry → timeout → circuit breaker` 的顺序）。那个文件当时绿着，
但 `resilient()` 在 `app/` 里**零调用** —— 机制正确、却没接上任何东西，
README 的"重试 / 超时 / 熔断"仍然是空头支票。

所以本文件只回答一个问题：**从节点走公共入口调 LLM 时，这三层保护是否真的在链路上生效？**
覆盖四个接线点：
1. `app/agents/nodes/_common.py` 的 `structured_invoke` / `text_invoke`（全部 6 个 LLM 节点的公共入口）；
2. `app/ingest/arxiv_client.py` 的对外 HTTP 出口（重试在里、熔断在外）；
3. 熔断状态迁移 → 结构化日志 + SSE 带外事件（`app/observability/resilience_events.py`）；
4. `RESILIENCE_ENABLED=false` 的旁路（"宁可旁路也不许把跑通链路堵死"）。

全部离线：LLM 用 mock provider，HTTP 用假 `_retry_loop`，**不发起任何网络请求**。
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog

from app.agents.nodes import _common
from app.agents.schemas import PlanOutput
from app.core.config import get_settings
from app.core.resilience import (
    BREAKER_HTTP,
    BREAKER_LLM,
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitOpenError,
    TransientError,
    breaker_snapshot,
    get_breaker,
    is_retryable,
    reset_breakers,
)
from app.observability.events import SSEEmitter, bind_emitter, emit_out_of_band, unbind_emitter

# 退避全部归零：熔断用例要连打好几次失败，不能真的睡够指数退避
NO_WAIT_ENV = {
    "LLM_PROVIDER": "mock",
    "EMBEDDING_PROVIDER": "mock",
    "RESILIENCE_ENABLED": "true",
    "RESILIENCE_ATTEMPTS": "2",
    "RESILIENCE_BACKOFF": "0",
    "RESILIENCE_MAX_BACKOFF": "0",
    "RESILIENCE_JITTER": "0",
    "RESILIENCE_TIMEOUT": "30",
    "BREAKER_FAILURE_THRESHOLD": "3",
    "BREAKER_RECOVERY_TIMEOUT": "60",
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


class _FailingChatModel:
    """假 ChatModel：每次调用都记一次数然后抛"可重试"错误。

    同时覆盖两条调用形态：
    - `with_structured_output(schema, include_raw=True).ainvoke(messages)`
    - `ainvoke(messages)`（text_invoke 走这条）
    """

    def __init__(self, counter: dict[str, int], exc: BaseException | None = None) -> None:
        self.counter = counter
        self.exc = exc or TransientError("provider down")

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FailingChatModel:
        return self

    async def ainvoke(self, messages: Any) -> Any:
        self.counter["n"] += 1
        raise self.exc


# ---------------------------------------------------------------------------
# 1) ★ LLM 公共入口上：连续失败会打开熔断，打开后不再触碰下游
# ---------------------------------------------------------------------------
async def test_structured_invoke_is_protected_by_breaker(monkeypatch):
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    threshold = get_settings().breaker_failure_threshold
    # 前 threshold 次：每次都真的打到下游（内部还会重试 attempts 次），并抛出原始异常
    for _ in range(threshold):
        with pytest.raises(TransientError):
            await _common.structured_invoke(PlanOutput, [HumanMessage(content="研究问题：RAG 重排")])

    assert get_breaker(BREAKER_LLM).state == STATE_OPEN

    # 打开之后：快速失败，**不再触碰下游**（这是熔断的全部意义）
    before = calls["n"]
    with pytest.raises(CircuitOpenError):
        await _common.structured_invoke(PlanOutput, [HumanMessage(content="研究问题：RAG 重排")])
    assert calls["n"] == before, "熔断打开后不该继续打下游"

    snapshot = breaker_snapshot()[BREAKER_LLM]
    assert snapshot["state"] == STATE_OPEN
    assert snapshot["short_circuited"] >= 1


async def test_breaker_counts_one_failure_per_call_even_with_retries(monkeypatch):
    """★ 接线后的顺序不变式：内部重试 N 次，断路器只记 1 次失败。

    这是"熔断会不会误开"的关键。若接线时把断路器放到重试**里面**
    （比如给 `_structured_call` 外面再套一层 breaker），这里 failure_count
    会变成 threshold×attempts 而不再是 1 —— 用例会立刻红。
    """
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    with pytest.raises(TransientError):
        await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])

    attempts = get_settings().resilience_attempts
    assert calls["n"] == attempts, f"内部应重试满 {attempts} 次"
    assert get_breaker(BREAKER_LLM).failure_count == 1, "一次调用只能记 1 次失败"
    assert get_breaker(BREAKER_LLM).state == STATE_CLOSED, "远未达阈值，不该打开"


async def test_text_invoke_is_protected_too(monkeypatch):
    """write 节点走的是 `text_invoke`，它必须和 `structured_invoke` 同一条保护链路。

    只包一个入口 = 一半的 LLM 调用没被保护，而 README 的话依然"看起来没错"。
    """
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    threshold = get_settings().breaker_failure_threshold
    for _ in range(threshold):
        with pytest.raises(TransientError):
            await _common.text_invoke([HumanMessage(content="写一段综述")])
    assert get_breaker(BREAKER_LLM).state == STATE_OPEN
    with pytest.raises(CircuitOpenError):
        await _common.text_invoke([HumanMessage(content="写一段综述")])


# ---------------------------------------------------------------------------
# 2) ★ mock / 离线：链路必须照旧跑通，且熔断计数为零
# ---------------------------------------------------------------------------
async def test_mock_provider_unaffected_by_resilience():
    """LLM_PROVIDER=mock 时，公共入口经过三层保护后**仍然正常返回结构化结果**。

    这条是 README "mock 下全链路可跑"承诺的护栏：
    如果接线把 mock 路径搞挂了（比如给 mock 加了个会超时的小预算），红的是这里。
    """
    from langchain_core.messages import HumanMessage

    parsed, usage = await _common.structured_invoke(
        PlanOutput, [HumanMessage(content="研究问题：hybrid retrieval bm25 reranker fusion")]
    )
    assert parsed is not None
    assert len(parsed.sub_questions) == 3
    assert usage["total_tokens"] > 0

    text, text_usage = await _common.text_invoke([HumanMessage(content="写一段综述")])
    assert text and text_usage["total_tokens"] > 0

    breaker = get_breaker(BREAKER_LLM)
    assert breaker.state == STATE_CLOSED
    assert breaker.failure_count == 0
    assert breaker.short_circuited == 0, "mock 永不失败 → 一次都不该被拦"


# ---------------------------------------------------------------------------
# 3) ★ 迁移要"可观测"：结构化日志 + SSE 带外事件
# ---------------------------------------------------------------------------
async def test_state_transitions_emit_structured_logs(monkeypatch):
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    with structlog.testing.capture_logs() as logs:
        threshold = get_settings().breaker_failure_threshold
        for _ in range(threshold):
            with pytest.raises(TransientError):
                await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])

    changes = [e for e in logs if e["event"] == "resilience.state_changed"]
    assert changes, "熔断打开必须留下结构化日志"
    last = changes[-1]
    assert last["to_state"] == STATE_OPEN
    assert last["breaker"] == BREAKER_LLM
    assert last["failures"] == threshold


async def test_circuit_open_emits_sse_event_to_running_stream(monkeypatch):
    """在 SSE 流的上下文里，熔断打开必须变成一条用户看得见的 `error` 事件。

    验证的是"事件真的到了这条流"：`stream()` 绑定了发射器（contextvar），
    容错层通过带外队列投喂，最后由 `drain_out_of_band()` 取出。
    """
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    emitter = SSEEmitter("run-observe-1")
    token = bind_emitter(emitter)
    try:
        threshold = get_settings().breaker_failure_threshold
        for _ in range(threshold):
            with pytest.raises(TransientError):
                await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])
        with pytest.raises(CircuitOpenError):
            await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])
    finally:
        unbind_emitter(token)

    events = emitter.drain_out_of_band()
    types = [e.type for e in events]
    assert "error" in types, "熔断迁移必须变成前端能渲染的 error 事件"

    payload = next(e.data for e in events if e.type == "error")
    assert payload["node"] == "resilience"
    assert payload["code"] in {"circuit_open", "llm_circuit_open"}
    assert payload["recoverable"] is True, "熔断是可恢复故障，前端要提示「稍后再试」"

    # 侧信道事件同样受"15 种类型"的契约约束：非法类型直接报错
    with pytest.raises(ValueError):
        emitter.emit_out_of_band("circuit_open")


def test_out_of_band_event_without_stream_is_harmless():
    """不在任何流上下文里（脚本 / 单测直接调函数）：只降级为日志，绝不抛异常。"""
    assert emit_out_of_band("error", node="resilience", code="circuit_open", message="x") is False


async def test_stream_actually_carries_out_of_band_events_to_the_wire():
    """★ 端到端的一根线：`stream()` 绑定发射器 → "节点内部"投喂的事件必须真的变成 SSE 帧。

    上一用例只证明事件进了队列；这里证明**用户收得到**。
    少了这一步的话，"熔断发 SSE 事件"会停在"队列里躺着没人取"这种最难查的状态
    （日志有、测试绿、前端什么都没有）。
    """

    class _FakeGraph:
        async def astream(self, inputs, config, *, stream_mode=None):  # noqa: ANN001
            # 模拟节点执行期间的容错层投喂：同一个 asyncio 任务 → 能读到 stream() 绑定的 contextvar
            assert emit_out_of_band(
                "error", node="resilience", code="circuit_open", message="下游被熔断", recoverable=True
            )
            yield {"plan": {"events": [{"type": "plan_created", "count": 3}]}}

    emitter = SSEEmitter("run-wire-1")
    frames = [f async for f in emitter.stream(_FakeGraph(), {"question": "q"}, {"configurable": {"thread_id": "t"}})]

    joined = "".join(frames)
    assert "event: error" in joined, "带外事件必须落到 SSE 帧里"
    assert "circuit_open" in joined
    assert "event: plan_created" in joined, "常规 state 事件不受影响"
    assert emitter.last_status == "ok"


# ---------------------------------------------------------------------------
# 4) ★ 旁路开关：RESILIENCE_ENABLED=false 时必须完全不拦
# ---------------------------------------------------------------------------
async def test_resilience_disabled_bypasses_entire_chain(monkeypatch):
    from langchain_core.messages import HumanMessage

    monkeypatch.setenv("RESILIENCE_ENABLED", "false")
    get_settings.cache_clear()

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    with pytest.raises(TransientError):
        await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])

    assert calls["n"] == 1, "旁路时必须一次透传：不重试、不计熔断"
    assert BREAKER_LLM not in breaker_snapshot(), "旁路时连熔断器都不该被创建"


# ---------------------------------------------------------------------------
# 5) ★ 对外 HTTP 出口：重试在里、熔断在外（arXiv 客户端）
# ---------------------------------------------------------------------------
async def test_arxiv_http_egress_is_wrapped_by_breaker(monkeypatch):
    from app.ingest.arxiv_client import ArxivClient, ArxivServerError

    calls = {"n": 0}
    client = ArxivClient(get_settings())

    async def fake_retry_loop(params: dict) -> dict:
        calls["n"] += 1
        raise ArxivServerError("arxiv http 503")

    monkeypatch.setattr(client, "_retry_loop", fake_retry_loop)

    threshold = get_settings().breaker_failure_threshold
    for _ in range(threshold):
        with pytest.raises(ArxivServerError):
            await client._request_with_retry({"start": 0})

    assert get_breaker(BREAKER_HTTP).state == STATE_OPEN
    before = calls["n"]
    with pytest.raises(CircuitOpenError):
        await client._request_with_retry({"start": 0})
    assert calls["n"] == before, "arXiv 已被熔断：不该再发第 4 次整段请求"
    # ensure_corpus 的既有降级分支吃的是"异常"，CircuitOpenError 是 Exception 的子孙，
    # 所以熔断打开时它会走 corpus_ready=False 的降级路径而不是崩掉（见 nodes/ensure_corpus.py）
    assert issubclass(CircuitOpenError, Exception)


async def test_arxiv_breaker_can_be_switched_off(monkeypatch):
    from app.ingest.arxiv_client import ArxivClient, ArxivServerError

    monkeypatch.setenv("BREAKER_FOR_HTTP", "false")
    get_settings.cache_clear()

    calls = {"n": 0}
    client = ArxivClient(get_settings())

    async def fake_retry_loop(params: dict) -> dict:
        calls["n"] += 1
        raise ArxivServerError("arxiv http 503")

    monkeypatch.setattr(client, "_retry_loop", fake_retry_loop)

    threshold = get_settings().breaker_failure_threshold
    for _ in range(threshold + 1):
        with pytest.raises(ArxivServerError):
            await client._request_with_retry({"start": 0})
    assert BREAKER_HTTP not in breaker_snapshot(), "关掉开关后不应再有 HTTP 熔断器"


# ---------------------------------------------------------------------------
# 6) 半开恢复：冷却结束 → 试探成功 → 闭合（并且整条链路重新可用）
# ---------------------------------------------------------------------------
async def test_half_open_trial_recovers_llm_path(monkeypatch):
    from langchain_core.messages import HumanMessage

    calls = {"n": 0}
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _FailingChatModel(calls))

    threshold = get_settings().breaker_failure_threshold
    for _ in range(threshold):
        with pytest.raises(TransientError):
            await _common.structured_invoke(PlanOutput, [HumanMessage(content="q")])
    assert get_breaker(BREAKER_LLM).state == STATE_OPEN

    # 把冷却时间"过掉"：直接改 breaker 的打开时刻，比 sleep 快也稳
    breaker = get_breaker(BREAKER_LLM)
    breaker._opened_at -= breaker.recovery_timeout + 1  # noqa: SLF001 - 测试内推进熔断时钟

    # 恢复后的真实 mock：这次调用应当成功并把断路器拉回 closed
    from app.core.llm import get_llm

    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: get_llm(temperature))

    parsed, _usage = await _common.structured_invoke(
        PlanOutput, [HumanMessage(content="研究问题：hybrid retrieval bm25 reranker fusion")]
    )
    assert parsed is not None
    assert breaker.state == STATE_CLOSED, "半开试探成功必须闭合"
    assert breaker.last_transition["from"] == STATE_HALF_OPEN


# ---------------------------------------------------------------------------
# 7) provider SDK 异常的可重试判定（接线时最容易漏的一类）
# ---------------------------------------------------------------------------
def _fake_provider_error(name: str, status_code: int | None = None) -> Any:
    class _Err(Exception):
        pass

    _Err.__name__ = name
    _Err.__module__ = "openai._exceptions"
    if status_code is not None:
        _Err.status_code = status_code
    return _Err()


@pytest.mark.parametrize(
    ("name", "status", "expected"),
    [
        ("APIConnectionError", None, True),
        ("APITimeoutError", None, True),
        ("RateLimitError", 429, True),  # 429 是唯一值得重试的 4xx
        ("APIStatusError", 503, True),
        ("BadRequestError", 400, False),  # context 超长之类：重试也是白费
        ("UnprocessableEntityError", 422, False),
    ],
)
def test_is_retryable_covers_provider_sdk_errors(name: str, status: int | None, expected: bool):
    assert is_retryable(_fake_provider_error(name, status)) is expected


# ---------------------------------------------------------------------------
# 8) ★ 恢复方向的迁移：是进度（thought），不是故障（error）
# ---------------------------------------------------------------------------
async def test_recovery_transitions_are_reported_as_thought_not_error():
    """`sse_observer` 的迁移表：`open→half_open`（正在试探恢复）与 `half_open→closed`
    （已恢复）必须走 `thought`；半开试探失败**重新打开**才走 `error`。

    为什么值得单独钉（架构 §8 的事件契约）：事件类型是前端用来决定"怎么渲染"的分支键。
    恢复信号若被写成 `error`，用户看到的是"系统出故障了"，实际发生的是"系统好了" ——
    而这类反向渲染在日志里完全正常，只有对着事件流才看得出来。
    """
    from app.observability.resilience_events import RESILIENCE_NODE, sse_observer

    breaker = get_breaker(BREAKER_LLM)
    emitter = SSEEmitter("run-transition-1")
    token = bind_emitter(emitter)
    try:
        sse_observer(STATE_OPEN, STATE_HALF_OPEN, breaker)
        sse_observer(STATE_HALF_OPEN, STATE_CLOSED, breaker)
        sse_observer(STATE_HALF_OPEN, STATE_OPEN, breaker)
        sse_observer(STATE_CLOSED, STATE_CLOSED, breaker)  # 未登记的迁移：静默
    finally:
        unbind_emitter(token)

    events = emitter.drain_out_of_band()
    thoughts = [e.data for e in events if e.type == "thought"]
    errors = [e.data for e in events if e.type == "error"]

    assert len(thoughts) == 2, "恢复过程必须被当成进度说明，而不是故障"
    assert "circuit_half_open" in thoughts[0]["delta"]
    assert "circuit_closed" in thoughts[1]["delta"]

    assert len(errors) == 1, "只有'又被打穿了'才该发 error"
    assert errors[0]["code"] == "circuit_reopened"
    assert errors[0]["recoverable"] is True
    assert errors[0]["node"] == RESILIENCE_NODE
    assert len(events) == 3, "映射表里没有的迁移不该凭空造出一个事件类型"


def test_sse_observer_is_harmless_outside_a_stream():
    """脱离 SSE 流（脚本 / CLI / 单测直调函数）：只降级为日志，绝不抛异常。

    这条同时守住"绝不新增事件类型"的另一半 —— 未登记的迁移也必须是静默而不是报错。
    """
    from app.observability.resilience_events import sse_observer

    breaker = get_breaker(BREAKER_LLM)
    assert sse_observer(STATE_CLOSED, STATE_OPEN, breaker) is None
    assert sse_observer("未知态", "另一个态", breaker) is None  # 映射表外的迁移：静默返回


async def test_observer_crash_does_not_become_a_new_incident():
    """观测器自己抛错时，绝不能把主调用一起带崩（熔断器不该成为新的故障源）。

    `CircuitBreaker._set_state` 里的 `try/except` 是这条承诺的唯一落点：
    没有它，一次 SSE 发射失败（队列关了、事件类型拼错了）会让
    "下游本来只是慢"升级成"整条链路 500"，而且真实异常还会被观测器的异常盖掉。
    """
    def noisy_observer(*_args: Any) -> None:
        raise RuntimeError("观测器炸了")

    breaker = CircuitBreaker("noisy", failure_threshold=1, observer=noisy_observer)

    async def _fail() -> None:
        raise TransientError("provider down")

    # closed→open 会通知观测器：真实故障必须原样抛出，观测器的异常不许顶掉它
    with pytest.raises(TransientError):
        await breaker.call(_fail)
    assert breaker.state == STATE_OPEN, "观测失败不影响状态迁移本身"

    # 解熔断（open→closed 又一次通知观测器）也必须照常成功
    breaker.reset()
    assert breaker.state == STATE_CLOSED

"""API 层的可观测面测试：SSE 对外契约 + LangSmith 配置真的生效（交付目标 C、D）。

两块内容各自要证明的事实：

**SSE 契约**（`app/observability/events.py` + `app/api/v1/research.py`）
- 一条流里出现的**每一种**事件类型都在 `EVENT_TYPES` 白名单内 ——
  这是对外契约，Streamlit 面板与用户脚本都按它分支；
- 每一帧的 `data` 都带 `schema_version`（前端据此做兼容判断，而不是靠字段在不在猜版本）；
- 容错层的降级信号能穿过这条流（`error` 事件带 code），
  而不是只躺在服务端日志里。

**LangSmith**（`app/observability/tracing.py`）
以前的状态很尴尬：`Settings` 从 `.env` 读到了 `LANGCHAIN_*`，
但那些值**从没进过 `os.environ`**，而 langsmith 只看进程环境决定要不要上报
—— 于是"填了 key 却没有 trace"是一句静默无效的承诺。
这里测的就是"同步确实发生了"，以及"没配就安静跳过、且绝不把 key 端出去"。

⚠️ 本文件**不会**在开启追踪的情况下跑图：真开 trace 会让 langsmith 在后台尝试
往自己的端点上报，而单测的承诺是"零网络"。开启态只查 `/healthz`（它不执行任何 runnable）。
"""

from __future__ import annotations

import json
import os

import structlog

from app.core.config import get_settings
from app.observability.events import EVENT_TYPES, SCHEMA_VERSION
from app.observability.tracing import (
    ENV_API_KEY,
    ENV_ENDPOINT,
    ENV_PROJECT,
    ENV_TRACING,
    configure_tracing,
    tracing_status,
)
from tests.api_support import STREAM_URL, api_client, assert_no_secret_values, parse_sse

# 明显是假的占位 key：只用于验证"配置值确实进了环境"，绝不会拿去联网
FAKE_LSMITH_KEY = "lsmith__fake-key-for-tests-only"


# ---------------------------------------------------------------------------
# SSE 对外契约
# ---------------------------------------------------------------------------
async def test_stream_only_uses_contract_event_types_and_carries_schema_version(monkeypatch, tmp_path):
    async with api_client(monkeypatch, tmp_path) as client:
        body = (await client.post(STREAM_URL, json={"question": "RAG 重排的代价"})).text
        parsed = parse_sse(body)
        assert parsed, "流里一帧都没有，契约无从谈起"

        allowed = set(EVENT_TYPES)
        for etype, data in parsed:
            assert etype in allowed, f"出现了未登记的事件类型 {etype}（会让老前端静默丢帧）"
            assert data.get("schema_version") == SCHEMA_VERSION, "每帧都必须自带版本号"

        # run_id 必须在 run_started 里，否则 HITL 无从发起（架构 §7 必踩坑②）
        assert next(d for t, d in parsed if t == "run_started")["run_id"]


async def test_stream_survives_a_degraded_llm_call_and_stays_on_contract(monkeypatch, tmp_path):
    """LLM 一路失败时，流仍然只吐契约内的事件，并以 `error` + `done(status=error)` 收尾。

    为什么专门测这条：容错层会往流里塞"带外事件"，一旦它拼错事件名，
    坏的方式是**整条流炸掉**（`emit()` 对未知类型直接抛），用户看到的是断流而不是报错说明。

    两次请求把容错链路的两个阶段都覆盖掉（熔断器是**进程内共享**的，
    这正是它该有的语义：一次研究失败不该把计数清零重数）：
    - 第 1 次：真的打到下游并失败 → `internal_error`（不可恢复，重试没意义）；
    - 第 2 次：阈值已达（这里设 1）→ 快速失败 → `llm_circuit_open`（可恢复，稍后再试有意义）。
    """
    from app.agents.nodes import _common
    from app.core.resilience import TransientError

    class _Failing:
        def with_structured_output(self, schema, **kwargs):
            return self

        async def ainvoke(self, messages):
            raise TransientError("provider down")

    monkeypatch.setenv("BREAKER_FAILURE_THRESHOLD", "1")
    monkeypatch.setattr(_common, "get_llm", lambda temperature=None: _Failing())

    async with api_client(monkeypatch, tmp_path) as client:
        codes: list[tuple[str, bool]] = []
        for _ in range(2):
            body = (await client.post(STREAM_URL, json={"question": "RAG 重排的代价"})).text
            parsed = parse_sse(body)
            assert {t for t, _ in parsed} <= set(EVENT_TYPES), "断流之前也不能冒出未登记的事件类型"
            assert parsed[-1][0] == "done"
            err = next(d for t, d in parsed if t == "error")
            codes.append((err["code"], err["recoverable"]))

    assert codes[0] == ("internal_error", False)
    assert codes[1] == ("llm_circuit_open", True)


# ---------------------------------------------------------------------------
# ★ 带外事件必须能穿过真实的 LangGraph 节点任务到达前端
# ---------------------------------------------------------------------------
async def test_node_side_out_of_band_event_reaches_the_stream(monkeypatch, tmp_path):
    """容错层的降级信号走的是 contextvar + 侧信道，这条用例证明**它在真图里成立**。

    为什么单独立一条：`stream()` 在 API 请求任务里绑定发射器，
    而 `structured_invoke`（熔断器要通知它的地方）跑在 **LangGraph 派生的节点子任务**里。
    contextvar 靠"任务创建时复制上下文"传递 —— 这一步如果断了，
    症状是"日志有、测试绿、前端什么都没有"，属于最难查的一类假绿。
    所以这里直接断言：节点内部投喂的事件确实变成了 SSE 帧。
    """
    from app.agents import graph as graph_mod
    from app.observability.events import emit_out_of_band

    original_plan = graph_mod.plan_node

    async def plan_with_degradation(state):
        out = await original_plan(state)
        # 返回 False 就等于"发射器没跟着进到节点任务里"，用例必须红
        assert emit_out_of_band(
            "error",
            node="resilience",
            code="circuit_half_open",
            message="下游冷却结束，正在试探性恢复",
            recoverable=True,
        ), "节点子任务读不到 stream() 绑定的发射器"
        return out

    monkeypatch.setattr(graph_mod, "plan_node", plan_with_degradation)

    async with api_client(monkeypatch, tmp_path) as client:
        body = (await client.post(STREAM_URL, json={"question": "RAG 重排的代价"})).text

    assert "circuit_half_open" in body, "带外事件必须出现在 SSE 帧里"
    assert {t for t, _ in parse_sse(body)} <= set(EVENT_TYPES)


# ---------------------------------------------------------------------------
# ★ LangSmith：配置 → 进程环境
# ---------------------------------------------------------------------------
def test_configure_tracing_writes_settings_into_environ_when_enabled():
    """"配了就有 trace"的确切含义：三个值真的被写进了进程环境。"""
    s = get_settings().model_copy(
        update={
            "langchain_tracing_v2": True,
            "langchain_api_key": FAKE_LSMITH_KEY,
            "langchain_project": "paperpilot-ci-test",
            "langchain_endpoint": "https://langsmith.invalid.example",
        }
    )
    fake_env: dict[str, str] = {}
    status = configure_tracing(s, environ=fake_env)

    assert fake_env[ENV_TRACING] == "true"
    assert fake_env[ENV_API_KEY] == FAKE_LSMITH_KEY
    assert fake_env[ENV_PROJECT] == "paperpilot-ci-test"
    assert fake_env[ENV_ENDPOINT] == "https://langsmith.invalid.example"
    assert status["enabled"] is True
    assert status["project"] == "paperpilot-ci-test"


def test_configure_tracing_switches_tracing_off_explicitly():
    """没配 → 安静跳过，并且**显式写 false**。

    为什么不是"什么都不做"：宿主 shell 里残留的 `LANGCHAIN_TRACING_V2=true`
    （别人的机器、CI runner 都可能）会让 langsmith 在没有 key 时反复告警甚至试图上报。
    显式写 false 才能把"我这没配"这件事确定下来。
    """
    s = get_settings().model_copy(
        update={"langchain_tracing_v2": False, "langchain_api_key": "", "langchain_endpoint": ""}
    )
    fake_env: dict[str, str] = {ENV_TRACING: "true", ENV_PROJECT: "leftover-from-host-shell"}
    status = configure_tracing(s, environ=fake_env)

    assert fake_env[ENV_TRACING] == "false"
    assert ENV_API_KEY not in fake_env, "没有 key 就不该往环境里写空串冒充有 key"
    assert ENV_ENDPOINT not in fake_env, "endpoint 留空应交给 langsmith 的默认值"
    assert status["enabled"] is False
    assert status["hint"], "未开启时要给出怎么开启的说明"


def test_configure_tracing_requested_without_key_is_loud_once():
    """开了开关却没 key —— 这是最常见的配错方式，日志必须说清楚缺哪一项。"""
    s = get_settings().model_copy(
        update={"langchain_tracing_v2": True, "langchain_api_key": ""}
    )
    with structlog.testing.capture_logs() as logs:
        status = configure_tracing(s, environ={})

    assert status["enabled"] is False
    assert status["has_api_key"] is False
    assert any(e["event"] == "tracing.skipped_no_api_key" for e in logs)


def test_configure_tracing_never_leaks_the_key(monkeypatch):
    """key 的值不许出现在返回字典、日志、或任何可外发的字符串里（交付约束：不打印 key）。"""
    s = get_settings().model_copy(
        update={"langchain_tracing_v2": True, "langchain_api_key": FAKE_LSMITH_KEY}
    )
    env: dict[str, str] = {}
    with structlog.testing.capture_logs() as logs:
        status = configure_tracing(s, environ=env)

    # 环境里必须有（这是"生效"的定义），但对外表达只允许布尔 has_api_key
    assert env[ENV_API_KEY] == FAKE_LSMITH_KEY
    assert FAKE_LSMITH_KEY not in json.dumps(status, ensure_ascii=False)
    assert FAKE_LSMITH_KEY not in json.dumps(logs, default=str, ensure_ascii=False)
    assert status["has_api_key"] is True


def test_configure_tracing_is_idempotent():
    """lifespan、脚本、测试会反复调它：多次调用的结论必须一致（不叠加、不来回改）。"""
    s = get_settings().model_copy(
        update={"langchain_tracing_v2": True, "langchain_api_key": FAKE_LSMITH_KEY, "langchain_endpoint": ""}
    )
    env: dict[str, str] = {}
    first = configure_tracing(s, environ=env)
    second = configure_tracing(s, environ=env)
    third = configure_tracing(s, environ=env)
    assert first == second == third
    assert env[ENV_TRACING] == "true"


def test_tracing_status_defaults_to_reading_settings(monkeypatch):
    """不传 enabled 时，状态由 settings 推导（/healthz 用的就是这条无参路径）。"""
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGCHAIN_API_KEY", FAKE_LSMITH_KEY)
    get_settings.cache_clear()
    try:
        assert tracing_status()["enabled"] is True
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        get_settings.cache_clear()
        assert tracing_status()["enabled"] is False
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# ★ 端到端：lifespan 真的把配置同步进了 os.environ
# ---------------------------------------------------------------------------
async def test_lifespan_syncs_langchain_env_when_configured(monkeypatch, tmp_path):
    """这一条把"README 里的 LangSmith 可选追踪"从口号变成事实的**唯一硬证据**。

    检查的是**真实 `os.environ`**（而不是返回字典）：因为 langsmith 只看这里。
    跑之前先确认本机环境是干净的（BASE_ENV 已把两个值置空/置 false），
    结束后交给 monkeypatch 还原，不会把 key 留在进程里。
    """
    # 前置条件：先把本机 shell 可能残留的值压干净，再断言"是 lifespan 把它写成 true 的"
    monkeypatch.setenv(ENV_TRACING, "false")
    monkeypatch.setenv(ENV_API_KEY, "")
    assert os.environ.get(ENV_API_KEY, "") == ""

    async with api_client(
        monkeypatch,
        tmp_path,
        LANGCHAIN_TRACING_V2="true",
        LANGCHAIN_API_KEY=FAKE_LSMITH_KEY,
        LANGCHAIN_PROJECT="paperpilot-lifespan-test",
    ) as client:
        assert os.environ[ENV_TRACING] == "true"
        assert os.environ[ENV_PROJECT] == "paperpilot-lifespan-test"

        resp = await client.get("/healthz")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tracing"]["enabled"] is True
        # /healthz 是无需鉴权的公开端点：这里断言的是"它绝不把 key 的值端出去"
        assert FAKE_LSMITH_KEY not in resp.text
        assert_no_secret_values(body)


async def test_lifespan_stays_quiet_when_not_configured(monkeypatch, tmp_path):
    """没配 key 时：`/healthz` 明确说"没在上报"，且 os.environ 被显式压成 false。"""
    async with api_client(monkeypatch, tmp_path) as client:
        assert os.environ[ENV_TRACING] == "false"
        body = (await client.get("/healthz")).json()
        assert body["tracing"]["enabled"] is False
        assert body["tracing"]["has_api_key"] is False
        assert body["tracing"]["project"] == "paperpilot-test"

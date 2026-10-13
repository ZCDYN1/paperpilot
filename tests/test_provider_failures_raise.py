"""真实 provider 失败必须**以异常形态**到达容错层 —— 否则重试/超时/熔断全是摆设。

## 这个文件为什么存在
熔断的验收很容易自欺：把 `get_llm` 换成一个"计数 + 必抛异常"的假模型，
测试就绿了。但那只能证明「给定异常时，熔断逻辑是对的」，
证明不了**真实 provider 失败时到底抛不抛异常** ——
如果 langchain 像某些版本那样把 401 咽掉、返回一条 `content="Error: …"` 的 AIMessage，
那么 `guarded()` 里的 tenacity 与 `CircuitBreaker` 在结构上永远等不到失败，
`failure_count` 恒为 0，而且那句错误文本会被当成**正常模型输出**继续流向抽取与校验。

所以这里把失败注入放在**传输层**（`httpx.MockTransport`）：
用的是真的 `ChatOpenAI` / `ChatOllama` 类，只是网络对面是假的。
这样"抛异常"这件事不是我喂给它的，是 provider 集成自己的行为。

全部离线、确定性：不联网、不需要任何 key。
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.agents.nodes import _common
from app.core import resilience as res
from app.core.config import get_settings


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[str] = []


def _json(status: int, payload: dict) -> httpx.Response:
    return httpx.Response(status, json=payload, headers={"content-type": "application/json"})


def _client_with(handler, **kwargs):
    """真 ChatOpenAI + 假传输：provider 的错误处理走它自己的代码。"""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-not-a-real-key",
        base_url="https://provider.invalid/v1",
        http_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5),
        max_retries=0,
        **kwargs,
    )


# 各种真实失败形态：状态码 + provider 常见的错误体
FAILURE_CASES = [
    ("401", lambda r: _json(401, {"error": {"message": "Authentication error", "type": "invalid_request_error"}})),
    ("403", lambda r: _json(403, {"error": {"message": "Forbidden", "type": "permission_error"}})),
    ("429", lambda r: _json(429, {"error": {"message": "Rate limit reached", "type": "rate_limit_error"}})),
    ("500", lambda r: _json(500, {"error": {"message": "server boom", "type": "server_error"}})),
    ("503", lambda r: _json(503, {"error": {"message": "Service Unavailable", "type": "api_error"}})),
    # 200 但响应体不对：这类最危险 —— 不报错、只是没有内容
    ("200-malformed", lambda r: _json(200, {"unexpected": "payload"})),
    ("200-empty-choices", lambda r: _json(200, {"choices": []})),
]


@pytest.mark.parametrize(("label", "handler"), FAILURE_CASES, ids=[c[0] for c in FAILURE_CASES])
async def test_provider_failure_raises_instead_of_returning_error_text(label, handler):
    """★ 前提断言：真实 provider 类遇到这些失败时必须抛，而不是返回"错误文本"。

    这里刻意用最宽的 `Exception`：要证的正是"不以异常形态出现"这一种失效，
    具体异常类型属于 provider 集成细节，锁死它反而会让升级时误红。

    这条一旦变红（比如升级 langchain-openai 后行为改了），整个容错层就立刻失效 ——
    所以它必须常驻测试，而不是靠某次人工验收。
    """
    client = _client_with(handler)
    with pytest.raises(Exception) as excinfo:  # noqa: B017, PT011
        await client.ainvoke("hi")
    assert not isinstance(excinfo.value, AssertionError)


async def test_structured_output_path_also_raises():
    """read/verify 实际走的是 `with_structured_output` 那条路，同样必须抛。"""
    client = _client_with(
        lambda r: _json(401, {"error": {"message": "Authentication error", "type": "invalid_request_error"}})
    )
    with pytest.raises(Exception):  # noqa: B017, PT011
        await client.with_structured_output(Out).ainvoke("hi")


async def test_connection_error_raises():
    def handler(request):  # noqa: ANN001, ARG001
        raise httpx.ConnectError("connection refused")

    client = _client_with(handler)
    with pytest.raises(Exception):  # noqa: B017, PT011
        await client.ainvoke("hi")


async def test_ollama_provider_raises_on_failure():
    """本机演示默认走 ollama：它失败时也必须抛，不能吞成文本。"""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:  # pragma: no cover - 没装就跳过，不该让测试红
        pytest.skip("langchain_ollama 未安装")

    client = ChatOllama(model="qwen2.5:7b", base_url="http://127.0.0.1:9", client_timeout=5)
    with pytest.raises(Exception):  # noqa: B017, PT011
        await client.ainvoke("hi")


# ---------------------------------------------------------------------------
# 端到端：真实失败打进熔断器，并证明"打开后不再打下游"
# ---------------------------------------------------------------------------
@pytest.fixture
def tight_breaker(monkeypatch):
    """把阈值压到 2、重试 1 次、不退避，并清掉进程级熔断状态。"""
    monkeypatch.setenv("RESILIENCE_ENABLED", "true")
    monkeypatch.setenv("RESILIENCE_ATTEMPTS", "1")
    monkeypatch.setenv("RESILIENCE_BACKOFF", "0")
    monkeypatch.setenv("RESILIENCE_MAX_BACKOFF", "0")
    monkeypatch.setenv("RESILIENCE_JITTER", "0")
    monkeypatch.setenv("RESILIENCE_TIMEOUT", "10")
    monkeypatch.setenv("BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("BREAKER_RECOVERY_TIMEOUT", "300")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    get_settings.cache_clear()
    res.reset_breakers()
    yield
    res.reset_breakers()
    get_settings.cache_clear()


async def test_real_provider_failures_open_the_breaker(tight_breaker, monkeypatch):
    """三轮：前两次真打下游并抛异常 → 熔断打开 → 第三次**请求数不再增长**。

    这是最接近线上的证明：失败来自传输层，不是来自我们自制的假模型。
    """
    requests = {"n": 0}

    def handler(request):  # noqa: ANN001
        requests["n"] += 1
        return _json(401, {"error": {"message": "Authentication error", "type": "invalid_request_error"}})

    monkeypatch.setattr(
        _common,
        "get_llm",
        lambda temperature=None: _client_with(handler),
    )

    for expected_calls in (1, 2):
        with pytest.raises(Exception):  # noqa: B017, PT011
            await _common.structured_invoke(Out, [("user", "hi")])
        assert requests["n"] == expected_calls

    breaker = res.get_breaker(res.BREAKER_LLM)
    assert breaker.state == "open", f"两次真实失败后应当打开，实际 {breaker.state}"

    # 关键一步：熔断打开后不该再把请求发出去
    before = requests["n"]
    with pytest.raises(res.CircuitOpenError):
        await _common.structured_invoke(Out, [("user", "hi")])
    assert requests["n"] == before, "熔断打开后仍然打到了下游 —— 保护没生效"

"""容错层测试（架构 §11）。

重点有两块：
1. 重试 / 超时 / 熔断各自的行为边界（只重试该重试的、超时真的中断、熔断三态正确迁移）；
2. ★ **依赖顺序 `retry → timeout → circuit breaker`** —— 用"重试 3 次只记 1 次失败、
   连 5 次调用才打开"的用例把它锁死。顺序写反时这个用例会立刻变红（见文件末尾注释）。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from app.core.resilience import (
    STATE_CLOSED,
    STATE_HALF_OPEN,
    STATE_OPEN,
    CircuitBreaker,
    CircuitOpenError,
    PermanentError,
    TransientError,
    is_retryable,
    resilient,
    retry_async,
    with_timeout,
)

# 测试里把退避与抖动全部归零，避免真的等待
NO_WAIT = dict(multiplier=0.0, max_wait=0.0, jitter=(0.0, 0.0))


class FakeClock:
    """可推进的单调时钟，用于免 sleep 地验证熔断冷却。"""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://example.invalid/x")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(f"HTTP {status}", request=request, response=response)


# =============================================================================
# 重试
# =============================================================================
async def test_retry_retries_retryable_until_attempts():
    calls = {"n": 0}

    @retry_async(attempts=3, **NO_WAIT)
    async def flaky():
        calls["n"] += 1
        raise TransientError("boom")

    with pytest.raises(TransientError):
        await flaky()
    assert calls["n"] == 3  # stop_after_attempt(3)


async def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    @retry_async(attempts=3, **NO_WAIT)
    async def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TransientError("transient")
        return "ok"

    assert await flaky() == "ok"
    assert calls["n"] == 2


async def test_retry_does_not_retry_permanent_error():
    """4xx 类错误重试也无用，必须只尝试一次。"""
    calls = {"n": 0}

    @retry_async(attempts=3, **NO_WAIT)
    async def bad_request():
        calls["n"] += 1
        raise PermanentError("context too long (4xx)")

    with pytest.raises(PermanentError):
        await bad_request()
    assert calls["n"] == 1


async def test_retry_retries_5xx_but_not_4xx():
    calls = {"n": 0}

    @retry_async(attempts=3, **NO_WAIT)
    async def server_error():
        calls["n"] += 1
        raise _http_status_error(503)

    with pytest.raises(httpx.HTTPStatusError):
        await server_error()
    assert calls["n"] == 3  # 5xx 重试

    calls["n"] = 0

    @retry_async(attempts=3, **NO_WAIT)
    async def client_error():
        calls["n"] += 1
        raise _http_status_error(400)

    with pytest.raises(httpx.HTTPStatusError):
        await client_error()
    assert calls["n"] == 1  # 4xx 不重试


def test_is_retryable_classification():
    assert is_retryable(TransientError("x"))
    assert is_retryable(TimeoutError())
    assert is_retryable(_http_status_error(500))
    assert is_retryable(httpx.ConnectError("no route"))
    assert not is_retryable(PermanentError("x"))
    assert not is_retryable(_http_status_error(404))
    assert not is_retryable(ValueError("bad input"))


# =============================================================================
# 超时
# =============================================================================
async def test_with_timeout_raises_on_slow_call():
    async def slow():
        await asyncio.sleep(1.0)
        return "late"

    with pytest.raises(asyncio.TimeoutError):
        await with_timeout(slow(), 0.02)


async def test_with_timeout_passes_through_when_fast():
    async def fast():
        return "quick"

    assert await with_timeout(fast(), 1.0) == "quick"
    # seconds=None 表示不设超时
    assert await with_timeout(fast(), None) == "quick"


# =============================================================================
# 熔断：三态迁移
# =============================================================================
async def test_breaker_opens_after_threshold_and_short_circuits():
    breaker = CircuitBreaker("t", failure_threshold=3, recovery_timeout=60, time_source=FakeClock())
    calls = {"n": 0}

    async def failing():
        calls["n"] += 1
        raise TransientError("down")

    for _ in range(3):
        with pytest.raises(TransientError):
            await breaker.call(failing)
    assert breaker.state == STATE_OPEN

    # 打开后不再打扰下游：直接快速失败
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)
    assert calls["n"] == 3


async def test_breaker_half_open_then_closed_on_success():
    clock = FakeClock()
    breaker = CircuitBreaker("t", failure_threshold=2, recovery_timeout=60, time_source=clock)

    async def failing():
        raise TransientError("down")

    for _ in range(2):
        with pytest.raises(TransientError):
            await breaker.call(failing)
    assert breaker.state == STATE_OPEN

    # 冷却未到：仍拒绝
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)

    # 冷却结束 → 半开试探；试探成功 → 关闭
    clock.advance(60)
    assert await breaker.call(lambda: _ok()) == "ok"
    assert breaker.state == STATE_CLOSED
    assert breaker.failure_count == 0


async def test_breaker_reopens_on_half_open_failure():
    clock = FakeClock()
    breaker = CircuitBreaker("t", failure_threshold=2, recovery_timeout=30, time_source=clock)

    async def failing():
        raise TransientError("down")

    for _ in range(2):
        with pytest.raises(TransientError):
            await breaker.call(failing)

    clock.advance(30)  # 进入半开
    with pytest.raises(TransientError):
        await breaker.call(failing)
    assert breaker.state == STATE_OPEN
    # 半开试探失败要重置冷却：立刻再调用仍是拒绝
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing)


async def test_breaker_returns_fallback_when_open():
    breaker = CircuitBreaker("t", failure_threshold=1, recovery_timeout=60, time_source=FakeClock())

    async def failing():
        raise TransientError("down")

    with pytest.raises(TransientError):
        await breaker.call(failing)
    assert breaker.is_open

    # 打开期间返回降级结果，而不是抛错
    result = await breaker.call(failing, fallback=lambda: "degraded")
    assert result == "degraded"
    assert await breaker.call(failing, fallback="cached") == "cached"


async def _ok():
    return "ok"


# =============================================================================
# ★ 依赖顺序验证：retry → timeout → circuit breaker
# =============================================================================
async def test_order_retry_inside_breaker_counts_one_failure_per_call():
    """核心用例：**一次调用内部重试 3 次，但对外只应记为 1 次失败**。

    若顺序写反（断路器在重试里面），一次调用会记 3 次失败，
    阈值 5 时**第 2 次调用就会误开** —— 那时下面的断言会立刻失败。
    因此本用例直接把 §11 要求的依赖顺序钉死为可回归的事实。
    """
    clock = FakeClock()
    breaker = CircuitBreaker("order", failure_threshold=5, recovery_timeout=60, time_source=clock)

    attempts = {"n": 0}

    async def always_fails():
        attempts["n"] += 1
        raise TransientError("boom")

    guarded = resilient(breaker, attempts=3, **NO_WAIT)(always_fails)

    # 前 4 次调用：每次都内部尝试 3 次，但断路器只累计 4 次失败，仍未打开
    for call_no in range(1, 5):
        attempts["n"] = 0
        with pytest.raises(TransientError):
            await guarded()
        assert attempts["n"] == 3, "内部应重试满 3 次"
        assert breaker.failure_count == call_no, "一次调用只能记 1 次失败"
        assert breaker.state == STATE_CLOSED, f"第 {call_no} 次调用后不该打开"

    # 第 5 次：累计到阈值 5 → 打开
    with pytest.raises(TransientError):
        await guarded()
    assert breaker.failure_count == 5
    assert breaker.state == STATE_OPEN

    # 打开后：直接短路，不再触碰下游（attempts 不增加）
    attempts["n"] = 0
    with pytest.raises(CircuitOpenError):
        await guarded()
    assert attempts["n"] == 0


def test_half_open_state_constant_is_exported():
    # 保证三态常量对外可见（便于上层与前端展示一致的状态名）
    assert {STATE_CLOSED, STATE_OPEN, STATE_HALF_OPEN} == {"closed", "open", "half_open"}

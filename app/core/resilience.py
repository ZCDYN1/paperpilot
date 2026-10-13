"""容错层：重试 / 超时 / 熔断（架构 §11）。

三者解决的是三种**不同**的故障，混着做会互相打架：

| 组件 | 针对的故障 | 核心问题 |
|---|---|---|
| `retry_async` | 偶发网络抖动、5xx | 重试就够，不需要额外机制 |
| `with_timeout` | 调用挂死（连接建立后迟迟不返回） | 重试救不了挂死，只会挂更久 |
| `CircuitBreaker` | 下游**持续**不可用 | 重试 / 超时都拦不住"雪崩" |

## ★ 依赖顺序：`retry → timeout → circuit breaker`
即**断路器包在最外层**（重试在里面）：

```
CircuitBreaker(          # 最外层：整段调用只算"一次成败"
    with_timeout(        # 中间层：给整段重试一个总时间预算
        retry_async(fn)  # 最内层：每次尝试失败可再试
    )
)
```

**为什么断路器必须在外面**（架构 §11 的原话是"否则每次重试都会触发断路判断，导致误开"）：
`CircuitBreaker` 的打开条件是"连续 N 次失败"。若把断路器放在重试**里面**，
那么一次业务失败会经历 3 次尝试 = **记 3 次失败**；阈值 N=5 时，
**2 次业务失败**（2×3=6）就会把断路器打开 —— 而按设计它本该能容忍 5 次业务失败。
结果是一旦网络稍微抖动，断路器就误开，把本来能自愈的请求直接降级。

正确顺序下，一次调用无论内部重试几次，**对外只表现为一次成败**，
断路器要连续 5 次"整段调用失败"才打开。`tests/test_resilience.py` 用
「调用 4 次仍关闭、第 5 次才打开」这个用例把顺序**锁死**（见该文件注释）。

## 为什么超时放在中间（而不是每次尝试各给一份）
时间预算应该是**整次调用**的：若按每次尝试计，3 次尝试最坏能拖 `3×seconds`，
上层请求的等待时间不可预期（对 SSE 这类交互式场景是致命的）。
放在中间层，一次 `call` 最多耗时 `seconds`，把最坏延迟钉死。
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

T = TypeVar("T")

# =============================================================================
# 异常分类：决定"要不要重试"
# =============================================================================
class TransientError(Exception):
    """可重试错误：语义等价于"网络抖动 / 5xx"。

    真实链路里我们会把 httpx 的 5xx、连接中断等映射成它；
    单测里直接抛它就能验证重试逻辑，不必伪造整个 HTTP 栈。
    """


class PermanentError(Exception):
    """不可重试错误：语义等价于"4xx（如 context 超长、参数错误）"。

    重试一个 4xx 永远只会得到同样的 4xx，白白浪费 3 倍配额与延迟。
    """


class CircuitOpenError(Exception):
    """熔断打开期间直接拒绝，避免对已知不可用的下游持续打请求（雪崩防护）。"""


def is_retryable(exc: BaseException) -> bool:
    """是否值得重试。**只对网络类错误与 5xx 重试，4xx 直接失败。**

    分类依据（架构 §11）：
    - 网络层：`httpx.TransportError` / `httpx.TimeoutException` / `ConnectionError` / 超时
    - HTTP 5xx：下游服务端故障，可能自愈 → 重试
    - HTTP 4xx：请求本身有问题（参数错、context 超长）→ **不重试**，重试也是白费
    """
    if isinstance(exc, PermanentError):
        return False
    if isinstance(exc, (TransientError, asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    if isinstance(exc, (httpx.TransportError, httpx.TimeoutException, httpx.HTTPError)):
        return True
    if isinstance(exc, ConnectionError):  # 含 ConnectionResetError 等
        return True
    return False


# =============================================================================
# 重试
# =============================================================================
def retry_async[T](
    func: Callable[..., Awaitable[T]] | None = None,
    *,
    attempts: int = 3,
    multiplier: float = 1.0,
    max_wait: float = 8.0,
    jitter: tuple[float, float] = (0.0, 1.0),
    retry_on: Callable[[BaseException], bool] = is_retryable,
) -> Callable[..., Awaitable[T]]:
    """装饰器：失败最多重试到 `attempts` 次（默认 3），指数退避 + 随机抖动。

    - **指数退避**：`1s → 2s → 4s…`（上限 `max_wait`）。给下游一点恢复时间，
      而不是立刻猛冲（立刻猛冲会把刚缓过来的下游再次打趴）。
    - **随机抖动**：多个并行分支（本产品 N 个子问题会并行检索）如果退避时间完全一致，
      会在同一时刻集体重试，形成"惊群"。抖动把它们的重试时刻错开。
    - **只重试 `retry_on` 为真的异常**：4xx 不重试（见 `is_retryable`）。

    既支持 `@retry_async`，也支持 `@retry_async(attempts=5)` 两种写法。
    单测会把 `multiplier/max_wait/jitter` 设为 0 以避免真的等待。
    """

    def decorate(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # tenacity 的异步迭代写法：每次迭代代表一次尝试，
            # `with attempt:` 捕获本轮异常并根据 retry/stop 决定是否继续。
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(attempts),
                wait=wait_exponential(multiplier=multiplier, max=max_wait) + wait_random(*jitter),
                retry=retry_if_exception(retry_on),
                reraise=True,  # 用尽后抛原始异常，而不是包一层 RetryError（上层更好处理）
            ):
                with attempt:
                    return await fn(*args, **kwargs)
            raise AssertionError("unreachable: tenacity 循环必须以 return 或抛异常结束")

        return wrapper

    # 裸装饰器用法：@retry_async
    if func is not None:
        return decorate(func)
    # 带参用法：@retry_async(attempts=5)
    return decorate


# =============================================================================
# 超时
# =============================================================================
async def with_timeout[T](coro: Awaitable[T], seconds: float | None) -> T:
    """给一个协程套上超时（`asyncio.wait_for`）。

    防的是什么：调用**挂死** —— 连接建起来了，但服务端迟迟不返回，
    调用方会一直等下去，把整个 SSE 流的体验拖垮。
    `seconds=None` 表示不设超时（供确实需要长等待的场景显式选择）。
    """
    if seconds is None:
        return await coro
    return await asyncio.wait_for(coro, timeout=seconds)


# =============================================================================
# 熔断
# =============================================================================
STATE_CLOSED = "closed"
STATE_OPEN = "open"
STATE_HALF_OPEN = "half_open"

_UNSET: Any = object()


class CircuitBreaker:
    """熔断器：连续 N 次失败打开，冷却 T 秒后放行**一次**试探（半开）。

    三态语义：
    - `closed`：正常放行。失败累计到 `failure_threshold` → `open`
    - `open`：**直接拒绝**（或返回降级结果），不去打扰已经不可用的下游
    - `half_open`：冷却结束后放行**一次**试探。成功 → `closed`；失败 → 再次 `open`

    防的是什么（雪崩）：下游整个挂了的时候，重试与超时都拦不住"每个请求都去打一遍"——
    大量并发请求持续冲击一个已经瘫痪的服务，会把它彻底压死，也耗尽本服务自身的
    连接与线程资源。熔断让请求在**本服务内部**就被快速失败，
    同时给下游一段喘息窗口去恢复。

    `time_source` 可注入（默认 `time.monotonic`）：单测用假时钟推进冷却时间，
    避免真的 `sleep(60)`。
    """

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold 必须 >= 1")
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._time = time_source

        self._state = STATE_CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_trial_used = False

    # ----------------------------- 只读视图 -----------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failures

    @property
    def is_open(self) -> bool:
        return self._state == STATE_OPEN

    # ----------------------------- 状态迁移 -----------------------------
    def _allow_request(self) -> bool:
        """判断本次请求能否放行；顺带完成 open→half_open 的时间迁移。"""
        if self._state == STATE_OPEN:
            if self._time() - self._opened_at >= self.recovery_timeout:
                # 冷却结束：切到半开，放行且**只放行一次**试探
                self._state = STATE_HALF_OPEN
                self._half_open_trial_used = True
                return True
            return False
        if self._state == STATE_HALF_OPEN:
            if self._half_open_trial_used:
                return False  # 试探已在途/已用掉，其余请求继续快速失败
            self._half_open_trial_used = True
            return True
        return True  # closed

    def _on_success(self) -> None:
        self._failures = 0
        self._state = STATE_CLOSED
        self._half_open_trial_used = False

    def _on_failure(self) -> None:
        self._failures += 1
        if self._state == STATE_HALF_OPEN:
            # 半开试探失败：立即重新打开，重新计冷却
            self._state = STATE_OPEN
            self._opened_at = self._time()
            self._half_open_trial_used = False
            return
        if self._failures >= self.failure_threshold:
            self._state = STATE_OPEN
            self._opened_at = self._time()
            self._half_open_trial_used = False

    def reset(self) -> None:
        """手动复位（测试与运维手动解熔断用）。"""
        self._state = STATE_CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_trial_used = False

    # ----------------------------- 调用入口 -----------------------------
    async def call(
        self,
        func: Callable[[], Awaitable[T]],
        *args: Any,
        fallback: Any = _UNSET,
        **kwargs: Any,
    ) -> T:
        """受熔断保护的调用。

        `func` 可以是零参协程函数，也可以带 `*args/**kwargs`（便于直接转发）。
        `fallback` 为熔断打开时的降级结果：**可调用**则 await 它，否则原样返回；
        未提供时抛 `CircuitOpenError`。

        ⚠️ 计数语义：**一次 `call` 只记一次成败**，这正是"断路器包在重试外面"的体现。
        """
        if not self._allow_request():
            if fallback is _UNSET:
                raise CircuitOpenError(
                    f"熔断器[{self.name}] 已打开，拒绝请求（连续失败 {self._failures} 次）"
                )
            # 降级结果支持三种形态：静态值 / 同步函数 / 异步函数 —— 统一归一成"可能可 await 的值"
            value = fallback() if callable(fallback) else fallback
            if inspect.isawaitable(value):
                value = await value
            return value

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._on_failure()
            raise
        else:
            self._on_success()
            return result


# =============================================================================
# 组合：把三层按正确顺序叠起来
# =============================================================================
def resilient(
    breaker: CircuitBreaker,
    *,
    seconds: float | None = None,
    attempts: int = 3,
    multiplier: float = 1.0,
    max_wait: float = 8.0,
    jitter: tuple[float, float] = (0.0, 1.0),
    fallback: Any = _UNSET,
    retry_on: Callable[[BaseException], bool] = is_retryable,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """装饰器工厂：按 `retry → timeout → circuit breaker` 的**正确顺序**叠加三层保护。

    组装出来的结构（从外到内）：

        breaker.call(  with_timeout(  retry_async(fn)(...)  )  )

    这个顺序是架构 §11 明确要求的，也是本模块存在的核心价值：
    **断路器在最外层 → 一次调用只给它记一次失败 → 不会因为内部重试而误开。**
    详见模块开头 docstring 的说明。
    """

    def decorate(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        # 最内层：重试
        retried = retry_async(
            attempts=attempts,
            multiplier=multiplier,
            max_wait=max_wait,
            jitter=jitter,
            retry_on=retry_on,
        )(fn)

        @functools.wraps(fn)
        async def guarded(*args: Any, **kwargs: Any) -> T:
            async def once() -> T:
                # 中间层：给整段重试一个总时间预算
                return await with_timeout(retried(*args, **kwargs), seconds)

            # 最外层：熔断
            if fallback is _UNSET:
                return await breaker.call(once)
            return await breaker.call(once, fallback=fallback)

        return guarded

    return decorate


__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "PermanentError",
    "STATE_CLOSED",
    "STATE_HALF_OPEN",
    "STATE_OPEN",
    "TransientError",
    "is_retryable",
    "resilient",
    "retry_async",
    "with_timeout",
]

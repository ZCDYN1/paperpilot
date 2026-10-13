"""把容错层的状态迁移"翻译"成可观测信号（结构化日志 + SSE 侧信道事件）。

## 为什么单独一个模块
`app/core/resilience.py` 是纯机制：它不知道 SSE 协议、也不该知道。
`app/observability/events.py` 是协议：它不知道什么叫熔断。
"熔断打开了要告诉用户"是两者之间的一层翻译，放在这里正好（架构 §8 的事件契约 + §11 的容错）。

## 为什么不新增一种 SSE 事件类型
事件类型枚举是**对外契约**（Streamlit 面板、未来的前端、用户自己写的消费脚本都在按它分支），
加一个 `circuit_open` 类型会让老前端走进"未知事件"分支 —— 而多数实现是静默丢弃。
所以复用既有的两类：
- `error`（`recoverable=True`）= "下游现在被打穿了，这次请求没真去调用" ——
  前端本来就要渲染 error，语义完全对得上；
- `thought` = "半开试探中 / 已恢复" —— 属于进度说明，不该被当成故障。

具体是熔断还是超时，看 `code` 字段（`circuit_open` / `circuit_half_open` / `circuit_closed`），
它写进 `error.code` / `thought.delta` 里，前端可以原样展示。
"""

from __future__ import annotations

from typing import Any

import structlog

from app.observability.events import emit_out_of_band

logger = structlog.get_logger(__name__)

# 事件里统一用这个 node 名，前端能把它归到"系统状态"而不是某个 Agent 节点
RESILIENCE_NODE = "resilience"

# 状态 → 事件形态。写在表里而不是散在 if 里：加一态时只用改这张表。
_TRANSITIONS: dict[tuple[str, str], dict[str, Any]] = {
    ("closed", "open"): {"event": "error", "code": "circuit_open", "recoverable": True},
    ("half_open", "open"): {"event": "error", "code": "circuit_reopened", "recoverable": True},
    ("open", "half_open"): {"event": "thought", "code": "circuit_half_open"},
    ("half_open", "closed"): {"event": "thought", "code": "circuit_closed"},
    ("open", "closed"): {"event": "thought", "code": "circuit_closed"},
}


def sse_observer(old_state: str, new_state: str, breaker: Any) -> None:
    """`CircuitBreaker(observer=...)` 用的观测器。

    三条硬性要求：
    1. **绝不抛异常** —— 观测失败绝不能把主调用带崩（熔断器自己不该成为故障源）；
    2. 不在流上下文时只打日志（脚本 / 单测直接调函数时走这条）；
    3. 事件里带 `breaker` 名与失败计数，用户/运维能判断是哪个下游、被拦了几次。
    """
    spec = _TRANSITIONS.get((old_state, new_state))
    if spec is None:
        logger.debug(
            "resilience.transition_unmapped",
            breaker=getattr(breaker, "name", "?"),
            from_state=old_state,
            to_state=new_state,
        )
        return

    name = getattr(breaker, "name", "default")
    failures = int(getattr(breaker, "failure_count", 0) or 0)
    message = f"熔断器[{name}] {old_state} → {new_state}（连续失败 {failures} 次）"

    if spec["event"] == "error":
        sent = emit_out_of_band(
            "error",
            node=RESILIENCE_NODE,
            code=spec["code"],
            message=message,
            recoverable=bool(spec.get("recoverable", True)),
            breaker=name,
            failures=failures,
        )
    else:
        sent = emit_out_of_band("thought", node=RESILIENCE_NODE, delta=f"{message} · {spec['code']}")

    logger.info("resilience.sse_emitted", breaker=name, code=spec["code"], streamed=sent)


__all__ = ["RESILIENCE_NODE", "sse_observer"]

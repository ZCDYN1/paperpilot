"""SSE 事件发射器（架构 §8）。

## 为什么把事件协议单独成模块、并强约束成 15 种
SSE 协议**一旦上线就是对外契约**：Streamlit 面板、未来的 React 前端、以及
用户自己写的消费脚本都依赖它。加字段容易、改字段难，所以要满足三点：
1. 类型**枚举化**（`EVENT_TYPES`）：拼错的事件名会被当场拒绝，而不是静默丢失；
2. 每条 data 带 `schema_version`：前端据此做兼容判断，而不是靠"字段在不在"猜版本；
3. 帧格式固定 `event: <type>\\ndata: <json>\\n\\n`：SSE 规范要求，两处换行缺一不可。

## 两个必须发出来的"隐形"事件
- `interrupted`：HITL 中断时**必须**补发。否则前端只看到流"卡住不动"，
  用户以为程序死了（架构 §7 的必踩坑②）。
- `token`：增量文本。前端靠它做打字机效果，而不是等整段返回再一次性渲染。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = "1"

# 15 种事件类型（架构 §8 事件表）。顺序与文档一致，便于逐条核对。
EVENT_TYPES: tuple[str, ...] = (
    "run_started",
    "node_started",
    "plan_created",
    "corpus",
    "tool_call",
    "tool_result",
    "retrieval",
    "thought",
    "token",
    "verification",
    "interrupted",
    "final_report",
    "token_usage",
    "error",
    "done",
)

_EVENT_TYPE_SET = frozenset(EVENT_TYPES)


@dataclass(slots=True)
class SSEEvent:
    """一条 SSE 事件。`data` 里会自动带上 `schema_version`。"""

    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """最终写进 `data:` 的 JSON 对象。"""
        return {"schema_version": SCHEMA_VERSION, **self.data}

    def to_sse(self) -> str:
        """转成标准 SSE 帧。

        ⚠️ `ensure_ascii=False`：默认的 ASCII 转义会把中文变成 `\\uXXXX`，
        帧体积膨胀 3~6 倍且人眼完全不可读（调试时最痛）。
        SSE 规范要求 UTF-8 编码，中文原样输出是安全的。
        """
        return f"event: {self.type}\ndata: {json.dumps(self.payload(), ensure_ascii=False)}\n\n"


def event_from_state(raw: dict[str, Any]) -> SSEEvent:
    """把节点塞进 `state["events"]` 的裸 dict 转成 `SSEEvent`。

    约定：节点产出 `{"type": "...", ...其余作为 data}`。这样节点不必依赖本模块，
    保持"节点只写数据、发射器只做协议"的分层。
    """
    payload = dict(raw)
    etype = payload.pop("type", "thought")
    return SSEEvent(type=etype, data=payload)


class SSEEmitter:
    """事件发射器。

    同时支持两种用法：
    1. **节点主动发**：节点把事件 dict 放进 `state["events"]`（见 `app/agents/state.py`），
       由 `drain_state_events` 取出。好处是节点签名保持 `(state) -> dict`，
       不依赖任何 emitter 实例，单测时不用 mock 网络层。
    2. **编排层流式发**：`stream()` 包住 `graph.astream`，把状态更新翻译成
       `node_started` / `plan_created` / `interrupted` 等框架级事件。
    """

    def __init__(self, run_id: str = "", *, schema_version: str = SCHEMA_VERSION) -> None:
        self.run_id = run_id
        self.schema_version = schema_version
        self._buffer: list[SSEEvent] = []
        self._token_usage: dict[str, Any] = {}

    # ------------------------------------------------------------ 基础发射
    def emit(self, event_type: str, **data: Any) -> SSEEvent:
        """发射一条事件。

        未知类型**直接抛错**而不是放行：静默放行会让"前端少了一行日志"这类问题
        拖到联调阶段才发现，而那时你已经不知道是哪次改动引入的。
        """
        if event_type not in _EVENT_TYPE_SET:
            raise ValueError(
                f"未定义的 SSE 事件类型 {event_type!r}；可用类型：{list(EVENT_TYPES)}"
            )
        ev = SSEEvent(type=event_type, data=data)
        self._buffer.append(ev)
        return ev

    def drain(self) -> list[SSEEvent]:
        """取走并清空缓冲区（节点事件通常这么消费）。"""
        events, self._buffer = self._buffer, []
        return events

    def drain_state_events(self, state: dict[str, Any]) -> list[SSEEvent]:
        """把 `state["events"]` 里的裸 dict 转成事件（不修改 state）。"""
        return [event_from_state(e) for e in (state.get("events") or [])]

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    # ----------------------------------------------------- 15 种事件的便捷方法
    # 之所以给每种事件都写一个方法（而不是让调用方手写字符串）：
    # 参数名即契约，IDE 能补全、改字段时能找到所有调用点。
    def run_started(self, question: str, *, extra: dict[str, Any] | None = None) -> SSEEvent:
        return self.emit("run_started", run_id=self.run_id, question=question, **(extra or {}))

    def node_started(self, node: str, label: str = "") -> SSEEvent:
        return self.emit("node_started", node=node, label=label or node, ts=self._now_ms())

    def plan_created(self, sub_questions: list[dict], rationale: str = "") -> SSEEvent:
        return self.emit(
            "plan_created", sub_questions=sub_questions, rationale=rationale, count=len(sub_questions)
        )

    def corpus(
        self, *, fetched: int, cached: int, total: int, corpus_ready: bool, **extra: Any
    ) -> SSEEvent:
        return self.emit(
            "corpus",
            fetched=fetched,
            cached=cached,
            total=total,
            corpus_ready=corpus_ready,
            **extra,
        )

    def tool_call(self, node: str, tool: str, args: dict, call_id: str) -> SSEEvent:
        return self.emit("tool_call", node=node, tool=tool, args=args, call_id=call_id)

    def tool_result(
        self, call_id: str, *, status: str, duration_ms: int, count: int = 0, **extra: Any
    ) -> SSEEvent:
        return self.emit(
            "tool_result",
            call_id=call_id,
            status=status,
            duration_ms=duration_ms,
            count=count,
            **extra,
        )

    def retrieval(self, sub_question_id: str, hits: list[dict]) -> SSEEvent:
        """检索完成。`hits` 只带展示需要的字段（标题/分数/章节摘要），不带全文。"""
        compact = [
            {
                "arxiv_id": h.get("arxiv_id"),
                "title": h.get("title"),
                "score": h.get("score"),
                "section": (h.get("chunks") or [{}])[0].get("section"),
                "source": (h.get("chunks") or [{}])[0].get("source"),
            }
            for h in hits
        ]
        return self.emit("retrieval", sub_question_id=sub_question_id, hits=compact, count=len(compact))

    def thought(self, node: str, delta: str) -> SSEEvent:
        return self.emit("thought", node=node, delta=delta)

    def token(self, node: str, delta: str) -> SSEEvent:
        return self.emit("token", node=node, delta=delta)

    def verification(
        self,
        *,
        passed: bool,
        score: float,
        unsupported: list[str],
        revision_count: int,
    ) -> SSEEvent:
        return self.emit(
            "verification",
            passed=passed,
            score=score,
            unsupported=unsupported,
            revision_count=revision_count,
        )

    def interrupted(
        self, at: str, reason: str, payload: Any = None, resume_token: str | None = None
    ) -> SSEEvent:
        return self.emit(
            "interrupted",
            at=at,
            reason=reason,
            payload=payload,
            resume_token=resume_token or self.run_id,
        )

    def final_report(self, markdown: str, citations: list[dict]) -> SSEEvent:
        return self.emit("final_report", markdown=markdown, citations=citations)

    def token_usage_event(
        self, total: dict[str, int], *, cost_estimate: float | None = None, by_node: dict | None = None
    ) -> SSEEvent:
        return self.emit(
            "token_usage",
            total=total,
            cost_estimate=cost_estimate,
            by_node=by_node or {},
        )

    def error(self, node: str, code: str, message: str, *, recoverable: bool = True) -> SSEEvent:
        return self.emit(
            "error", node=node, code=code, message=message, recoverable=recoverable
        )

    def done(self, status: str = "ok") -> SSEEvent:
        return self.emit("done", run_id=self.run_id, status=status)

    # ------------------------------------------------------------- 图级流式
    async def stream(
        self,
        graph: Any,
        inputs: dict[str, Any] | None,
        config: dict[str, Any],
        *,
        resume: bool = False,
    ) -> AsyncIterator[str]:
        """把图的状态流翻译成 SSE 帧。

        `inputs=None` 表示**从中断点恢复**（HITL），这是 LangGraph 的约定，
        与"新起一次运行"必须区分 —— 传空 dict 会从 START 重跑，白烧一遍 token。

        只消费 `stream_mode="updates"`：每个 chunk 是 `{节点名: 该节点返回的增量}`。
        用 `values` 模式会每个超级步吐一份**全量** state（含所有 chunk 全文），
        在 20+ 篇论文的规模下会把带宽打满。
        """
        if not resume:
            # ⚠️ 必须 `yield` 出去，不能只 `emit`。
            # `emit` 只是把事件塞进内部 buffer，而 `stream()` 从头到尾**不 drain 这个 buffer** ——
            # 写成 `self.run_started(...)` 的话事件会静默消失，前端拿不到 run_id，
            # 也就无法在中断后调 resume（表现为"HITL 功能没反应"，很难查）。
            yield self.run_started(inputs.get("question", "") if inputs else "").to_sse()

        interrupted: dict[str, Any] | None = None
        async for chunk in graph.astream(inputs, config, stream_mode="updates"):
            # 中断信号：LangGraph 把它放在 `__interrupt__` 键下
            if "__interrupt__" in chunk:
                interrupted = {"node": "interrupt"}
                # ⚠️ 中断时必须**补发一个 interrupted 事件**，否则前端只看到流"卡住不动"，
                # 用户会以为程序死了（架构 §7 必踩坑②）。
                # payload 里带上**当前 state 的子问题**，前端才能立刻渲染"编辑子问题"面板 ——
                # 只报"中断了"而不给可编辑内容，等于没实现 HITL。
                payload: Any = None
                try:
                    snapshot = await graph.aget_state(config)
                    values = getattr(snapshot, "values", None) or {}
                    payload = {
                        "sub_questions": values.get("sub_questions"),
                        "stage": values.get("stage"),
                    }
                except Exception as exc:  # noqa: BLE001 - 拿不到 state 不该让流断掉
                    logger.warning("sse.interrupt_state_unavailable", error=str(exc)[:200])
                yield self.interrupted(
                    at="after_plan",
                    reason="用户在拆解后可编辑子问题",
                    payload=payload,
                ).to_sse()
                continue

            for node_name, update in chunk.items():
                yield self.node_started(node_name).to_sse()
                if isinstance(update, dict):
                    for ev in self.drain_state_events(update):
                        yield ev.to_sse()

        status = "interrupted" if interrupted else "ok"
        yield self.done(status=status).to_sse()

    # ------------------------------------------------------------------ 收尾
    async def flush_token_usage(self, by_node: dict[str, Any] | None = None) -> None:
        """流结束时把 token 统计刷出去。

        API 层在 `finally` 里调用它（架构 §7 的 SSE 实现要点）。这里只记日志，
        真正的落库在 `finalize` 节点完成 —— 因为**只有 finalize 知道 run 是否真的成功**，
        发射器在连接断开时也会走到这里，不该由它决定"要不要写库"。
        """
        if by_node:
            self._token_usage = by_node
        logger.info("sse.flushed", run_id=self.run_id, token_usage=self._token_usage)


__all__ = ["EVENT_TYPES", "SCHEMA_VERSION", "SSEEmitter", "SSEEvent", "event_from_state"]

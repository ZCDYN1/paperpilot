"""SSE 事件发射器（架构 §8）。

## 为什么把事件协议单独成模块、并强约束成枚举（架构 §8 的 15 种 + 第 16 种 `citation_rescue`）
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
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SCHEMA_VERSION = "1"

# 16 种事件类型：前 15 种与架构 §8 事件表逐条对应（顺序一致，便于核对）；
# 第 16 种 `citation_rescue` 是观测面补强（引用定位的救回/剔除统计），追加在末尾，
# 不打乱文档顺序。它必须**每次都发**（哪怕救回 0 条）—— 只在非零时出现的事件
# 无法区分"根因已修掉"与"统计断线了"，而这条统计存在的意义正是验证根因修复没有回退
# （见 `app/agents/nodes/evidence.py` 的 `Resolution.how`）。
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
    "citation_rescue",
)

_EVENT_TYPE_SET = frozenset(EVENT_TYPES)

# ---------------------------------------------------------------------------
# "带外事件"（out-of-band）：当前 SSE 流绑定 + 侧信道队列
# ---------------------------------------------------------------------------
# 为什么需要侧信道：事件通常由节点写进 `state["events"]`、由 `stream()` 取出。
# 但有些事实发生在**节点内部、且必须立刻让用户看到** —— 典型就是容错层：
# LLM 下游被熔断、本次调用被降级。这类信息没有 state 字段可放
# （为了它去改 `ResearchState` 与各节点的返回值，代价远大于收益），
# 所以给发射器开一条"直接投喂"的口子，`stream()` 每收到一个 chunk 就顺带 drain 一次。
#
# 为什么用 contextvar 而不是把 emitter 一路传参：`structured_invoke` 埋在节点实现里，
# 逐层透传 emitter 会污染所有函数签名。而 asyncio 任务创建时会**复制当前上下文**，
# 所以 `stream()`（API 请求任务）里绑定的 emitter，节点子任务天然能读到；
# 读不到时（脚本、单测直接调函数）就只剩结构化日志 —— 这是刻意的降级，绝不报错。
_current_emitter: ContextVar[SSEEmitter | None] = ContextVar("paperpilot_current_emitter", default=None)


def bind_emitter(emitter: SSEEmitter | None) -> Token:
    """绑定当前 SSE 流，返回用于 `unbind_emitter` 的 token。"""
    return _current_emitter.set(emitter)


def unbind_emitter(token: Token) -> None:
    """解绑当前流。**必须容忍跨 Context**：

    `stream()` 是异步生成器，它的 `finally` 可能在**另一个 Context** 里执行
    （调用方提前 break、生成器被 `aclose()`、或 LangGraph 在子任务里推进迭代器）。
    那时 `Token.reset()` 会抛 `ValueError: was created in a different Context` ——
    而这个异常原本只会污染"流已经跑完"的正常收尾，把一次成功的运行变成 500。
    兜底做法是显式写回 None：值层面的正确性拿到了，跨 Context 的 token 语义只能放弃。
    """
    try:
        _current_emitter.reset(token)
    except ValueError:
        logger.debug("sse.emitter_unbind_cross_context")
        _current_emitter.set(None)


def get_emitter() -> SSEEmitter | None:
    """当前上下文绑定的发射器（不在流内时为 None）。"""
    return _current_emitter.get()


def emit_out_of_band(event_type: str, **data: Any) -> bool:
    """向"当前这条流"投喂一条事件；没有流在跑就只打日志并返回 False。

    给容错/安全层用：它们不该为了发一条事件而依赖 API 层，也不该因为发不出事件而失败。
    """
    emitter = _current_emitter.get()
    if emitter is None:
        logger.debug("sse.no_stream_bound", event_type=event_type, **data)
        return False
    emitter.emit_out_of_band(event_type, **data)
    return True


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
        # 侧信道队列（见模块级 `emit_out_of_band` 的说明）。与 `_buffer` 分开是有意的：
        # `_buffer` 里存的是"已经由便捷方法 yield 出去过"的事件，混用会重复发帧。
        self._oob: list[SSEEvent] = []
        self._token_usage: dict[str, Any] = {}
        # 这条流最后停在哪：`ok` / `interrupted` / `error`。
        # API 层据此决定 `research_runs.status` —— 以前流一结束就写 completed，
        # 于是"停在 HITL 等用户确认"的 run 也显示成已完成（状态说谎）。
        self.last_status: str = "ok"

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

    def emit_out_of_band(self, event_type: str, **data: Any) -> SSEEvent:
        """不经 state、由任意层直接投喂一条事件（容错/安全层用）。

        仍然走 `emit()` 的类型校验：**事件协议是对外契约**，
        侧信道不是"可以随便加类型"的后门 —— 未知类型照样报错。
        """
        ev = self.emit(event_type, **data)
        # emit() 已经把事件放进 `_buffer`；侧信道要的是"马上被 stream() 取走"，
        # 所以把它从 `_buffer` 里挪过来，避免将来有人 drain() 时二次发出。
        if self._buffer and self._buffer[-1] is ev:
            self._buffer.pop()
        self._oob.append(ev)
        return ev

    def drain_out_of_band(self) -> list[SSEEvent]:
        """取走并清空侧信道队列。"""
        events, self._oob = self._oob, []
        return events

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

    # ----------------------------------------------------- 16 种事件的便捷方法
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

    def citation_rescue(
        self,
        node: str,
        *,
        checked: int,
        rescued: int,
        dropped: int = 0,
        recovered_by: dict[str, int] | None = None,
        drop_reasons: dict[str, int] | None = None,
    ) -> SSEEvent:
        """引用定位的救回统计（`Resolution.how` 三条路径 + verify 重锚的对外出口）。

        `node` 标明统计来自哪一步（read=抽取侧兜底，verify=校验侧重锚）；
        `rescued=0` 也必须发 —— 这条事件的用途是"随时验证根因修复没有回退"，
        只在非零时出现等于把回归藏起来。
        """
        return self.emit(
            "citation_rescue",
            node=node,
            checked=checked,
            rescued=rescued,
            dropped=dropped,
            recovered_by=recovered_by or {},
            drop_reasons=drop_reasons or {},
        )

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

        # 把这条流绑进当前上下文：节点子任务（asyncio 会复制上下文）里的容错层
        # 就能通过 `emit_out_of_band()` 直接给用户发事件（熔断打开 / 降级等）。
        token = bind_emitter(self)
        interrupted: dict[str, Any] | None = None
        try:
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
                else:
                    for node_name, update in chunk.items():
                        yield self.node_started(node_name).to_sse()
                        if isinstance(update, dict):
                            for ev in self.drain_state_events(update):
                                yield ev.to_sse()

                # 每个 chunk 之后清一次侧信道：容错/安全层投喂的事件在这里落地。
                # 放在 chunk 末尾（而不是每个事件都判断）能让"降级"紧跟在它发生的那一步之后出现。
                for ev in self.drain_out_of_band():
                    yield ev.to_sse()
        finally:
            unbind_emitter(token)

        self.last_status = "interrupted" if interrupted else "ok"
        yield self.done(status=self.last_status).to_sse()

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


__all__ = [
    "EVENT_TYPES",
    "SCHEMA_VERSION",
    "SSEEmitter",
    "SSEEvent",
    "bind_emitter",
    "emit_out_of_band",
    "event_from_state",
    "get_emitter",
    "unbind_emitter",
]

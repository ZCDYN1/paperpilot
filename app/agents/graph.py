"""LangGraph 图装配（架构 §4.2 + §4.3 的六个修正点）。

```
START → plan → ensure_corpus ─┬─(Send×N)→ retrieve → read → verify ─┬─pass──→ write → finalize → END
                              │                                       ├─revise→ revise ─(Send×M)→ retrieve ─┘
                              └─(corpus_ready=False)→ write ──────────┴─give_up→ write
```

## 六个修正点落在哪（逐条对照任务书）
1. **`plan → ensure_corpus`**；中断点 A = `interrupt_after=["plan"]`（**默认开**，
   产品核心卖点：用户可删改子问题）；中断点 B = `interrupt_before=["write"]`
   由 `settings.confirm_outline` 控制（**默认关**，否则"3 分钟"体验会退化成半自动）。
2. **并行 fan-out**：`g.add_conditional_edges("ensure_corpus", route_after_corpus, [...])`。
   该路由函数同时承担两件事：语料充足则 `Send` 扇出，不足则**降级直连 write**
   （§2.2 的"语料不足"分支）。
3. **`revise` 也必须用 `Send` 扇出**：`g.add_conditional_edges("revise", route_revision_workers, ["retrieve"])`。
   用普通边的话 `retrieve` 拿不到 `sub_question`（Send 注入的字段不在全局 state 上），
   会带着空子问题去检索 —— 不报错，只是结果随机。
4. **并行汇合**：普通边 + `operator.add` reducer。**实测**（langgraph 1.2.14，3 个子问题）
   `retrieve` 执行 3 次、`read` 执行 **1** 次、`retrieved` 聚合长度 3。
   另在 `read` 里加了幂等哨兵作为回归护栏，详见 `nodes/reader.py` 的模块注释。
5. **checkpointer 用 `AsyncSqliteSaver`**（不是同步 `SqliteSaver`）：FastAPI + SSE 全程异步，
   同步 checkpointer 会阻塞事件循环（ADR D12）。用 `open_checkpointer()` 拿上下文管理器。
6. **`MAX_REVISION` + `give_up` 出口**：`decide_after_verify` 读 `settings.max_revision`；
   恢复用 `await graph.ainvoke(None, config)`（传 `None` 而不是 `{}`，
   后者会从 START 重跑，白烧一遍 token）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from app.agents.nodes import (
    compare_node,
    ensure_corpus_node,
    finalize_node,
    plan_node,
    read_node,
    retrieve_node,
    revise_node,
    verify_node,
    write_node,
)
from app.agents.state import ResearchState
from app.core.config import get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# 条件边路由函数（不是节点）
# ---------------------------------------------------------------------------
def route_after_corpus(state: ResearchState) -> list[Send] | str:
    """语料保障之后：充足则扇出并行检索，不足则降级直接写作。

    ⚠️ 返回 `Send` 列表而不是节点名：`Send` 让每个子问题成为**图上独立的并行分支**，
    各自有 checkpoint / trace / 重试（架构 §4.3-①）。这是选 LangGraph 而非
    `asyncio.gather` 的核心理由 —— 一个分支挂了不会连坐，且 LangSmith 上能看到分支粒度。
    """
    if not state.get("corpus_ready", True):
        # 语料不足分支：不让 Agent 在空索引上硬编答案（架构 §2.2）
        logger.warning("graph.corpus_insufficient_degrade")
        return "write"

    sub_questions = state.get("sub_questions") or []
    return [
        Send("retrieve", {"sub_question": sq, "session_id": state.get("session_id", "")})
        for sq in sub_questions
    ]


def route_revision_workers(state: ResearchState) -> list[Send]:
    """`revise` 之后同样用 Send 扇出（修正点 3）。

    入参只带 `sub_question` + `session_id`：Send 的入参是**该分支的私有 state 覆盖**，
    多带字段没有意义（全局 state 本来就能读到），少带 `sub_question` 则是致命的。
    """
    revision_sub_questions = state.get("revision_sub_questions") or []
    return [
        Send("retrieve", {"sub_question": sq, "session_id": state.get("session_id", "")})
        for sq in revision_sub_questions
    ]


def decide_after_verify(state: ResearchState) -> Literal["pass", "revise", "give_up"]:
    """校验后的三岔口（架构 §4.3-③）。

    反思机制**必须有退出条件**，否则会无限重试烧 token。超过 `MAX_REVISION` 即
    `give_up` —— 降级输出，并在报告中显式标注哪些结论缺乏直接证据。
    反思能解决"偶尔漏检"，解决不了"文献里根本没有答案"，后者必须如实暴露。
    """
    verification = state.get("verification") or {}
    if verification.get("passed"):
        return "pass"
    if int(state.get("revision_count", 0)) >= get_settings().max_revision:
        logger.warning(
            "graph.give_up", revision_count=state.get("revision_count"), max_revision=get_settings().max_revision
        )
        return "give_up"
    return "revise"


# ---------------------------------------------------------------------------
# 装配
# ---------------------------------------------------------------------------
def build_graph(checkpointer: Any = None, *, settings: Any = None) -> Any:
    """装配并编译图。

    Args:
        checkpointer: **必须**是 `AsyncSqliteSaver`（见修正点 5）。传 None 时图能编译，
            但 HITL 中断**无法恢复**（没有 checkpoint 可续跑）—— 仅用于纯结构测试。
        settings: 覆盖配置（测试用）。默认读 `get_settings()`。
    """
    s = settings or get_settings()

    g: StateGraph = StateGraph(ResearchState)
    g.add_node("plan", plan_node)
    g.add_node("ensure_corpus", ensure_corpus_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("read", read_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)
    g.add_node("compare", compare_node)
    g.add_node("write", write_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "plan")
    # 修正点 1：中断点 A 挂在 plan 之后
    g.add_edge("plan", "ensure_corpus")
    # 修正点 2：并行 fan-out（同时承载"语料不足"降级分支）
    g.add_conditional_edges("ensure_corpus", route_after_corpus, ["retrieve", "write"])
    # 修正点 4：普通边 + reducer 汇合（实测 read 只执行 1 次）
    g.add_edge("retrieve", "read")
    g.add_edge("read", "verify")
    # 修正点 6：条件回边 + give_up 出口
    # pass 与 give_up 都先经过 compare：前者证据齐全、后者带"未找到支撑"的标注，
    # 两种情况下用户都需要那张对比表（缺的字段留空即可）。revise 仍走回边补检索。
    g.add_conditional_edges(
        "verify",
        decide_after_verify,
        {"pass": "compare", "revise": "revise", "give_up": "compare"},
    )
    # comparator：把已校验的证据组织成结构化对比表（产品定义的核心产出物 A）
    g.add_edge("compare", "write")
    # 修正点 3：revise 也必须 Send 扇出
    g.add_conditional_edges("revise", route_revision_workers, ["retrieve"])
    g.add_edge("write", "finalize")
    g.add_edge("finalize", END)

    # 修正点 1（续）：两个独立的 HITL 点，默认值不同
    # A（plan 之后）**恒开** —— 产品核心卖点：拆解错了后面全错，必须让用户能干预。
    # B（write 之前）由 CONFIRM_OUTLINE 控制，**默认关** —— 两个都开会把"3 分钟"拉成半自动。
    interrupt_after = ["plan"]
    interrupt_before = ["write"] if s.confirm_outline else []

    return g.compile(
        checkpointer=checkpointer,
        interrupt_after=interrupt_after,
        interrupt_before=interrupt_before,
    )


@asynccontextmanager
async def open_checkpointer(path: str | None = None) -> AsyncIterator[Any]:
    """打开 `AsyncSqliteSaver`（修正点 5）。

    为什么用 `from_conn_string`（异步上下文管理器）而不是自己 `aiosqlite.connect`：
    它内部把连接与 `setup()`（建 checkpoint 表）都管好了，且**支持 async 上下文**，
    在 FastAPI 的 lifespan 里可以安全地 acquire/release。
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    s = get_settings()
    s.ensure_dirs()
    conn = str(path or s.checkpoint_db)
    async with AsyncSqliteSaver.from_conn_string(conn) as saver:
        yield saver


@asynccontextmanager
async def open_graph(path: str | None = None, *, settings: Any = None) -> AsyncIterator[Any]:
    """便捷入口：打开 checkpointer 并编译好图，退出时自动释放。"""
    async with open_checkpointer(path) as saver:
        yield build_graph(saver, settings=settings)


def thread_config(run_id: str, *, emitter: Any = None) -> dict[str, Any]:
    """构造 LangGraph 的调用 config（`thread_id` 是断点续跑的钥匙）。

    客户端断开后重连时，只要带**同一个 thread_id** 走 resume，
    就能从最后一个 checkpoint 续跑，而不是从头发起把 token 白烧一遍（架构 §7 必踩坑③）。
    """
    configurable: dict[str, Any] = {"thread_id": run_id}
    if emitter is not None:
        configurable["emitter"] = emitter
    return {"configurable": configurable}


__all__ = [
    "build_graph",
    "decide_after_verify",
    "open_checkpointer",
    "open_graph",
    "route_after_corpus",
    "route_revision_workers",
    "thread_config",
]

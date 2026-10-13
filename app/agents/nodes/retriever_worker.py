"""`retrieve` 节点 —— 单子问题检索（**每个子问题一份，并行执行**）。

这个节点是 Send 扇出的落点，因此有两条极易踩坏的性质：

1. **它必须只读 `state["sub_question"]`**（Send 注入的单分支私有字段），
   不能去读 `state["sub_questions"]`（那是全量列表）。读全量的话每个分支都会
   检索全部子问题，N 倍 token 与 N² 倍的检索量 —— 而且结果"看起来也对"，
   只是慢和贵，靠 review 根本发现不了。
2. **它的返回值必须是 reducer 可聚合的形状**：`retrieved` 用 `operator.add`。
   并行分支同时写同一个 channel 时若没有 reducer，LangGraph 会抛
   `InvalidUpdateError`（这还算友好），而有 reducer 才能把 N 份结果拼起来。

⚠️ 它**不写 `stage`**：`stage` 没有 reducer，N 个并行分支同时写会直接冲突。
阶段标记由汇合后的 `read` 节点统一维护（这也是 §4.3-④ 幂等哨兵能成立的前提）。
"""

from __future__ import annotations

import time
import uuid

import structlog

from app.agents.nodes._common import node_event
from app.agents.state import ResearchState, SubQuestion
from app.agents.tools.hybrid_search import run_hybrid_search
from app.core.config import get_settings
from app.schemas.retrieval import HybridSearchArgs

logger = structlog.get_logger(__name__)


def _build_args(state: ResearchState, sub_question: SubQuestion) -> HybridSearchArgs:
    """把子问题 + 全局过滤条件拼成检索工具入参（经 Pydantic 强校验）。

    `HybridSearchArgs` 会把 `top_k>50`、年份倒置、越界分类这些**LLM 幻觉参数**
    挡在检索层之外（架构 §10「工具侧」）；把校验放在这里，
    错误会在第一时间以明确报错暴露，而不是变成"结果莫名其妙变少"。
    """
    settings = get_settings()
    filters = state.get("filters") or {}
    # 分类优先级：子问题自己的分类更具体，优先；否则退回 API 层传下来的全局过滤。
    category = sub_question.get("category")
    if not category:
        cats = filters.get("categories") or []
        category = cats[0] if len(cats) == 1 else None

    return HybridSearchArgs(
        query=sub_question.get("text") or state.get("question") or "",
        keywords=list(sub_question.get("keywords") or []),
        top_k=settings.rerank_top_n,
        year_gte=filters.get("year_gte"),
        year_lte=filters.get("year_lte"),
        category=category,
        sub_question_id=sub_question.get("id"),
    )


async def retrieve_node(state: ResearchState) -> dict:
    """对**单个**子问题执行一次混合检索。"""
    sub_question: SubQuestion | None = state.get("sub_question")
    if not sub_question:
        # Send 忘了注入 sub_question 时会走到这里。这正是架构 §4.2 点名的坑：
        # `revise` 若用普通边而非 Send，retrieve 就会拿到空子问题。
        # 这里显式返回空并记日志，让问题**可见**，而不是静默检索一个空串。
        logger.error("retrieve.missing_sub_question", hint="检查是否用了 Send 扇出")
        return {"retrieved": [], "events": []}

    args = _build_args(state, sub_question)
    call_id = f"ret-{sub_question.get('id', 'unknown')}-{uuid.uuid4().hex[:6]}"

    started = time.perf_counter()
    hits = await run_hybrid_search(args)
    duration_ms = int((time.perf_counter() - started) * 1000)

    logger.info(
        "retrieve.done",
        sub_question=sub_question.get("id"),
        papers=len(hits),
        duration_ms=duration_ms,
    )

    return {
        "retrieved": hits,
        # ⚠️ 这里**不能**返回 stage：它是无 reducer 的 channel，N 个并行分支同时写会抛
        #    InvalidUpdateError。阶段标记统一由汇合后的 read 节点维护。
        "events": [
            node_event(
                "tool_call",
                node="retrieve",
                tool="hybrid_search",
                args=args.model_dump(),
                call_id=call_id,
            ),
            node_event(
                "tool_result",
                call_id=call_id,
                status="ok",
                duration_ms=duration_ms,
                count=len(hits),
            ),
            node_event(
                "retrieval",
                sub_question_id=sub_question.get("id", ""),
                hits=[
                    {
                        "arxiv_id": h.get("arxiv_id"),
                        "title": h.get("title"),
                        "score": h.get("score"),
                        "section": (h.get("chunks") or [{}])[0].get("section"),
                        "source": (h.get("chunks") or [{}])[0].get("source"),
                    }
                    for h in hits
                ],
                count=len(hits),
            ),
        ],
    }


__all__ = ["retrieve_node"]

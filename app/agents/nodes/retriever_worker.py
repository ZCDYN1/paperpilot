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

    分类只取**子问题自己的**；API 传下来的全局分类走 `_build_extra_filters`。
    此前这里写的是"全局恰好 1 个分类才塞进单值 category"，多分类会被**静默整组丢弃**
    —— F2 承诺的多分类过滤实际从未生效过。`HybridSearchArgs.category` 是单值契约
    （LLM 工具入参，宁缺毋滥），多值过滤属于全局通道，不该挤进这个字段。

    `top_k` / `rerank` 优先取 **state**（请求级，由 `_initial_state` 转发），没传才退回配置。
    之前这两个只读 `settings`，于是 `ResearchRequest.top_k` 与 `.rerank` 是纯粹的装饰字段：
    用户把重排开关拨到 off，链路照旧重排，而且从 SSE 事件到日志都看不出差别。
    """
    settings = get_settings()
    filters = state.get("filters") or {}
    rerank_flag = state.get("rerank")
    return HybridSearchArgs(
        query=sub_question.get("text") or state.get("question") or "",
        keywords=list(sub_question.get("keywords") or []),
        # ⚠️ 这里给的是"最终留几篇"（`run_hybrid_search` 把它映射到 rerank_top_n），
        # 每一路的召回量由 `RETRIEVE_TOP_K` 决定，两者不要混。
        top_k=int(state.get("top_k") or settings.rerank_top_n),
        # None / 缺失 = 请求没表态，按"重排开"处理，与加这个字段之前的行为一致。
        # 全局 `RERANK_ENABLED=false` 仍然压得住：那时拿到的就是 NullReranker。
        rerank=True if rerank_flag is None else bool(rerank_flag),
        year_gte=filters.get("year_gte"),
        year_lte=filters.get("year_lte"),
        category=sub_question.get("category"),
        has_code=filters.get("has_code"),
        sub_question_id=sub_question.get("id"),
    )


def _build_extra_filters(state: ResearchState) -> dict:
    """全局过滤里 `HybridSearchArgs` 单值契约装不下的部分（多分类 / 指定论文集合）。

    `build_filter` 的合并语义是"args 里已有值就不覆盖"：子问题的单分类仍然胜出
    （更具体的约束优先，架构 §5.2），这里只补它没占住的键。
    """
    filters = state.get("filters") or {}
    return {
        "categories": list(filters.get("categories") or []) or None,
        "arxiv_ids": list(filters.get("arxiv_ids") or []) or None,
    }


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
    hits = await run_hybrid_search(args, extra_filter=_build_extra_filters(state))
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
                # 全局多分类 / F14 论文集合走的是 extra_filter，不记进事件就排查不了
                # "过滤到底有没有生效"这类问题（事件是 SSE 调试面板的数据源）。
                extra_filters=_build_extra_filters(state),
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

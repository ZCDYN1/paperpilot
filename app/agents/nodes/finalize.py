"""`finalize` 节点 —— 统计 token、落库、写长期记忆（架构 §4.2 / §12）。

## 一句话原则：**收尾失败不能毁掉整次研究**
用户已经拿到报告了（`write` 节点产出的 `draft` 就在 state 里），
这时候因为"写 token_usage 表失败"而让整个 SSE 流以 `error` 收场，
是典型的**用次要功能绑架主要功能**。所以这里的每一步副作用都单独 try/except，
失败只记日志 + 发一个 `recoverable=True` 的 error 事件。

## 落库的口径
- `research_runs.langgraph_state` **只存轻量摘要**（论文数/引用数/token），
  不存全量 state —— 全量里含 retrieved/readings 全文，整包塞进 SQLite 会让库迅速膨胀。
  完整状态由 LangGraph 的 checkpointer 负责（架构 §6 的注释）。
- `citations` 独立成表：让"引用可溯源"变成可查询、可校验的事实。
- `token_usage` **按节点各落一行**（`node_name="plan"/"read"/...`）外加一行 `all` 总量：
  架构 §12 要的是"成本瓶颈在哪个节点"，只落一条总量答不了这个问题。

## 记忆（架构 §9 / 产品定义 F10）
`finalize` 是长期记忆的**写入时机**（架构 §9 原话）。这里同时落两类记忆：
- 长期：本次关注的子方向（问题）、已读论文 id、输出偏好（如产出了对比表）；
- 短期：把本轮"用户提问 + 最终报告"追加进 `messages` 表，供"会话内追问"跨 run 复用。
两者都是副作用，各自 try/except —— 记忆丢了可以下次补，报告丢了不能重来。
"""

from __future__ import annotations

import structlog

from app.agents.nodes._common import node_event
from app.agents.state import STAGE_DONE, ResearchState
from app.core.llm import llm_identity
from app.observability.token_meter import get_token_meter

logger = structlog.get_logger(__name__)

# 粗略单价（美元 / 1K tokens），仅用于"成本可量化展示"的数量级参考。
# 真实报价随 provider 与模型版本变动很大，务必按 .env 里选的模型自行校准 ——
# 这里给的是数量级，不是账单。
_COST_PER_1K_PROMPT = 0.00015
_COST_PER_1K_COMPLETION = 0.0006


def _estimate_cost(usage: dict[str, int]) -> float:
    prompt = int(usage.get("prompt_tokens", 0))
    completion = int(usage.get("completion_tokens", 0))
    return round(prompt / 1000 * _COST_PER_1K_PROMPT + completion / 1000 * _COST_PER_1K_COMPLETION, 6)


async def _persist(state: ResearchState, usage: dict[str, int], by_node: dict | None = None) -> None:
    """把 run / citations / token_usage 写库（幂等 upsert）。

    `by_node` 是 token_meter 汇总出的分节点用量：除了原有的 `all` 总量行，
    每个节点再单独落一行 —— 这样"哪个节点最烧钱"是一条 SQL 就能查出来的事实，
    而不是埋在 JSON 里靠肉眼翻。
    """
    from sqlalchemy import select

    from app.db.models import Citation, PaperChunk, ResearchRun, ResearchSession, TokenUsage
    from app.db.session import init_db, session_scope

    await init_db()
    run_id = state.get("run_id") or state.get("session_id") or "run-unknown"
    session_id = state.get("session_id") or run_id
    citations = list(state.get("citations") or [])
    by_node = by_node or {}

    async with session_scope() as session:
        # 会话必须先存在：research_runs.session_id 有外键约束（SQLite 已开 foreign_keys=ON）
        if (await session.get(ResearchSession, session_id)) is None:
            session.add(ResearchSession(id=session_id, title=(state.get("question") or "")[:200]))

        run = await session.get(ResearchRun, run_id)
        summary = {
            "papers": len({p.get("arxiv_id") for p in (state.get("retrieved") or [])}),
            "claims": len(state.get("readings") or []),
            "citations": len(citations),
            "verified": sum(1 for c in citations if c.get("verified")),
            "token_usage": usage,
            "token_by_node": by_node,
            "corpus_ready": state.get("corpus_ready"),
        }
        if run is None:
            session.add(
                ResearchRun(
                    id=run_id,
                    session_id=session_id,
                    thread_id=state.get("session_id") or run_id,
                    question=state.get("question") or "",
                    status="done",
                    langgraph_state=summary,  # 只存轻量摘要（架构 §6）
                )
            )
        else:
            run.status = "done"
            run.langgraph_state = summary

        # ⚠️ chunk_id 有外键约束：引用指向一个库里不存在的 chunk 会直接抛错。
        # 先取一次合法的 chunk id 集合，越界的一律置 None ——
        # 宁可丢一个"定位字段"，也不能让收尾整体失败。
        valid_chunk_ids = {
            int(x) for x in (await session.execute(select(PaperChunk.id))).scalars().all()
        }
        for cit in citations:
            # ⚠️ verify 纠正过定位的引用，真实片段在 `resolved_chunk_id` 里（键名原因见
            # `app/agents/state.py` 的 Evidence）：优先采用它，否则这条引用会落成 NULL，
            # `/citations` 与人工抽检里的原文列就是空的。
            cid = cit.get("resolved_chunk_id") or cit.get("chunk_id")
            session.add(
                Citation(
                    run_id=run_id,
                    claim=cit.get("claim") or "",
                    chunk_id=int(cid) if cid is not None and int(cid) in valid_chunk_ids else None,
                    arxiv_id=cit.get("resolved_paper_id") or cit.get("paper_id"),
                    quote=cit.get("quote") or "",
                    locator=cit.get("resolved_locator") or cit.get("locator"),
                    source=cit.get("resolved_source") or cit.get("source"),
                    verified=bool(cit.get("verified")),
                    # 校验失败原因（硬校验/软校验的具体说明）落进 note，供人工抽检口径核对；
                    # 通过的引用没有失败原因，保持 None 而不是空串，便于区分"没写"与"写了但为空"。
                    note=(cit.get("verify_note") or None) if not cit.get("verified") else None,
                )
            )

        session.add(
            TokenUsage(
                run_id=run_id,
                node_name="all",
                model=llm_identity()["model"],
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                cost_estimate=_estimate_cost(usage),
            )
        )
        # 分节点行：one row per node，让"成本瓶颈在哪个节点"可直接 SQL 聚合查询。
        # 跳过 "calls" 这类非 token 字段（表里没有对应列）。
        for node_name, bucket in by_node.items():
            node_usage = {
                "prompt_tokens": int(bucket.get("prompt_tokens", 0)),
                "completion_tokens": int(bucket.get("completion_tokens", 0)),
            }
            session.add(
                TokenUsage(
                    run_id=run_id,
                    node_name=str(node_name)[:32],
                    model=llm_identity()["model"],
                    prompt_tokens=node_usage["prompt_tokens"],
                    completion_tokens=node_usage["completion_tokens"],
                    cost_estimate=_estimate_cost(node_usage),
                )
            )


def _read_paper_ids(state: ResearchState) -> list[str]:
    """本次**实际精读**过的论文 id。

    为什么不用 `retrieved`（检索到的全部）：产品的语义是"已读论文清单，避免下次重复精读"，
    而检索结果里大部分论文只是被召回、并没进 Reader。用 `readings` 才名副其实。
    `readings` 为空（例如语料不足降级）时退回 `retrieved`，让"至少读过些什么"仍有记录 ——
    有记录比精确更重要，何况降级场景下 retrieved 就是全部可得信息。
    """
    read_ids = [r.get("paper_id") for r in (state.get("readings") or []) if r.get("paper_id")]
    if read_ids:
        return sorted(set(read_ids))
    return sorted({p.get("arxiv_id") for p in (state.get("retrieved") or []) if p.get("arxiv_id")})


async def _write_memory(state: ResearchState) -> None:
    """写长期 + 短期记忆（架构 §9 / 产品定义 F10）。

    落三类长期记忆（关注方向 / 已读论文 / 输出偏好）+ 一轮短期对话：
    - 关注方向、已读论文走 `long_term.remember_run`（它内部再分型并各自容错）；
    - 输出偏好单独写：只有本次真的产出了对比表才写"偏好表格对比"，
      避免把"偶然一次"当成"长期偏好"。
    每段都独立 try/except —— 记忆是"下次更好"，绝不该拿它绑架本次结果。
    """
    session_id = state.get("session_id")
    run_id = state.get("run_id")
    draft = state.get("draft") or ""

    try:
        from app.memory import long_term
    except Exception:  # noqa: BLE001 - 记忆模块不可用则整体跳过
        logger.info("finalize.memory_skipped", reason="long_term 模块未就绪")
    else:
        try:
            await long_term.remember_run(
                session_id=session_id,
                run_id=run_id,
                question=state.get("question"),
                paper_ids=_read_paper_ids(state),
                summary=draft[:500],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("finalize.memory_write_failed", error=str(exc)[:200])

        # 输出偏好只在"确实用了对比表"时记：把偶发当偏好会让记忆很快变成噪声。
        if state.get("comparison"):
            try:
                await long_term.remember(
                    "用户偏好：研究结论以「对比表 + 引用列表」形式呈现。",
                    session_id=session_id,
                    memory_type=long_term.TYPE_PREFERENCE,
                    importance=0.7,
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("finalize.preference_write_failed", error=str(exc)[:200])

    # 短期记忆：把本轮问答追加进会话，供"会话内追问"跨 run 复用（架构 §9）。
    try:
        from app.memory import short_term

        await short_term.append_message(
            session_id or run_id or "session-unknown",
            "user",
            state.get("question") or "",
            run_id=run_id,
            node_name="finalize",
        )
        if draft:
            await short_term.append_message(
                session_id or run_id or "session-unknown",
                "assistant",
                draft,
                run_id=run_id,
                node_name="writer",
            )
    except Exception as exc:  # noqa: BLE001 - 短期记忆落库失败同样不该中断收尾
        logger.warning("finalize.short_term_write_failed", error=str(exc)[:200])


async def finalize_node(state: ResearchState) -> dict:
    """统计 token、落库、写长期记忆。"""
    usage = dict(state.get("token_usage") or {})
    meter = get_token_meter()
    by_node = meter.snapshot()  # ★ 按节点细分：{plan/read/write/...: {prompt_tokens, ...}}
    cost = _estimate_cost(usage)
    errors: list[dict] = []

    try:
        await _persist(state, usage, by_node)
    except Exception as exc:  # noqa: BLE001 - 收尾失败不能毁掉整次研究
        logger.warning("finalize.persist_failed", error=str(exc)[:300])
        errors.append({"step": "persist", "error": str(exc)[:200]})

    await _write_memory(state)

    logger.info(
        "finalize.done",
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cost=cost,
        by_node=list(by_node),
        errors=len(errors),
    )

    events = [
        node_event(
            "token_usage",
            total=usage,
            cost_estimate=cost,
            by_node=by_node,  # 按节点细分（架构 §12）：前端可据此画"成本瓶颈"条形图
        )
    ]
    for err in errors:
        events.append(
            node_event(
                "error",
                node="finalize",
                code="persist_failed",
                message=err["error"],
                recoverable=True,
            )
        )

    return {
        "stage": STAGE_DONE,
        # ★ 写进 state 供上层（API/评测/UI）读取；该字段无 reducer，
        #   而 finalize 只会执行一次，所以一次性覆盖语义是正确的。
        "token_by_node": by_node,
        "events": events,
    }


__all__ = ["finalize_node"]

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
- `comparison_rows` 独立成表（每行一篇论文）：让"对比表字段完整率"变成可统计的事实，
  也让旧 run 的表格能取回来。留空语义原样入库，不做任何填充。
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
    """把 run / citations / comparison_rows / token_usage 写库（幂等 upsert）。

    `by_node` 是 token_meter 汇总出的分节点用量：除了原有的 `all` 总量行，
    每个节点再单独落一行 —— 这样"哪个节点最烧钱"是一条 SQL 就能查出来的事实，
    而不是埋在 JSON 里靠肉眼翻。

    对比表逐行落 `comparison_rows`：产品定义 §8 的「字段完整率」此前**没有可测数据**
    （行只活在 graph state 与 SSE 帧里），旧 run 的表格也取不回。落库口径照抄
    `ComparisonRow` 的留空语义 —— 空串/None 原样存"没找到依据"，不做任何兜底填充。
    """
    from sqlalchemy import delete, select

    from app.db.models import (
        Citation,
        ComparisonRecord,
        PaperChunk,
        ResearchRun,
        ResearchSession,
        TokenUsage,
    )
    from app.db.session import init_db, session_scope

    await init_db()
    run_id = state.get("run_id") or state.get("session_id") or "run-unknown"
    session_id = state.get("session_id") or run_id
    citations = list(state.get("citations") or [])
    comparison = list(state.get("comparison") or [])
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
            "comparison_rows": len(comparison),
        }
        # 表级信息（生成时间 / 缺失说明）不进 comparison_rows —— 那张表的契约是
        # "每行一篇论文"，而这两项是**整表**的属性。`comparison` state 只装得下行，
        # 它们被 compare 节点塞进了 thought 事件（见 comparator.compare_node 的同名注释），
        # 这里从事件里取回来放进轻量摘要，否则"表何时生成、哪些列普遍缺失"会随 thread 一起丢。
        for event in reversed(list(state.get("events") or [])):
            if "comparison_notes" in event or "comparison_generated_at" in event:
                summary["comparison_generated_at"] = event.get("comparison_generated_at")
                summary["comparison_notes"] = event.get("comparison_notes")
                break
        # ⚠️ 接住 API 层在中断点写下的决策位：本函数用**新摘要整体覆盖** langgraph_state
        # （JSON 列必须赋新值才触发 UPDATE）。生产方在 resume 入口与流结束后各写一次
        # （`app/api/v1/research.py::_record_plan_edit`），中间恰好隔着本次覆盖 ——
        # 不接住的话"进程在两段之间死掉"就会把信号整个吞掉，而 §8 可用率只看跑完的 run。
        # 流结束后生产方会用权威值再覆盖一次，这里带旧值不会让它变陈旧。
        if isinstance(run, ResearchRun) and isinstance(run.langgraph_state, dict):
            for _k in ("plan_edited", "plan_edit_detail"):
                if _k in run.langgraph_state:
                    summary[_k] = run.langgraph_state[_k]
        if run is None:
            session.add(
                ResearchRun(
                    id=run_id,
                    session_id=session_id,
                    # ★ thread_id 必须是 run 级，和 `research.py::_register_run` 同一条不变量。
                    # 这里曾经写成 `session_id or run_id`：走"API 层没登记、由 finalize 补插行"
                    # 的路径时，同一会话的多个 run 会共用一个 checkpoint thread，
                    # 于是第二次提问不是从 START 起跑，而是接着上一轮已跑完的 state 走 ——
                    # 表现为 readings/citations 被上一轮污染（不报错，只是结果不对）。
                    thread_id=run_id,
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
                    # 校验说明落进 note，供人工抽检口径核对。两种内容共用同一字段：
                    # ① 失败时的"为什么没过"（硬校验/软校验的具体原因）；
                    # ② 通过时的"怎么对上的" —— 只有靠抽取残留容忍才通过的引用才非空
                    #    （精确通过保持空串→None），这样"这轮有多少条是归一化救回来的"
                    #    可以直接对表统计，不必 grep 服务端日志。
                    # 保持 None 而不是空串，便于区分"没写"与"写了但为空"。
                    note=cit.get("verify_note") or None,
                    # 定位路径与"verify 是否重锚过"落库：此前这两个事实只在后端日志里，
                    # `/citations` 与抽检工具都看不到"这条引用是靠哪条兜底路径救回来的"。
                    # resolved_chunk_id 只有 verify 真的纠正过定位时才存在（见 verifier._anchor）。
                    resolution_how=cit.get("resolution_how"),
                    verify_reanchored=bool(cit.get("resolved_chunk_id") is not None),
                )
            )

        # ---- 对比表落库（产品定义 §8「字段完整率」的数据源）----
        # 先删本 run 旧行再插新行：run 恢复重跑时 finalize 可能执行第二次，
        # 表里留"最新一次的表"才对得上报告；delete+insert 比逐行 upsert 简单且幂等。
        # 空串 / None 一律原样存 —— 那是「没找到依据」的产品承诺，不是缺失数据。
        await session.execute(delete(ComparisonRecord).where(ComparisonRecord.run_id == run_id))
        for idx, row in enumerate(comparison):
            session.add(
                ComparisonRecord(
                    run_id=run_id,
                    row_index=idx,
                    arxiv_id=str(row.get("arxiv_id") or ""),
                    title=str(row.get("title") or ""),
                    method_name=str(row.get("method_name") or ""),
                    core_idea=str(row.get("core_idea") or ""),
                    datasets=str(row.get("datasets") or ""),
                    improvement=str(row.get("improvement") or ""),
                    cost=str(row.get("cost") or ""),
                    limitation=str(row.get("limitation") or ""),
                    code_url=str(row.get("code_url") or ""),
                    venue=str(row.get("venue") or ""),
                    cited_by_count=(
                        int(row["cited_by_count"])
                        if isinstance(row.get("cited_by_count"), int)
                        else None
                    ),
                    cited_at=(str(row["cited_at"]) if row.get("cited_at") else None),
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

    ⚠️ `readings` 为空时**如实返回空**，不退回 `retrieved`。
    退回等于把"这次一篇都没读成"（抽取失败、语料不足降级）记成"这 24 篇全读过了"。
    在已读清单还没被任何地方消费时这只是不精确；但 `read` 节点现在会用它排除候选
    （产品定义 F10），一次降级运行就能把下一次的候选压光，形成持续拿不到候选的负反馈。
    降级场景下"读过什么"这个信息本来就不存在，不编造比编造有用。
    """
    return sorted({r.get("paper_id") for r in (state.get("readings") or []) if r.get("paper_id")})


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

"""`plan` 节点 —— 研究问题 → 3~5 个子问题（架构 §4.2）。

三条设计要点：

1. **结构化输出**：用 Pydantic schema 约束，而不是"让模型输出 JSON 再正则解析"。
   后者在模型多说一句"好的，以下是"时就会崩，且崩在运行期而不是编译期。
2. **关键词必须英文**：这由 `app.schemas.research.SubQuestion` 的 field_validator 兜底。
   但校验失败会**抛出异常**，所以节点里必须给它一条生路 —— 见 `_salvage`。
   否则一个中文关键词就能让整次研究直接失败，而用户看到的只是一个红字错误。
3. **节点不做检索**：拆解与检索是两个阶段（架构 §2.2），
   把 arXiv 抓取放在 `ensure_corpus`，plan 保持"纯 LLM、秒级完成"的定位 ——
   这样中断点 A（用户编辑子问题）才是真的"秒回"。

## 长期记忆接入（架构 §9，产品定义 F10）
架构 §9 规定"召回时机：`plan` 节点开始时，检索与当前问题相关的记忆注入 prompt"。
落地在这里做两件事：
- 调 `long_term.recall(question, top_k=N)` 召回相关记忆，写进 `state["memory_context"]`；
- 把记忆渲染成一段上下文，**注入 planner 的 prompt**（"用户已读过的论文……请避免重复推荐"），
  让拆解真的受历史影响，而不是"召回了却没人看"。

⚠️ 记忆召回失败**绝不能**让 plan 挂掉：plan 是整条链的起点，它挂了整次研究就没了。
所以这里对"模块不可用"和"召回抛异常"分别 try/except，一律降级为空列表继续。
这一条与 finalize 的"收尾失败不能毁掉整次研究"是同一条原则的两端。
"""

from __future__ import annotations

import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import node_event, structured_invoke
from app.agents.prompts import load_prompt
from app.agents.schemas import PlanOutput
from app.agents.state import STAGE_PLANNED, ResearchState, SubQuestion

logger = structlog.get_logger(__name__)

_CJK_RE = re.compile(r"[\u3000-\u9fff]")

# 召回多少条长期记忆。定小是有意的：记忆是"辅助上下文"，条数一多就会稀释当前问题的信号
# （架构 §9 点名的"越用越吵"）。5 条足够覆盖"方向 / 已读 / 偏好"三类。
DEFAULT_MEMORY_TOP_K = 5


async def _recall_memory(question: str, session_id: str | None) -> list[str]:
    """召回与当前问题相关的长期记忆；**任何失败都降级为空列表**。

    分两层 try/except 而不是包一层：这样"模块压根没就绪"与"模块在但召回出错"
    在日志里是两条不同的信息，排查时不用猜。
    """
    if not question:
        return []
    try:
        from app.memory import long_term
    except Exception:  # noqa: BLE001 - 记忆模块不可用不应阻断拆解
        logger.info("planner.memory_unavailable")
        return []
    try:
        hits = await long_term.recall(question, top_k=DEFAULT_MEMORY_TOP_K, session_id=session_id)
    except Exception as exc:  # noqa: BLE001 - 召回失败同样不该阻断拆解
        logger.warning("planner.memory_recall_failed", error=str(exc)[:200])
        return []
    memories = [str(h) for h in (hits or []) if h]
    if memories:
        logger.info("planner.memory_recalled", count=len(memories))
    return memories


def _memory_block(memories: list[str]) -> str:
    """把召回的记忆渲染成注入 prompt 的上下文块。

    ⚠️ 这段文本**必须放在 `研究问题：` 这一行之前**：mock 模型靠 `rfind("研究问题：")`
    截取"提问+文档"来抽英文关键词，把记忆放在提问之后会让记忆里的中文/杂词污染关键词。
    放在前面则天然被截掉，既完成了注入、又不影响关键词质量。
    """
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "以下是该用户的历史长期记忆（拆解子问题时请参考，"
        "并**避免重复推荐用户已读过的论文**）：\n"
        f"{lines}\n\n"
    )


def _salvage(sub_questions: list[SubQuestion], question: str) -> list[SubQuestion]:
    """把不合规的子问题"救"回来，而不是让整次研究失败。

    具体做两件事：
    1. 过滤掉中文关键词（BM25 用不了，留着只会注水）；
    2. 若某子问题一个英文关键词都不剩，则用**问题本身**里的英文词补位
       （用户问 "RAG 重排" 时，`RAG` 就是可用的英文词）。

    为什么不直接把校验失败的子问题丢掉：拆解数量直接决定并行度，
    丢到少于 3 个就失去了 Manager-Worker 的意义。补位比丢弃划算。
    """
    fallback = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", question)][:5]
    fixed: list[SubQuestion] = []
    for sq in sub_questions:
        kws = [k for k in (sq.get("keywords") or []) if k and not _CJK_RE.search(k)]
        if not kws:
            kws = fallback or ["retrieval"]
            logger.warning("planner.keywords_fallback", sub_question=sq.get("id"), keywords=kws)
        fixed.append({**sq, "keywords": kws})
    return fixed


async def plan_node(state: ResearchState) -> dict:
    """把研究问题拆成 3~5 个带英文关键词的子问题。

    返回的 `sub_questions` 会被 `route_to_workers` 用 Send 扇出成 N 个并行检索分支。
    """
    question = (state.get("question") or "").strip()
    if not question:
        # 没有研究问题就没有可拆解的东西。这里不抛异常，而是给一个单子问题兜底 ——
        # 让图继续跑出"无有效输入"的报告，比直接抛栈更容易定位问题（栈会淹没在 SSE 里）。
        logger.warning("planner.empty_question")
        sub_questions: list[SubQuestion] = [
            {"id": "sq-1", "text": "（缺少研究问题）", "keywords": ["retrieval"], "category": None}
        ]
        return {
            "sub_questions": sub_questions,
            "plan_rationale": "输入为空，已降级为单子问题。",
            "memory_context": [],  # 没有提问就无从召回；显式置空，避免下游读到 None
            "stage": STAGE_PLANNED,
            "revision_count": 0,
            "events": [node_event("plan_created", sub_questions=sub_questions, count=1)],
        }

    # ★ 长期记忆召回：放在调 LLM 之前，召回结果同时写进 state 与 prompt（架构 §9）。
    memories = await _recall_memory(question, state.get("session_id"))
    messages = [
        SystemMessage(content=load_prompt("planner")),
        # 记忆块在前、`研究问题：` 在后（见 `_memory_block` 的注释）。
        HumanMessage(content=f"{_memory_block(memories)}研究问题：{question}"),
    ]
    parsed, usage = await structured_invoke(PlanOutput, messages)

    sub_questions = _salvage([sq.model_dump() for sq in parsed.sub_questions], question)
    logger.info("planner.done", count=len(sub_questions), memories=len(memories))

    return {
        "sub_questions": sub_questions,
        "plan_rationale": parsed.rationale,
        # 把召回结果透出到 state：finalize 可据此判断"记忆确实被用上了"，
        # 也让 SSE/评测能观测到"这一轮参考了哪些历史"。
        "memory_context": memories,
        # revision_count 归零：这是新一次研究的起点；若沿用上一轮的值，
        # 第一次校验失败就会直接命中 give_up，反思机制形同虚设。
        "revision_count": 0,
        "stage": STAGE_PLANNED,
        "token_usage": usage,
        "events": [
            node_event(
                "plan_created",
                sub_questions=sub_questions,
                rationale=parsed.rationale,
                count=len(sub_questions),
                memory_count=len(memories),  # 让前端能显示"参考了 N 条历史记忆"
            )
        ],
    }


__all__ = ["plan_node"]

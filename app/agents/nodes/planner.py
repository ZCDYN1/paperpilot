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

## 短期记忆接入（产品定义 F9「会话内追问」）
"那方法 B 在小数据集上表现如何？"这句话单独看是**无法检索**的 —— B 是什么、
"小数据集"指哪批数据集，全在同一个会话的前一轮里。所以 plan 节点还要读本会话的历史问答
（`finalize` 每轮都把"提问 + 报告"写进 `messages` 表，数据一直都在，缺的只是读取这一半）。

注入点选 plan 而不是 write，理由是**这里改一处、后面全链都受益**：
子问题一旦被补全成自包含的表述（"方法 B = cross-encoder reranking 在小数据集上的表现"），
retrieve / read / verify / compare / write 拿到的就都是明确的检索任务，
不需要每个节点各自再灌一遍历史。反过来只喂 write 的话，检索阶段仍然按"那方法 B"去查，
召回的是噪声，历史上下文只能在一份已经跑偏的证据集上打转。
代价是 plan 的 prompt 变长（有预算上限，见 `_HISTORY_MAX_CHARS`）。

⚠️ 长期记忆**不看 session_id**（它属于用户、跨会话，见 `app/memory/long_term.py` 的★小节），
短期记忆**才**是按 session_id 取的。两者别混：把长期记忆改回按会话过滤，
在 F9 接线之后症状会从"恒为空"变成"只在本会话内为空"，依旧静默、依旧全绿。
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

# 会话历史的注入预算（窗口/截断长度）收在 `app/memory/short_term.py`：
# 读侧现在有两个消费方，预算只能有一处定义，理由见那边的注释。


async def _recall_memory(question: str) -> list[str]:
    """召回与当前问题相关的长期记忆；**任何失败都降级为空列表**。

    分两层 try/except 而不是包一层：这样"模块压根没就绪"与"模块在但召回出错"
    在日志里是两条不同的信息，排查时不用猜。

    ⚠️ **跨会话召回，不按 `session_id` 过滤**：`research.py` 里 `session_id = run_id`，
    而记忆要等 `finalize` 才写入，按当前会话过滤必然拿到空集 —— 那样"长期记忆已接线"
    就只是接了线、读不到东西，而且单测全绿（它们把 recall 打了桩），靠 review 发现不了。
    详见 `app/memory/long_term.py` 模块 docstring 的★小节。
    """
    if not question:
        return []
    try:
        from app.memory import long_term
    except Exception:  # noqa: BLE001 - 记忆模块不可用不应阻断拆解
        logger.info("planner.memory_unavailable")
        return []
    try:
        hits = await long_term.recall(question, top_k=DEFAULT_MEMORY_TOP_K)
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


async def _recall_session_history(session_id: str) -> list[dict]:
    """读本会话此前的问答（短期记忆，F9）。实现收在 `app/memory/short_term.recall_history`。

    这里只剩一层薄委托，因为**同一个读取现在有两个消费方**：`plan` 拆子问题、
    `followup` 答追问。窗口大小、摘要行标记、降级原则各写一份的话，
    表现就是"拆解时记得住、追问时记不住"这种没法解释的差别。
    保留函数名是因为有用例按这个名字打桩（`tests/test_planner_parse_degradation.py`）。

    降级原则与 `_recall_memory` 一致：历史是辅助信息，读不到最多是"这次没带上下文"，
    但 plan 一挂整次研究就没了 —— 两者的代价差着量级。
    """
    try:
        from app.memory import short_term
    except Exception:  # noqa: BLE001 - 记忆模块不可用不应阻断拆解
        logger.info("planner.short_term_unavailable")
        return []
    history = await short_term.recall_history(session_id)
    if history:
        logger.info("planner.history_recalled", session_id=session_id, messages=len(history))
    return history


def _history_block(history: list[dict]) -> str:
    """渲染会话历史上下文块（放在 `研究问题：` 之前，理由见 `_memory_block`）。

    标签包裹与摘要行标注这两条边界都定义在 `short_term.render_history_block` 里，
    那边有完整说明；这里不重复，免得两处口径分叉。
    """
    from app.memory.short_term import render_history_block

    return render_history_block(history)


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
            "events": [
                # history/memory 计数恒为 0 也要带上：事件字段"只在非零时出现"会让前端
                # 分不清"首轮提问"与"上下文读取断了线"。
                node_event(
                    "plan_created",
                    sub_questions=sub_questions,
                    count=1,
                    memory_count=0,
                    history_count=0,
                )
            ],
        }

    # ★ 两个上下文块都在调 LLM 之前取，且都排在 `研究问题：` 之前（原因见 `_memory_block`）。
    # 顺序是"会话历史 → 长期记忆 → 研究问题"：本轮指代（"那方法 B"）只有会话历史能解，
    # 所以它离提问最近；长期记忆是跨会话的用户事实，负责"别重复推荐"。
    session_id = (state.get("session_id") or "").strip()
    history = await _recall_session_history(session_id)
    memories = await _recall_memory(question)
    messages = [
        SystemMessage(content=load_prompt("planner")),
        HumanMessage(
            content=(
                f"{_history_block(history)}{_memory_block(memories)}研究问题：{question}"
            )
        ),
    ]
    parsed, usage = await structured_invoke(PlanOutput, messages)
    degraded = parsed is None
    if degraded:
        # `include_raw=True` 下"模型输出不合规"**不抛异常**，只把 `parsed` 置成 None。
        # 必须在这里就当失败处理：否则下一行 `parsed.sub_questions` 抛 AttributeError，
        # 把一个本可降级的事件变成整次研究 500（reader 已按同一口径处置）。
        #
        # 只降级这一种情况。**传输层异常照原样上抛**（超时/熔断/连不上）：那是
        # 容错层的产物 —— 它要给流里送 `internal_error` / `llm_circuit_open`，
        # 让用户看到"下游挂了、稍后重试有意义"。在这里一并接住的话，
        # 用户会拿到一份"模型根本没跑起来"的报告，而熔断器明明告诉他现在别试。
        logger.warning("planner.degraded", reason="structured_output_unparsed")
        rationale = "模型未产出合规拆解，已降级为「原问题单路检索」。"
        # 兜底走 `_salvage` 而不是自己拼关键词：补位规则（只留英文词、空则用 "retrieval"）
        # 只该有一份定义，两处各写一遍迟早会漂移。
        sub_questions = _salvage(
            [{"id": "sq-1", "text": question, "keywords": [], "category": None}], question
        )
    else:
        rationale = str(parsed.rationale or "")
        sub_questions = _salvage([sq.model_dump() for sq in parsed.sub_questions], question)

    logger.info(
        "planner.done",
        count=len(sub_questions),
        memories=len(memories),
        history=len(history),
        degraded=degraded,
    )

    return {
        "sub_questions": sub_questions,
        "plan_rationale": rationale,
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
                rationale=rationale,
                count=len(sub_questions),
                memory_count=len(memories),  # 让前端能显示"参考了 N 条历史记忆"
                # F9 的观测位：本轮带进了几段会话上下文（0 = 首轮提问，或短期记忆降级）。
                # 走既有 `plan_created` 事件的附加字段，不新增事件类型 ——
                # 事件协议是对外契约，加字段容易、加类型要前端同步改分支。
                history_count=len(history),
            )
        ]
        + (
            # 降级必须显式出现在流里：只改 `plan_rationale` 的文案，用户在界面上
            # 看到的是"拆解只有 1 个子问题"，分不清模型保守还是拆解坏了。
            [
                node_event(
                    "error",
                    node="plan",
                    code="plan_degraded",
                    message=rationale,
                    recoverable=True,
                )
            ]
            if degraded
            else []
        ),
    }


__all__ = ["plan_node"]

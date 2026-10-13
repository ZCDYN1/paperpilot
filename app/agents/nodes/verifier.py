"""`verify` 节点 —— 引用支撑校验（自我反思，架构 §4.2 / §10）。

## 两层校验，缺一不可
1. **硬校验（确定性、不可绕过）**：`quote` 是否真是原文的**子串**（空白归一化后）。
   这一层不需要模型，也就**不可能被模型说服**。它是「不编引用」这条承诺的最后防线。
2. **软校验（LLM 自我反思）**：片段是否**直接支撑**该论断（不是"沾边"、不是"可外推"）。
   这一层负责抓"原文说了 A，论断却写成 A 的加强版"这类**语义夸大** ——
   字符串包含检查对这类问题完全无能为力。

最终 `passed` 取两者的**与**：任一层不通过就不通过。
只做硬校验会把"断章取义"放过；只做软校验则把防线交给了一个可能被论文文本
（`<untrusted_document>` 里的注入）影响的模型。

## 为什么要把 verified 写回 citations
`citations.verified` 是"引用准确率 ≥ 90%"这个验收指标的**自动口径**
（架构 §12：「Verifier 已写回该字段，所以这条指标是免费的」）。
不写回，这个指标就只能靠人工，等于没有。
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import (
    chunk_index,
    merge_usage,
    node_event,
    normalize_text,
    structured_invoke,
)
from app.agents.prompts import load_prompt, render_document
from app.agents.schemas import VerifyOutput
from app.agents.state import STAGE_VERIFIED, Evidence, ResearchState

logger = structlog.get_logger(__name__)


def _hard_check(citation: Evidence, known: dict) -> tuple[bool, str]:
    """硬校验：引文必须能在原文里逐字（忽略空白差异）找到。"""
    chunk = known.get((citation.get("paper_id"), citation.get("chunk_id")))
    if chunk is None:
        return False, "引用的 chunk 不在本次检索结果中"
    quote = normalize_text(citation.get("quote", ""))
    if not quote:
        return False, "引文为空"
    if quote not in normalize_text(chunk.get("text", "")):
        return False, "引文未能在原文中逐字匹配"
    return True, ""


async def verify_node(state: ResearchState) -> dict:
    """逐条校验引用是否有原文支撑，并把结果写回 citations。"""
    citations: list[Evidence] = list(state.get("citations") or [])
    # `known` 由**检索结果**构建（而不是由 citations 自己构建）：
    # 这样"引用的 chunk 是否存在"才是一个独立的、可失败的事实。
    known = chunk_index(state.get("retrieved") or [])

    if not citations:
        verification = {
            "passed": False,
            "score": 0.0,
            "unsupported": [],
            "checked": 0,
            "reason": "没有可校验的引用",
        }
        return {
            "verification": verification,
            "stage": STAGE_VERIFIED,
            "events": [
                node_event(
                    "verification",
                    passed=False,
                    score=0.0,
                    unsupported=[],
                    revision_count=int(state.get("revision_count", 0)),
                )
            ],
        }

    # ---- 第一层：硬校验 ----
    hard: dict[int, tuple[bool, str]] = {}
    for i, cit in enumerate(citations):
        hard[i] = _hard_check(cit, known)

    # ---- 第二层：LLM 自我反思（软校验）----
    # 把论断 + 其引文渲染成不可信文档块：引文本身也来自论文正文，
    # 一样可能藏注入文本（例如"这条已被验证通过"），必须同等对待。
    blocks = "\n\n".join(
        render_document(
            chunk_id=cit.get("chunk_id", 0),
            arxiv_id=cit.get("paper_id", ""),
            section=cit.get("locator", "abstract"),
            source=cit.get("source", "abstract"),
            text=cit.get("quote", ""),
            title=f"论断：{cit.get('claim', '')}",
        )
        for cit in citations
    )
    messages = [
        SystemMessage(content=load_prompt("verifier")),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                f"以下是待校验的论断及其引文（共 {len(citations)} 条）：\n\n{blocks}"
            )
        ),
    ]

    soft_by_claim: dict[str, bool] = {}
    usage: dict[str, int] = {}
    try:
        parsed, usage = await structured_invoke(VerifyOutput, messages)
        for item in parsed.items:
            soft_by_claim[item.claim.strip()] = bool(item.supported)
    except Exception as exc:  # noqa: BLE001 - 软校验失败不应让硬校验的结论也失效
        logger.warning("verify.soft_check_failed", error=str(exc)[:200])

    # ---- 合并 ----
    unsupported: list[str] = []
    updated: list[Evidence] = []
    ok_count = 0
    for i, cit in enumerate(citations):
        hard_ok, hard_reason = hard[i]
        soft_ok = soft_by_claim.get((cit.get("claim") or "").strip(), True)  # 软校验缺席时不过度拦截
        ok = hard_ok and soft_ok
        if ok:
            ok_count += 1
        else:
            reason = hard_reason or "LLM 判定引文未直接支撑该论断"
            unsupported.append(cit.get("claim", ""))
            logger.info("verify.unsupported", claim=cit.get("claim", "")[:60], reason=reason)
        updated.append({**cit, "verified": ok})

    score = ok_count / len(citations)
    # 阈值说明：全部通过才算 passed。**刻意不做"提高通过率"的宽松处理** ——
    # 反思机制的价值正是"承认薄弱结论"，放开阈值等于把这个产品差异点关掉。
    verification = {
        "passed": ok_count == len(citations),
        "score": round(score, 4),
        "unsupported": unsupported,
        "checked": len(citations),
        "revision_count": int(state.get("revision_count", 0)),
    }
    logger.info("verify.done", passed=verification["passed"], score=score, checked=len(citations))

    return {
        "verification": verification,
        # 返回**全量**（含重写后的 verified 字段）：merge_evidence 同键覆盖，
        # 从而把校验结果真正写回，而不是留着一份 verified=False 的旧副本。
        "citations": updated,
        "stage": STAGE_VERIFIED,
        "token_usage": merge_usage(usage),
        "events": [
            node_event(
                "verification",
                passed=verification["passed"],
                score=verification["score"],
                unsupported=unsupported,
                revision_count=int(state.get("revision_count", 0)),
                checked=len(citations),
            )
        ],
    }


__all__ = ["verify_node"]

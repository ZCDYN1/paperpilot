"""`write` 节点 —— 生成中文综述 + 引用列表（中断点 B 挂在这里之前）。

## 核心原则：**不把合规性交给模型**
产品定义 §4 的产出物 B/C/D 有三条硬要求：
「每条结论带 `[n]`」「引用要能点回原文」「无支撑的结论必须显式标注」。
prompt 里当然要写清楚，但**只在 prompt 里写是幼稚的**：
模型漏一次，产品的核心承诺就崩一次，而且崩得毫无征兆。

所以这里做**后处理强制**：
1. 参考文献列表由 `citations` **程序生成**（不让模型编标题/作者/年份），
   模型正文里的 `[n]` 由我们建立"引用 → 序号"的映射来对齐；
2. 「⚠️ 未找到原文支撑的结论」一节由代码**追加**，内容直接取自
   `verification.unsupported`，模型删不掉也改不了；
3. 页脚"本文由 AI 生成，引用请以原文为准"固定追加（产品定义 §7 合规要求）。

## 语料不足的降级
`corpus_ready=False` 时，报告开头**显式声明检索覆盖度**。
架构 §2.2：「不能让 Agent 在空索引上硬编答案」。
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import dedup_papers, merge_usage, node_event, text_invoke
from app.agents.prompts import load_prompt, render_papers_block
from app.agents.state import STAGE_WRITTEN, Evidence, ResearchState

logger = structlog.get_logger(__name__)

_FOOTER = "> 本文由 AI 生成，引用请以原文为准。"


def _reference_list(citations: list[Evidence], titles: dict[str, str]) -> tuple[str, dict[str, int]]:
    """生成参考文献列表，并返回 `(paper_id, chunk_id) → 序号` 的映射。

    序号由**去重后的引用顺序**决定，而不是让模型自己数 ——
    模型数错一个，全文引用标记就整体错位，而报告读起来依然通顺（最难发现的错）。
    """
    lines: list[str] = []
    order: dict[str, int] = {}
    idx = 0
    for cit in citations:
        key = f"{cit.get('paper_id')}|{cit.get('chunk_id')}"
        if key in order:
            continue
        idx += 1
        order[key] = idx
        title = titles.get(cit.get("paper_id", ""), cit.get("paper_id", ""))
        locator = cit.get("locator", "abstract")
        source = cit.get("source", "abstract")
        strength = "摘要级证据" if source == "abstract" else "正文原文"
        lines.append(f"[{idx}] {title} — arXiv:{cit.get('paper_id', '')}（章节：{locator}｜{strength}）")
    return "\n".join(lines), order


def _coverage_notice(state: ResearchState) -> str:
    """语料不足时的显式声明块（空串表示语料充足）。"""
    if state.get("corpus_ready", True):
        return ""
    stats = state.get("corpus_stats") or {}
    found = stats.get("total_papers", stats.get("total", 0))
    threshold = stats.get("threshold", "?")
    return (
        "> ⚠️ **检索覆盖度声明**：本地语料仅命中 "
        f"{found} 篇（低于阈值 {threshold} 篇），本次综述的覆盖度有限，"
        "结论仅供参考，请勿直接用于正式文献综述。\n\n"
    )


async def write_node(state: ResearchState) -> dict:
    """生成最终中文综述（Markdown）。"""
    citations: list[Evidence] = list(state.get("citations") or [])
    retrieved = state.get("retrieved") or []
    papers = dedup_papers(retrieved)
    titles = {p["arxiv_id"]: p.get("title", "") for p in papers}

    reference_block, order = _reference_list(citations, titles)

    # 把"引用 → 序号"的映射一并交给模型，避免它自己编号错位
    numbered_evidence = "\n".join(
        f"[{order[f'{c.get('paper_id')}|{c.get('chunk_id')}']}] 论断：{c.get('claim')}｜"
        f"原文：{c.get('quote', '')[:120]}（arXiv:{c.get('paper_id')}｜{c.get('source')}）"
        for c in citations
        if f"{c.get('paper_id')}|{c.get('chunk_id')}" in order
    )

    block = render_papers_block(papers, max_chunks_per_paper=1)
    messages = [
        SystemMessage(content=load_prompt("writer")),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                f"请基于下列**已通过校验的证据**撰写中文综述。"
                f"引用序号请严格使用已给编号：\n\n{numbered_evidence}\n\n"
                f"（以下为对应论文原文片段，仅供核对措辞，不要引入未在上面列出的结论）\n\n{block}"
            )
        ),
    ]

    usage: dict[str, int] = {}
    try:
        body, usage = await text_invoke(messages)
    except Exception as exc:  # noqa: BLE001 - 写作失败也要给用户"部分结果"（架构 §11 整体降级）
        logger.warning("write.llm_failed", error=str(exc)[:200])
        body = "（综述生成失败，以下为本次检索与抽取的结构化结果）"

    unsupported = list((state.get("verification") or {}).get("unsupported") or [])
    sections: list[str] = [_coverage_notice(state), body.strip()]

    # ⚠️ 未支撑论断一节：**由代码追加**，模型删不掉（产品定义 §4 产出物 D）
    if unsupported:
        lines = "\n".join(f'- "{c}"' for c in unsupported)
        sections.append(
            "## ⚠️ 未找到原文支撑的结论\n\n"
            "以下结论未在检索到的文献中找到直接原文支撑，请勿直接引用：\n\n"
            f"{lines}"
        )
    elif citations:
        sections.append(
            "## ✅ 引用校验结果\n\n"
            f"本次共校验 {len(citations)} 条引用，全部在原文中定位到支撑片段。"
        )

    if reference_block:
        sections.append(f"## 参考文献\n\n{reference_block}")

    # 免责声明必须**恰好出现一次**：模型有时会在正文末尾自己抄一遍页脚，
    # 无条件追加就会出现两条一模一样的声明（读起来像 bug）。去重交给代码，不指望 prompt。
    if _FOOTER not in body:
        sections.append(_FOOTER)
    markdown = "\n\n".join(s.strip() for s in sections if s and s.strip())

    logger.info("write.done", chars=len(markdown), citations=len(citations))
    return {
        "draft": markdown,
        "stage": STAGE_WRITTEN,
        "token_usage": merge_usage(usage),
        "events": [
            node_event(
                "final_report",
                markdown=markdown,
                citations=[
                    {
                        "claim": c.get("claim"),
                        "paper_id": c.get("paper_id"),
                        "chunk_id": c.get("chunk_id"),
                        "quote": c.get("quote"),
                        "locator": c.get("locator"),
                        "source": c.get("source"),
                        "verified": c.get("verified"),
                    }
                    for c in citations
                ],
            )
        ],
    }


__all__ = ["write_node"]

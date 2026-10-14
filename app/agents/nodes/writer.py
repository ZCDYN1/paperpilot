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

## F6「剔除」的三态口径（为什么不是"只留 verified=True"）
`Evidence.verified` 是三态：True=校验通过、False=判为无支撑、**缺失/None=没校验过**
（give_up 降级路径与校验整体失败时 verifier 根本没写回）。
产品定义 §5 对 F6 的验收是「无支撑的论断被标记并**从正文剔除**」，
所以这里只剔除 `verified is False` 的条目，保留缺失/None 的 ——
若按"只留 True"处理，降级路径下所有引用都会被当成"无支撑"，
整份报告的正文与参考文献会被掏空，用户拿到一篇没有引用的空壳，
比"带着一两条未校验引用"损伤大得多。宁可少剔除，不可错杀。

## 语料不足的降级
`corpus_ready=False` 时，报告开头**显式声明检索覆盖度**。
架构 §2.2：「不能让 Agent 在空索引上硬编答案」。

## ★ 这里是 `token` 事件的真实发送点（架构 §8「LLM 输出增量」）
前端 `useResearchStream.ts` 一直在消费 `token` 事件做打字机效果，但真实模式下从未收到过
一条（只有 mock replay 里有），因为**全仓没有发送点**。现在由本节点补上：
`text_invoke(..., on_delta=sink.feed)` 走底层 `llm.astream`，
每个**真实生成出来的**增量经 LangGraph 的 `custom` 通道顶出图、立刻变成 SSE 帧
（机制与"为什么不能走 `state["events"]`"见 `app/agents/nodes/_common.py` 的 `TokenSink`）。

三条边界，别误会这个流的范围：
1. 流的是**模型生成的正文**，不含下面程序追加的 ⚠️ 一节 / 参考文献 / 页脚 ——
   把它们也吐成 `token` 就是"成稿切片假装流"，正是这次改动禁止的做法；
   完整成稿一律走 `final_report` 事件（前端渲染报告也只认它）。
2. 只给 write 开流式：plan/read/verify/compare 走 structured output，
   增量是半截 JSON，没有合法的 token 流可言。
3. 重试与流式的冲突按"**宁少流、不重流**"处置，见 `TokenSink` docstring。
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import (
    dedup_papers,
    make_token_sink,
    merge_usage,
    node_event,
    text_invoke,
)
from app.agents.prompts import load_prompt, render_papers_block
from app.agents.state import STAGE_WRITTEN, Evidence, ResearchState

logger = structlog.get_logger(__name__)

# 事件里 `node` 字段的取值：与图上的节点名一致（`app/agents/graph.py` 的 `add_node("write", …)`），
# 前端按它把增量归到 write 这一条流（`EventTimeline.tsx` 的 `collapseTokens` 也按 node 分段）。
WRITE_NODE = "write"

_FOOTER = "> 本文由 AI 生成，引用请以原文为准。"

# F17 输出语言切换：目前只显式认识 zh / en，其余取值把语言码原样写进指令
# （小模型对"用 fr 撰写"这类指令的遵循度不高，但这属于"默认行为不变 + 尽力而为"的范围）。
_LANG_LABELS = {"zh": "中文", "en": "英文（English）"}


def _output_language(state: ResearchState) -> str:
    """本次运行的输出语言码。缺省 zh —— 与加 language 字段之前的行为完全一致。"""
    return (state.get("language") or "zh").strip().lower() or "zh"


def _language_rule(lang: str) -> str:
    """渲染 writer prompt 里的 `{{LANGUAGE_RULE}}` 占位符。"""
    if lang == "en":
        return (
            "综述正文一律使用 **English** 撰写；"
            "方法名 / 数据集名 / 指标名保持英文原文。"
        )
    if lang == "zh":
        return "综述正文一律使用**中文**撰写；方法名 / 数据集名 / 指标名保留英文原文。"
    return f"综述正文一律使用 {lang} 撰写；方法名 / 数据集名 / 指标名保留其在原文中的写法。"


def _split_by_verification(
    citations: list[Evidence],
) -> tuple[list[Evidence], list[Evidence]]:
    """按三态口径把引用拆成 `(保留, 剔除)`（判据与原因见模块 docstring）。"""
    kept = [c for c in citations if c.get("verified") is not False]
    dropped = [c for c in citations if c.get("verified") is False]
    return kept, dropped


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
    """生成最终综述（Markdown，语言由 `state["language"]` 决定，默认中文）。"""
    citations: list[Evidence] = list(state.get("citations") or [])
    # F6 剔除：`verified is False` 的论断不进 prompt、不进参考文献（三态口径见模块 docstring）
    kept, dropped = _split_by_verification(citations)
    retrieved = state.get("retrieved") or []
    papers = dedup_papers(retrieved)
    titles = {p["arxiv_id"]: p.get("title", "") for p in papers}

    reference_block, order = _reference_list(kept, titles)

    # 把"引用 → 序号"的映射一并交给模型，避免它自己编号错位
    numbered_evidence = "\n".join(
        f"[{order[f'{c.get('paper_id')}|{c.get('chunk_id')}']}] 论断：{c.get('claim')}｜"
        f"原文：{c.get('quote', '')[:120]}（arXiv:{c.get('paper_id')}｜{c.get('source')}）"
        for c in kept
        if f"{c.get('paper_id')}|{c.get('chunk_id')}" in order
    )

    lang = _output_language(state)
    lang_label = _LANG_LABELS.get(lang, lang)
    system_prompt = load_prompt("writer").replace("{{LANGUAGE_RULE}}", _language_rule(lang))
    block = render_papers_block(papers, max_chunks_per_paper=1)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                f"请基于下列**已通过校验的证据**撰写{lang_label}综述。"
                f"引用序号请严格使用已给编号：\n\n{numbered_evidence}\n\n"
                f"（以下为对应论文原文片段，仅供核对措辞，不要引入未在上面列出的结论）\n\n{block}"
            )
        ),
    ]

    usage: dict[str, int] = {}
    # ★ `token` 事件的发送点：`sink.feed` 收到的是 `llm.astream` 真实吐出来的增量。
    # 不在流上下文里（单测直接调本节点 / 脚本跑 ainvoke）时 sink 全程静默，
    # 返回值与流式与否无关 —— 所以"能不能流"永远不会影响"报告对不对"。
    sink = make_token_sink(WRITE_NODE)
    try:
        body, usage = await text_invoke(messages, on_delta=sink.feed, on_abort=sink.abort)
    except Exception as exc:  # noqa: BLE001 - 写作失败也要给用户"部分结果"（架构 §11 整体降级）
        logger.warning("write.llm_failed", error=str(exc)[:200])
        body = "（综述生成失败，以下为本次检索与抽取的结构化结果）"

    logger.info(
        "write.streamed",
        node=WRITE_NODE,
        streamed_chunks=sink.chunks_sent,
        streamed_chars=sink.chars_sent,
        discarded=sink.closed,
    )

    unsupported = list((state.get("verification") or {}).get("unsupported") or [])
    # 兜底合并：verification.unsupported 正常就来自 `verified is False` 的那批引用
    # （见 verifier 的写回），这里只补"被剔除却没进 unsupported"的漏网条目，
    # 按论断文本去重，保证同一条不会在 ⚠️ 一节里列两遍。
    _listed = {(u or "").strip() for u in unsupported}
    for c in dropped:
        claim = (c.get("claim") or "").strip()
        if claim and claim not in _listed:
            unsupported.append(c.get("claim"))
            _listed.add(claim)
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

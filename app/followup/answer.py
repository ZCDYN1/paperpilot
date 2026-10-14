"""F9「会话内追问」的省钱那一半：基于**上一轮已经读过并校验过的证据**回答，不重新检索。

## 为什么这是一个独立服务，而不是往主图里加一条边
产品定义 F9 的原话是"基于已读论文继续问，不重新检索"。主图里加分支要同时改
`route_after_corpus`、HITL 中断点、checkpointer 线程语义，而这三处每加一个条件
都会让"3 分钟"那条验收标准变得不可预测。更麻烦的是：追问需要的东西其实**已经在库里了**
—— 上一轮 `citations` 表里存着逐字引文、章节定位、`verified` 结果。
所以这里直接读那张表，一次 LLM 调用出答案，全程不碰 FAISS/BM25、不碰 arXiv。

## 三条不许越过的线
1. **只用 `verified is not False` 的引用**（与 `nodes/writer.py` 同一套三态口径）：
   上一轮被判无支撑的论断不会因为"它已经在库里"就变成可复用的事实。
2. **答案里没有 `[n]` 标记的句子一律列进 `ungrounded`**。这是本产品对外承诺的形态：
   主动承认哪几句没有依据，而不是等用户去核对。
3. **上一轮没有可用证据时如实返回 `status="no_evidence"`**，不退化成"让模型凭参数记忆回答"
   —— 那正好是产品定义 §2 痛点 4 里被点名的问题。
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import select

from app.agents.nodes._common import text_invoke
from app.db.models import Citation, Paper, ResearchRun
from app.db.session import session_scope

logger = structlog.get_logger(__name__)

_CITATION_MARK_RE = re.compile(r"\[(\d{1,3})\]")
_QUOTE_MAX_CHARS = 420


class RunNotFoundError(LookupError):
    """指定 run 不存在。与 `app/export/markdown.py` 里的同名异常同语义（路由映射 404）。"""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} 不存在，无法基于它的已读内容回答")
        self.run_id = run_id


def _evidence_block(rows: list[Citation], titles: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    """把引用编成 `[n]` 号给模型，并返回同号的对外结构。

    编号由代码定、不让模型自己数：它数错一个，全文引用标记就整体错位，
    而报告读起来依然通顺（这是本项目记录过的最难发现的一类错）。
    """
    lines: list[str] = []
    used: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        quote = (row.quote or "").strip()
        if len(quote) > _QUOTE_MAX_CHARS:
            quote = quote[:_QUOTE_MAX_CHARS] + "…"
        title = titles.get(row.arxiv_id or "", row.arxiv_id or "")
        strength = "摘要级证据" if (row.source or "abstract") == "abstract" else "正文原文"
        lines.append(
            f"[{idx}] {title}（arXiv:{row.arxiv_id}｜章节：{row.locator or 'abstract'}｜{strength}）"
            f"\n    论断：{row.claim}\n    原文：{quote}"
        )
        used.append(
            {
                "index": idx,
                "claim": row.claim,
                "arxiv_id": row.arxiv_id,
                "chunk_id": row.chunk_id,
                "locator": row.locator,
                "source": row.source,
                "quote": quote,
            }
        )
    return "\n".join(lines), used


def _ungrounded_sentences(answer: str, known_indexes: set[int]) -> list[str]:
    """答案里没带任何有效 `[n]` 标记的句子。

    只认"指向已知编号"的标记：模型偶尔会写出 `[7]` 而证据只有 5 条，
    那种引用比不带引用更糟（它会看起来可核查）。
    """
    out: list[str] = []
    for sentence in re.split(r"(?<=[。！？\n])", answer or ""):
        body = sentence.strip()
        if not body or body.startswith("#"):
            continue
        marks = {int(m) for m in _CITATION_MARK_RE.findall(body)}
        if not (marks & known_indexes):
            out.append(body)
    return out


async def answer_from_read_papers(run_id: str, question: str) -> dict[str, Any]:
    """在**这次研究已经读过并校验过的内容**上回答追问。不检索、不联网、不建索引。

    Raises:
        RunNotFoundError: run 不存在（路由映射 404）。
    """
    async with session_scope() as session:
        run = await session.get(ResearchRun, run_id)
        if run is None:
            raise RunNotFoundError(run_id)
        rows = list(
            (
                await session.execute(
                    select(Citation).where(Citation.run_id == run_id).order_by(Citation.id)
                )
            )
            .scalars()
            .all()
        )
        # 三态口径：只排除显式判为 False 的（None/缺失 = 当时没校验过，仍可用）
        usable = [r for r in rows if r.verified is not False]
        ids = {r.arxiv_id for r in usable if r.arxiv_id}
        titles = (
            {
                p.arxiv_id: p.title
                for p in (
                    await session.execute(select(Paper).where(Paper.arxiv_id.in_(list(ids))))
                )
                .scalars()
                .all()
            }
            if ids
            else {}
        )

    base: dict[str, Any] = {
        "run_id": run_id,
        "question": question,
        "citations_total": len(rows),
        "citations_usable": len(usable),
        "papers": len({r.arxiv_id for r in usable if r.arxiv_id}),
        "retrieved_again": False,  # 这条接口存在的全部意义：不重新检索
    }
    if not usable:
        return {
            **base,
            "status": "no_evidence",
            "answer": "",
            "used_citations": [],
            "ungrounded": [],
            "note": (
                "这次研究没有可复用的、带原文支撑的引用，因此无法在不重新检索的前提下回答。"
                "请发起一次新的研究（会重新检索），或先让上一轮跑完校验。"
            ),
        }

    evidence, used = _evidence_block(usable, titles)
    known = {item["index"] for item in used}
    messages = [
        SystemMessage(
            content=(
                "你是文献问答助手。只允许使用下面给出的证据回答，"
                "每一条依据都必须在句尾用对应的 [n] 标记指出。"
                "证据里没有的内容一律不要写进去 —— "
                "如果证据不足以回答，就直接说明不足，不要用你自己的知识补全。"
            )
        ),
        HumanMessage(content=f"证据：\n{evidence}\n\n问题：{question}"),
    ]
    try:
        body, usage = await text_invoke(messages)
    except Exception as exc:  # noqa: BLE001 - 追问失败要给用户可读的降级，不是 500
        logger.warning("followup.llm_failed", run_id=run_id, error=str(exc)[:200])
        return {
            **base,
            "status": "error",
            "answer": "",
            # 没有任何内容被生成，就不能声称"用过了哪些证据"；
            # 可用证据的数量在 `citations_usable` 里，那个字段才是"库存"的意思。
            "used_citations": [],
            "ungrounded": [],
            "note": f"生成失败（{type(exc).__name__}），未产生任何结论。",
        }

    answer = (body or "").strip()
    return {
        **base,
        "status": "ok",
        "answer": answer,
        "used_citations": [item for item in used if item["index"] in _cited_in(answer, known)],
        "ungrounded": _ungrounded_sentences(answer, known),
        "token_usage": usage or {},
    }


def _cited_in(answer: str, known: set[int]) -> set[int]:
    return {int(m) for m in _CITATION_MARK_RE.findall(answer or "")} & known


__all__ = ["RunNotFoundError", "answer_from_read_papers"]

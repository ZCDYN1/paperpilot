"""F16「引用关系图」：谁引用了谁，以及这张图能说什么、不能说什么。

## 边的唯一来源是 OpenAlex 的 `referenced_works`
这是论文自己参考文献列表的机器可读版，属于**结构性事实**。
⚠️ 刻意禁止两类"看起来也能用"的边：
- **年份先后 + 主题相似** → 那是把"可能读过"写成"引用了"，F12 的演进叙事会整体失真；
- **正文里提到某个方法名** → 提到 ≠ 引用，而且抽取本身就可能出错。
宁可图上的边少一半，也不要一条假的。

## 只存两端都能解析成 arXiv id 的边
`citation_edges` 的两端是 arXiv 短 id。引到期刊论文、书、GitHub 仓库的参考文献
解析不出 arXiv id，就**不建边**（`resolve_arxiv_ids` 会返回它们，`arxiv_id=None`）。
这是范围限制，不是数据缺失：本产品管的是 arXiv 语料，画一条指向"库里根本没有、
也不是 arXiv 论文"的节点，除了好看没有任何可核查价值。

## `in_corpus` 是这张表最重要的一个布尔
库里有的论文才有正文证据可点（F5 的引用溯源）；库里没有的只是一个名字。
前端把两类节点画成同一种样式，就等于把"这篇引了 ColBERT"和"这篇引了库里某篇、
可以点开看原文"混为一谈。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import structlog
from sqlalchemy import or_, select

from app.db.models import CitationEdge, Paper
from app.db.session import session_scope
from app.ingest.openalex_client import OpenAlexClient

logger = structlog.get_logger(__name__)


def _short_id(value: str) -> str:
    """剥掉 `arxiv/`、版本号后缀，回到 `papers.arxiv_id` 用的那种短 id。

    与 `arxiv_client.parse_arxiv_id` 同一口径 —— 边表的主键两端必须和 `papers` 表
    用同一种写法，否则 `in_corpus` 的判定会恒为 False（表现是"图里全是无名节点"）。
    """
    text = (value or "").strip()
    for prefix in ("arxiv/", "math/", "physics/", "q-bio/", "q-fin/", "stat/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.lower().startswith("arxiv:"):
        text = text[len("arxiv:") :]
    for suffix_len in range(2, 4):  # v2 / v10 这类版本后缀
        if len(text) > suffix_len and text[-suffix_len] == "v" and text[-(suffix_len - 1) :].isdigit():
            text = text[: -suffix_len]
            break
    return text


async def build_reference_edges(
    arxiv_ids: Sequence[str], *, client: OpenAlexClient | None = None
) -> dict[str, Any]:
    """抓取并落库这些论文的引用边。返回一份可直接回给调用方的统计。

    **永不抛异常**：引用边是增强信息，OpenAlex 挂了不能影响已入库的语料
    （与 `openalex_client.enrich` 同一条降级原则）。
    """
    ids = [_short_id(i) for i in (arxiv_ids or []) if _short_id(i)]
    stats: dict[str, Any] = {
        "requested": len(ids),
        "works_found": 0,
        "edges_new": 0,
        "edges_known": 0,
        "refs_unresolved": 0,
        "errors": 0,
    }
    if not ids:
        return stats

    own_client = client is None
    client = client or OpenAlexClient()
    try:
        for arxiv_id in ids:
            try:
                found = await client.fetch_referenced_works(arxiv_id)
                if not found:
                    continue
                citing_wid, refs = found
                stats["works_found"] += 1
                resolved = await client.resolve_arxiv_ids(refs)
                # ⚠️ 必须是**本篇自己的**未解析数：下面 `edges_known` 要拿它减本篇的
                # `len(refs)`，用累计值相减会在几篇之后变成负数并一路累加
                # （实测 24 篇跑完报出 `edges_known: -292`）。
                unresolved_here = sum(1 for r in refs if not resolved.get(r, {}).get("arxiv_id"))
                stats["refs_unresolved"] += unresolved_here
                new_here = await _persist_edges(
                    citing_arxiv_id=arxiv_id,
                    citing_openalex_id=citing_wid,
                    resolved=resolved,
                )
                stats["edges_new"] += new_here
                # 解析出了 arXiv id 但边早就在库里 = "已知"，即本轮没有新增的那部分
                stats["edges_known"] += max(0, len(refs) - unresolved_here - new_here)
            except Exception as exc:  # noqa: BLE001 - 单篇失败不连坐整批
                stats["errors"] += 1
                logger.warning("references.edge_failed", arxiv_id=arxiv_id, error=str(exc)[:200])
    finally:
        if own_client:
            await client.aclose()

    logger.info("references.build_done", **stats)
    return stats


async def _persist_edges(
    *, citing_arxiv_id: str, citing_openalex_id: str, resolved: dict[str, dict[str, Any]]
) -> int:
    """写入本批新边，返回新增条数。已存在的边跳过（唯一约束兜住并发双插）。"""
    targets = {
        _short_id(info["arxiv_id"]): wid
        for wid, info in resolved.items()
        if info.get("arxiv_id") and _short_id(info["arxiv_id"]) != citing_arxiv_id
    }
    if not targets:
        return 0

    async with session_scope() as session:
        known = set(
            (
                await session.execute(
                    select(CitationEdge.cited_arxiv_id).where(
                        CitationEdge.citing_arxiv_id == citing_arxiv_id,
                        CitationEdge.cited_arxiv_id.in_(list(targets)),
                        CitationEdge.source == "openalex",
                    )
                )
            )
            .scalars()
            .all()
        )
        in_corpus = set(
            (
                await session.execute(
                    select(Paper.arxiv_id).where(Paper.arxiv_id.in_(list(targets)))
                )
            )
            .scalars()
            .all()
        )
        added = 0
        for cited, cited_wid in targets.items():
            if cited in known:
                continue
            session.add(
                CitationEdge(
                    citing_arxiv_id=citing_arxiv_id,
                    cited_arxiv_id=cited,
                    source="openalex",
                    citing_openalex_id=citing_openalex_id,
                    cited_openalex_id=cited_wid,
                    in_corpus=cited in in_corpus,
                )
            )
            added += 1
        return added


async def get_reference_graph(paper_ids: Sequence[str]) -> dict[str, Any]:
    """以这些论文为中心取一层的引用图（**只取一层，不做递归展开**）。

    递归展开会把整张 OpenAlex 图拉进内存，而产品要的是"这几篇之间谁补了谁的坑"
    （产品定义 F16 的原话是"谁引用了谁"，不是"给我整个引文网络"）。
    方向语义：`citing` 指向被引方，所以 `edges[*].source` 是引用发起方。
    """
    ids = [_short_id(i) for i in (paper_ids or []) if _short_id(i)]
    if not ids:
        return {"nodes": [], "edges": [], "papers": len(ids)}

    async with session_scope() as session:
        rows = (
            await session.execute(
                select(CitationEdge).where(
                    or_(
                        CitationEdge.citing_arxiv_id.in_(ids),
                        CitationEdge.cited_arxiv_id.in_(ids),
                    )
                )
            )
        ).scalars().all()
        node_ids = {e.citing_arxiv_id for e in rows} | {e.cited_arxiv_id for e in rows} | set(ids)
        papers = {
            p.arxiv_id: p
            for p in (
                await session.execute(select(Paper).where(Paper.arxiv_id.in_(list(node_ids))))
            )
            .scalars()
            .all()
        }

    nodes = [
        {
            "arxiv_id": nid,
            "title": papers[nid].title if nid in papers else "",
            "year": (
                papers[nid].published.year
                if nid in papers and papers[nid].published
                else None
            ),
            "venue": papers[nid].venue if nid in papers else None,
            "cited_by_count": papers[nid].cited_by_count if nid in papers else None,
            # 库里没有的节点没有正文证据可点开（F5 的溯源对它不成立），必须让前端区分
            "in_corpus": nid in papers,
            "focused": nid in ids,
        }
        for nid in sorted(node_ids)
    ]
    edges = [
        {
            "source": e.citing_arxiv_id,
            "target": e.cited_arxiv_id,
            "evidence": e.source,
        }
        for e in sorted(rows, key=lambda r: (r.citing_arxiv_id, r.cited_arxiv_id))
    ]
    return {"nodes": nodes, "edges": edges, "papers": len(ids)}


__all__ = ["build_reference_edges", "get_reference_graph"]

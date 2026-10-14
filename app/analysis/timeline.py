"""F12 —— 方法演进时间线：某方向的方法随时间如何迭代（"谁补了谁的坑"）。

## 数据来源纪律（产品定义 §4 字段来源规则 / §7 边界）
时间线条目里的每个字段都来自**库里已有的事实**，不做任何生成：
- `title / venue / cited_by_count / cited_at / published` ← `papers`（arXiv 官方 + OpenAlex 补数字）；
- `core_idea / limitation` ← `comparison_rows`（comparator 逐字段回原文校验过的产物）；
- `venue_tier` ← `app.analysis.venue_tiers.lookup`（纯查表，判不准给 unknown）；
- `cited_by_count` 沿用对比表的**成对规则**（架构 D10）：没有 `cited_at` 日期就把数字一并留空，
  孤零零的 "0" 会被误读成"没人引用"。

## ⚠️ `addresses_gap` 只认引用边证据，不许用年份推断
"谁补了谁的坑"是一个**演进关系断言**，它的合法证据只有一种：引用边
（A 引用了 B ⇒ A 建立在 B 的工作之上，可能针对 B 的局限做改进）。
本模块按主控 Agent 正在落的 `citation_edges` 表契约预留查询（字段
`citing_arxiv_id / cited_arxiv_id / source`），模型没建好时用
`try: from app.db.models import CitationEdge except ImportError` 兜住。

**没有引用边证据时该字段恒为空（`[]`），不允许用年份先后推断演进关系。**
"2024 的论文晚于 2022"只说明时间顺序，不说明它在补谁的坑 ——
把它写成演进关系就是编造，会直接击穿 F12 的立身之本。

## 排序
条目按 `year` 升序（早→晚，演进叙事的时间轴方向）；年份缺失的排最后、按 arxiv_id 定序，
保证同样输入永远同样输出（可复现，同导出层口径）。
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from sqlalchemy import select

from app.analysis.venue_tiers import lookup
from app.db import session as db_session
from app.db.models import ComparisonRecord, Paper
from app.export.markdown import RunNotFoundError, load_run_facts

logger = structlog.get_logger(__name__)

# arXiv 新式 id 的前四位天然就是投稿年月（"2401.00001" → 2024-01）。
# 只用作 published 缺失时的**事实性兜底**（id 编码是 arXiv 官方规则，不是推断内容）。
_ID_YEAR_RE = re.compile(r"^(\d{2})\d{2}\.\d{4,5}(v\d+)?$")


async def _load_edges(paper_ids: list[str]) -> list[dict[str, Any]]:
    """按主控的 `citation_edges` 契约读引用边；**表/模型不存在时返回 []，永不抛异常**。

    查询契约（主控加表时必须对上这三个字段名）：
    `CitationEdge.citing_arxiv_id / CitationEdge.cited_arxiv_id / CitationEdge.source`。
    只取"引用方和被引方都在本次时间线集合内"的边 —— 时间线讲的是**这组论文内部**的
    迭代关系，集合外的被引论文没有对比表行可讲，收了也讲不清。
    """
    if not paper_ids:
        return []
    try:
        from app.db.models import CitationEdge
    except ImportError:
        return []
    try:
        async with db_session.session_scope() as session:
            rows = (
                await session.execute(
                    select(
                        CitationEdge.citing_arxiv_id,
                        CitationEdge.cited_arxiv_id,
                        CitationEdge.source,
                    ).where(
                        CitationEdge.citing_arxiv_id.in_(paper_ids),
                        CitationEdge.cited_arxiv_id.in_(paper_ids),
                    )
                )
            ).all()
    except Exception as exc:  # noqa: BLE001 - 边表没建/列名不符时降级为"没有证据"，不拖垮整个时间线
        logger.warning("timeline.edges_unavailable", error=str(exc)[:200])
        return []
    return [
        {"citing": a, "cited": b, "source": src}
        for a, b, src in rows
        if a and b and a != b
    ]


def _year_of(paper: Paper | None, arxiv_id: str) -> int | None:
    """发表年份：papers.published → papers.updated → arXiv id 前缀（官方编码），都拿不到就 None。"""
    for p in (paper.published, paper.updated) if paper else ():
        if p is not None:
            return p.year
    m = _ID_YEAR_RE.match(arxiv_id or "")
    if m:
        return 2000 + int(m.group(1))
    return None


def _cited_pair(paper: Paper | None, row: ComparisonRecord | None) -> tuple[int | None, str | None]:
    """被引数与截至日期，沿用 comparator 的成对规则：缺日期 → 数字也留空。papers 优先。"""
    if paper is not None and paper.cited_by_count is not None:
        if paper.cited_at is None:
            return None, None
        return paper.cited_by_count, paper.cited_at.isoformat()
    if row is not None and row.cited_by_count is not None:
        # 对比表行落库前已被 comparator 成对清洗过（无日期即无数），这里双保险照抄口径。
        return (row.cited_by_count, row.cited_at) if row.cited_at else (None, None)
    return None, None


def _assemble(
    ids: list[str],
    papers_by_id: dict[str, Paper],
    rows_by_id: dict[str, ComparisonRecord],
    edges: list[dict[str, Any]],
) -> dict[str, Any]:
    """把"论文元数据 + 对比表行 + 引用边"拼成时间线载荷（纯函数，无 IO，可单测）。"""
    titles = {
        pid: (papers_by_id[pid].title if pid in papers_by_id else "")
        or (rows_by_id[pid].title if pid in rows_by_id else "")
        or pid
        for pid in ids
    }
    gaps_by_citing: dict[str, list[dict[str, Any]]] = {}
    for e in edges:
        gaps_by_citing.setdefault(e["citing"], []).append(
            {
                "paper_id": e["cited"],
                "title": titles.get(e["cited"], e["cited"]),
                "edge_source": e["source"],
            }
        )

    entries: list[dict[str, Any]] = []
    for pid in ids:
        paper = papers_by_id.get(pid)
        row = rows_by_id.get(pid)
        venue = ((paper.venue if paper else None) or (row.venue if row else None) or "").strip()
        tier_result = lookup(
            venue,
            comment=(paper.comment if paper else None),
        )
        count, cited_at = _cited_pair(paper, row)
        entries.append(
            {
                "arxiv_id": pid,
                "title": titles[pid],
                "year": _year_of(paper, pid),
                "venue": venue,
                "venue_tier": tier_result["tier"],
                "venue_evidence": tier_result["evidence"],
                "core_idea": (row.core_idea if row else "") or "",
                "limitation": (row.limitation if row else "") or "",
                "cited_by_count": count,
                "cited_at": cited_at,
                # ⚠️ 无引用边证据时恒为 []，不允许用年份先后推断演进关系（模块 docstring）。
                "addresses_gap": gaps_by_citing.get(pid, []),
            }
        )

    entries.sort(key=lambda e: (e["year"] is None, e["year"] or 0, e["arxiv_id"]))
    notes = (
        "条目按年份升序；年份缺失的排最后。"
        "addresses_gap 只来自 citation_edges 引用边证据（A→B 表示 A 引用了 B，"
        "即 A 可能针对 B 的局限做改进）—— 本次"
    ) + (
        "未取到任何引用边，该字段全部为空；不允许用年份先后推断演进关系。"
        if not edges
        else f"取到 {len(edges)} 条集合内引用边。"
    )
    return {
        "entries": entries,
        "paper_ids": ids,
        "edge_evidence_available": bool(edges),
        "notes": notes,
    }


async def build_timeline(*, run_id: str | None = None, paper_ids: list[str] | None = None) -> dict:
    """构建 F12 方法演进时间线。

    两种取数入口**二选一**（都不给或都给 → `ValueError`；这是用法错误，不该被静默兜住）：
    - `run_id`：论文集合 = 该 run 对比表的行（按 row_index 去重取值），run 不存在抛
      `RunNotFoundError`（与导出接口同一异常约定，路由直接映射 404）；
    - `paper_ids`：直接指定论文集合（可以没有跑过 run；对比表字段取各 run 中最新落库的一行，
      查不到对比行就只给 papers 里的事实，自由文本字段留空）。

    返回 `{entries, paper_ids, edge_evidence_available, notes, run_id}`；
    `entries` 各项字段见模块 docstring，缺失一律留空（None/空串/空列表），不编造。
    """
    if (run_id is None) == (paper_ids is None):
        raise ValueError("run_id 与 paper_ids 必须二选一（且只给一个）")

    ids: list[str] = []
    papers_by_id: dict[str, Paper] = {}
    rows_by_id: dict[str, ComparisonRecord] = {}

    if run_id is not None:
        facts = await load_run_facts(run_id)
        for r in facts["comparison_rows"]:
            if not r.arxiv_id:
                continue
            # 已按 row_index 排序：同 id 多行时取首行（表内行序即精读序）。
            rows_by_id.setdefault(r.arxiv_id, r)
            if r.arxiv_id not in ids:
                ids.append(r.arxiv_id)
        async with db_session.session_scope() as session:
            if ids:
                for p in (
                    await session.execute(select(Paper).where(Paper.arxiv_id.in_(ids)))
                ).scalars():
                    papers_by_id[p.arxiv_id] = p
        payload = _assemble(ids, papers_by_id, rows_by_id, await _load_edges(ids))
        payload["run_id"] = run_id
        return payload

    ids = list(dict.fromkeys(pid for pid in (paper_ids or []) if pid))
    if not ids:
        payload = _assemble([], {}, {}, [])
        payload["run_id"] = None
        return payload
    async with db_session.session_scope() as session:
        for p in (await session.execute(select(Paper).where(Paper.arxiv_id.in_(ids)))).scalars():
            papers_by_id[p.arxiv_id] = p
        # 跨 run 取"最新一行"：按自增 id 升序遍历、后来者覆盖（同论文多次被对比时，
        # 最后落库的对应最近一次研究；comparison_rows 本身不记 run 优先级）。
        for r in (
            await session.execute(
                select(ComparisonRecord)
                .where(ComparisonRecord.arxiv_id.in_(ids))
                .order_by(ComparisonRecord.id)
            )
        ).scalars():
            rows_by_id[r.arxiv_id] = r
    payload = _assemble(ids, papers_by_id, rows_by_id, await _load_edges(ids))
    payload["run_id"] = None
    return payload


__all__ = ["RunNotFoundError", "build_timeline"]

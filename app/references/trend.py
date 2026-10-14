"""F19 的另一半：被引次数的**时间序列**。

## 为什么必须单独存快照，而不是直接读 `papers.cited_by_count`
`cited_by_count` 是一个标量，每次抓取就被覆盖 —— 覆盖之后"上周多少"这个问题
在库里就没有答案了。架构 D10 定的规矩是"被引数必须携带截至日期"，
那张表把这条规矩从"一个日期"扩展成"一串日期"，趋势才有数据来源。

## 只有 1 个采样点时不报趋势
`citation_trend()` 在点数 < 2 时返回 `verdict="insufficient_data"`，
而不是画一条从原点到今天的假斜率。arXiv 新论文普遍被引 0~3 次，
两个点之间的随机波动就足以编造出"稳步上升" —— 这类图在学术场景里是有害的。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import structlog
from sqlalchemy import select

from app.db.models import CitationSnapshot, Paper
from app.db.session import session_scope
from app.ingest.openalex_client import OpenAlexClient

logger = structlog.get_logger(__name__)


async def record_citation_snapshots(
    arxiv_ids: Sequence[str], *, client: OpenAlexClient | None = None
) -> dict[str, Any]:
    """把这几篇论文**今天**的被引数记进快照表，并把 `papers` 上的最新值一起刷新。

    同一天重复调用是幂等的（唯一约束 `(arxiv_id, taken_on)`，第二次只更新数字）。
    走标题 + 作者交叉校验的 `enrich()`：被引数来自第三方聚合源，
    不校验就可能把别的论文的数字记到这篇头上（`openalex_client` 模块 docstring 记过现场）。
    """
    ids = [str(i).strip() for i in (arxiv_ids or []) if str(i).strip()]
    stats: dict[str, Any] = {"requested": len(ids), "recorded": 0, "no_match": 0, "errors": 0}
    if not ids:
        return stats

    own_client = client is None
    client = client or OpenAlexClient()
    today = date.today()
    try:
        async with session_scope() as session:
            papers = {
                p.arxiv_id: p
                for p in (
                    await session.execute(select(Paper).where(Paper.arxiv_id.in_(ids)))
                )
                .scalars()
                .all()
            }
        for arxiv_id in ids:
            paper = papers.get(arxiv_id)
            if paper is None:
                stats["no_match"] += 1
                continue
            try:
                metrics = await client.enrich(
                    paper.title, list(paper.authors or []), arxiv_id=arxiv_id
                )
            except Exception as exc:  # noqa: BLE001 - 补充源一律不致命
                stats["errors"] += 1
                logger.warning("trend.enrich_failed", arxiv_id=arxiv_id, error=str(exc)[:200])
                continue
            if metrics is None:
                # 交叉校验没过 = 拿不到可信数字。宁可不记，也不要记一个错的日子
                stats["no_match"] += 1
                continue
            await _upsert_snapshot(arxiv_id, metrics.cited_by_count, today, paper.title)
            stats["recorded"] += 1
    finally:
        if own_client:
            await client.aclose()

    logger.info("trend.recorded", **stats)
    return stats


async def _upsert_snapshot(
    arxiv_id: str, count: int, taken_on: date, title: str
) -> None:
    async with session_scope() as session:
        row = (
            await session.execute(
                select(CitationSnapshot).where(
                    CitationSnapshot.arxiv_id == arxiv_id,
                    CitationSnapshot.taken_on == taken_on,
                )
            )
        ).scalars().first()
        if row is None:
            session.add(
                CitationSnapshot(
                    arxiv_id=arxiv_id, cited_by_count=int(count), taken_on=taken_on
                )
            )
        else:
            row.cited_by_count = int(count)
        # 顺带刷新 papers 上的"最新值 + 截至日期"：这是对比表与 /papers 详情读的那一列，
        # 不刷新就会出现"快照表里有今天的数、对比表里还是上个月的数"两套口径。
        paper = await session.get(Paper, arxiv_id)
        if paper is not None:
            paper.cited_by_count = int(count)
            paper.cited_at = taken_on


async def citation_trend(arxiv_id: str) -> dict[str, Any]:
    """取某篇论文的被引序列，并**如实说明它够不够判断趋势**。"""
    async with session_scope() as session:
        points = (
            (
                await session.execute(
                    select(CitationSnapshot)
                    .where(CitationSnapshot.arxiv_id == arxiv_id)
                    .order_by(CitationSnapshot.taken_on)
                )
            )
            .scalars()
            .all()
        )
        paper = await session.get(Paper, arxiv_id)

    series = [{"on": p.taken_on.isoformat(), "cited_by_count": p.cited_by_count} for p in points]
    result: dict[str, Any] = {
        "arxiv_id": arxiv_id,
        "title": paper.title if paper else "",
        "points": series,
        "verdict": "insufficient_data",
        "delta": None,
        "span_days": None,
    }
    if len(series) < 2:
        result["note"] = (
            f"只有 {len(series)} 个采样点，无法判断趋势。"
            "再调用一次 POST /api/v1/papers/snapshot-citations 才会多出一天。"
        )
        return result

    first, last = series[0], series[-1]
    span = (date.fromisoformat(last["on"]) - date.fromisoformat(first["on"])).days
    result.update(
        {
            "delta": last["cited_by_count"] - first["cited_by_count"],
            "span_days": span,
            "verdict": (
                "rising"
                if last["cited_by_count"] > first["cited_by_count"]
                else "flat"
                if last["cited_by_count"] == first["cited_by_count"]
                else "falling"
            ),
        }
    )
    # 不需要判"同一天两个点"：唯一约束 `(arxiv_id, taken_on)` 让那种数据进不来
    return result


__all__ = ["citation_trend", "record_citation_snapshots"]

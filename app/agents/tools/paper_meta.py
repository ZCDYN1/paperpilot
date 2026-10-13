"""`get_paper_detail` 工具 —— 按 arxiv_id 取论文元数据与分块。

存在的理由：Reader / Writer 需要论文的元数据（标题、年份、venue、被引数），
而这些字段**不在检索结果里**（检索结果只带 chunk 文本与代表分）。
让它们直接查 DB 会破坏"Agent 层不碰 DB"的分层，所以包成工具。

⚠️ 被引数必须连同 `cited_at` 一起返回：没有"截至日期"的数字会随抓取时间漂移、
跨论文不可比（架构 §2.1 / ADR D10）。工具层负责把这两个字段绑在一起，
上层就不可能只拿到一半。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from sqlalchemy import select

from app.db.models import Paper, PaperChunk
from app.db.session import session_scope


async def run_get_paper_detail(arxiv_id: str, *, max_chunks: int = 20) -> dict[str, Any] | None:
    """读单篇论文的元数据 + 前 `max_chunks` 个分块。查不到返回 None（不抛错）。

    返回 None 而不是抛异常：调用方（Reader/Writer）在引用一篇已从库里消失的论文时，
    应该能优雅降级为"该篇证据不可用"，而不是让整条链挂掉。
    """
    async with session_scope() as session:
        paper = (
            await session.execute(select(Paper).where(Paper.arxiv_id == arxiv_id))
        ).scalar_one_or_none()
        if paper is None:
            return None
        chunks = (
            (
                await session.execute(
                    select(PaperChunk)
                    .where(PaperChunk.arxiv_id == arxiv_id)
                    .order_by(PaperChunk.chunk_index)
                    .limit(max_chunks)
                )
            )
            .scalars()
            .all()
        )

    return {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "abstract": paper.abstract,
        "authors": list(paper.authors or []),
        "published": paper.published.isoformat() if paper.published else None,
        "year": paper.published.year if paper.published else None,
        "primary_category": paper.primary_category,
        "venue": paper.venue,               # 从 arXiv comment 抽取（ADR D9）
        "cited_by_count": paper.cited_by_count,
        "cited_at": paper.cited_at.isoformat() if paper.cited_at else None,  # ⚠️ 必须与数字同时给出
        "pdf_url": paper.pdf_url,
        "html_url": paper.html_url,
        "has_code": paper.has_code,
        "chunks": [
            {
                "chunk_id": c.id,
                "section": c.section,
                "source": c.source,
                "chunk_type": c.chunk_type,
                "page": c.page,
                "text": c.text,
            }
            for c in chunks
        ],
    }


@tool("get_paper_detail")
async def get_paper_detail(arxiv_id: str) -> dict[str, Any] | None:
    """按 arXiv id 查询论文元数据与分块正文（用于引用溯源与对比表字段）。

    Args:
        arxiv_id: 形如 `2401.00001` 的 arXiv 标识。

    Returns:
        元数据 + chunks；论文不在本地语料库时返回 None。
    """
    return await run_get_paper_detail(arxiv_id)


__all__ = ["get_paper_detail", "run_get_paper_detail"]

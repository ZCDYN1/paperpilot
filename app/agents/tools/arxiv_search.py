"""`arxiv_search` 工具 —— 直连 arXiv 元数据检索（**只查元数据，不入库**）。

和 `ingest` 的关系（架构 §2.2 的两阶段）：
- `arxiv_search`：**轻量探针**。只问"这个方向 arXiv 上有多少篇"，用于
  `ensure_corpus` 判断是否需要真的抓取；不写库、不建索引。
- `ingest`：重活。抓正文 → 分块 → 落库 → 建索引，**串行受 3s 限速**。

⚠️ 两者都受 arXiv 限速约束（约 1 req / 3s），`ArxivClient` 内部已做串行限速与重试，
调用方**不要**再并发调用它 —— 并发只会让 arXiv 返回 429，重试更慢。
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from app.core.config import get_settings


async def run_arxiv_search(
    query: str,
    *,
    categories: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """在 arXiv 上检索论文元数据（不入库）。返回精简字段列表。"""
    from app.ingest.arxiv_client import ArxivClient

    settings = get_settings()
    async with ArxivClient(settings) as client:
        papers = await client.search(
            query,
            categories=categories,
            year_from=year_from,
            year_to=year_to,
            max_results=max_results,
        )
    return [
        {
            "arxiv_id": p.arxiv_id,
            "title": p.title,
            "abstract": p.abstract,
            "authors": list(p.authors),
            "published": p.published.isoformat() if p.published else None,
            "primary_category": p.primary_category,
            "categories": list(p.categories),
            "comment": p.comment,
            "pdf_url": p.pdf_url,
            "html_url": p.html_url,
        }
        for p in papers
    ]


@tool("arxiv_search")
async def arxiv_search(
    query: str,
    max_results: int = 10,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    """在 arXiv 官方 API 上检索论文元数据（不写库、不建索引）。

    Args:
        query: 检索式（英文效果最好，支持 arXiv 的字段前缀如 `ti:`、`abs:`）。
        max_results: 最多返回多少条（≤ 50，避免触发限速）。
        year_from / year_to: 提交年份范围。

    Returns:
        论文元数据列表。
    """
    return await run_arxiv_search(
        query,
        year_from=year_from,
        year_to=year_to,
        max_results=min(int(max_results), 50),
    )


__all__ = ["arxiv_search", "run_arxiv_search"]

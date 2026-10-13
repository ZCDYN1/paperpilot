"""metadata 过滤的单元测试。

过滤的正确性标准不是"实现了过滤"，而是**不满足条件的论文一个都不能出现**。
因此每个用例都先确认目标论文本来能被召回（对照组），再确认过滤后它消失 ——
否则测试可能在"这篇论文本来就召回不到"的假象下通过。
"""

from __future__ import annotations

from app.retrieval.hybrid_retriever import MetadataFilter
from tests.conftest import BROAD_KEYWORDS, FOCUSED_QUERY, PAPER_A, PAPER_B, PAPER_C


async def test_no_filter_returns_all_papers(retriever):
    """对照组：不加过滤时三篇都在，后面的过滤测试才有意义。"""
    hits = await retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10)
    assert {h.arxiv_id for h in hits} == {PAPER_A, PAPER_B, PAPER_C}


async def test_category_filter(retriever):
    """按 arXiv 分类过滤：只保留 cs.IR 的那篇。"""
    hits = await retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(categories=["cs.IR"])
    )
    assert {h.arxiv_id for h in hits} == {PAPER_B}

    hits_cl = await retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(categories=["cs.CL"])
    )
    assert {h.arxiv_id for h in hits_cl} == {PAPER_A}


async def test_year_gte_filter(retriever):
    """year_gte=2023 应排除 2021 年的论文 B。"""
    hits = await retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(year_gte=2023)
    )
    ids = {h.arxiv_id for h in hits}
    assert ids == {PAPER_A, PAPER_C}
    assert PAPER_B not in ids


async def test_year_lte_filter(retriever):
    """year_lte=2021 只应留下 2021 年的论文 B。"""
    hits = await retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(year_lte=2021)
    )
    assert {h.arxiv_id for h in hits} == {PAPER_B}


async def test_year_range_and_arxiv_id_filter(retriever):
    """年份区间与按 arxiv_id 精确过滤都要生效。"""
    ranged = await retriever.search(
        FOCUSED_QUERY,
        BROAD_KEYWORDS,
        top_k=10,
        filters=MetadataFilter(year_gte=2023, year_lte=2023),
    )
    assert {h.arxiv_id for h in ranged} == {PAPER_C}

    by_id = await retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(arxiv_ids=[PAPER_C])
    )
    assert {h.arxiv_id for h in by_id} == {PAPER_C}


async def test_filter_matching_nothing_returns_empty(retriever):
    """过滤条件与语料无交集时应安全返回空，而不是回退成全量结果。"""
    hits = await retriever.search(
        FOCUSED_QUERY,
        BROAD_KEYWORDS,
        top_k=10,
        filters=MetadataFilter(categories=["math.NT"]),
    )
    assert hits == []

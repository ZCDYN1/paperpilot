"""schemas 的校验行为测试。

"用 BGE 抽了 Pydantic 模型"本身没有价值，有价值的是**它真的能拦住错误入参**。
所以这里逐条验证边界：范围、白名单、以及"中文关键词"这个 BM25 高频坑。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.retrieval.hybrid_retriever import MetadataFilter
from app.schemas import HybridSearchArgs, ResearchFilters, ResearchRequest, SubQuestion


def test_hybrid_search_args_valid_and_maps_to_metadata_filter():
    args = HybridSearchArgs(
        query="混合检索如何融合 BM25 与向量",
        keywords=["hybrid", "retrieval", "bm25", "reranker"],
        top_k=10,
        year_gte=2022,
        category="cs.CL",
    )
    f = MetadataFilter(**args.to_filter_kwargs())
    assert f.categories == ["cs.CL"]
    assert f.year_gte == 2022


@pytest.mark.parametrize("bad_top_k", [0, 51, -3])
def test_hybrid_search_args_top_k_bounds(bad_top_k):
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", top_k=bad_top_k)


@pytest.mark.parametrize("bad_year", [1990, 2101])
def test_hybrid_search_args_year_bounds(bad_year):
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", year_gte=bad_year)


def test_hybrid_search_args_rejects_unknown_category():
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", category="cs.AI.LLM")


def test_hybrid_search_args_rejects_chinese_keywords():
    """中文关键词打英文 BM25 索引会静默失效，必须在工具边界拦下（§4.1 注①）。"""
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", keywords=["重排", "检索"])


def test_hybrid_search_args_rejects_year_inversion_and_extra_field():
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", year_gte=2024, year_lte=2020)
    with pytest.raises(ValidationError):
        HybridSearchArgs(query="q", unexpected=1)


def test_research_request_and_filters():
    req = ResearchRequest(
        question="对比主流混合检索方案",
        filters=ResearchFilters(categories=["cs.IR"], year_gte=2021),
        top_k=12,
        rerank=False,
    )
    assert req.filters.year_gte == 2021
    assert req.top_k == 12
    assert req.rerank is False

    # 空问题 / 超长问题 / 非法分类都要被拒
    with pytest.raises(ValidationError):
        ResearchRequest(question="")
    with pytest.raises(ValidationError):
        ResearchRequest(question="x", filters=ResearchFilters(categories=["nonsense.XX"]))


def test_sub_question_defaults_and_validation():
    sq = SubQuestion(id="s1", text="如何进行混合检索", keywords=["hybrid", "retrieval"])
    assert sq.category is None
    with pytest.raises(ValidationError):
        SubQuestion(id="s2", text="x", keywords=["混合检索"])


def test_pdf_source_is_a_valid_evidence_strength():
    """上传件的字面量必须能被对外结构接住。

    `app/ingest/pdf.py` 把入库片段的 source 写成 "pdf"，而 `ChunkOut` / `CitationOut`
    早先只认 html|abstract：第一个人上传 PDF，`GET /research/{id}/citations` 就会在
    响应模型校验上炸掉（本地库只有 abstract/html，所以这条一直没人碰到）。
    """
    from app.agents.schemas import ExtractionItem
    from app.schemas.research import ChunkOut, CitationOut

    ChunkOut(chunk_id=1, arxiv_id="2401.00001", source="pdf", text="t")
    CitationOut(
        claim="c",
        paper_id="2401.00001",
        chunk_id=1,
        quote="t",
        locator="§1",
        source="pdf",
    )
    ExtractionItem(paper_id="2401.00001", chunk_id=1, claim="c", quote="t", source="pdf")

    # 反过来也要拦得住：拼错的值不能悄悄当成 abstract 出去
    with pytest.raises(ValidationError):
        ChunkOut(chunk_id=1, arxiv_id="2401.00001", source="pdfe", text="t")

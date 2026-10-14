"""检索与过滤层的"请求 → 检索 → 抓取"接线测试（F2 三类过滤 + F14 指定论文）。

这些用例的共同标准：**字段从请求传进去了、且真的减少了计算量**，
而不是"参数名存在但中途被丢弃"。本文件里的失败用例对应过三个真实缺口：
- `has_code` 在抓取侧硬编码 None、在 schemas 里根本没有字段；
- 多分类全局过滤在 `_build_args` 被静默丢弃（只支持恰好 1 个）；
- API 的 year 过滤落不到抓取环节（语料抓取与过滤条件无关，白耗 arXiv 限速配额）。
全部离线：mock embedding（conftest 环境变量）+ 不触网的纯函数与内存索引。
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from app.agents.nodes import retriever_worker
from app.agents.nodes.ensure_corpus import collect_categories, collect_ingest_params
from app.agents.nodes.retriever_worker import _build_args, _build_extra_filters, retrieve_node
from app.agents.tools.hybrid_search import build_filter
from app.ingest.arxiv_client import ArxivPaper
from app.ingest.pipeline import _paper_row, detect_has_code
from app.retrieval.hybrid_retriever import HybridRetriever, MetadataFilter, corpus_digest
from app.schemas.research import ResearchFilters, ResearchRequest
from app.schemas.retrieval import HybridSearchArgs
from tests.conftest import BROAD_KEYWORDS, FOCUSED_QUERY, PAPER_A, PAPER_B, PAPER_C, make_corpus


# conftest 语料本身不带 has_code（保持既有测试不变），这里在其上叠加：
# A=确认有仓库（True）、C=明确没有（False）、B 保持 None（未判定），
# 三种取值齐活，才能把过滤语义钉死。
def _corpus_with_code() -> tuple[list, dict[str, str]]:
    records, titles = make_corpus()
    for r in records:
        if r.arxiv_id == PAPER_A:
            r.has_code = True
        elif r.arxiv_id == PAPER_C:
            r.has_code = False
    return records, titles


@pytest.fixture
async def code_retriever(tmp_path) -> HybridRetriever:
    records, titles = _corpus_with_code()
    r = HybridRetriever(index_dir=tmp_path / "faiss")
    await r.build_index(records, titles)
    return r


# ------------------------------------------------------------------ T1 抓取侧
def test_detect_has_code_positive_cases():
    """四大仓库/模型托管域，出现在 comment 或 abstract 任一处都算真阳性。"""
    assert detect_has_code("Code: https://github.com/foo/bar", None) is True
    assert detect_has_code(None, "model at https://huggingface.co/org/model") is True
    assert detect_has_code("repo: gitlab.com/a/b", None) is True
    assert detect_has_code(None, "see https://bitbucket.org/x/y") is True
    # 大小写混用是 comment 常态
    assert detect_has_code("GitHub.com/Foo/Bar released", None) is True


def test_detect_has_code_negative_stays_none_not_false():
    """识别不到 = None（未判定）。返回 False 就是编造"这篇论文没有代码"。"""
    assert detect_has_code("Accepted at NeurIPS 2024", "We propose a method...") is None
    assert detect_has_code(None, None) is None
    assert detect_has_code("", "") is None
    # 裸域名行文不是仓库链接：没有 `/路径`，误判成真阳性比漏判更坑
    assert detect_has_code("GitHub.com is a hosting platform", None) is None
    # 只认四个域，其它外链（项目主页、docs）不算
    assert detect_has_code("demo at https://project.example.org/code", None) is None


def test_paper_row_fills_has_code_from_arxiv_metadata():
    """`pipeline` 的硬编码 None 已被替换：Paper 行与元数据检测结果一致（ADR D7）。"""
    paper = ArxivPaper(
        arxiv_id="2401.00001",
        title="t",
        abstract="code available at https://github.com/a/b",
        authors=["x"],
        published=None,
        updated=None,
        primary_category="cs.CL",
        categories=["cs.CL"],
        comment=None,
        journal_ref=None,
        pdf_url=None,
        html_url="https://arxiv.org/html/2401.00001",
    )
    assert _paper_row(paper, None).has_code is True
    assert _paper_row(replace(paper, abstract="no links here"), None).has_code is None


# ------------------------------------------------------------------ T2 检索侧
async def test_has_code_filter_restricts_hits(code_retriever):
    """对照组 + 过滤组：has_code=True 只留 A，False 留 B(未判定)+C(明确无)。

    B 进 false 一侧是刻意的口径：入库检测只产出 True/None，若 false 过滤要求
    显式 False，则该方向永远为空、F2 的"是否有代码"只剩一半。保守方向是
    "没见过链接"，它不虚构任何字段值（展示层仍是空白）。
    """
    hits = await code_retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(has_code=True)
    )
    assert {h.arxiv_id for h in hits} == {PAPER_A}

    hits_no = await code_retriever.search(
        FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(has_code=False)
    )
    assert {h.arxiv_id for h in hits_no} == {PAPER_B, PAPER_C}

    hits_all = await code_retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10)
    assert {h.arxiv_id for h in hits_all} == {PAPER_A, PAPER_B, PAPER_C}


def test_has_code_prefilter_shrinks_whitelist(code_retriever):
    """过滤必须在 `_prefilter` 减少白名单（FAISS/BM25 拿到的 allow_ids 变小），
    而不是全量检索后丢弃 —— 后者会让 recall 静默下降（架构 §5.2 明文警告）。"""
    allow = code_retriever._prefilter(MetadataFilter(has_code=True))
    assert allow is not None
    assert allow == {r.chunk_id for r in code_retriever._chunks.values() if r.has_code}
    assert 0 < len(allow) < code_retriever.size


def test_has_code_survives_meta_roundtrip(tmp_path):
    """has_code 必须进索引元数据落盘：重启进程后过滤结果不变，
    否则"同进程能用、重启后失效"会成为又一类静默 bug。"""
    records, titles = _corpus_with_code()
    d = tmp_path / "faiss"

    async def _scenario() -> set[str]:
        r1 = HybridRetriever(index_dir=d)
        await r1.build_index(records, titles)
        r2 = HybridRetriever(index_dir=d)  # 模拟进程重启：只靠磁盘 npz 恢复元数据
        assert r2.size == r1.size
        assert r2._prefilter(MetadataFilter(has_code=True)) == r1._prefilter(
            MetadataFilter(has_code=True)
        )
        hits = await r2.search(
            FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=MetadataFilter(has_code=True)
        )
        return {h.arxiv_id for h in hits}

    assert asyncio.run(_scenario()) == {PAPER_A}


def test_old_index_without_has_codes(tmp_path):
    """老索引 npz 缺 has_codes 列 → load_meta 优雅降级为全 None，不报错。

    （与 embedding_identity 同口径：拦死老用户升级的代价远大于收益；
    corpus_digest 已把 has_code 算进指纹，下一次 ensure_corpus 必然重建。）
    """
    records, titles = _corpus_with_code()
    d = tmp_path / "faiss"

    async def _build() -> HybridRetriever:
        r = HybridRetriever(index_dir=d)
        await r.build_index(records, titles)
        return r

    r1 = asyncio.run(_build())
    with np.load(r1.meta_path, allow_pickle=True) as f:
        payload = {k: f[k] for k in f.files}
    payload.pop("has_codes")  # 还原改动前的老索引形态
    np.savez_compressed(r1.meta_path, **payload)

    r2 = HybridRetriever(index_dir=d)
    assert r2.size == len(records)
    assert all(rec.has_code is None for rec in r2._chunks.values())


def test_has_code_change_invalidates_index_digest():
    """过滤字段变了但指纹不变 = 索引静默陈旧，这个项目反复踩的坑，用测试钉死。"""
    records, titles = make_corpus()
    before = corpus_digest(records, titles)
    for r in records:
        if r.arxiv_id == PAPER_A:
            r.has_code = True
    assert corpus_digest(records, titles) != before


# ------------------------------------------------------------- T3 请求侧透传
def test_hybrid_search_args_has_code_passthrough():
    args = HybridSearchArgs(query="q", has_code=True)
    assert args.to_filter_kwargs()["has_code"] is True
    assert MetadataFilter(**args.to_filter_kwargs()).has_code is True
    # 不传 = 不过滤（None 直通，不能悄悄变成 False）
    assert HybridSearchArgs(query="q").to_filter_kwargs()["has_code"] is None


def test_research_filters_has_code_and_dump():
    f = ResearchFilters(has_code=True)
    assert f.model_dump()["has_code"] is True
    assert ResearchFilters().has_code is None
    # API 层用 `filters.model_dump()` 喂图 state：字段必须在 dump 里出现才传得下去
    assert "has_code" in ResearchFilters().model_dump()


# ------------------------------------------------------------------ T4 多分类
def _state(**filters) -> dict:
    return {
        "question": "q",
        "sub_question": {"id": "s1", "text": "t", "keywords": ["x"], "category": None},
        "filters": filters,
    }


def test_multicategory_global_filter_not_dropped():
    """F2 回归锁：多分类全局过滤曾被 `_build_args` 静默整组丢弃（只支持恰好 1 个）。"""
    state = _state(categories=["cs.CL", "cs.IR"])
    args = _build_args(state, state["sub_question"])
    # 单值 category 契约不被全局多分类侵占（那是 LLM 工具入参，宁缺毋滥）
    assert args.category is None
    f = build_filter(args, _build_extra_filters(state))
    assert f is not None
    assert sorted(f.categories) == ["cs.CL", "cs.IR"]


def test_single_category_and_sub_question_precedence():
    """单分类全局过滤（旧行为）不能退化；子问题分类仍优先于全局（更具体的约束）。"""
    state = _state(categories=["cs.CL"])
    f = build_filter(_build_args(state, state["sub_question"]), _build_extra_filters(state))
    assert f is not None and list(f.categories) == ["cs.CL"]

    state2 = _state(categories=["cs.IR"])
    state2["sub_question"]["category"] = "cs.CL"
    f2 = build_filter(_build_args(state2, state2["sub_question"]), _build_extra_filters(state2))
    assert f2 is not None and list(f2.categories) == ["cs.CL"]


async def test_retrieve_node_passes_filters_to_tool(monkeypatch):
    """端到端锁：节点调用 `run_hybrid_search` 时 has_code 走 args、多分类/arxiv_ids 走 extra。"""
    captured: dict = {}

    async def fake_run(args, *, retriever=None, extra_filter=None):
        captured["args"] = args
        captured["extra"] = extra_filter
        return []

    monkeypatch.setattr(retriever_worker, "run_hybrid_search", fake_run)
    state = _state(categories=["cs.CL", "cs.IR"], has_code=True, arxiv_ids=["2401.00002"])
    out = await retrieve_node(state)
    assert out["retrieved"] == []
    assert captured["args"].has_code is True
    assert captured["extra"]["categories"] == ["cs.CL", "cs.IR"]
    assert captured["extra"]["arxiv_ids"] == ["2401.00002"]


# ------------------------------------------------------------------ T5 抓取侧
def test_collect_ingest_params_wires_request_filters():
    """year 过滤必须落到 `ingest()` 的抓取参数（收窄串行请求，不是加并发）。"""
    state = {
        "sub_questions": [{"category": "cs.CL"}],
        "filters": {"categories": ["cs.IR"], "year_gte": 2022, "year_lte": 2024},
    }
    p = collect_ingest_params(state)
    assert p == {
        "categories": ["cs.CL", "cs.IR"],  # 子问题 ∪ 全局（OR 集合，并集安全）
        "year_from": 2022,
        "year_to": 2024,
    }
    assert collect_categories(state) == ["cs.CL", "cs.IR"]


def test_collect_ingest_params_empty_state():
    p = collect_ingest_params({})
    assert p == {"categories": None, "year_from": None, "year_to": None}


# ------------------------------------------------------------------ T6 F14
def test_request_arxiv_ids_merges_into_filters():
    """顶层 `arxiv_ids` 必须并进 `filters.arxiv_ids`：API 只透传 `filters.model_dump()`，
    不并进去，F14 就又回到"检索层支持但请求传不进去"的原状。"""
    req = ResearchRequest(question="q", arxiv_ids=["2305.15334v2"])
    assert req.filters is not None
    assert req.filters.arxiv_ids == ["2305.15334"]  # 版本后缀被剥离（主键是无版本短 id）
    assert req.filters.model_dump()["arxiv_ids"] == ["2305.15334"]

    # 两处都给 → 取并集、按序去重
    req2 = ResearchRequest(
        question="q", arxiv_ids=["2401.00002"], filters=ResearchFilters(arxiv_ids=["2305.15334"])
    )
    assert req2.filters.arxiv_ids == ["2305.15334", "2401.00002"]

    # 没给 → filters 保持 None（对既有请求零影响）
    assert ResearchRequest(question="q").filters is None


@pytest.mark.parametrize("bad", ["2305", "cs/0701001", "https://arxiv.org/abs/2305.15334", ""])
def test_request_arxiv_ids_rejects_bad_shapes(bad):
    with pytest.raises(ValidationError):
        ResearchRequest(question="q", arxiv_ids=[bad])
    with pytest.raises(ValidationError):
        ResearchFilters(arxiv_ids=[bad])


async def test_arxiv_ids_end_to_end_restricts_search(code_retriever):
    """请求 → state.filters → extra_filter → build_filter → MetadataFilter → 检索只剩指定论文。"""
    req = ResearchRequest(question="q", arxiv_ids=[PAPER_B])
    state = _state(**req.filters.model_dump())
    args = _build_args(state, state["sub_question"])
    f = build_filter(args, _build_extra_filters(state))
    assert f is not None and list(f.arxiv_ids) == [PAPER_B]

    hits = await code_retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, filters=f)
    assert {h.arxiv_id for h in hits} == {PAPER_B}


# ------------------------------------------------------------------ T7 language
def test_language_field():
    assert ResearchRequest(question="q").language == "zh"  # 默认中文（主用法）
    assert ResearchRequest(question="q", language="en").language == "en"
    with pytest.raises(ValidationError):
        ResearchRequest(question="q", language="fr")

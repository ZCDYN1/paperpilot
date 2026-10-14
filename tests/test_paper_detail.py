"""F13 单篇深挖的离线单测（T1 详情 + T2 深度问答）。

全部离线：LLM/embedding 用 mock（api_support.BASE_ENV + conftest 环境变量），
SQLite / FAISS 指向 pytest 的 tmp_path，不碰 data/。

判断口径刻意**不追模型输出质量**（本地 7B 不在场），只锁三条产品承诺：
1. 详情返回元数据 + 章节级 chunks（带 source）+ 历史引用/对比行；
2. 问答只可能引用**这一篇**的片段（定向通道 + DB 兜底都在 arxiv_id 之内）；
3. 引文逐字对不上就整条剔除，剔完没有支撑就如实回答"不足以回答"——
   这条必须由代码保证，prompt 里写了不算（同 verifier 的硬校验思路）。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from sqlalchemy import select

import app.papers.detail as detail
from app.agents.tools.hybrid_search import reset_shared_retriever
from app.core.config import get_settings
from app.db.models import Citation, ComparisonRecord, Paper, PaperChunk
from app.db.session import dispose_engine, init_db, session_scope
from app.papers import PaperNotFoundError, ask_about_paper, build_paper_detail
from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever
from tests.api_support import BASE_ENV

_ARXIV_ID = "2401.00001"
_OTHER_ID = "2401.00002"


@pytest.fixture
async def offline_env(monkeypatch, tmp_path):
    """临时库 + 临时索引目录 + 全 mock。用完拆进程内单例，防止串味。"""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SQLITE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'f13.db').as_posix()}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SQLITE_DIR", str(tmp_path / "data" / "sqlite"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "data" / "faiss"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "data" / "sqlite" / "checkpoints.db"))
    get_settings.cache_clear()
    await dispose_engine()
    await init_db()
    yield tmp_path
    reset_shared_retriever()
    await dispose_engine()
    get_settings.cache_clear()


async def _seed_paper(
    arxiv_id: str = _ARXIV_ID,
    title: str = "Hybrid Retrieval with BM25 and Reranker Fusion",
    chunks: tuple[tuple[str, str, str], ...] = (
        ("method", "The hybrid retrieval pipeline uses bm25 and a cross encoder reranker.", "html"),
        ("results", "Reranker fusion lifts recall over the bm25 baseline.", "html"),
        ("abstract", "We propose hybrid retrieval combining dense and sparse scoring.", "abstract"),
    ),
    *,
    with_history: bool = False,
) -> list[int]:
    """种一篇论文（可带 chunks / 历史 citations+comparison_rows）。返回 chunk id 列表。"""
    ids: list[int] = []
    async with session_scope() as session:
        session.add(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                abstract="stub abstract",
                authors=["Ada Lovelace"],
                published=datetime(2024, 1, 5, tzinfo=UTC),
                venue="NeurIPS 2024",
                cited_by_count=42,
                cited_at=date(2026, 1, 1),
                has_code=True,
                primary_category="cs.CL",
            )
        )
        await session.flush()
        for i, (section, text, source) in enumerate(chunks):
            row = PaperChunk(
                arxiv_id=arxiv_id,
                chunk_index=i,
                section=section,
                text=text,
                source=source,
                chunk_type="text",
                token_count=max(1, len(text) // 4),
            )
            session.add(row)
            await session.flush()
            ids.append(row.id)
        if with_history:
            session.add(
                Citation(
                    run_id="run-old",
                    claim="该方法使用了重排器。",
                    chunk_id=ids[0],
                    arxiv_id=arxiv_id,
                    quote="cross encoder reranker",
                    locator="method",
                    source="html",
                    verified=True,
                )
            )
            session.add(
                ComparisonRecord(
                    run_id="run-old",
                    row_index=0,
                    arxiv_id=arxiv_id,
                    title=title,
                    method_name="HybridFusion",
                    core_idea="双路召回 + RRF",
                )
            )
    return ids


async def _index_all(tmp_path) -> HybridRetriever:
    """把库里全部 chunks 建成临时索引（口径同 load_corpus，但索引文件落 tmp）。"""
    async with session_scope() as session:
        papers = (await session.execute(select(Paper))).scalars().all()
        titles = {p.arxiv_id: p.title for p in papers}
        rows = (await session.execute(select(PaperChunk))).scalars().all()
    records = [
        ChunkRecord(
            chunk_id=r.id,
            arxiv_id=r.arxiv_id,
            section=r.section,
            text=r.text,
            source=r.source,
            chunk_type=r.chunk_type,
            page=r.page,
        )
        for r in rows
    ]
    r = HybridRetriever(index_dir=tmp_path / "data" / "faiss")
    await r.build_index(records, titles)
    return r


# --------------------------------------------------------------------------- T1 详情
async def test_detail_returns_metadata_chunks_and_history(offline_env):
    ids = await _seed_paper(with_history=True)
    d = await build_paper_detail(_ARXIV_ID)

    assert d["title"].startswith("Hybrid Retrieval")
    assert d["authors"] == ["Ada Lovelace"]
    assert d["year"] == 2024
    assert d["venue"] == "NeurIPS 2024"
    # 被引数与"截至日期"成对出现（models.py 的同一条产品承诺）
    assert d["cited_by_count"] == 42
    assert d["cited_at"] == "2026-01-01"
    assert d["has_code"] is True
    assert d["is_upload"] is False

    assert [c["chunk_id"] for c in d["chunks"]] == ids
    # source 原样带出：前端要能区分正文原文与摘要级证据
    assert [c["source"] for c in d["chunks"]] == ["html", "html", "abstract"]
    assert d["stats"]["weak_evidence_chunks"] == 1

    assert len(d["citations"]) == 1 and d["citations"][0]["run_id"] == "run-old"
    assert d["stats"]["verified_citations"] == 1
    assert len(d["comparison_rows"]) == 1
    assert d["comparison_rows"][0]["method_name"] == "HybridFusion"


async def test_detail_unknown_id_raises_not_none(offline_env):
    """路由靠这个异常映射 404；返回 None 会把"没这篇"混成"有但没内容"。"""
    with pytest.raises(PaperNotFoundError) as ei:
        await build_paper_detail("2499.99999")
    assert "2499.99999" in str(ei.value)
    assert ei.value.arxiv_id == "2499.99999"


# --------------------------------------------------------------------------- T2 问答
async def test_ask_answers_only_on_this_paper(offline_env, monkeypatch):
    """索引里有两篇论文，问答只可能引用被点名的那篇。"""
    a_ids = await _seed_paper()
    await _seed_paper(
        arxiv_id=_OTHER_ID,
        title="Vision Transformer Image Classification",
        chunks=(("method", "The vision transformer patch embedding is followed by attention.", "html"),),
    )
    r = await _index_all(offline_env)
    monkeypatch.setattr(detail, "get_shared_retriever", lambda: r)

    res = await ask_about_paper(_ARXIV_ID, "这篇论文的方法是什么？")

    assert res["supported"] is True
    assert res["retrieval"] == "targeted"
    assert res["citations"], "mock 输出与真实文档同构，应能落回至少一条引用"
    for c in res["citations"]:
        assert c["chunk_id"] in a_ids  # 绝不越篇
        assert c["source"] in ("html", "abstract")
        assert c["quote"]
    # 引用必须逐字能在被引用的片段里找到（硬校验通过才是"通过"）
    async with session_scope() as session:
        rows = (await session.execute(select(PaperChunk))).scalars().all()
    text_by_id = {row.id: row.text for row in rows}
    for c in res["citations"]:
        assert detail.normalize_text(c["quote"]) in detail.normalize_text(text_by_id[c["chunk_id"]])


async def test_ask_uses_targeted_channel_not_whole_corpus(offline_env, monkeypatch):
    """锁接线：走 run_hybrid_search 且 extra_filter 带 arxiv_ids、top_k=1。

    （`arxiv_ids` 定向通道 F14 已打通，这里验证 F13 真的在用，而不是偷偷全库检索。）
    """
    await _seed_paper()
    captured: dict = {}

    async def fake_run(args, *, retriever=None, extra_filter=None):
        captured["args"] = args
        captured["extra"] = extra_filter
        return []

    monkeypatch.setattr(detail, "run_hybrid_search", fake_run)
    res = await ask_about_paper(_ARXIV_ID, "reranker fusion 的收益有多大？")

    assert captured["extra"] == {"arxiv_ids": [_ARXIV_ID]}
    assert captured["args"].top_k == 1
    # 检索层没东西时降级为"本篇 DB 片段"，仍然不出这一篇
    assert res["retrieval"] == "fallback_db"
    assert res["supported"] is True


async def test_ask_falls_back_to_db_chunks_without_index(offline_env):
    """新进程冷启动（索引还没建）时 F13 不能瞎：仍然只用本篇 chunks 作答。"""
    ids = await _seed_paper()
    # 不 monkeypatch 检索器：fixture 的 FAISS 目录为空 → 检索自然返回 []
    res = await ask_about_paper(_ARXIV_ID, "方法部分怎么做的？")
    assert res["retrieval"] == "fallback_db"
    assert res["supported"] is True
    assert {c["chunk_id"] for c in res["citations"]} <= set(ids)


async def test_ask_drops_fabricated_quotes_and_answers_honestly(offline_env, monkeypatch):
    """核心承诺：模型编了引文 → 整条剔除；剔完没有支撑 → 答案换成如实的"不足以回答"。"""
    await _seed_paper()

    async def fake_invoke(schema, messages, *, temperature=None):
        parsed = schema(
            answer="这篇论文提出了惊天的新架构，碾压所有已知方法。",
            claims=[
                detail._GroundedClaim(
                    claim="编造的论断",
                    quote="这句话不存在于任何原文片段中",
                    chunk_id=999999,
                )
            ],
        )
        return parsed, {"prompt_tokens": 1, "completion_tokens": 1}

    monkeypatch.setattr(detail, "run_hybrid_search", _empty_search)
    monkeypatch.setattr(detail, "structured_invoke", fake_invoke)
    res = await ask_about_paper(_ARXIV_ID, "它提出了什么新方法？")

    assert res["supported"] is False
    assert res["citations"] == []
    assert res["answer"] == detail.INSUFFICIENT_ANSWER
    assert "惊天" not in res["answer"]  # 编造的答案一个字都不能出去


async def test_ask_honest_when_paper_has_no_chunks(offline_env, monkeypatch):
    """正文都没入库的论文：如实回答且**不烧一次 LLM**（没有可 grounding 的东西）。"""
    await _seed_paper(chunks=())
    called = False

    async def guard(schema, messages, *, temperature=None):
        nonlocal called
        called = True
        raise AssertionError("不该调用模型")

    monkeypatch.setattr(detail, "structured_invoke", guard)
    res = await ask_about_paper(_ARXIV_ID, "这篇论文的方法？")
    assert res["supported"] is False
    assert res["answer"] == detail.INSUFFICIENT_ANSWER
    assert res["retrieval"] == "no_chunks"
    assert not called


async def test_ask_unknown_paper_raises(offline_env):
    with pytest.raises(PaperNotFoundError):
        await ask_about_paper("2499.99999", "任何问题")


async def test_ask_empty_question_rejected(offline_env):
    await _seed_paper()
    with pytest.raises(ValueError):
        await ask_about_paper(_ARXIV_ID, "   ")


async def _empty_search(args, *, retriever=None, extra_filter=None):
    return []

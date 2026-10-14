"""F12 `app/analysis/timeline.build_timeline` 的离线测试。

锁死五条：
1. `run_id` / `paper_ids` 二选一，用法错误抛 ValueError；run 不存在抛 RunNotFoundError；
2. 条目字段全部来自库里事实（papers + comparison_rows），按年份升序、缺失排最后；
3. `venue_tier` 走 venue_tiers 查表；被引数沿用"与截至日期成对"规则（缺日期→数字也空）；
4. **addresses_gap 只认引用边**：`citation_edges` 模型不存在（ImportError）时恒为 []，
   且不许用年份推断 —— 用 fake CitationEdge 注入的方式验证契约字段
   `citing_arxiv_id / cited_arxiv_id / source` 对上时才有内容；
5. 缺 papers 行 / 缺对比行的论文也保留条目，字段如实留空，不编造。

临时 SQLite（tmp_path），不碰真实 data/。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, datetime

import pytest
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 注册全部表到 metadata
from app.analysis import timeline as timeline_mod
from app.analysis.timeline import build_timeline
from app.db import session as db_session
from app.db.base import Base
from app.db.models import ComparisonRecord, Paper, ResearchRun, ResearchSession
from app.export.markdown import RunNotFoundError

# citation_edges 契约预留：主控加真实表之前，测试自己定义同形模型；
# 真实模型一旦出现（表名已注册）就直接复用，保证测试跟着真实 schema 走。
if "citation_edges" in Base.metadata.tables:  # pragma: no cover - 取决于主控进度
    FakeCitationEdge = app.db.models.CitationEdge
else:

    class FakeCitationEdge(Base):  # type: ignore[no-redef]
        __tablename__ = "citation_edges"
        id = Column(Integer, primary_key=True)
        citing_arxiv_id = Column(String(64), nullable=False, index=True)
        cited_arxiv_id = Column(String(64), nullable=False, index=True)
        source = Column(String(32))


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    yield


@pytest.fixture
async def temp_db(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'timeline.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(db_session, "session_scope", _scope)
    yield _scope
    await engine.dispose()


async def _seed(scope) -> str:
    run_id = "run-tl-001"
    async with scope() as s:
        s.add(ResearchSession(id="sess-tl", title="t"))
        s.add(
            ResearchRun(
                id=run_id, session_id="sess-tl", thread_id="sess-tl",
                question="重排的演进", status="done",
            )
        )
        s.add(
            Paper(
                arxiv_id="2401.00001", title="Ranker A", venue="ACL 2024",
                comment="Accepted at ACL 2024", published=datetime(2024, 1, 5),
                cited_by_count=40, cited_at=date(2026, 10, 1),
            )
        )
        s.add(
            Paper(
                arxiv_id="2305.00002", title="Ranker B", venue=None,
                published=datetime(2023, 5, 1), cited_by_count=99, cited_at=None,
            )
        )
        # 2501.00003：没有 published，年份从 arXiv id 前缀兜底；也不进 papers（测兜底）
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=0, arxiv_id="2401.00001", title="Ranker A",
                core_idea="cross encoder 打分", limitation="latency 高",
                venue="ACL 2024", cited_by_count=40, cited_at="2026-10-01",
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=1, arxiv_id="2305.00002", title="Ranker B",
                core_idea="bi encoder 检索", limitation="",
                venue="", cited_by_count=None, cited_at=None,
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=2, arxiv_id="2501.00003", title="Ranker C 未入库",
                core_idea="listwise 蒸馏", limitation="",
                venue="arXiv", cited_by_count=None, cited_at=None,
            )
        )
    return run_id


def _by_id(payload: dict) -> dict[str, dict]:
    return {e["arxiv_id"]: e for e in payload["entries"]}


async def test_argument_exclusivity(temp_db):
    with pytest.raises(ValueError):
        await build_timeline()
    with pytest.raises(ValueError):
        await build_timeline(run_id="x", paper_ids=["y"])


async def test_run_not_found(temp_db):
    with pytest.raises(RunNotFoundError):
        await build_timeline(run_id="ghost")


async def test_entries_sorted_by_year_with_facts_only(temp_db):
    run_id = await _seed(temp_db)
    payload = await build_timeline(run_id=run_id)
    entries = payload["entries"]
    assert [e["arxiv_id"] for e in entries] == ["2305.00002", "2401.00001", "2501.00003"]
    assert [e["year"] for e in entries] == [2023, 2024, 2025]  # 2025 来自 id 前缀兜底

    a = _by_id(payload)["2401.00001"]
    assert a["title"] == "Ranker A"
    assert a["venue"] == "ACL 2024"
    assert a["venue_tier"] == "top"
    assert a["venue_evidence"]["matched"] == "ACL"
    assert "ACL" in (a["venue_evidence"]["comment_excerpt"] or "")  # comment 证据定位
    assert a["core_idea"] == "cross encoder 打分"
    assert a["limitation"] == "latency 高"
    assert a["cited_by_count"] == 40 and a["cited_at"] == "2026-10-01"

    b = _by_id(payload)["2305.00002"]
    # 成对规则：papers 里有数字但没有 cited_at → 数字一并留空（照抄 comparator 口径）
    assert b["cited_by_count"] is None and b["cited_at"] is None
    assert b["venue"] == "" and b["venue_tier"] == "preprint"
    assert b["limitation"] == ""  # 空字段如实空，不写占位

    c = _by_id(payload)["2501.00003"]
    # papers 表没有这行 → 标题用对比行兜底、venue 用对比行兜底
    assert c["title"] == "Ranker C 未入库"
    assert c["venue"] == "arXiv" and c["venue_tier"] == "preprint"


async def test_addresses_gap_empty_without_edge_table(temp_db, monkeypatch):
    """citation_edges 模型不存在（ImportError 路径）→ 该字段恒为 []，且 notes 说明原因。"""
    run_id = await _seed(temp_db)
    # 无论主控是否已把 CitationEdge 落进 models，这里都强制切到"无表"分支；
    # 用 monkeypatch 而不是 delattr：测试结束自动还原，不污染同进程后续测试。
    monkeypatch.delattr(app.db.models, "CitationEdge", raising=False)
    payload = await build_timeline(run_id=run_id)
    assert all(e["addresses_gap"] == [] for e in payload["entries"])
    assert payload["edge_evidence_available"] is False
    assert "不允许用年份先后推断演进关系" in payload["notes"]


async def test_addresses_gap_only_from_edges_contract(temp_db, monkeypatch):
    """注入符合契约的 CitationEdge（citing/cited/source）后，addresses_gap 才有内容。"""
    run_id = await _seed(temp_db)
    monkeypatch.setattr(app.db.models, "CitationEdge", FakeCitationEdge, raising=False)
    async with temp_db() as s:
        s.add(FakeCitationEdge(citing_arxiv_id="2401.00001", cited_arxiv_id="2305.00002", source="openalex"))
        # 集合外的被引论文不该出现在 addresses_gap（只取两端都在集合内的边）
        s.add(FakeCitationEdge(citing_arxiv_id="2401.00001", cited_arxiv_id="9999.99999", source="parse"))
        # 自环丢弃
        s.add(FakeCitationEdge(citing_arxiv_id="2305.00002", cited_arxiv_id="2305.00002", source="parse"))

    payload = await build_timeline(run_id=run_id)
    a = _by_id(payload)["2401.00001"]
    assert a["addresses_gap"] == [
        {"paper_id": "2305.00002", "title": "Ranker B", "edge_source": "openalex"}
    ]
    assert _by_id(payload)["2305.00002"]["addresses_gap"] == []
    assert payload["edge_evidence_available"] is True


async def test_paper_ids_entry_without_run(temp_db):
    await _seed(temp_db)
    payload = await build_timeline(paper_ids=["2401.00001", "old-0001"])
    ids = [e["arxiv_id"] for e in payload["entries"]]
    assert ids == ["2401.00001", "old-0001"]
    ghost = _by_id(payload)["old-0001"]
    # papers 与对比表都没有 → 字段全部如实留空，标题用 id 兜底；
    # 非新式数字 id 抽不出年份前缀 → year 也是 None（不猜）。
    assert ghost["title"] == "old-0001"
    assert ghost["year"] is None
    assert ghost["venue"] == "" and ghost["venue_tier"] == "preprint"
    assert ghost["core_idea"] == "" and ghost["limitation"] == ""
    assert ghost["addresses_gap"] == []
    assert payload["run_id"] is None


async def test_latest_comparison_row_wins_for_paper_ids(temp_db):
    await _seed(temp_db)
    async with temp_db() as s:
        # 同论文第二个 run 又落了一行（id 更大）→ paper_ids 入口取最新一行
        s.add(
            ComparisonRecord(
                run_id="run-tl-002", row_index=0, arxiv_id="2401.00001", title="Ranker A",
                core_idea="新版：listwise 校准", limitation="", venue="",
                cited_by_count=None, cited_at=None,
            )
        )
    payload = await build_timeline(paper_ids=["2401.00001"])
    assert _by_id(payload)["2401.00001"]["core_idea"] == "新版：listwise 校准"
    # venue 字段：papers 有 ACL 2024，仍是 top（papers.venue 优先于新行的空 venue）
    assert _by_id(payload)["2401.00001"]["venue_tier"] == "top"


def test_assemble_year_none_last_deterministic():
    """纯函数层面：年份缺失排最后，同为缺失按 arxiv_id 定序（可复现）。"""
    payload = timeline_mod._assemble(
        ["2501.00009", "2401.00001", "abc/math.001"], {}, {}, []
    )
    assert [e["arxiv_id"] for e in payload["entries"]] == [
        "2401.00001",
        "2501.00009",
        "abc/math.001",
    ]

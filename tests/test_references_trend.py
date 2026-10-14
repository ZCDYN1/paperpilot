"""F16 引用边 与 F19 被引趋势的服务层测试。

钉住四件容易悄悄坏掉的事：
1. 边的来源只有 OpenAlex，且**幂等**（同一对论文 + 同一来源只有一条）；
2. 引不到 arXiv id 的参考文献**不建边**，也不许"就近挑一个像的"顶替；
3. `in_corpus` 必须真的是"库里有没有这篇"，前端靠它区分"能点开看原文"与"只是个名字"；
4. 被引趋势在采样点不足时**必须拒绝给结论** —— 这条是整个 F19 的底线。

全程不联网：外部调用一律换成鸭子类型的假 client。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 注册全部表到 metadata
from app.db.base import Base
from app.db.models import CitationEdge, CitationSnapshot, Paper
from app.references import (
    build_reference_edges,
    citation_trend,
    get_reference_graph,
    record_citation_snapshots,
)
from app.references import edges as edges_mod
from app.references import trend as trend_mod


@pytest.fixture
async def scope(monkeypatch, tmp_path):
    """临时库 + 把 `session_scope` 换过去（服务层调用时才取模块属性，所以可替换）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'refs.db'}")
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

    monkeypatch.setattr(edges_mod, "session_scope", _scope)
    monkeypatch.setattr(trend_mod, "session_scope", _scope)
    yield _scope
    await engine.dispose()


class FakeClient:
    """鸭子类型的 OpenAlex 客户端：只实现服务层真正用到的三个方法。"""

    def __init__(self, refs: dict[str, tuple[str, list[str]]] | None = None, resolved: dict | None = None):
        self._refs = refs or {}
        self._resolved = resolved or {}
        self.closed = False
        self.enrich_results: dict[str, object] = {}

    async def fetch_referenced_works(self, arxiv_id):
        return self._refs.get(arxiv_id)

    async def resolve_arxiv_ids(self, work_ids):
        return {w: self._resolved.get(w, {"arxiv_id": None, "title": ""}) for w in work_ids}

    async def enrich(self, title, authors, *, arxiv_id=None):  # noqa: ARG002
        return self.enrich_results.get(arxiv_id)

    async def aclose(self):
        self.closed = True


async def _seed_papers(scope, ids: list[str]) -> None:
    async with scope() as s:
        for i, aid in enumerate(ids):
            s.add(
                Paper(
                    arxiv_id=aid,
                    title=f"Paper {aid}",
                    abstract="",
                    authors=["Some Author"],
                    cited_by_count=10 + i,
                    cited_at=date(2026, 1, 1),
                )
            )


# --------------------------------------------------------------------------- F16 建边
async def test_edges_are_persisted_and_idempotent(scope):
    await _seed_papers(scope, ["2305.15334", "2401.00001"])
    client = FakeClient(
        refs={"2305.15334": ("W100", ["W200", "W300"])},
        resolved={
            "W200": {"arxiv_id": "2401.00001", "title": "Paper 2401.00001"},
            "W300": {"arxiv_id": "2201.00002", "title": "Not in corpus"},
        },
    )

    first = await build_reference_edges(["2305.15334"], client=client)
    assert first["works_found"] == 1
    assert first["edges_new"] == 2

    second = await build_reference_edges(["2305.15334"], client=client)
    assert second["edges_new"] == 0, "重复抓取在造重复的边（唯一约束没起作用）"

    async with scope() as s:
        rows = (await s.execute(select(CitationEdge))).scalars().all()
    assert len(rows) == 2
    by_target = {r.cited_arxiv_id: r for r in rows}
    # in_corpus 的判定直接决定前端能不能把节点画成"可点开溯源"
    assert by_target["2401.00001"].in_corpus is True
    assert by_target["2201.00002"].in_corpus is False
    assert {r.source for r in rows} == {"openalex"}


async def test_refs_without_arxiv_id_do_not_become_edges(scope):
    """解析不出 arXiv id 的参考文献（期刊 / 书 / 仓库）宁可不建边。"""
    await _seed_papers(scope, ["2305.15334"])
    client = FakeClient(
        refs={"2305.15334": ("W100", ["W900"])},
        resolved={"W900": {"arxiv_id": None, "title": "Some journal article"}},
    )
    stats = await build_reference_edges(["2305.15334"], client=client)
    assert stats["refs_unresolved"] == 1
    assert stats["edges_new"] == 0
    async with scope() as s:
        assert (await s.execute(select(func.count()).select_from(CitationEdge))).scalar() == 0


async def test_self_reference_is_dropped(scope):
    """OpenAlex 偶尔会把 DOI 记录本身混进 referenced_works，自环会让图与时间线都失真。"""
    await _seed_papers(scope, ["2305.15334"])
    client = FakeClient(
        refs={"2305.15334": ("W100", ["W101"])},
        resolved={"W101": {"arxiv_id": "2305.15334", "title": "same paper"}},
    )
    stats = await build_reference_edges(["2305.15334"], client=client)
    assert stats["edges_new"] == 0


async def test_missing_work_returns_stats_not_exception(scope):
    """OpenAlex 没有这篇的记录是常态（新论文尤其常见）→ 如实报告，不抛。"""
    await _seed_papers(scope, ["2305.15334"])
    stats = await build_reference_edges(["2305.15334"], client=FakeClient(refs={}))
    assert stats["works_found"] == 0 and stats["edges_new"] == 0


async def test_graph_returns_one_hop_with_node_metadata(scope):
    await _seed_papers(scope, ["2305.15334", "2401.00001"])
    client = FakeClient(
        refs={"2305.15334": ("W100", ["W200"])},
        resolved={"W200": {"arxiv_id": "2401.00001", "title": "Paper 2401.00001"}},
    )
    await build_reference_edges(["2305.15334"], client=client)

    graph = await get_reference_graph(["2305.15334"])
    assert {(e["source"], e["target"]) for e in graph["edges"]} == {("2305.15334", "2401.00001")}
    nodes = {n["arxiv_id"]: n for n in graph["nodes"]}
    assert nodes["2305.15334"]["focused"] is True
    assert nodes["2401.00001"]["title"] == "Paper 2401.00001"
    assert nodes["2401.00001"]["in_corpus"] is True


async def test_graph_with_no_edges_is_empty_not_error(scope):
    graph = await get_reference_graph(["2305.15334"])
    assert graph["edges"] == []
    assert {n["arxiv_id"] for n in graph["nodes"]} == {"2305.15334"}


# --------------------------------------------------------------------------- F19 趋势
class _Metrics:
    def __init__(self, count: int) -> None:
        self.cited_by_count = count


async def test_snapshot_updates_papers_and_is_idempotent_per_day(scope):
    """同一天重复记只留一条；`papers` 上的最新值必须一起刷新，否则两套口径会分叉。"""
    await _seed_papers(scope, ["2305.15334"])
    client = FakeClient()
    client.enrich_results = {"2305.15334": _Metrics(42)}

    await record_citation_snapshots(["2305.15334"], client=client)
    await record_citation_snapshots(["2305.15334"], client=client)

    async with scope() as s:
        snaps = (await s.execute(select(CitationSnapshot))).scalars().all()
        paper = await s.get(Paper, "2305.15334")
    assert len(snaps) == 1
    assert snaps[0].cited_by_count == 42
    assert paper.cited_by_count == 42 and paper.cited_at == date.today()


async def test_snapshot_skips_papers_without_a_validated_match(scope):
    """交叉校验没过（返回 None）就不记 —— 记错一天的数字会污染整条趋势。"""
    await _seed_papers(scope, ["2305.15334"])
    stats = await record_citation_snapshots(["2305.15334"], client=FakeClient())
    assert stats["no_match"] == 1 and stats["recorded"] == 0


async def test_trend_refuses_conclusion_with_one_point(scope):
    await _seed_papers(scope, ["2305.15334"])
    client = FakeClient()
    client.enrich_results = {"2305.15334": _Metrics(7)}
    await record_citation_snapshots(["2305.15334"], client=client)

    out = await citation_trend("2305.15334")
    assert out["verdict"] == "insufficient_data"
    assert out["delta"] is None
    assert "无法判断趋势" in out["note"]


async def test_trend_reports_direction_with_two_days(scope, monkeypatch):
    await _seed_papers(scope, ["2305.15334"])
    today = date.today()
    async with scope() as s:
        s.add(CitationSnapshot(arxiv_id="2305.15334", cited_by_count=5, taken_on=today - timedelta(days=7)))
        s.add(CitationSnapshot(arxiv_id="2305.15334", cited_by_count=9, taken_on=today))

    out = await citation_trend("2305.15334")
    assert out["verdict"] == "rising"
    assert out["delta"] == 4 and out["span_days"] == 7

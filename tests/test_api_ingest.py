"""`POST /api/v1/papers/ingest` 的对外契约（批量手动入库，架构 §7 补口）。

全部离线，一条真请求都不发 arXiv：
- id 模式：把 `manual` 命名空间里的 `ArxivClient` 换成桩（桩构造**真实的**
  `ArxivPaper` dataclass，None = 查无此篇，与真实 `get_paper` 的返回 schema 一致），
  分块/落库/索引刷新走真实 pipeline 零件 —— 这正是本接口该被验的方式；
- 关键词模式：把 `manual.ingest`（即 `pipeline.ingest`）换成桩，返回**真实的**
  `IngestResult/IngestStats` dataclass。桩返回结构凭想象造字段是这个文件最不能犯的错。

`FETCH_HTML=false / OPENALEX_ENABLED=false`：分块走摘要兜底（生产同一条路），
OpenAlex 补数分支按 settings 跳过 —— 与 pipeline 主流程的开关语义逐字一致。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

import app.db.models  # noqa: F401 - 注册表到 metadata
import app.ingest.manual as manual
from app.agents.nodes.ensure_corpus import load_corpus
from app.agents.tools.hybrid_search import get_shared_retriever, reset_shared_retriever
from app.api.v1 import papers as papers_module
from app.db.models import Paper, PaperChunk
from app.db.session import session_scope
from app.ingest.arxiv_client import ArxivPaper
from app.ingest.pipeline import IngestResult, IngestStats
from app.retrieval.hybrid_retriever import ChunkRecord
from tests.api_support import api_client

OFFLINE_ENV = {"FETCH_HTML": "false", "OPENALEX_ENABLED": "false"}

# 桩目录：两个合法 id + 一个 arXiv 上必然不存在的 id（get_paper 返回 None 的分支）
_KNOWN_IDS = {"2401.99001", "2401.99002"}
_MISSING_ID = "0000.00001"


@pytest.fixture
async def client(monkeypatch, tmp_path):
    # 单例检索器跨用例残留会指向别的 tmp 目录：入库前丢弃，索引断言才只属于本用例
    reset_shared_retriever()
    async with api_client(monkeypatch, tmp_path, **OFFLINE_ENV) as c:
        yield c
    reset_shared_retriever()


def _arxiv_paper(aid: str) -> ArxivPaper:
    """照抄 `arxiv_client.ArxivPaper` 的真实字段构造桩返回值（frozen dataclass，字段一个不差）。"""
    return ArxivPaper(
        arxiv_id=aid,
        title=f"Manual Ingest Fixture {aid}",
        abstract="We evaluate hybrid retrieval with a reranker; code at https://github.com/example/repo.",
        authors=["A. Author"],
        published=datetime(2024, 1, 15, tzinfo=UTC),
        updated=datetime(2024, 1, 16, tzinfo=UTC),
        primary_category="cs.CL",
        categories=["cs.CL", "cs.IR"],
        comment="Accepted at ACL 2024",
        journal_ref=None,
        pdf_url=f"https://arxiv.org/pdf/{aid}",
        html_url=f"https://arxiv.org/html/{aid}",
    )


class _FakeArxivClient:
    """ArxivClient 桩：只实现 manual 用到的 `get_paper`，**不发网络、不限速等待**。

    语义与真实客户端逐条对齐：未知 id 返回 None；构造签名兼容 `ArxivClient(settings)`；
    每次实际发出的 id 记进类级 `all_requested`，用于断言"缓存命中一篇都不发"。
    """

    all_requested: list[str] = []

    def __init__(self, settings=None, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def get_paper(self, arxiv_id: str) -> ArxivPaper | None:
        _FakeArxivClient.all_requested.append(arxiv_id)
        if arxiv_id in _KNOWN_IDS:
            return _arxiv_paper(arxiv_id)
        return None


# --------------------------------------------------------------------------- id 模式（端到端、离线）
async def test_ingest_ids_end_to_end_offline(client, monkeypatch):
    _FakeArxivClient.all_requested = []
    monkeypatch.setattr(manual, "ArxivClient", _FakeArxivClient)

    # 预先入库一篇（带 chunk）：它应当走缓存分支、一次网络都不发
    async with session_scope() as db:
        db.add(
            Paper(
                arxiv_id="2401.99003",
                title="Already Ingested Paper",
                abstract="cached body",
                authors=["Z. Author"],
                primary_category="cs.CL",
            )
        )
        db.add(
            PaperChunk(
                arxiv_id="2401.99003",
                chunk_index=0,
                section="abstract",
                source="abstract",
                chunk_type="text",
                text="[Already Ingested Paper] [abstract]\ncached body",
            )
        )

    resp = await client.post(
        "/api/v1/papers/ingest",
        # 重复项 + v2 后缀：归一化短 id 后按 4 篇唯一 id 处理
        json={"arxiv_ids": ["2401.99001", "2401.99001v2", "2401.99002", "2401.99003", _MISSING_ID]},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "ids"
    assert body["requested"] == 4  # 去重归一后
    assert body["fetched"] == 2  # 新增入库
    assert body["cached"] == 1  # 命中已有语料，未发网络请求
    assert body["failed"] == 1  # 查无此篇，逐条带 id，不静默吞
    assert body["failures"][0]["arxiv_id"] == _MISSING_ID
    assert "arxiv" in body["failures"][0]["error"].lower()
    by_id = {p["arxiv_id"]: p for p in body["papers"]}
    assert by_id["2401.99001"]["status"] == "fetched"
    assert by_id["2401.99003"]["status"] == "cached"

    # 桩只被调用了 2 次（缓存篇 99003 与查无此篇之外没有多余请求）
    assert sorted(_FakeArxivClient.all_requested) == sorted(_KNOWN_IDS | {_MISSING_ID})

    # 单篇失败不连坐：另两篇真实落库，且有可建索引的 chunk
    async with session_scope() as db:
        titles = (
            await db.execute(select(Paper.title).where(Paper.arxiv_id == "2401.99001"))
        ).scalars().all()
        n_chunks = (
            await db.execute(
                select(PaperChunk).where(PaperChunk.arxiv_id.in_(_KNOWN_IDS))
            )
        ).scalars().all()
    assert titles == ["Manual Ingest Fixture 2401.99001"]
    assert len(n_chunks) >= 2

    # 索引跟上：corpus_digest 判定与 ensure_corpus 同一通道，重建后必须"当前"
    assert body["index"]["refreshed"] is True
    assert body["index"]["action"] == "rebuilt"
    records, titles_map = await load_corpus()
    assert get_shared_retriever().index_is_current(records, titles_map) is True

    # 第二次全部命中缓存：fetched=0、一个网络请求都不补发、索引按指纹 reused（不是每次重建）
    _FakeArxivClient.all_requested = []
    resp2 = await client.post(
        "/api/v1/papers/ingest",
        json={"arxiv_ids": ["2401.99001", "2401.99002", "2401.99003"]},
    )
    assert resp2.status_code == 200, resp2.text
    body2 = resp2.json()
    assert (body2["fetched"], body2["cached"], body2["failed"]) == (0, 3, 0)
    assert _FakeArxivClient.all_requested == []
    assert body2["index"]["action"] == "reused"


# --------------------------------------------------------------------------- 路由委托与参数转接
async def test_ingest_ids_route_gates_and_delegates(client, monkeypatch):
    seen: dict = {}

    async def fake_by_ids(ids):
        seen["ids"] = ids
        return {"mode": "ids", "requested": len(ids), "fetched": 0, "cached": len(ids), "failed": 0, "failures": [], "papers": [], "index": {"refreshed": True, "action": "reused"}, "elapsed_s": 0.0}

    monkeypatch.setattr(papers_module, "ingest_by_ids", fake_by_ids)
    resp = await client.post(
        "/api/v1/papers/ingest", json={"arxiv_ids": [" 2305.15334 ", "2401.00001"]}
    )
    assert resp.status_code == 200
    # 闸门在前：strip 之后才进服务层（判的是一份、用的也必须是那一份）
    assert seen["ids"] == ["2305.15334", "2401.00001"]


async def test_ingest_query_route_translates_params(client, monkeypatch):
    seen: dict = {}

    async def fake_by_query(query, *, categories=None, year_gte=None, year_lte=None, max_results=50):
        seen.update(
            query=query, categories=categories, year_gte=year_gte, year_lte=year_lte, max_results=max_results
        )
        return {"mode": "query", "query": query, "requested": 0, "fetched": 0, "cached": 0, "failed": 0, "failures": [], "papers": [], "index": {"refreshed": True, "action": "reused"}, "elapsed_s": 0.0}

    monkeypatch.setattr(papers_module, "ingest_by_query", fake_by_query)
    resp = await client.post(
        "/api/v1/papers/ingest",
        json={"query": "hybrid retrieval", "categories": ["cs.CL", " cs.LG "], "year_gte": 2024, "max_results": 20},
    )
    assert resp.status_code == 200, resp.text
    assert seen["query"] == "hybrid retrieval"
    assert seen["categories"] == ["cs.CL", "cs.LG"]
    assert seen["year_gte"] == 2024 and seen["year_lte"] is None and seen["max_results"] == 20


async def test_ingest_query_service_maps_to_pipeline_ingest(client, monkeypatch):
    """关键词模式必须**整段转调 `pipeline.ingest()`**，且计数/失败逐条透出（桩照抄真实 schema）。"""
    seen: dict = {}

    async def fake_ingest(query, **kwargs):
        seen["query"] = query
        seen.update(kwargs)
        rec = ChunkRecord(
            chunk_id=1,
            arxiv_id="2401.99100",
            section="abstract",
            text="[T] [abstract]\nbody",
            source="abstract",
            chunk_type="text",
            year=2024,
            primary_category="cs.CL",
        )
        return IngestResult(
            records=[rec],
            titles={"2401.99100": "T", "2401.99101": "U"},
            corpus_ready=True,
            stats=IngestStats(
                fetched=1,
                cached=1,
                failed=1,
                errors=[{"arxiv_id": "2401.99102", "error": "no chunks produced"}],
            ),
        )

    monkeypatch.setattr(manual, "ingest", fake_ingest)
    resp = await client.post(
        "/api/v1/papers/ingest",
        json={"query": "hybrid retrieval", "categories": ["cs.CL"], "year_gte": 2024, "year_lte": 2025},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # API 侧 year_gte/year_lte 只在服务层翻回 pipeline 的 year_from/year_to 命名
    assert seen["categories"] == ["cs.CL"]
    assert seen["year_from"] == 2024 and seen["year_to"] == 2025
    assert (body["fetched"], body["cached"], body["failed"]) == (1, 1, 1)
    assert body["failures"] == [{"arxiv_id": "2401.99102", "error": "no chunks produced"}]
    assert body["requested"] == 2  # 命中并入库的论文数（缓存 + 新增）


# --------------------------------------------------------------------------- 422 形状闸门
async def test_ingest_requires_exactly_one_mode(client):
    neither = await client.post("/api/v1/papers/ingest", json={})
    assert neither.status_code == 422
    assert neither.json()["detail"]["code"] == "invalid_input"

    both = await client.post(
        "/api/v1/papers/ingest", json={"arxiv_ids": ["2305.15334"], "query": "hybrid retrieval"}
    )
    assert both.status_code == 422
    assert both.json()["detail"]["field"] == "arxiv_ids/query"


async def test_ingest_shape_gates(client, monkeypatch):
    """坏形状一律 422 且**不进服务层**（闸门在最外层，桩函数一次都不该被调到）。"""
    calls: list = []

    async def spy(payload):
        calls.append(payload)
        return {"mode": "ids", "ok": True}

    monkeypatch.setattr(papers_module, "ingest_by_ids", spy)
    monkeypatch.setattr(papers_module, "ingest_by_query", lambda *a, **k: spy(a))

    bad_bodies = [
        {"arxiv_ids": ["bad id"]},          # 空格 → _guard_ids 拦
        {"arxiv_ids": ["a/b"]},             # `/` → 旧式 id 明确不支持（hint 里给出）
        {"arxiv_ids": ["x" * 65]},          # 超长
        {"arxiv_ids": []},                  # 空列表 → pydantic
        {"arxiv_ids": [f"2401.{i:05d}" for i in range(51)]},  # 超过 MAX_IDS_PER_REQUEST
        {"query": "x"},                     # min_length=2
        {"query": "hybrid", "categories": ["cs CL"]},  # 分类含空格 → 不进 arXiv 检索式
        {"query": "hybrid", "year_gte": 2025, "year_lte": 2024},  # 区间倒挂
        {"query": "hybrid", "max_results": 51},  # 与 ids 上限同源
        {"arxiv_ids": ["2305.15334"], "top_k": 3},  # extra=forbid
    ]
    for body in bad_bodies:
        resp = await client.post("/api/v1/papers/ingest", json=body)
        assert resp.status_code == 422, body
    assert calls == []

    # 反向确认：好形状确实走到了服务层
    ok = await client.post("/api/v1/papers/ingest", json={"arxiv_ids": ["2305.15334"]})
    assert ok.status_code == 200 and ok.json()["ok"] is True
    assert calls == [["2305.15334"]]


async def test_ingest_route_is_registered_before_paper_id(client):
    """路由注册哨兵：静态路径 `/ingest` 若没声明在 `/{arxiv_id}` 之前的纪律被破坏，
    openapi 里这条路径会整段消失或形状走样（同 venue-tier 的教训， papers.py 头注）。"""
    spec = (await client.get("/openapi.json")).json()["paths"]
    assert "/api/v1/papers/ingest" in spec
    assert "post" in spec["/api/v1/papers/ingest"]

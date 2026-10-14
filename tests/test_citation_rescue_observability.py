"""引用定位救回统计的可观测面：SSE `citation_rescue` 事件 + `citations` 表两个新列。

此前这些事实只存在于后端日志（`read.done` 的 `reanchored` / `recovered_by`、
`verify.citation_reanchored`），事后想回答"根因是不是真的修掉了"只能 grep 日志。
本文件钉住四条链路，每条都要求 **rescued=0 也照发/照存** ——
只在非零时出现的指标无法区分"根因已修掉"与"统计断线了"：

1. read 节点：citations 携带 `resolution_how`，并发出 `citation_rescue` 事件；
2. verify 节点：重锚计数进 `citation_rescue`（node="verify"，带路径分布）；
3. finalize 落库：`citations.resolution_how` / `citations.verify_reanchored` 两个新列；
4. 最小迁移：`init_db` 对**已存在的旧 schema 表**补列，幂等、不丢数据
   （没有 Alembic，`create_all` 不会给已有表加列 —— 这段迁移是唯一通道）。

隔离方式沿用 `test_evidence_resolution.py`（provider 切 mock + 打桩 `structured_invoke`），
零 key、零联网。
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from app.agents.nodes import reader as reader_node
from app.agents.nodes import verifier as verifier_node
from app.agents.schemas import ExtractionItem, ReadOutput, VerificationItem, VerifyOutput
from app.core.config import get_settings

_A = "2401.00001"

QUOTE_101 = "BM25 scores are unbounded."
QUOTE_102 = "A cross-encoder reranker rescores the fused candidate list."

RETRIEVED: list[dict[str, Any]] = [
    {
        "arxiv_id": _A,
        "title": "Hybrid Retrieval at Work",
        "chunks": [
            {
                "chunk_id": 101,
                "section": "abstract",
                "source": "abstract",
                "text": f"RRF fusion combines dense and lexical rankings. {QUOTE_101}",
            },
            {
                "chunk_id": 102,
                "section": "methods",
                "source": "html",
                "text": QUOTE_102,
            },
        ],
    },
]


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _rescue_events(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in out.get("events") or [] if e.get("type") == "citation_rescue"]


# ---------------------------------------------------------------------------
# 1) read 节点：resolution_how 随证据走 + citation_rescue 事件（0 也照发）
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_read(monkeypatch):
    def _set(claims: list[ExtractionItem]) -> None:
        async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
            return ReadOutput(claims=claims), {}

        monkeypatch.setattr(reader_node, "structured_invoke", _invoke)

    return _set


def _read_state() -> dict[str, Any]:
    return {"question": "混合检索怎么做融合？", "retrieved": RETRIEVED, "stage": "corpus"}


async def test_read_emits_rescue_stats_even_when_all_exact(fake_read):
    """根因修掉后的常态：全部 exact、救回 0 条 —— 事件仍然必须发出来。"""
    fake_read(
        [
            ExtractionItem(
                paper_id=_A, chunk_id=101, claim="词面得分没有上界。", quote=QUOTE_101
            )
        ]
    )

    out = await reader_node.read_node(_read_state())

    assert out["citations"][0]["resolution_how"] == "exact"
    events = _rescue_events(out)
    assert len(events) == 1, "rescued=0 也必须发事件：只在非零时出现等于把回归藏起来"
    ev = events[0]
    assert ev["node"] == "read"
    assert ev["checked"] == 1
    assert ev["rescued"] == 0
    assert ev["dropped"] == 0
    assert ev["recovered_by"] == {}


async def test_read_rescue_stats_count_paths_and_drops(fake_read):
    """一条走 quote 重锚救回、一条真幻觉被剔除：两个数字都要如实进事件。"""
    fake_read(
        [
            ExtractionItem(
                paper_id=_A, chunk_id=999, claim="重排器重打融合候选。", quote=QUOTE_102
            ),
            ExtractionItem(
                paper_id=_A, chunk_id=999, claim="它拿下了所有榜单。",
                quote="state-of-the-art on every single benchmark",
            ),
        ]
    )

    out = await reader_node.read_node(_read_state())

    assert [c["resolution_how"] for c in out["citations"]] == ["quote"]
    ev = _rescue_events(out)[0]
    assert ev["checked"] == 2
    assert ev["rescued"] == 1
    assert ev["recovered_by"] == {"quote": 1}
    assert ev["dropped"] == 1
    assert ev["drop_reasons"] == {"quote_not_found": 1}


# ---------------------------------------------------------------------------
# 2) verify 节点：重锚计数进 citation_rescue（node="verify"）
# ---------------------------------------------------------------------------
def _stub_soft(monkeypatch, items: list[VerificationItem] | None = None):
    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        return VerifyOutput(items=items or []), {}

    monkeypatch.setattr(verifier_node, "structured_invoke", _invoke)


def _citation(**over: Any) -> dict[str, Any]:
    cit: dict[str, Any] = {
        "claim": "词面得分没有上界。",
        "paper_id": _A,
        "chunk_id": 101,
        "quote": QUOTE_101,
        "locator": "abstract",
        "source": "abstract",
        "verified": False,
        "resolution_how": "exact",
    }
    cit.update(over)
    return cit


def _verify_state(citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": "混合检索怎么做融合？", "retrieved": RETRIEVED, "citations": citations}


async def test_verify_emits_rescue_stats_with_zero_when_nothing_reanchored(monkeypatch):
    _stub_soft(monkeypatch, [VerificationItem(claim="词面得分没有上界。", supported=True)])

    out = await verifier_node.verify_node(_verify_state([_citation()]))

    events = _rescue_events(out)
    assert len(events) == 1
    assert events[0]["node"] == "verify"
    assert events[0]["checked"] == 1
    assert events[0]["rescued"] == 0
    assert events[0]["recovered_by"] == {}


async def test_verify_reports_reanchored_citation_with_its_path(monkeypatch):
    """编号打错 + 引文在未渲染片段里：verify 重锚救回，事件里带路径分布。"""
    _stub_soft(
        monkeypatch, [VerificationItem(claim="重排器重打融合候选。", supported=True)]
    )

    out = await verifier_node.verify_node(
        _verify_state(
            [_citation(claim="重排器重打融合候选。", chunk_id=999, quote=QUOTE_102)]
        )
    )

    cit = out["citations"][0]
    assert cit["resolved_chunk_id"] == 102
    # read 侧留下的定位路径不许被 verify 覆盖：两列语义不同（谁救的记谁头上）
    assert cit["resolution_how"] == "exact"
    ev = _rescue_events(out)[0]
    assert ev["node"] == "verify"
    assert ev["rescued"] == 1
    assert ev["recovered_by"] == {"quote": 1}


# ---------------------------------------------------------------------------
# 3) finalize 落库：citations.resolution_how / verify_reanchored
# ---------------------------------------------------------------------------
@pytest.fixture
async def db(monkeypatch, tmp_path):
    """临时库 + 替换 session_scope/init_db（沿用 test_verify_reanchor_and_note 的做法）。"""
    from contextlib import asynccontextmanager

    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.db.models  # noqa: F401 - 注册全部 ORM 表
    from app.db import session as session_mod
    from app.db.base import Base

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rescue.db'}")
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

    async def _noop() -> None:
        return None

    monkeypatch.setattr(session_mod, "session_scope", _scope)
    monkeypatch.setattr(session_mod, "init_db", _noop)
    yield _scope
    await engine.dispose()


async def test_finalize_persists_resolution_how_and_verify_reanchored(db, monkeypatch):
    from sqlalchemy import select

    from app.agents.nodes import finalize as finalize_mod
    from app.db.models import Citation, Paper, PaperChunk

    async def noop(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(finalize_mod, "_write_memory", noop)

    async with db() as session:
        session.add(Paper(arxiv_id=_A, title="Hybrid Retrieval at Work"))
        chunk = PaperChunk(arxiv_id=_A, chunk_index=0, section="abstract", text=f"x {QUOTE_101}")
        session.add(chunk)
        await session.flush()
        chunk_id = chunk.id

    await finalize_mod._persist(  # noqa: SLF001 - 测试内直调内部入口
        {
            "run_id": "r-rescue",
            "session_id": "s-rescue",
            "question": "混合检索怎么做融合？",
            "retrieved": [],
            "readings": [],
            "citations": [
                # read 侧 exact 命中、verify 没重锚
                {**_citation(), "chunk_id": chunk_id, "verified": True, "verify_note": ""},
                # read 侧 quote 救回、verify 又重锚过一次（真实定位在 resolved_* 里）
                {
                    **_citation(claim="重排器重打融合候选。", resolution_how="quote"),
                    "chunk_id": 999,
                    "resolved_chunk_id": chunk_id,
                    "verified": True,
                    "verify_note": "",
                },
            ],
        },
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    async with db() as session:
        rows = list((await session.execute(select(Citation).order_by(Citation.id))).scalars())

    assert [r.resolution_how for r in rows] == ["exact", "quote"]
    assert [r.verify_reanchored for r in rows] == [False, True]


# ---------------------------------------------------------------------------
# 4) 最小迁移：旧 schema 的既有库补列，幂等、不丢数据
# ---------------------------------------------------------------------------
# 迁移前 citations 表的真实 DDL（data/sqlite/paperpilot.db 里的形态）
_OLD_CITATIONS_DDL = """
CREATE TABLE citations (
    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
    run_id VARCHAR(64) NOT NULL,
    claim TEXT NOT NULL,
    chunk_id INTEGER,
    arxiv_id VARCHAR(64),
    quote TEXT NOT NULL,
    locator VARCHAR(128),
    source VARCHAR(16),
    verified BOOLEAN,
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL
)
"""


async def test_init_db_migrates_existing_citations_table_idempotently(monkeypatch, tmp_path):
    """`create_all` 不会给已存在的表加列 —— 迁移是唯一通道，且必须能重复跑。"""
    from app.db.session import dispose_engine, init_db

    db_path = tmp_path / "legacy.db"
    con = sqlite3.connect(db_path)
    con.execute(_OLD_CITATIONS_DDL)
    con.execute(
        "INSERT INTO citations (run_id, claim, quote, verified) VALUES (?, ?, ?, ?)",
        ("legacy-run", "旧数据论断", "legacy quote", 1),
    )
    con.commit()
    con.close()

    monkeypatch.setenv("SQLITE_URL", f"sqlite+aiosqlite:///{db_path}")
    get_settings.cache_clear()
    await dispose_engine()
    try:
        await init_db()
        await init_db()  # ★ 第二遍必须安静通过（幂等）
    finally:
        await dispose_engine()
        get_settings.cache_clear()

    con = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in con.execute("PRAGMA table_info(citations)")}
        assert {"resolution_how", "verify_reanchored"} <= columns
        rows = con.execute(
            "SELECT claim, verified, resolution_how, verify_reanchored FROM citations"
        ).fetchall()
        # 历史行一条不少，新列是 NULL（= 当时没记录），不是被回填成假值
        assert rows == [("旧数据论断", 1, None, None)]
    finally:
        con.close()


# ---------------------------------------------------------------------------
# 5) /citations 接口透出新字段
# ---------------------------------------------------------------------------
async def test_citations_endpoint_exposes_resolution_fields(monkeypatch, tmp_path):
    from app.db import session as db_session
    from app.db.models import Citation
    from tests.api_support import api_client

    async with api_client(monkeypatch, tmp_path) as client:
        async with db_session.session_scope() as session:
            session.add(
                Citation(
                    run_id="rescue-run",
                    claim="c1",
                    quote="q1",
                    verified=True,
                    resolution_how="swapped_chunk",
                    verify_reanchored=True,
                )
            )
            session.add(Citation(run_id="rescue-run", claim="c2", quote="q2", verified=True))

        body = (await client.get("/api/v1/research/rescue-run/citations")).json()
        by_claim = {c["claim"]: c for c in body["citations"]}
        assert by_claim["c1"]["resolution_how"] == "swapped_chunk"
        assert by_claim["c1"]["verify_reanchored"] is True
        # exact 是 read 侧写进来的常态值；没写过的（旧数据形态）如实返回 None
        assert by_claim["c2"]["resolution_how"] is None
        assert by_claim["c2"]["verify_reanchored"] is None

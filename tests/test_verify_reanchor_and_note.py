"""verify 侧的重锚口径 + 「校验结论写回并落库」这条链路的测试。

`tests/test_evidence_resolution.py` 把 **read 侧**的严格口径（只认渲染给模型看过的片段）
钉住了，但 **verify 侧的放宽**只活在 `nodes/evidence.py` 的一段注释里，从来没有节点级用例
看守（架构 §4.2 / §10：重写轮里 citations 可能来自上一轮的渲染集，而 verify 判的是
"这条引文在不在原文里"，与模型当时看到过什么无关）。口径一旦被人"顺手统一"成严格侧，
症状是 revise 轮把好引用二次杀掉、反思永远出不来 —— 日志里却看不出异常。

同一条链路上还有三条无人看守的承诺：
1. 两层校验取**与**：LLM 软校验能否决硬校验放行的"语义夸大"；反过来软校验自己挂了
   时不许把硬校验已经确认过的真引用一起判死（`except` 分支的存在理由）；
2. 结论必须写回 `citations[].verified` —— 它是架构 §12「引用准确率 ≥ 90%」自动口径
   **唯一的数据源**，不写回这个指标就只能靠人工，等于没有；
3. 失败原因要经 `verify_note` 一路落进 `citations.note`，否则"为什么没过"只存在于日志里，
   人工抽检（§12 的第二口径）无从查证。

隔离方式沿用 `test_evidence_resolution.py`（provider 切 mock + 打桩 `structured_invoke`）
与 `test_memory_in_graph.py`（临时 SQLite + 替换 `session_scope`），零 key、零联网。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.nodes import verifier as verifier_node
from app.agents.nodes.evidence import build_chunk_lookup, build_paper_chunks, resolve_citation
from app.agents.schemas import VerificationItem, VerifyOutput
from app.core.config import get_settings

_A = "2401.00001"
_B = "2401.00002"

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
            # 102 是"这一轮没渲染给模型看"的那一片（read 侧因此不许锚到它）
            {
                "chunk_id": 102,
                "section": "methods",
                "source": "html",
                "text": QUOTE_102,
            },
        ],
    },
    {
        "arxiv_id": _B,
        "title": "Survey of Literature Review Generation",
        "chunks": [
            {
                "chunk_id": 201,
                "section": "abstract",
                "source": "abstract",
                "text": "Outline-first pipelines produce section-level citations.",
            }
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


def _stub_soft(monkeypatch, items: list[VerificationItem] | None = None, *, boom: bool = False):
    """打桩软校验那次 LLM 调用：返回给定的判定，或直接抛异常（模拟 provider 挂了）。"""

    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        if boom:
            raise RuntimeError("provider down")
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
    }
    cit.update(over)
    return cit


def _state(citations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question": "混合检索怎么做融合？",
        "retrieved": RETRIEVED,
        "citations": citations,
        "revision_count": 1,  # 重写轮：渲染集与上一轮不同，正是 verify 侧要放宽的场景
    }


# ---------------------------------------------------------------------------
# 1) ★ verify 侧的放宽口径（与 read 侧刻意不同）
# ---------------------------------------------------------------------------
async def test_verify_reanchors_into_a_chunk_that_read_side_would_refuse(monkeypatch):
    """同一条"编号打错 + 引文属于未渲染片段"的引用：read 侧拒、verify 侧必须救。

    两侧口径不同是**设计**而不是疏漏（见 `nodes/evidence.py` 的 `restrict_to_rendered`），
    所以这里把"verify 节点真的走了宽松那条路"钉成行为断言，而不只是纯函数断言。
    """
    papers = build_paper_chunks(RETRIEVED)
    lookup = build_chunk_lookup(RETRIEVED)
    strict = resolve_citation(
        papers,
        paper_id=_A,
        chunk_id=999,  # 模型打错的自增编号
        quote=QUOTE_102,
        rendered_chunk_ids={101},
        restrict_to_rendered=True,
        chunk_lookup=lookup,
    )
    assert strict.ok is False, "前置条件：read 侧（只看渲染过的片段）应当判不出来"

    _stub_soft(monkeypatch, [VerificationItem(claim="交叉编码器会重打融合后的候选。", supported=True)])
    out = await verifier_node.verify_node(
        _state(
            [
                _citation(
                    claim="交叉编码器会重打融合后的候选。",
                    chunk_id=999,
                    quote=QUOTE_102,
                )
            ]
        )
    )

    assert out["verification"]["passed"] is True, "verify 侧应允许退到该篇其余片段"
    cit = out["citations"][0]
    assert cit["verified"] is True
    assert cit["verify_note"] == ""
    # ★ 定位被纠正过必须留下痕迹，否则这条引用在库里仍指向一个不存在的片段
    assert cit["chunk_id"] == 999, "merge_evidence 的去重键含 chunk_id，不许改它"
    assert cit["resolved_chunk_id"] == 102


async def test_corrected_anchor_does_not_duplicate_the_citation(monkeypatch):
    """写回定位**不能**让同一条引用变成两条。

    `merge_evidence` 按 `(paper_id, chunk_id, claim)` 覆盖，所以 verify 直接改写
    `chunk_id` 会让"覆盖"变成"追加"：状态里同时留着纠正前（指向空片段、会被外键保护
    写成 NULL）与纠正后两条，症状是引用数凭空变多、准确率被自己造的那条假记录拉低。
    """
    from app.agents.state import merge_evidence

    _stub_soft(monkeypatch, [VerificationItem(claim="交叉编码器会重打融合后的候选。", supported=True)])
    original = _citation(claim="交叉编码器会重打融合后的候选。", chunk_id=999, quote=QUOTE_102)
    out = await verifier_node.verify_node(_state([original]))

    merged = merge_evidence([original], out["citations"])
    assert len(merged) == 1, f"纠正定位应当是覆盖而不是追加，实得 {len(merged)} 条"
    assert merged[0]["resolved_chunk_id"] == 102
    assert merged[0]["verified"] is True


# ---------------------------------------------------------------------------
# 2) 结论写回 verified + 失败原因写进 verify_note
# ---------------------------------------------------------------------------
async def test_verify_writes_back_verified_and_reason(monkeypatch):
    """一条真、一条编造 → verified 分别是 True/False，失败那条带得上"为什么没过"。"""
    _stub_soft(
        monkeypatch,
        [
            VerificationItem(claim="词面得分没有上界。", supported=True),
            VerificationItem(claim="检索完全可以省掉。", supported=True),  # 软校验放行也拦不住硬校验
        ],
    )

    out = await verifier_node.verify_node(
        _state(
            [
                _citation(),
                _citation(claim="检索完全可以省掉。", quote="we prove that retrieval is unnecessary"),
            ]
        )
    )

    by_claim = {c["claim"]: c for c in out["citations"]}
    assert by_claim["词面得分没有上界。"]["verified"] is True
    assert by_claim["词面得分没有上界。"]["verify_note"] == ""

    bad = by_claim["检索完全可以省掉。"]
    assert bad["verified"] is False
    assert bad["verify_note"] == "引文未能在原文中逐字匹配", "失败原因必须随证据带下去（finalize 据此落 citations.note）"

    verification = out["verification"]
    assert verification["passed"] is False  # 全部通过才算 passed（阈值刻意不放宽）
    assert verification["score"] == pytest.approx(0.5)
    assert verification["unsupported"] == ["检索完全可以省掉。"]


async def test_llm_soft_veto_overrides_a_hard_pass(monkeypatch):
    """逐字对得上但语义被夸大：软校验必须能否决硬校验的"通过"。

    这是"两层校验缺一不可"的后一半 —— 只有硬校验时，断章取义/加强版论断会一路绿灯。
    """
    _stub_soft(monkeypatch, [VerificationItem(claim="所有榜单都被刷爆了。", supported=False)])

    out = await verifier_node.verify_node(
        _state([_citation(claim="所有榜单都被刷爆了。", quote=QUOTE_101)])
    )

    cit = out["citations"][0]
    assert cit["verified"] is False
    assert "LLM" in cit["verify_note"]
    assert out["verification"]["unsupported"] == ["所有榜单都被刷爆了。"]


async def test_soft_check_failure_does_not_kill_hard_verdict(monkeypatch):
    """软校验那次 LLM 调用整体挂了：硬校验的结论照常成立，不许"判不了就当假"。

    否则 provider 一抖动，一次研究里所有真引用都会被写成 verified=False，
    自动口径（§12）跟着归零 —— 那是把基础设施抖动放大成质量事故。
    """
    _stub_soft(monkeypatch, boom=True)

    out = await verifier_node.verify_node(
        _state(
            [
                _citation(),  # 真引文：硬校验通过 → 仍然 True
                _citation(claim="检索完全可以省掉。", quote="we prove that retrieval is unnecessary"),
            ]
        )
    )

    by_claim = {c["claim"]: c for c in out["citations"]}
    assert by_claim["词面得分没有上界。"]["verified"] is True
    assert by_claim["检索完全可以省掉。"]["verified"] is False
    assert out["verification"]["score"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3) ★ verify_note → citations.note 的落库这一程
# ---------------------------------------------------------------------------
async def test_persist_stores_verified_and_note(monkeypatch, tmp_path):
    """finalize 落库时必须把 `verified` 与失败原因写进 `citations` 表。

    接口 `/research/{run_id}/citations` 与抽检脚本导出的 `auto_verified`/`verify_note` 两列
    都直接吃这张表：这一程断了，前端与人工口径看到的就永远是空值。
    """
    from contextlib import asynccontextmanager

    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.db.models  # noqa: F401 - 注册全部 ORM 表
    from app.agents.nodes import finalize as finalize_mod
    from app.db import session as session_mod
    from app.db.base import Base
    from app.db.models import Citation, Paper, PaperChunk

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'note.db'}")
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

    async def noop(*args: Any, **kwargs: Any) -> None:
        return None

    # `_persist` 在函数体内 import session_scope / init_db → 打桩模块属性即可生效
    monkeypatch.setattr(session_mod, "session_scope", _scope)
    monkeypatch.setattr(session_mod, "init_db", _noop)
    monkeypatch.setattr(finalize_mod, "_write_memory", noop)

    async with _scope() as session:
        session.add(Paper(arxiv_id=_A, title="Hybrid Retrieval at Work"))
        chunk = PaperChunk(arxiv_id=_A, chunk_index=0, section="abstract", text=f"x {QUOTE_101}")
        session.add(chunk)
        await session.flush()
        chunk_id = chunk.id

    await finalize_mod._persist(  # noqa: SLF001 - 测试内直调内部入口
        {
            "run_id": "r-note",
            "session_id": "s-note",
            "question": "混合检索怎么做融合？",
            "retrieved": [],
            "readings": [],
            "citations": [
                {**_citation(), "chunk_id": chunk_id, "verified": True, "verify_note": ""},
                {
                    **_citation(claim="检索完全可以省掉。", verified=False),
                    "chunk_id": chunk_id,
                    "verify_note": "引文未能在原文中逐字匹配",
                },
                # verify 重锚过的那条：原 chunk_id 在库里不存在，真实定位在 resolved_* 里
                {
                    **_citation(claim="交叉编码器会重打融合后的候选。"),
                    "chunk_id": 999,
                    "resolved_chunk_id": chunk_id,
                    "verified": True,
                    "verify_note": "",
                },
            ],
        },
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )

    async with _scope() as session:
        rows = list((await session.execute(select(Citation).order_by(Citation.id))).scalars().all())
    await engine.dispose()

    assert [r.verified for r in rows] == [True, False, True]
    # 通过的引用保持 None（区分"没写"与"写了但为空"）；失败的必须带上原因
    assert rows[0].note is None
    assert rows[1].note == "引文未能在原文中逐字匹配"
    # ★ 重锚那条落的是**纠正后**的片段，而不是被外键保护写成 NULL
    assert rows[2].chunk_id == chunk_id

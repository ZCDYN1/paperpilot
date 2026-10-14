"""F9 追问服务 `app/followup/answer.py` 的单元测试。

接口层测试见 `tests/test_api_corpus_routes.py`；这里钉的是三件只有服务层能证明的事：
1. **绝不重新检索**（`retrieved_again=False` 是这条接口存在的全部意义）；
2. 上一轮被判无支撑的论断不会因为"它已经在库里"就变成可复用的事实；
3. 答案里没有有效 `[n]` 的句子会被自己列出来 —— 包括模型编出来的 `[7]`（越界编号）。
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 注册表到 metadata
from app.db.base import Base
from app.db.models import Citation, ResearchRun, ResearchSession
from app.followup import RunNotFoundError, answer_from_read_papers
from app.followup import answer as answer_mod
from app.followup.answer import _ungrounded_sentences


@pytest.fixture
async def scope(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'followup.db'}")
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

    monkeypatch.setattr(answer_mod, "session_scope", _scope)
    yield _scope
    await engine.dispose()


async def _seed(scope, run_id: str, citations: list[dict]) -> None:
    async with scope() as s:
        if await s.get(ResearchSession, "s1") is None:
            s.add(ResearchSession(id="s1", title="t"))
        s.add(
            ResearchRun(
                id=run_id, session_id="s1", thread_id=run_id, question="q", status="completed"
            )
        )
        for row in citations:
            s.add(Citation(run_id=run_id, arxiv_id="2999.00001", quote="On BEIR nDCG 0.62", **row))


def _fake_llm(monkeypatch, answer: str) -> None:
    async def fake(messages, **kwargs):  # noqa: ARG001
        return answer, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    monkeypatch.setattr(answer_mod, "text_invoke", fake)


async def test_unknown_run_raises_typed_error(scope):
    with pytest.raises(RunNotFoundError) as excinfo:
        await answer_from_read_papers("nope", "追问")
    assert excinfo.value.run_id == "nope"
    assert isinstance(excinfo.value, LookupError)


async def test_all_unverified_means_no_evidence_and_no_retrieval(scope):
    """库里只剩被判无支撑的引用时，正确行为是拒绝回答，而不是"退化成重新检索一次"。"""
    await _seed(scope, "r1", [{"claim": "方法 D 更优", "verified": False, "locator": "discussion"}])
    out = await answer_from_read_papers("r1", "方法 D 真的更优吗")

    assert out["status"] == "no_evidence"
    assert out["retrieved_again"] is False
    assert out["answer"] == ""
    assert out["citations_usable"] == 0 and out["citations_total"] == 1


async def test_unverified_field_is_three_state(scope, monkeypatch):
    """`verified=None`（当时没校验过）仍然可用 —— 与 `nodes/writer.py` 同一套三态口径。"""
    await _seed(
        scope,
        "r2",
        [
            {"claim": "通过校验的论断", "verified": True, "source": "html"},
            {"claim": "没校验过的论断", "verified": None, "source": "abstract"},
            {"claim": "被判无支撑的论断", "verified": False, "source": "html"},
        ],
    )
    _fake_llm(monkeypatch, "它达到 nDCG 0.62 [1]。另一条也有依据 [2]。")

    out = await answer_from_read_papers("r2", "它的指标是多少")
    assert out["status"] == "ok"
    assert out["citations_usable"] == 2 and out["citations_total"] == 3
    assert {c["claim"] for c in out["used_citations"]} == {"通过校验的论断", "没校验过的论断"}


async def test_llm_failure_degrades_without_losing_the_evidence(scope, monkeypatch):
    async def boom(messages, **kwargs):  # noqa: ARG001
        raise RuntimeError("下游不可用")

    monkeypatch.setattr(answer_mod, "text_invoke", boom)
    await _seed(scope, "r3", [{"claim": "有依据的论断", "verified": True, "source": "html"}])

    out = await answer_from_read_papers("r3", "问题")
    assert out["status"] == "error"
    assert out["answer"] == ""
    assert out["used_citations"] == []  # 没生成内容就不该声称"用过了哪些证据"


def test_ungrounded_sentences_flags_missing_and_out_of_range_marks():
    answer = (
        "# 标题不算句子\n"
        "有依据的一句 [1]。\n"
        "没带依据的一句。\n"
        "引了一个不存在的编号 [7]。\n"
    )
    out = _ungrounded_sentences(answer, {1})
    assert any("没带依据" in s for s in out)
    # 越界编号比不带编号更糟（它看起来可核查），所以同样算无依据
    assert any("不存在的编号" in s for s in out)
    assert not any("有依据的一句" in s for s in out)

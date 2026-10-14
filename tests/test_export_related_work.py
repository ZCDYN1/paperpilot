"""F18 `app/export/related_work.render_related_work_markdown` 的离线测试。

锁死五条：
1. run 不存在 → 抛 `RunNotFoundError`（与 F8 同一个异常类，不是新建的第二个）；
2. **按主题分组**：共享英文锚点的论文进同一小节（确定性聚类，非流水账），
   落单的进「其他工作」；
3. 每篇三句式：做了什么(core_idea) + 相对优势(improvement) + 作者自述局限(limitation)，
   空字段的分句**整个不出现**（不写"未提及"占位，不编造）；
4. `[n]` 按正文首次出现顺序编号、同篇恒同号，结尾一份「## 参考文献」（且只有一份，
   去重口径照抄 markdown.py）；证据强度沿用三态口径：只有 verified is False 剔除，
   None/缺失保留并如实标"摘要级证据"；
5. 对比表为空 → 草稿如实为空话术，不退回临时生成。

临时 SQLite（tmp_path）+ mock provider，不碰真实 data/。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 注册全部表到 metadata
from app.db import session as db_session
from app.db.base import Base
from app.db.models import Citation, ComparisonRecord, Paper, ResearchRun, ResearchSession
from app.export.markdown import RunNotFoundError as MarkdownRunNotFoundError
from app.export.related_work import RunNotFoundError, render_related_work_markdown


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    yield


@pytest.fixture
async def temp_db(monkeypatch, tmp_path):
    """把 `db_session.session_scope` 换到 tmp_path 的临时库（同 test_export_markdown 接法）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'related.db'}")
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


async def _seed_run(scope) -> str:
    run_id = "run-rw-001"
    async with scope() as s:
        s.add(ResearchSession(id="sess-rw", title="t"))
        s.add(
            ResearchRun(
                id=run_id,
                session_id="sess-rw",
                thread_id="sess-rw",
                question="重排方法有哪些演进？",
                status="done",
                started_at=datetime(2026, 10, 12, 10, 0, 0),
            )
        )
        s.add(Paper(arxiv_id="2999.00001", title="CrossRanker: Cross Encoder Reranking", abstract=""))
        s.add(Paper(arxiv_id="2999.00002", title="ListRank: Listwise Reranker Distillation", abstract=""))
        s.add(Paper(arxiv_id="2999.00003", title="VisGPT for Classification", abstract=""))
        # 行 0 / 行 1 共享锚点 "reranker" → 同主题；行 2 落单 → 其他工作；行 3 字段全空。
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=0, arxiv_id="2999.00001", title="CrossRanker",
                method_name="CrossRanker", core_idea="cross encoder reranker 逐对打分",
                datasets="BEIR", improvement="nDCG +3.2", cost="",
                limitation="latency 高", code_url="", venue="EMNLP 2026",
                cited_by_count=3, cited_at="2026-10-12",
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=1, arxiv_id="2999.00002", title="ListRank",
                method_name="ListRank", core_idea="listwise reranker 蒸馏",
                datasets="", improvement="", cost="", limitation="",
                code_url="", venue="", cited_by_count=None, cited_at=None,
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=2, arxiv_id="2999.00003", title="VisGPT",
                method_name="VisGPT", core_idea="vision transformer 分类",
                datasets="", improvement="", cost="", limitation="",
                code_url="", venue="", cited_by_count=None, cited_at=None,
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id, row_index=3, arxiv_id="2999.00004", title="",
                method_name="", core_idea="", datasets="", improvement="",
                cost="", limitation="", code_url="", venue="",
                cited_by_count=None, cited_at=None,
            )
        )
        # 引用证据：001 有正文原文；002 只有 None（三态保留 → 摘要级）；
        # 003 唯一一条是 verified=False（剔除 → 该论文没有可用逐句引用证据）。
        s.add(
            Citation(run_id=run_id, claim="c1", arxiv_id="2999.00001",
                     locator="method", source="html", verified=True)
        )
        s.add(
            Citation(run_id=run_id, claim="c2", arxiv_id="2999.00002",
                     locator=None, source=None, verified=None)
        )
        s.add(
            Citation(run_id=run_id, claim="c3", arxiv_id="2999.00003",
                     locator="abstract", source="html", verified=False, note="未支撑")
        )
    return run_id


async def test_run_not_found_raises_same_error_class(temp_db):
    with pytest.raises(RunNotFoundError) as excinfo:
        await render_related_work_markdown("nope")
    assert excinfo.value.run_id == "nope"
    # 复用 markdown.py 的那个异常类，不是新建的第二个
    assert RunNotFoundError is MarkdownRunNotFoundError


async def test_grouped_by_theme_not_paper_order(temp_db):
    run_id = await _seed_run(temp_db)
    md = await render_related_work_markdown(run_id)

    assert "## 主题：reranker" in md
    assert "## 其他工作" in md
    # reranker 组（行0、行1）出现在其他工作之前；空行也归在其他工作里（不编内容）
    assert md.index("主题：reranker") < md.index("## 其他工作")
    assert md.index("CrossRanker") < md.index("ListRank") < md.index("VisGPT")
    # 同组两篇都在主题小节内、在「其他工作」之前
    assert md.index("ListRank 提出") < md.index("## 其他工作")


async def test_three_clause_sentence_shape(temp_db):
    run_id = await _seed_run(temp_db)
    md = await render_related_work_markdown(run_id)
    line = next(s for s in md.splitlines() if "CrossRanker 提出" in s)
    assert "cross encoder reranker 逐对打分" in line          # 做了什么
    assert "相对 baseline 的提升：nDCG +3.2" in line          # 相对谁的优势
    assert "作者自述局限：latency 高" in line                  # 作者自述局限
    # 空字段的分句整个不出现（ListRank 行没有 improvement/limitation）
    l2 = next(s for s in md.splitlines() if "ListRank 提出" in s)
    assert "相对 baseline" not in l2 and "局限" not in l2
    # 全空行：如实标留空，不编造内容
    empty_line = next(s for s in md.splitlines() if "留空" in s)
    assert "2999.00004" in empty_line


async def test_reference_numbering_and_tri_state(temp_db):
    run_id = await _seed_run(temp_db)
    md = await render_related_work_markdown(run_id)

    # 只有一份「## 参考文献」（去重口径照抄 markdown.py）
    assert md.count("## 参考文献") == 1
    refs = md.split("## 参考文献")[1]
    # [n] 按正文首现顺序：001→[1]、002→[2]、003→[3]、004→[4]
    assert "[1] CrossRanker: Cross Encoder Reranking — arXiv:2999.00001（章节：method｜正文原文）" in refs
    # None 三态保留：source 缺失如实标"摘要级证据"、locator 兜底 abstract
    assert "[2] ListRank: Listwise Reranker Distillation — arXiv:2999.00002（章节：abstract｜摘要级证据）" in refs
    # verified=False 的证据被剔除：003 没有可用引用行 → 如实写"无逐句引用记录"
    assert "[3] VisGPT for Classification — arXiv:2999.00003" in refs
    assert "无该论文的逐句引用记录" in refs
    assert "method｜正文原文）" not in refs.split("[3]")[1].split("\n")[0]
    # 004 不在 papers 表 → 标题用 arxiv_id 兜底，不编标题
    assert "[4] 2999.00004 — arXiv:2999.00004" in refs
    # 正文句尾带 [n]
    assert "[1]。" in md and "[2]。" in md
    # 固定声明在
    assert "> 本文由 AI 生成，引用请以原文为准。" in md


async def test_empty_run_renders_honest_placeholder(temp_db):
    run_id = "run-rw-empty"
    async with temp_db() as s:
        s.add(ResearchSession(id="sess-rw2", title="t"))
        s.add(
            ResearchRun(
                id=run_id, session_id="sess-rw2", thread_id="sess-rw2",
                question="只有问题", status="running",
            )
        )
    md = await render_related_work_markdown(run_id)
    assert "# 相关工作" in md
    assert "草稿为空" in md
    assert "## 参考文献" not in md

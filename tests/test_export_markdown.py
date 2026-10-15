"""F8 导出服务 `app/export/markdown.py` 的离线测试。

锁死四条：
1. run 不存在 → 抛 `RunNotFoundError`（路由层据此映射 404，语义比返回 None 直白）；
2. 完整 run 的导出内容顺序与口径：标题/问题 → 对比表 → 正文 → 参考文献
   （带章节定位 + 摘要级/正文原文强度标注）→ ⚠️ 未支撑结论 → 成本耗时 → 固定声明；
3. **缺失一律留空不编造**：没落库的部分对应小节直接不出现，空字段渲染成空单元格；
4. 三态口径与 writer 同源：`verified=None/缺失` 的行留在参考文献，只有 False 进 ⚠️ 节。

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
from app.db.models import (
    Citation,
    ComparisonRecord,
    Message,
    Paper,
    PaperChunk,
    ResearchRun,
    ResearchSession,
    TokenUsage,
)
from app.export import RunNotFoundError, render_run_markdown


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    yield


@pytest.fixture
async def temp_db(monkeypatch, tmp_path):
    """把 `db_session.session_scope` 换到 tmp_path 的临时库（同 test_spotcheck 接法）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'export.db'}")
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


async def _seed_full_run(scope) -> str:
    run_id = "run-full-001"
    async with scope() as s:
        s.add(ResearchSession(id="sess-1", title="t"))
        s.add(
            ResearchRun(
                id=run_id,
                session_id="sess-1",
                thread_id="sess-1",
                question="RAG 重排有哪些方法？",
                status="done",
                started_at=datetime(2026, 10, 12, 10, 0, 0),
                ended_at=datetime(2026, 10, 12, 10, 2, 51),
            )
        )
        s.add(
            Paper(
                arxiv_id="2999.00001",
                title="CrossRanker: Cross Encoder Reranking",
                abstract="",
            )
        )
        s.add(
            PaperChunk(
                id=101, arxiv_id="2999.00001", chunk_index=0, section="method", source="html", text="x"
            )
        )
        # 三条引用：True（正文原文）/ False（进 ⚠️ 节）/ None（三态保留，source 缺失→标摘要级）
        s.add(
            Citation(
                run_id=run_id,
                claim="CrossRanker 在 BEIR 上超过 BM25",
                chunk_id=101,
                arxiv_id="2999.00001",
                quote="On BEIR it reaches nDCG 0.62",
                locator="method",
                source="html",
                verified=True,
            )
        )
        s.add(
            Citation(
                run_id=run_id,
                claim="方法 D 在长文档场景更优",
                chunk_id=None,
                arxiv_id="2999.00001",
                quote="maybe better",
                locator="discussion",
                source="html",
                verified=False,
                note="引文为推测性表述，未直接支撑论断",
            )
        )
        s.add(
            Citation(
                run_id=run_id,
                claim="摘要级证据的论断",
                chunk_id=None,
                arxiv_id="2999.00002",  # 未入库的论文 → 标题用 arxiv_id 兜底
                quote="",
                locator=None,
                source=None,
                verified=None,
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id,
                row_index=0,
                arxiv_id="2999.00001",
                title="CrossRanker",
                method_name="CrossRanker",
                core_idea="交叉编码器逐对打分",
                datasets="BEIR",
                improvement="nDCG +3.2",
                cost="",                # 刻意留空（原文没写）
                limitation="",
                code_url="https://github.com/Example/CrossRanker",
                venue="EMNLP 2026",
                cited_by_count=3,
                cited_at="2026-10-12",
            )
        )
        s.add(
            ComparisonRecord(
                run_id=run_id,
                row_index=1,
                arxiv_id="2999.00002",
                title="|空行|测试",     # 竖线必须被转义，否则表格被截断
                method_name="",
                core_idea="",
                datasets="",
                improvement="",
                cost="",
                limitation="",
                code_url="",
                venue="",
                cited_by_count=None,
                cited_at=None,
            )
        )
        s.add(TokenUsage(run_id=run_id, node_name="all", model="mock", prompt_tokens=4000, completion_tokens=2180, cost_estimate=0.63))
        s.add(
            Message(
                session_id="sess-1",
                run_id=run_id,
                role="assistant",
                content="综述正文一句带 [1] 引用。",
                node_name="writer",
            )
        )
    return run_id


async def test_run_not_found_raises_typed_error(temp_db):
    with pytest.raises(RunNotFoundError) as excinfo:
        await render_run_markdown("run-does-not-exist")
    assert excinfo.value.run_id == "run-does-not-exist"
    assert isinstance(excinfo.value, LookupError)  # 路由层 `except LookupError → 404` 可用


async def test_render_full_run_sections_and_order(temp_db):
    run_id = await _seed_full_run(temp_db)
    md = await render_run_markdown(run_id)

    # 章节出现顺序 = 产品定义 §4 产出物 A~E 的口径
    order = [
        "RAG 重排有哪些方法？",
        "## 结构化对比表",
        "## 综述正文",
        "## 参考文献",
        "## ⚠️ 未找到原文支撑的结论",
        "## 成本与耗时",
        "> 本文由 AI 生成，引用请以原文为准。",
    ]
    positions = [md.index(h) for h in order]
    assert positions == sorted(positions)

    # 对比表：空字段留空（不是"未提及"），日期并进被引列，竖线被转义
    assert "| CrossRanker | 交叉编码器逐对打分 | BEIR | nDCG +3.2 |  |  |" in md
    assert "3（截至 2026-10-12）" in md
    assert "\\|空行\\|测试" in md

    # 参考文献：True 排 [1] 标正文原文；None 的三态保留、source 缺失如实标摘要级；False 的不出现
    refs = md.split("## 参考文献")[1].split("## ⚠️")[0]
    assert "[1] CrossRanker: Cross Encoder Reranking — arXiv:2999.00001（章节：method｜正文原文）" in refs
    assert "摘要级证据" in refs
    assert "arXiv:2999.00002" in refs      # 未入库标题用 id 兜底，不编造
    assert "长文档场景更优" not in refs

    # ⚠️ 节：只有 verified=False 的行，带失败原因
    unsup = md.split("## ⚠️")[1].split("## 成本")[0]
    assert "方法 D 在长文档场景更优" in unsup
    assert "引文为推测性表述" in unsup
    assert "CrossRanker 在 BEIR" not in unsup

    # 成本与耗时
    assert "prompt 4000 / completion 2180" in md
    assert "2 分 51 秒" in md


async def test_footer_not_duplicated_when_draft_already_has_it(temp_db):
    run_id = await _seed_full_run(temp_db)
    async with temp_db() as s:
        s.add(
            Message(
                session_id="sess-1",
                run_id=run_id,
                role="assistant",
                content="最新正文。\n\n> 本文由 AI 生成，引用请以原文为准。",
                node_name="writer",
            )
        )
    md = await render_run_markdown(run_id)
    assert md.count("> 本文由 AI 生成，引用请以原文为准。") == 1
    assert "最新正文。" in md and "综述正文一句" not in md  # 取最新一条


async def test_reference_and_unsupported_not_duplicated_when_draft_is_full_report(temp_db):
    """生产形态的 draft：writer 落库的就是**完整报告**，里面已带「## 参考文献」与
    「⚠️ 未找到原文支撑的结论」两节（见 `nodes/writer.py` 的 sections 装配）。
    此时导出必须沿用正文里的那两份 —— 它和正文的 `[n]` 是同一套编号；
    再从库里的行拼一遍会出现"两份参考文献、编号还不一致"，读者无从判断哪份对。
    """
    run_id = await _seed_full_run(temp_db)
    full_report = (
        "综述正文一句带 [1] 引用。\n\n"
        "## ⚠️ 未找到原文支撑的结论\n\n"
        '以下结论未在检索到的文献中找到直接原文支撑，请勿直接引用：\n\n'
        '- "方法 D 在长文档场景更优"\n\n'
        "## 参考文献\n\n"
        "[1] CrossRanker: Cross Encoder Reranking — arXiv:2999.00001（章节：method｜正文原文）"
    )
    async with temp_db() as s:
        s.add(
            Message(
                session_id="sess-1",
                run_id=run_id,
                role="assistant",
                content=full_report,
                node_name="writer",
            )
        )
    md = await render_run_markdown(run_id)
    assert md.count("## 参考文献") == 1
    assert md.count("未找到原文支撑的结论") == 1


async def test_missing_parts_render_blanks_not_fabricated(temp_db):
    """只有 run 行：其余表全空 → 正文/表/引用/成本小节如实缺席，标题与声明仍在。"""
    run_id = "run-empty-001"
    async with temp_db() as s:
        s.add(ResearchSession(id="sess-2", title="t"))
        s.add(
            ResearchRun(
                id=run_id, session_id="sess-2", thread_id="sess-2",
                question="只有问题", status="running",
            )
        )
    md = await render_run_markdown(run_id)
    assert "# 文献研究报告：只有问题" in md
    for section in ("## 结构化对比表", "## 综述正文", "## 参考文献", "## 成本与耗时"):
        assert section not in md
    assert "未找到原文支撑" not in md
    assert "> 本文由 AI 生成，引用请以原文为准。" in md


async def test_hitl_wait_seconds_are_subtracted_from_the_reported_duration(temp_db):
    """`started_at~ended_at` 是墙钟，HITL 等待必须从中扣掉，否则"耗时"是运行+人工的混合值。"""
    run_id = await _seed_full_run(temp_db)
    async with temp_db() as s:
        run = await s.get(ResearchRun, run_id)
        # 墙钟 2 分 51 秒，其中 2 分 05 秒是用户在改子问题
        run.langgraph_state = {"hitl_wait_seconds": 125.0}

    md = await render_run_markdown(run_id)
    assert "已扣除等待用户确认拆解的 2 分 5 秒" in md
    assert "耗时：0 分 46 秒" in md, "报告里的耗时仍等于墙钟 —— 扣除没生效"


async def test_runs_without_hitl_tracking_report_plain_wall_clock(temp_db):
    """没有等待记录（没走过 HITL，或历史数据）时不该凭空加一句"已扣除"。"""
    run_id = await _seed_full_run(temp_db)
    md = await render_run_markdown(run_id)
    assert "耗时：2 分 51 秒" in md and "已扣除" not in md

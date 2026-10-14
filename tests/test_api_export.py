"""F8 导出接口 `GET /api/v1/research/{run_id}/export.md` 的对外契约。

只测**这一层该负责的事**（渲染内容的正确性由 `tests/test_export_markdown.py` 覆盖）：
1. 未知 run → 404，而不是 200 + 一份空报告（静默的空文件比报错更难发现）；
2. 响应是 `text/markdown` 且带 `attachment` 下载头 —— 浏览器直接打开会把 .md 渲染成网页，
   用户以为"导出成功"其实拿到的是格式化后的页面；
3. `run_id` 的字符集闸门在这个接口上同样生效（它和别的接口一样直接进 DB 查询）。

离线夹具与 `tests/api_support.py` 完全一致：mock provider + tmp_path 临时库，不碰 data/。
"""

from __future__ import annotations

import pytest
from sqlalchemy import select

import app.db.models  # noqa: F401 - 注册表到 metadata
from app.db.models import Citation, Paper, ResearchRun, ResearchSession
from app.db.session import session_scope
from tests.api_support import api_client

EXPORT_URL = "/api/v1/research/{run_id}/export.md"


@pytest.fixture
async def client(monkeypatch, tmp_path):
    async with api_client(monkeypatch, tmp_path) as c:
        yield c


async def _seed_run(run_id: str, *, with_citation: bool = False) -> None:
    async with session_scope() as db:
        if await db.get(ResearchSession, "sess-export") is None:
            db.add(ResearchSession(id="sess-export", title="导出用例"))
        db.add(
            ResearchRun(
                id=run_id,
                session_id="sess-export",
                thread_id=run_id,
                question="RAG 重排的代价是什么",
                status="done",
            )
        )
        if with_citation:
            # 参考文献的标题要从 papers 表查得到，才测得出"标题用 arxiv_id 兜底"以外的另一条路径
            if await db.get(Paper, "2999.00001") is None:
                db.add(Paper(arxiv_id="2999.00001", title="CrossRanker: Cross Encoder Reranking", abstract=""))
            db.add(
                Citation(
                    run_id=run_id,
                    claim="CrossRanker 在 BEIR 上超过 BM25",
                    arxiv_id="2999.00001",
                    quote="On BEIR it reaches nDCG 0.62",
                    locator="method",
                    source="html",
                    verified=True,
                )
            )


async def test_export_unknown_run_is_404_not_an_empty_report(client):
    resp = await client.get(EXPORT_URL.format(run_id="run-not-there"))
    assert resp.status_code == 404
    assert "run-not-there" in resp.text


async def test_export_returns_markdown_attachment(client):
    await _seed_run("run-export-1", with_citation=True)

    resp = await client.get(EXPORT_URL.format(run_id="run-export-1"))
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/markdown")
    # 文件名带 run_id：用户存下十几份报告之后还能对回是哪一次研究
    assert 'attachment; filename="paperpilot-run-export-1.md"' in resp.headers["content-disposition"]
    body = resp.text
    assert "# 文献研究报告：RAG 重排的代价是什么" in body
    assert "CrossRanker" in body
    assert "> 本文由 AI 生成，引用请以原文为准。" in body


async def test_export_guards_run_id_shape(client):
    """含空白的 id 走不通字符集闸门：它和别的接口一样把 id 直接拼进 DB 查询。"""
    resp = await client.get(EXPORT_URL.format(run_id="bad id"))
    assert resp.status_code == 422
    assert resp.json()["detail"]["field"] == "run_id"


async def test_export_is_read_only(client):
    """导出不许产生副作用：反复点"下载"不能改变 run 状态，也不能写库。

    这条值得单独钉：报告渲染看着是纯读，但导出路径很容易顺手加一句
    "顺便把 status 改成 exported"，那会让 HITL 与断点续跑的语义多一个隐式分支。
    """
    await _seed_run("run-export-2")
    for _ in range(2):
        assert (await client.get(EXPORT_URL.format(run_id="run-export-2"))).status_code == 200

    async with session_scope() as db:
        run = await db.get(ResearchRun, "run-export-2")
        assert run.status == "done"
        assert (await db.execute(select(Citation).where(Citation.run_id == "run-export-2"))).scalars().all() == []

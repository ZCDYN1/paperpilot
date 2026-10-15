"""papers 路由对外错误形状：稳定 code + 一句人话，异常原文不外泄。

`GET /papers/{id}` 与 `POST /papers/{id}/ask` 曾把 `PaperNotFoundError` 的
`str(exc)` 原样放进 404 detail —— 消息文本一旦由底层拼进查询上下文 / 文件路径，
就是"clone 即可跑"承诺上的信息泄露面（本项目在 Windows 上跑，路径含中文用户名）。
本文件钉住两件事：
1. 404 对外是 `{code, reason}` 结构化 detail（与本文件其余错误形状同口径）；
2. 服务端异常文本即使带 SQL 片段 / 连接串 / 本机路径，也不会出现在响应里。
"""

from __future__ import annotations

import pytest

from tests.api_support import api_client


@pytest.fixture
async def client(monkeypatch, tmp_path):
    async with api_client(monkeypatch, tmp_path) as c:
        yield c


_LEAK_FRAGMENTS = ("SELECT", "hunter2", "paperpilot.db", "张三", "sqlite")


def _assert_no_leak(payload: object) -> None:
    text = str(payload)
    for frag in _LEAK_FRAGMENTS:
        assert frag not in text, f"异常原文泄出对外响应: {frag!r}"


async def test_paper_detail_404_is_structured_and_sanitized(client, monkeypatch):
    from app.api.v1 import papers as papers_mod
    from app.papers import PaperNotFoundError

    # 底层异常消息带泄漏面：路由不得 `detail=str(exc)` 透传，只回显过了闸门的 id。
    leaky = PaperNotFoundError("2999.99999")
    leaky.args = (
        "论文不存在：'2999.99999' | sqlite3.OperationalError [SQL: SELECT * FROM papers] "
        "C:\\Users\\张三\\data\\sqlite\\paperpilot.db password=hunter2",
    )

    async def boom(_doc_id: str):
        raise leaky

    monkeypatch.setattr(papers_mod, "build_paper_detail", boom)

    resp = await client.get("/api/v1/papers/2999.99999")
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "paper_not_found"
    assert detail["reason"]
    _assert_no_leak(detail)


async def test_paper_ask_404_is_structured_and_sanitized(client):
    # 真实读库路径（不 monkeypatch）：未入库的 id 也要拿到 {code, reason} 而不是裸文本。
    resp = await client.post(
        "/api/v1/papers/2999.88888/ask", json={"question": "它怎么决定何时检索"}
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["code"] == "paper_not_found"
    assert detail["reason"]
    _assert_no_leak(detail)


async def test_paper_detail_real_missing_id_still_404(client):
    resp = await client.get("/api/v1/papers/2999.99999")
    assert resp.status_code == 404
    assert resp.json()["detail"]["code"] == "paper_not_found"

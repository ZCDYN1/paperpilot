"""F16 用到的两个 OpenAlex 取数方法（离线，`httpx.MockTransport`）。

这里测的是"请求发对了没有"——URL 参数、分批、以及**解析不出 arXiv id 时不许猜**。
服务层的落库逻辑在 `tests/test_references_trend.py`。
"""

from __future__ import annotations

import json

import httpx

from app.ingest.openalex_client import OpenAlexClient, arxiv_id_from_work


def _client(handler, settings=None) -> OpenAlexClient:
    return OpenAlexClient(
        settings,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5),
    )


# --------------------------------------------------------------------------- id 解析
def test_arxiv_id_from_work_three_sources():
    assert arxiv_id_from_work({"ids": {"arxiv": "https://arxiv.org/abs/2305.15334v2"}}) == "2305.15334"
    assert arxiv_id_from_work({"doi": "https://doi.org/10.48550/arXiv.2401.00001"}) == "2401.00001"
    assert (
        arxiv_id_from_work(
            {"ids": {}, "primary_location": {"url": "http://arxiv.org/abs/2202.11111"}}
        )
        == "2202.11111"
    )
    # 期刊论文：解析不出就是 None，绝不返回一个"看着像"的 id
    assert arxiv_id_from_work({"doi": "https://doi.org/10.1145/3576840.3576874", "ids": {}}) is None
    assert arxiv_id_from_work({}) is None
    assert arxiv_id_from_work(None) is None


# --------------------------------------------------------------------------- referenced_works
async def test_fetch_referenced_works_uses_doi_filter():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={"results": [{"id": "https://openalex.org/W100", "referenced_works": [
                "https://openalex.org/W200", "https://openalex.org/W300"
            ]}]},
        )

    found = await _client(handler).fetch_referenced_works("2305.15334")
    assert found == ("W100", ["W200", "W300"])
    # 精确 DOI 查找，而不是标题搜索：引用边是结构性事实，错一条比少一条糟得多
    assert seen["params"]["filter"] == "doi:10.48550/arxiv.2305.15334"


def _json_handler(payload: str):
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, json=json.loads(payload))

    return handler


async def test_fetch_referenced_works_returns_none_on_missing_or_error():
    cases = {
        "没有这篇的记录": '{"results": []}',
        "有记录但没有参考文献": '{"results": [{"id": "W1", "referenced_works": []}]}',
    }
    for label, payload in cases.items():
        assert await _client(_json_handler(payload)).fetch_referenced_works("1") is None, label

    def boom(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(500)

    assert await _client(boom).fetch_referenced_works("1") is None

    def explode(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        raise httpx.ConnectError("网络不通")

    # 永不抛异常：引用边是增强信息，外部源挂了不能影响已入库的语料
    assert await _client(explode).fetch_referenced_works("1") is None


async def test_version_suffix_is_stripped_before_the_doi_lookup():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["filter"] = dict(request.url.params)["filter"]
        return httpx.Response(200, json={"results": []})

    await _client(handler).fetch_referenced_works("arxiv/2305.15334v3")
    assert seen["filter"] == "doi:10.48550/arxiv.2305.15334"


# --------------------------------------------------------------------------- 批量解析
async def test_resolve_batches_requests_and_keeps_unresolved_entries():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        ids = params["filter"].removeprefix("openalex_id:").split("|")
        results = [
            {
                "id": f"https://openalex.org/{wid}",
                "title": f"work {wid}",
                "ids": {"arxiv": "https://arxiv.org/abs/2305.00001"} if wid == "W1" else {},
            }
            for wid in ids
        ]
        return httpx.Response(200, json={"results": results})

    work_ids = [f"W{i}" for i in range(1, 121)]  # 120 条参考文献：一篇长论文的常态
    out = await _client(handler).resolve_arxiv_ids(work_ids)

    # 每批 50 → 3 个请求。拼成一条超长 query 会被 OpenAlex 400 拒掉且看不出原因
    assert len(calls) == 3
    assert all(c["filter"].startswith("openalex_id:") for c in calls)
    assert out["W1"]["arxiv_id"] == "2305.00001"
    # 解析不出 arXiv id 的条目**照样返回**（arxiv_id=None）：
    # "这篇参考文献存在但不在 arXiv 上"与"这条没查到"是两件事
    assert out["W2"] == {"arxiv_id": None, "title": "work W2"}

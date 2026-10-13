"""`tests/test_api_*.py` 共用的离线 ASGI 客户端与 SSE 解析助手。

放在单独模块而不是重复两份：两个 API 测试文件测的是同一套对外契约，
夹具必须**完全一致**（尤其"不许联网"那几条环境变量），否则一个文件绿、另一个红
会浪费掉很久去查"到底哪儿不一样"。

三条离线约束的实现位置：
1. `LLM_PROVIDER=mock` + `EMBEDDING_PROVIDER=mock` + `RERANK_PROVIDER=lexical`；
2. `MIN_CORPUS=0` —— `ensure_corpus` 节点的抓网条件是"已有论文数 < min_corpus"，
   设 0 让它直接跳过联网分支（`INGEST_ENABLED=false` 同时表达意图）；
3. SQLite / checkpoint / FAISS 全部指向 pytest 的 `tmp_path`。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from app.core.config import get_settings
from app.core.resilience import reset_breakers
from app.db.session import dispose_engine
from app.observability.token_meter import reset_token_meter

STREAM_URL = "/api/v1/research/stream"

# 每个 API 用例都从这里起步：离线 + 临时存储 + 显式关掉追踪
# （本机 `.env` 里可能填了真 LANGCHAIN_API_KEY，测试绝不能把它读进断言、
#  也绝不能因为它的存在而开始往 LangSmith 上报。）
BASE_ENV = {
    "LLM_PROVIDER": "mock",
    "EMBEDDING_PROVIDER": "mock",
    "RERANK_PROVIDER": "lexical",
    "RERANK_ENABLED": "true",
    "MIN_CORPUS": "0",
    "INGEST_ENABLED": "false",
    "CONFIRM_OUTLINE": "false",
    "MAX_REVISION": "1",
    # 容错层保持开启：这条链路在 mock 下也必须跑得通（这是被测事实，不是为了方便）
    "RESILIENCE_ENABLED": "true",
    "RESILIENCE_BACKOFF": "0",
    "RESILIENCE_MAX_BACKOFF": "0",
    "RESILIENCE_JITTER": "0",
    "INPUT_GUARD_ENABLED": "true",
    "INPUT_INJECTION_ACTION": "reject",
    "LANGCHAIN_TRACING_V2": "false",
    "LANGCHAIN_API_KEY": "",
    "LANGCHAIN_PROJECT": "paperpilot-test",
}


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """把 SSE 文本切成 `[(事件类型, data 字典)]`（只做协议要求的最小解析）。"""
    out: list[tuple[str, dict]] = []
    current = "message"
    for line in text.splitlines():
        if line.startswith("event:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            raw = line.split(":", 1)[1].strip()
            try:
                out.append((current, json.loads(raw)))
            except json.JSONDecodeError:
                out.append((current, {"_unparsed": raw}))
    return out


def event_types(text: str) -> list[str]:
    return [t for t, _ in parse_sse(text)]


def first_data(text: str, event_type: str) -> dict:
    for t, data in parse_sse(text):
        if t == event_type:
            return data
    raise AssertionError(f"SSE 流里没有 {event_type} 事件；实际类型：{event_types(text)}")


# 字段名里出现这些词就意味着"这里可能承载密钥"：值必须是布尔/None，不能是字符串。
_SECRETISH_FIELDS = ("api_key", "apikey", "secret", "token", "password")


def assert_no_secret_values(payload: object, _path: str = "$") -> None:
    """递归断言一个响应体里**没有任何密钥值**（交付约束：不打印 / 不外发 key）。

    为什么要写成递归断言而不是"看一眼 healthz"：这类接口以后一定会加字段，
    而"顺手把 settings.openai_api_key 也报出去"看起来人畜无害，实际上
    是把密钥挂到一个无需鉴权的公开端点上。这条断言会在改动的那一刻替我们拦住。
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            lane = f"{_path}.{key}"
            if isinstance(key, str) and any(s in key.lower() for s in _SECRETISH_FIELDS):
                assert value in (None, True, False) or value == "", f"{lane} 看起来是密钥，但带着真实值"
            assert_no_secret_values(value, lane)
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_no_secret_values(item, f"{_path}[{i}]")


@asynccontextmanager
async def api_client(monkeypatch, tmp_path, **extra_env) -> AsyncIterator[httpx.AsyncClient]:
    """起一个"跑完 lifespan 的 app + ASGI 客户端"，用完把进程内单例全部拆掉。"""
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    for k, v in extra_env.items():
        monkeypatch.setenv(k, str(v))

    data_dir = tmp_path / "data"
    monkeypatch.setenv("SQLITE_URL", f"sqlite+aiosqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("CHECKPOINT_DB", str(data_dir / "sqlite" / "checkpoints.db"))
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(data_dir / "faiss"))
    monkeypatch.setenv("SQLITE_DIR", str(data_dir / "sqlite"))

    get_settings.cache_clear()
    await dispose_engine()  # engine 是进程内单例：不拆掉就会带着上一个用例的库地址
    reset_breakers()
    reset_token_meter()

    from app.main import create_app, lifespan

    app = create_app()
    async with lifespan(app):  # 启动自检（LANGCHAIN_* 同步、熔断器建实例）也在被测范围
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client
    await dispose_engine()
    reset_breakers()
    get_settings.cache_clear()

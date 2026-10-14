"""F14「定向对比模式」：请求点名的论文 id 要一路走到检索，并且**不要**触发关键词抓取。

保护的行为：
1. 顶层 `arxiv_ids` 并进 `filters`（`_initial_state` 只把 `filters.model_dump()` 喂给图，
   并进失败 = 这个字段在 API 层就蒸发了，检索层再支持也没用）；
2. 点名模式下 `ensure_corpus` 跳过联网抓取 —— 抓回来的语料进不了对比表
   （检索侧已被 arxiv_ids 限死），却要串行吃 arXiv 的 1 req/3s 配额；
3. 没点名时抓取分支行为不变（否则这条"跳过"会把主流程一起改掉）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.agents.nodes.ensure_corpus as ec
from app.api.v1.research import _initial_state
from app.schemas.research import ResearchRequest


def _clear_settings_cache() -> None:
    from app.core.config import get_settings

    get_settings.cache_clear()


# --------------------------------------------------------------------------- 请求 → state
def test_top_level_arxiv_ids_merge_into_filters_state():
    req = ResearchRequest(question="对比这几篇", arxiv_ids=["2305.15334v2", "2401.00001"])
    state = _initial_state(req, run_id="r1", session_id="r1", question="对比这几篇")

    # v2 后缀被剥掉、顺序保留、并且**在 filters 里**（图只看得见 filters）
    assert state["filters"]["arxiv_ids"] == ["2305.15334", "2401.00001"]


def test_arxiv_ids_inside_filters_still_work():
    req = ResearchRequest(question="q", filters={"arxiv_ids": ["2305.15334"]})
    state = _initial_state(req, run_id="r2", session_id="r2", question="q")
    assert state["filters"]["arxiv_ids"] == ["2305.15334"]


def test_no_arxiv_ids_leaves_filters_empty():
    """没点名时 state 里不该冒出 arxiv_ids 键（`filters` 为 None 时就是空 dict，
    检索侧靠 `.get()` 取值，多塞一个 `[]` 反而会让"有没有过滤"的判定变模糊）。"""
    state = _initial_state(ResearchRequest(question="q"), run_id="r3", session_id="r3", question="q")
    assert not state["filters"].get("arxiv_ids")


# --------------------------------------------------------------------------- 抓取分支
class _FakeRetriever:
    def index_is_current(self, records, titles) -> bool:  # noqa: ARG002
        return True


@pytest.fixture
def corpus_probe(monkeypatch):
    """把 ensure_corpus 的三个外部依赖换成"能告诉我它走没走抓取分支"的探针。"""

    calls: list[str] = []

    async def fake_init_db() -> None:
        return None

    async def fake_load_corpus():
        return [], {}  # 本地一篇都没有：抓取分支的唯一触发条件是"语料不足"

    async def fake_ingest(query, **kwargs):  # noqa: ARG001
        calls.append("ingest")
        return SimpleNamespace(
            records=[], titles={}, stats=SimpleNamespace(fetched=0, cached=0)
        )

    monkeypatch.setattr(ec, "init_db", fake_init_db)
    monkeypatch.setattr(ec, "load_corpus", fake_load_corpus)
    monkeypatch.setattr(ec, "get_shared_retriever", lambda: _FakeRetriever())
    # 节点里是函数内 import，所以要顶掉模块本身（不在 sys.modules 时先 import 一次）
    import app.ingest.pipeline as pipeline

    monkeypatch.setattr(pipeline, "ingest", fake_ingest)

    monkeypatch.setenv("MIN_CORPUS", "8")
    monkeypatch.setenv("INGEST_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    _clear_settings_cache()
    yield calls
    _clear_settings_cache()


async def test_directed_compare_skips_keyword_ingest(corpus_probe):
    state = {
        "question": "只对比 cross-encoder 与 late interaction 这两篇",
        "sub_questions": [{"id": "sq-1", "text": "t", "keywords": ["reranking"], "category": "cs.IR"}],
        "filters": {"arxiv_ids": ["2305.15334", "2401.00001"]},
    }
    out = await ec.ensure_corpus_node(state)
    assert corpus_probe == [], "点名论文模式下仍然去按关键词抓 arXiv（白烧限速配额）"
    # 如实反映"本地没抓到东西"：语料不足必须让下游声明覆盖度，而不是假装就绪
    assert out["corpus_stats"]["total_papers"] == 0


async def test_without_arxiv_ids_ingest_still_runs(corpus_probe):
    """对照组：没点名时抓取分支必须照旧，否则这条"跳过"就把主流程一起改坏了。"""
    state = {
        "question": "RAG 重排有哪些方法",
        "sub_questions": [{"id": "sq-1", "text": "t", "keywords": ["reranking"], "category": "cs.IR"}],
        "filters": {},
    }
    await ec.ensure_corpus_node(state)
    assert corpus_probe == ["ingest"]

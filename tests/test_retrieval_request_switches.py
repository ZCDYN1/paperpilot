"""请求级检索开关（`top_k` / `rerank`）与两个 top_k 口径的映射。

这个文件存在的理由是两个具体的坑：

1. **开关是装饰器**：`ResearchRequest.top_k` / `.rerank` 早就在请求体里了，
   但节点只读 `get_settings()`，用户拨到 off 链路照旧重排 —— 而且 SSE 与日志上
   完全看不出差别，这类缺陷不会以"报错"的形式暴露，只会以"结论不符"的形式回来。
2. **两个 top_k 混为一谈**：`HybridRetriever.search(top_k=)` 是**每一路**（FAISS / BM25）
   各召回多少条，而 `HybridSearchArgs.top_k` 的语义是**最终留几篇**。
   曾经节点把 `RERANK_TOP_N`(=8) 直接当 `search(top_k=)` 传下去，于是每路只召回 8 条
   就去融合；而 `app/eval/ablation.py` 与 README 的消融表量的是每路 50 条 ——
   对外公布的是一条产品没在跑的链路。下面的断言把映射钉死。
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.agents.nodes.retriever_worker import _build_args, _build_extra_filters
from app.agents.tools.hybrid_search import build_filter, run_hybrid_search
from app.retrieval.hybrid_retriever import ChunkRecord, MetadataFilter, PaperHit
from app.schemas.retrieval import HybridSearchArgs


def _hit(i: int) -> PaperHit:
    return PaperHit(
        arxiv_id=f"2999.0000{i}",
        title=f"Paper {i}",
        score=1.0 / (i + 1),
        rep_chunk=ChunkRecord(
            chunk_id=i, arxiv_id=f"2999.0000{i}", section="method", text=f"text {i}"
        ),
        chunks=[],
        matched_by=["vector"],
    )


class RecordingRetriever:
    """只记录入参、返回固定篇数的假检索器（不碰 FAISS / BM25 / embedding）。"""

    def __init__(self, hits: list[PaperHit] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._hits = hits if hits is not None else [_hit(i) for i in range(1, 11)]

    async def search(self, query, keywords=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls.append({"query": query, "keywords": keywords, **kwargs})
        # 模拟"重排分支才按 rerank_top_n 截断"：没重排时返回全量
        cfg = kwargs.get("config")
        limit = kwargs.get("rerank_top_n") or len(self._hits)
        return self._hits[: limit if getattr(cfg, "use_rerank", True) else len(self._hits)]


def _args(**over: Any) -> HybridSearchArgs:
    base = dict(query="reranking 方法有哪些", keywords=["reranking"], top_k=8)
    base.update(over)
    return HybridSearchArgs(**base)


# --------------------------------------------------------------------------- 映射口径
async def test_request_top_k_maps_to_final_paper_count_not_per_path_recall():
    """`args.top_k` 必须落到 `rerank_top_n`（最终留几篇），每路召回交回 RETRIEVE_TOP_K。"""
    fake = RecordingRetriever()
    hits = await run_hybrid_search(_args(top_k=3), retriever=fake)

    call = fake.calls[0]
    assert call["top_k"] is None, "每路召回量被请求级 top_k 抢走了（RETRIEVE_TOP_K 会失效）"
    assert call["rerank_top_n"] == 3
    assert len(hits) == 3


async def test_paper_count_is_enforced_even_when_rerank_is_off():
    """关掉重排时 `search()` 不按 rerank_top_n 截断，工具层要补上这一刀。

    少这一刀的后果是"top_k 说了不算"：rerank=false 时返回 30 篇，
    而下游 Reader 又按 READ_TOP_K 自己挑，用户看不出任何异常。
    """
    fake = RecordingRetriever()
    hits = await run_hybrid_search(_args(top_k=3, rerank=False), retriever=fake)

    assert fake.calls[0]["config"].use_rerank is False
    assert len(hits) == 3


async def test_rerank_defaults_to_on():
    """默认值必须是"重排开" —— 与加这个字段之前的线上行为一致。"""
    fake = RecordingRetriever()
    await run_hybrid_search(_args(), retriever=fake)
    assert fake.calls[0]["config"].use_rerank is True


# --------------------------------------------------------------------------- 请求开关进得了 state
def test_build_args_prefers_request_level_switches_over_settings():
    state = {"question": "q", "top_k": 5, "rerank": False, "filters": {}}
    args = _build_args(state, {"id": "sq-1", "text": "子问题", "keywords": ["reranking"]})
    assert args.top_k == 5
    assert args.rerank is False


def test_build_args_falls_back_to_settings_when_request_says_nothing():
    """没表态时退回配置，行为与"加这两个字段之前"逐字一致（不能让默认路径跟着变）。"""
    args = _build_args({"question": "q", "filters": {}}, {"id": "sq-1", "text": "子问题"})
    assert args.top_k == 8
    assert args.rerank is True


# --------------------------------------------------------------------------- 过滤条件的拼接
def test_has_code_false_is_a_real_filter_not_a_falsy_nothing():
    """`has_code=False` 是"只看没确认开源仓库的论文"，是真过滤。

    用 `any(kwargs.values())` 判空会把它连同整条过滤器一起丢掉，
    表现成"这个过滤器只在想要 True 的时候生效"。
    """
    filt = build_filter(_args(has_code=False))
    assert isinstance(filt, MetadataFilter), "has_code=False 被当成没设过滤"
    assert filt.has_code is False

    assert build_filter(_args(has_code=True)).has_code is True
    assert build_filter(_args()) is None  # 全空仍然不构造过滤器


def test_arxiv_ids_from_global_channel_reaches_the_filter():
    """F14：`HybridSearchArgs` 没有 arxiv_ids 字段（单值契约），它走全局 extra 通道。"""
    state = {"question": "q", "filters": {"arxiv_ids": ["2305.15334"], "categories": ["cs.CL", "cs.LG"]}}
    extra = _build_extra_filters(state)
    filt = build_filter(_args(), extra)

    assert filt is not None
    assert list(filt.arxiv_ids) == ["2305.15334"]
    assert sorted(filt.categories) == ["cs.CL", "cs.LG"]  # 多分类不再被整组丢弃


def test_sub_question_category_wins_over_global_categories():
    """更具体的约束优先：子问题自带分类时，全局多分类不该覆盖它。"""
    state = {"question": "q", "filters": {"categories": ["cs.LG"]}}
    args = _build_args(state, {"id": "sq-1", "text": "t", "category": "cs.IR"})
    filt = build_filter(args, _build_extra_filters(state))
    assert filt.categories == ["cs.IR"]


@pytest.mark.parametrize("bad", [{"top_k": 0}, {"top_k": 51}, {"has_code": "maybe"}])
def test_tool_schema_still_rejects_hallucinated_params(bad: dict):
    """工具入参的强校验不能因为加字段而放松（架构 §10「工具侧」）。"""
    with pytest.raises(ValidationError):
        _args(**bad)

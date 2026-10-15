"""`hybrid_search` 工具 —— Agent 与检索层之间**唯一**的通道。

架构 §2：「Agent 层不允许直接碰 FAISS / BM25 / DB」。
这条不是洁癖，而是为了让检索策略能被单独做消融实验（§5.3）、被单测、被替换实现。

这里提供两个入口，职责刻意分开：
- `run_hybrid_search(args)`：**给节点直接调用**。入参是已经构造好的 `HybridSearchArgs`。
- `hybrid_search`（`@tool`）：**给 LLM 调用**。LangChain 会把 `HybridSearchArgs`
  的 JSON schema 塞给模型，并在调用时做 Pydantic 强校验（§10「工具侧」）。

为什么两者都要：节点侧走工具函数会白搭一层 LLM 往返（又慢又贵，还引入不确定性），
而 LLM 侧必须有严格 schema 兜住"top_k 传 500"这类幻觉参数。同一个底层实现，
两种调用姿势。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from langchain_core.tools import tool

from app.retrieval.hybrid_retriever import (
    DEFAULT_SEARCH_CONFIG,
    HybridRetriever,
    MetadataFilter,
    PaperHit,
)
from app.schemas.retrieval import HybridSearchArgs

# 进程内共享的检索器单例。
# 为什么共享：`HybridRetriever.__init__` 会 load FAISS 索引 + 反序列化 BM25 模型
# （几百 MB 级），每个 retrieve 分支各建一个 = N 倍内存与 N 倍加载时间。
# 检索是**只读**的，并行共享安全；写路径（ingest）由 ensure_corpus 串行独占。
_shared: HybridRetriever | None = None


def get_shared_retriever() -> HybridRetriever:
    """拿到进程内共享的检索器（懒加载）。"""
    global _shared
    if _shared is None:
        _shared = HybridRetriever()
    return _shared


def reset_shared_retriever() -> None:
    """丢弃单例。索引被重建（ingest 之后）或测试切换索引目录时必须调用，
    否则会继续用旧的内存索引 —— 表现是"刚抓的论文检索不到"，且不报错。"""
    global _shared
    _shared = None


def serialize_hit(hit: PaperHit) -> dict[str, Any]:
    """`PaperHit` → 可进 state / 可 JSON 序列化的 dict。

    ⚠️ 必须转成纯 dict 再放进 state：LangGraph 的 checkpointer 会序列化 state，
    把 dataclass 塞进去在 sqlite 后端上会失败（或退化成不可回读的形态）。
    """
    return {
        "arxiv_id": hit.arxiv_id,
        "title": hit.title,
        "score": float(hit.score),
        "matched_by": list(hit.matched_by),
        "rep_chunk_id": hit.rep_chunk.chunk_id if hit.rep_chunk else None,
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "arxiv_id": c.arxiv_id,
                "section": c.section,
                "text": c.text,
                "source": c.source,
                "chunk_type": c.chunk_type,
                "page": c.page,
            }
            for c in hit.chunks
        ],
    }


def _filter_is_active(value: Any) -> bool:
    """区分"没设这个过滤"与"设了一个取值本身就是 False / 空集合的过滤"。

    `has_code=False`（只召回未确认有开源仓库的论文）是一条**真过滤条件**，
    但 `any(kwargs.values())` 会把它当 falsy 整个丢掉，于是 `MetadataFilter` 变 None、
    过滤形同没开 —— 表现是"要 True 的时候好用、要 False 的时候静默无效"，
    最难发现的那一类半实现。
    """
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def build_filter(args: HybridSearchArgs, extra: dict[str, Any] | None = None) -> MetadataFilter | None:
    """把工具入参拼成检索层的过滤条件。

    `extra` 用于把 API 层传下来的全局过滤（如 `filters.categories` 多分类）
    与子问题自己的单分类合并 —— 子问题是**更具体**的约束，优先级更高。
    """
    kwargs = args.to_filter_kwargs()
    if extra:
        for key, value in extra.items():
            if _filter_is_active(value) and not _filter_is_active(kwargs.get(key)):
                kwargs[key] = value
    if not any(_filter_is_active(v) for v in kwargs.values()):
        return None
    return MetadataFilter(**kwargs)


async def run_hybrid_search(
    args: HybridSearchArgs,
    *,
    retriever: HybridRetriever | None = None,
    extra_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """执行一次混合检索并返回 paper 级结果（已序列化）。

    ★ 两个 `top_k` 的映射关系在这里，不在别处：
    `HybridSearchArgs.top_k` 的语义是「最终留几篇」，所以它落到 `rerank_top_n`；
    每一路的召回量留给 `search(top_k=None)` 退回 `settings.retrieve_top_k`（默认 50）。
    曾经这里写成 `top_k=args.top_k`，于是每路只召回 8 条就去融合，
    而 `app/eval/ablation.py` 与 README 那张消融表量的是每路 50 条 ——
    等于对外公布了一条产品根本没在跑的链路。改回来之后两边才是同一条路。
    """
    retriever = retriever or get_shared_retriever()
    filters = build_filter(args, extra_filter)
    hits = await retriever.search(
        args.query,
        args.keywords,
        top_k=None,
        rerank_top_n=args.top_k,
        filters=filters,
        # 用 `replace` 而不是就地改 `DEFAULT_SEARCH_CONFIG`：那是模块级共享实例，
        # 一次提问把 `use_rerank` 改脏了，之后所有引用"生产配置"的地方都跟着变。
        config=replace(DEFAULT_SEARCH_CONFIG, use_rerank=args.rerank),
    )
    # 关掉重排时 `search()` 不会按 `rerank_top_n` 截断（那个参数只在重排分支里生效），
    # 这里统一补一刀，`rerank=false` 才不至于让"留几篇"这个承诺失效。
    return [serialize_hit(h) for h in hits[: args.top_k]]


@tool("hybrid_search", args_schema=HybridSearchArgs)
async def hybrid_search(
    query: str,
    keywords: list[str] | None = None,
    top_k: int = 8,
    rerank: bool = True,
    year_gte: int | None = None,
    year_lte: int | None = None,
    category: str | None = None,
    has_code: bool | None = None,
    sub_question_id: str | None = None,
) -> list[dict[str, Any]]:
    """在本地论文语料库上做「向量 + BM25 + RRF 融合 + 重排」混合检索。

    Args:
        query: 自然语言查询（中文亦可，向量侧支持跨语言）。
        keywords: **英文**术语列表，供 BM25 使用。中文关键词打英文索引会失效。
        top_k: 返回论文数上限（1~50），指最终留几篇，不是每路召回量。
        rerank: 是否启用交叉编码器重排；延迟敏感可设 false。
        year_gte / year_lte: 年份范围过滤。
        category: arXiv 分类（必须在白名单内，如 cs.CL）。
        has_code: 是否只看有开源仓库的论文（true/false/不传=不过滤）。
        sub_question_id: 所属子问题 id，便于结果回填到 SSE retrieval 事件。

    Returns:
        paper 级结果列表，每项含 arxiv_id / title / score / chunks（含 source 与章节）。

    ⚠️ 函数签名必须与 `HybridSearchArgs` 的字段**逐一对应**：LangChain 按 schema
    把模型给的参数逐个传进这个函数，schema 里有而签名里没有的字段会直接 TypeError
    （表现为"模型一旦想用 has_code，这次检索就炸"）。
    """
    args = HybridSearchArgs(
        query=query,
        keywords=keywords or [],
        top_k=top_k,
        rerank=rerank,
        year_gte=year_gte,
        year_lte=year_lte,
        category=category,
        has_code=has_code,
        sub_question_id=sub_question_id,
    )
    return await run_hybrid_search(args)


__all__ = [
    "build_filter",
    "get_shared_retriever",
    "hybrid_search",
    "reset_shared_retriever",
    "run_hybrid_search",
    "serialize_hit",
]

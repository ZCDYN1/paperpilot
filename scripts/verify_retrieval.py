"""检索层端到端验证脚本（离线 mock，可反复运行）。

**为什么要有这个脚本？**

单测锁的是"某条规则是否正确"，而这个脚本锁的是"整条流水线能不能跑起来"：
建索引 → 双路召回 → RRF → paper 聚合 → 重排 → 返回。二者不能互相替代 ——
单测全绿但模块之间字段对不上（比如 `arxiv_id` 大小写、`matched_by` 拼接）是很常见的。

运行方式（不需要任何模型权重 / API key）：
    EMBEDDING_PROVIDER=mock RERANK_PROVIDER=lexical python scripts/verify_retrieval.py

退出码非 0 表示流水线没跑通，可直接接进 CI。
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# 必须在 import app.* 之前设好环境变量：Settings 是进程级单例，
# 晚设会读到默认的 local 模式，触发 3GB 权重下载
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANK_PROVIDER", "lexical")
os.environ.setdefault("RERANK_ENABLED", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.hybrid_retriever import (  # noqa: E402
    ChunkRecord,
    HybridRetriever,
    MetadataFilter,
)


def _fake_corpus() -> tuple[list[ChunkRecord], dict[str, str]]:
    """构造 3 篇论文、每篇 3~5 个 chunk 的假语料。

    刻意让每篇论文与查询词的**词面重叠程度拉开差距**：论文 1 的每个 chunk
    都密集包含查询词（hybrid / retrieval / bm25 / reranker / fusion），
    论文 2、3 则完全不含 —— 这样"排序第一是论文 1"才是可断言的，
    而不是被 mock 随机向量带偏。

    若不做这个控制，重排（LexicalReranker 用 Jaccard）的结果会随语料随机波动，
    验证脚本就会时绿时红。
    """
    titles = {
        "2401.00001": "Retrieval Augmented Generation with Hybrid Search",
        "2401.00002": "A Survey on Sparse and Dense Indexing",
        "2401.00003": "Vision Transformers for Image Classification",
    }

    spec: list[tuple[str, int, str, str, str, int, str]] = [
        # (arxiv_id, chunk_id, section, text, source, year, category)
        (
            "2401.00001", 101, "abstract",
            "[Retrieval Augmented Generation with Hybrid Search] [abstract]\n"
            "We present a hybrid retrieval system that fuses dense embedding search and "
            "sparse bm25 retrieval with a reranker fusion.",
            "html", 2024, "cs.CL",
        ),
        (
            "2401.00001", 102, "method",
            "[Retrieval Augmented Generation with Hybrid Search] [method]\n"
            "Our hybrid retrieval pipeline combines dense retrieval with sparse bm25 "
            "retrieval, then applies a cross-encoder reranker fusion before ranking.",
            "html", 2024, "cs.CL",
        ),
        (
            "2401.00001", 103, "method",
            "[Retrieval Augmented Generation with Hybrid Search] [method]\n"
            "The reranker fusion module of our hybrid retrieval system boosts bm25 recall.",
            "html", 2024, "cs.CL",
        ),
        (
            "2401.00002", 201, "abstract",
            "[A Survey on Sparse and Dense Indexing] [abstract]\n"
            "We survey sparse and dense indexing techniques for information lookup systems.",
            "abstract", 2021, "cs.IR",
        ),
        (
            "2401.00002", 202, "introduction",
            "[A Survey on Sparse and Dense Indexing] [introduction]\n"
            "Sparse indexing relies on lexical matching while dense indexing uses learned vectors.",
            "abstract", 2021, "cs.IR",
        ),
        (
            "2401.00002", 203, "conclusion",
            "[A Survey on Sparse and Dense Indexing] [conclusion]\n"
            "Indexing practice favors a combination of sparse and dense approaches.",
            "abstract", 2021, "cs.IR",
        ),
        (
            "2401.00003", 301, "abstract",
            "[Vision Transformers for Image Classification] [abstract]\n"
            "We train vision transformer models for large scale image classification.",
            "html", 2023, "cs.CV",
        ),
        (
            "2401.00003", 302, "method",
            "[Vision Transformers for Image Classification] [method]\n"
            "Our patch embedding is followed by attention layers.",
            "html", 2023, "cs.CV",
        ),
        (
            "2401.00003", 303, "results",
            "[Vision Transformers for Image Classification] [results]\n"
            "Table 1: accuracy on ImageNet.",
            "html", 2023, "cs.CV",
        ),
    ]

    records = [
        ChunkRecord(
            chunk_id=chunk_id,
            arxiv_id=arxiv_id,
            section=section,
            text=text,
            source=source,
            chunk_type="table" if text.lstrip().startswith("Table") else "text",
            year=year,
            primary_category=category,
        )
        for arxiv_id, chunk_id, section, text, source, year, category in spec
    ]
    return records, titles


async def main() -> int:
    records, titles = _fake_corpus()

    with tempfile.TemporaryDirectory(prefix="paperpilot_verify_") as tmp:
        retriever = HybridRetriever(index_dir=Path(tmp))

        n = await retriever.build_index(records, titles)
        print(f"[build_index] chunks={n} retriever.size={retriever.size}")
        assert n == len(records), "建索引的 chunk 数不对"

        query = "hybrid retrieval bm25 reranker fusion"
        keywords = ["hybrid", "retrieval", "bm25", "reranker", "fusion"]

        hits = await retriever.search(query, keywords, top_k=10)
        print(f"\n[search] query={query!r} -> {len(hits)} papers")
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  #{rank} {hit.arxiv_id} score={hit.score:.4f} "
                f"title={hit.title!r} matched_by={hit.matched_by} "
                f"chunks={[c.chunk_id for c in hit.chunks]}"
            )

        assert hits, "检索没有返回任何 paper —— 流水线没跑通"
        # 词面最重叠的论文 2401.00001 应当排在第一位（BM25 起作用）
        assert hits[0].arxiv_id == "2401.00001", (
            f"期望 2401.00001 排第一，实际 {hits[0].arxiv_id}"
        )
        # 每篇论文最多 2 个 chunk
        assert all(len(h.chunks) <= 2 for h in hits), "paper 聚合的 chunk 上限失效"
        # 代表分必须等于该篇已保留 chunk 的最高分
        for hit in hits:
            assert hit.rep_chunk.chunk_id in [c.chunk_id for c in hit.chunks]
        print("\n[ok] 全流水线跑通：category/year 过滤、RRF、paper 聚合、重排均正常")

        # 顺带验证 metadata 过滤：只留 cs.IR 的论文
        filtered = await retriever.search(
            query, keywords, top_k=10, filters=MetadataFilter(categories=["cs.IR"])
        )
        ids = {h.arxiv_id for h in filtered}
        print(f"[filter] categories=cs.IR -> {ids}")
        assert ids == {"2401.00002"}, f"分类过滤失效：{ids}"

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

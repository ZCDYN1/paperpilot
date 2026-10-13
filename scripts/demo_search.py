"""检索效果演示 —— 快速肉眼确认索引与检索链路是否正常。

用法：
    # 最简：中文提问 + 英文关键词（本产品的核心用法）
    python scripts/demo_search.py --query "RAG 的重排技术" --keywords "reranking,rerank,reranker"

    # 带 metadata 过滤
    python scripts/demo_search.py --query "long context" --category cs.CL --year-gte 2024

    用 mock embedding 也能跑（结果以 BM25 为主）：
    EMBEDDING_PROVIDER=mock RERANK_PROVIDER=lexical python scripts/demo_search.py --query "..."

用途：
- 确认索引没有坏（返回的是不是合理的论文）
- 手工检查「中文提问 + 英文关键词」这条路径的效果
- 消融实验前的肉眼对照
"""

from __future__ import annotations

import argparse
import asyncio

from app.core.config import get_settings
from app.retrieval.hybrid_retriever import HybridRetriever, MetadataFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaperPilot 检索效果演示")
    parser.add_argument("--query", required=True, help="查询语句（可以是中文）")
    parser.add_argument(
        "--keywords",
        default="",
        help="英文关键词，逗号分隔。BM25 只认词面匹配，中文关键词打英文索引基本无效",
    )
    parser.add_argument("--top", type=int, default=5, help="返回论文数")
    parser.add_argument("--category", default="", help="arXiv 分类过滤，如 cs.CL")
    parser.add_argument("--year-gte", type=int, default=0, help="起始年份")
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    settings = get_settings()

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()] or None
    filters = MetadataFilter(
        categories=[args.category] if args.category else None,
        year_gte=args.year_gte or None,
    )

    retriever = HybridRetriever()
    print(f"索引 chunk 数：{retriever.size}")
    if retriever.is_empty:
        print("⚠️  索引是空的，先跑：python scripts/build_index.py")
        return 1
    print(
        f"embedding：{retriever.embedder.__class__.__name__} | "
        f"reranker：{retriever.reranker.name} | "
        f"mock_mode：{settings.mock_mode}\n"
    )

    hits = await retriever.search(
        args.query,
        keywords,
        rerank_top_n=args.top,
        filters=filters if (filters.categories or filters.year_gte) else None,
    )

    print(f"命中 {len(hits)} 篇论文（按重排分数）：\n")
    for rank, hit in enumerate(hits, start=1):
        print(f"{rank}. [{hit.score:.4f}] {hit.arxiv_id}  ← {'/'.join(hit.matched_by)}")
        print(f"   {hit.title[:88]}")
        print(
            f"   section={hit.rep_chunk.section} | 证据来源={hit.rep_chunk.source} | "
            f"命中 chunk 数={len(hit.chunks)}"
        )
        snippet = hit.rep_chunk.text.replace("\n", " ")
        print(f"   {snippet[:140]}...")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

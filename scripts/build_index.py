"""从数据库重建检索索引。

用法：
    python scripts/build_index.py

**为什么要单独一个脚本？**

抓取（ingest）和建索引是两件事。抓到的语料落在 SQLite（这是**事实来源**），
而 FAISS / BM25 索引只是从库里派生出来的**可重建数据**。把它们分开有三个好处：

1. 索引文件损坏、或换了 embedding 模型、或调了分块策略 —— 都能靠重跑这个脚本恢复；
2. 离线部署成为可能：把 SQLite 拷过去，跑一次就有了索引，不必重新抓 arXiv；
3. 让"抓取"和"检索"可以各自独立测试（这也是架构文档 §2 的分层原则）。
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy import select

from app.db.models import Paper, PaperChunk
from app.db.session import session_scope
from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever

logger = structlog.get_logger(__name__)


async def load_chunk_records() -> tuple[list[ChunkRecord], dict[str, str]]:
    """把库里所有 chunk 读成检索层需要的视图。

    注意这里做了一次 **ORM → 检索层数据结构** 的显式转换：
    检索层不认识 ORM 模型（架构文档 §2 的分层原则），
    这样它才能被单独测试、也才能在换存储时不受影响。
    """
    async with session_scope() as session:
        papers = (await session.execute(select(Paper))).scalars().all()
        titles = {p.arxiv_id: p.title for p in papers}
        years = {p.arxiv_id: (p.published.year if p.published else None) for p in papers}
        categories = {p.arxiv_id: p.primary_category for p in papers}

        rows = (
            (await session.execute(select(PaperChunk).order_by(PaperChunk.id)))
            .scalars()
            .all()
        )

    records = [
        ChunkRecord(
            chunk_id=row.id,
            arxiv_id=row.arxiv_id,
            section=row.section,
            text=row.text,
            source=row.source,
            chunk_type=row.chunk_type,
            year=years.get(row.arxiv_id),
            primary_category=categories.get(row.arxiv_id),
            page=row.page,
        )
        for row in rows
    ]
    return records, titles


async def main() -> int:
    started = time.perf_counter()
    records, titles = await load_chunk_records()

    if not records:
        print("⚠️  库里还没有语料，请先抓取：python -m app.ingest.pipeline --query \"...\" --max-results 10")
        return 1

    retriever = HybridRetriever()
    count = await retriever.build_index(records, titles)
    elapsed = time.perf_counter() - started

    print(
        f"✓ 索引重建完成：{count} 个 chunk / {len(titles)} 篇论文 / 耗时 {elapsed:.2f}s\n"
        f"  索引目录：{retriever.index_dir}\n"
        f"  索引指纹：{retriever.embedder.identity}\n"
        f"  reranker ：{retriever.reranker.name}\n"
        "\n"
        f"  提示：索引与 embedding 模型绑定。以后若更换 EMBEDDING_PROVIDER / 模型，\n"
        f"        检索会明确报错提示重建 —— 重跑本脚本即可，不需要重新抓 arXiv。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

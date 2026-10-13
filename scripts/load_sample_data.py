"""导入仓库内的离线样例语料（`sample_data/corpus.json`）。

**这个脚本存在的理由**：新用户 clone 下来时 `data/` 是空的（索引与库都不入库），
而第一次提问会触发联网抓 arXiv —— 那需要一个合规的 User-Agent、几分钟限速等待，
还要求能访问 arXiv。这条路径不适合"我想先看看它能不能跑"。
所以这里提供一份**零 key、零联网**的起步语料。

幂等：已存在的 `arxiv_id` 直接跳过，不会重复插入、不会覆盖你真实抓取的数据。

用法：
    python scripts/load_sample_data.py            # 语料为空时导入
    python scripts/load_sample_data.py --force    # 库里有 papers 也补缺失的
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import func, select

from app.core.console import force_utf8_output
from app.db.models import Paper, PaperChunk
from app.db.session import init_db, session_scope

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS = REPO_ROOT / "sample_data" / "corpus.json"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    # 被引数必须带"截至日期"（见 db/models.py），否则这个数字会被读者误读成实时值
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


async def load(force: bool = False) -> dict:
    if not CORPUS.exists():
        raise SystemExit(f"找不到样例语料文件：{CORPUS}")
    data = json.loads(CORPUS.read_text(encoding="utf-8"))
    papers = data.get("papers") or []

    await init_db()
    inserted_papers = inserted_chunks = skipped = 0

    async with session_scope() as session:
        existing = set(
            (await session.execute(select(Paper.arxiv_id))).scalars().all()
        )
        total = (await session.execute(select(func.count(Paper.arxiv_id)))).scalar() or 0

    if total and not force:
        print(f"库里已有 {total} 篇论文，样例语料导入跳过（要补缺失项用 --force）")
        return {"inserted_papers": 0, "inserted_chunks": 0, "skipped": len(papers)}

    for rec in papers:
        arxiv_id = rec["arxiv_id"]
        if arxiv_id in existing:
            skipped += 1
            continue
        async with session_scope() as session:
            session.add(
                Paper(
                    arxiv_id=arxiv_id,
                    title=rec["title"],
                    abstract=rec.get("abstract", ""),
                    authors=rec.get("authors") or [],
                    published=_parse_dt(rec.get("published")),
                    primary_category=rec.get("primary_category"),
                    categories=rec.get("categories") or [],
                    venue=rec.get("venue"),
                    cited_by_count=rec.get("cited_by_count"),
                    cited_at=_parse_date(rec.get("cited_at")),
                    referenced_works_count=rec.get("referenced_works_count"),
                    is_open_access=rec.get("is_open_access"),
                    oa_status=rec.get("oa_status"),
                    pdf_url=rec.get("pdf_url"),
                    html_url=rec.get("html_url"),
                    has_code=rec.get("has_code"),
                )
            )
            for chunk in rec.get("chunks") or []:
                session.add(
                    PaperChunk(
                        arxiv_id=arxiv_id,
                        chunk_index=chunk["chunk_index"],
                        section=chunk.get("section", "abstract"),
                        chunk_type=chunk.get("chunk_type", "text"),
                        source=chunk.get("source", "abstract"),
                        text=chunk["text"],
                        token_count=chunk.get("token_count", 0),
                    )
                )
                inserted_chunks += 1
            inserted_papers += 1

    return {"inserted_papers": inserted_papers, "inserted_chunks": inserted_chunks, "skipped": skipped}


async def main(force: bool) -> None:
    stats = await load(force=force)
    print(
        f"样例语料：新增 {stats['inserted_papers']} 篇 / {stats['inserted_chunks']} chunk，"
        f"已存在跳过 {stats['skipped']} 篇"
    )


if __name__ == "__main__":
    force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="库里已有论文时仍补齐缺失的篇目")
    args = parser.parse_args()
    asyncio.run(main(args.force))

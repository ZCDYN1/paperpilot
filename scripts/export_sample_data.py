"""把本地库里的论文元数据 + 摘要级 chunk 导出成仓库内的离线样例语料。

产物 `sample_data/corpus.json` 是**仓库交付物**：新用户 clone 下来不必联网抓 arXiv、
不必申请任何 key，就能跑通「检索 → 抽取 → 校验 → 撰写」全链路。

刻意只用**摘要**构造 chunk（正文 chunk 不进仓库：体积大、且属于作者版权内容）：
1. `paper_chunks.source="abstract"` 会在引用卡片上如实标出"摘要级证据"，
   不会让演示用户误以为看到的是正文原文；
2. 摘要逐字来自 arXiv 官方书目记录，引用校验的"逐字子串比对"在摘要上同样成立。

用法（只读本地库，不写库）：
    python scripts/export_sample_data.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.core.console import force_utf8_output
from app.db.models import Paper
from app.db.session import session_scope

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "sample_data" / "corpus.json"


async def export(limit: int) -> dict:
    async with session_scope() as session:
        papers = (await session.execute(select(Paper).order_by(Paper.arxiv_id))).scalars().all()

    picked: list[str] = []
    for p in papers:
        if not p.abstract.strip():
            continue
        picked.append(p.arxiv_id)
        if len(picked) >= limit:
            break
    chosen = set(picked)

    records = []
    for p in papers:
        if p.arxiv_id not in chosen:
            continue
        records.append(
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "abstract": p.abstract,
                "authors": p.authors or [],
                "published": p.published.isoformat() if p.published else None,
                "primary_category": p.primary_category,
                "categories": p.categories or [],
                "venue": p.venue,
                "cited_by_count": p.cited_by_count,
                "cited_at": str(p.cited_at) if p.cited_at else None,
                "referenced_works_count": p.referenced_works_count,
                "is_open_access": p.is_open_access,
                "oa_status": p.oa_status,
                "pdf_url": p.pdf_url,
                "html_url": p.html_url,
                "has_code": p.has_code,
                "chunks": [
                    {
                        "chunk_index": 0,
                        "section": "abstract",
                        "chunk_type": "text",
                        "source": "abstract",
                        "text": p.abstract.strip(),
                        "token_count": max(1, len(p.abstract) // 4),
                    }
                ],
            }
        )

    return {
        "kind": "paperpilot-sample-corpus",
        "version": 1,
        "note": (
            "离线样例语料：真实 arXiv 论文的元数据 + 摘要级分块（不含正文）。"
            "用于零 key、零联网跑通全链路；检索/引用质量以摘要级证据为准。"
        ),
        "papers": records,
    }


async def main(limit: int) -> None:
    data = await export(limit)
    OUTPUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    chunks = sum(len(p["chunks"]) for p in data["papers"])
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"导出 {len(data['papers'])} 篇 / {chunks} chunk → {OUTPUT} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    force_utf8_output()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=24, help="最多导出多少篇（默认 24）")
    args = parser.parse_args()
    asyncio.run(main(args.limit))

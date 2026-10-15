"""引用硬校验**离线重算** —— 判据升级后，用新口径重评库里已有的 citations（默认 dry-run）。

为什么能离线重算（背景见 `app/agents/nodes/_common.py` 的两段式判定注释）：
硬校验是纯字符串比较，不碰模型。判据从"精确比较"升级为"精确比较 → 抽取残留容忍"
之后，历史库里攒着一批假阴性（证据在、只是当时比对方式过严），它们被列进了
「未找到原文支撑的结论」，把自动口径（`citations.verified` 占比）系统性压低。
重跑一次研究要再花一遍 token，而这里一次数据库读取就够 —— 脚本因此只做一件事：
拿 `quote_match()` 对每条 citation 的 quote × chunk 原文重算一遍，报告
多少条 False→True、分别属于哪类归一化。

口径与边界（务必读懂再 --write）：
- 只重算**硬校验层**。LLM 软校验离线跑不了也不该重跑，因此：
  `note` 里写着"LLM 判定"的 False 行**永不翻转** —— 字符串层再宽也救不了软校验的否决。
- 允许翻转的行：`verified=False` 且失败原因来自硬校验（note 为逐字匹配失败），
  或 note 为空的历史行（当时没落原因，无从考证 —— 报告里单独计数，建议抽检复核）。
  硬原因当年盖过了软原因，所以"翻转=假定软校验也会通过"这一前提是已知残余不确定性，
  报告如实标注，不装作用户看不见。
- `verified=True` 的行只重算并报告异常（原判定通过但新口径连容忍层都对不上，
  通常是 chunk 已被删），**绝不回写** —— 本脚本只把 False 翻 True，不反向改判既有结论。
- 判定只在"这一块原文里"重算，同篇重锚作为兜底路径一并模拟（与 verify 的
  `_hard_check` 两条路径同口径），但落库不动 `chunk_id`/`resolution_how`：
  定位是当时的事实，本脚本没有检索结果可回写，改定位是越界。

用法（连 `.env` 里配置的 SQLite，路径由 `app.core.config.get_settings()` 决定）：

    python scripts/recheck_citations.py            # dry-run（默认）：只报告，不改库
    python scripts/recheck_citations.py --write    # 把允许翻转的行真正落库
    python scripts/recheck_citations.py --run-id <run_id>   # 只重算某个 run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def rejudge(quote: str, direct_text: str | None, same_paper_texts: list[str]) -> str:
    """按新口径重算一条引用的**逐字比较路径**（返回 `quote_match` 的路径码）。

    先查引用的那个 chunk，再退到同篇其余片段 —— 与 verify `_hard_check` 的
    两条定位路径同口径（判定只在 `quote_match` 一处定义，这里不重写规则）。
    """
    if direct_text is not None:
        match = _quote_match(quote, direct_text)
        if match:
            return match
    for text in same_paper_texts:
        match = _quote_match(quote, text)
        if match:
            return match
    return ""


def _quote_match(quote: str, chunk_text: str) -> str:
    # 延迟 import：与 spotcheck_citations.py 同风格，--help 不必拖起整个 app。
    from app.agents.nodes._common import quote_match

    return quote_match(quote, chunk_text)


async def fetch_rows(run_id: str | None) -> list[dict[str, Any]]:
    """读 citations 与其 chunk 原文（**只读**查询；--write 的回写在主流程里做）。"""
    from sqlalchemy import select

    from app.db import session as db_session
    from app.db.models import Citation, PaperChunk

    async with db_session.session_scope() as session:
        stmt = (
            select(
                Citation.id,
                Citation.run_id,
                Citation.arxiv_id,
                Citation.chunk_id,
                Citation.quote,
                Citation.verified,
                Citation.note,
                PaperChunk.text.label("chunk_text"),
            )
            .join(PaperChunk, PaperChunk.id == Citation.chunk_id, isouter=True)
        )
        if run_id:
            stmt = stmt.where(Citation.run_id == run_id)
        rows = (await session.execute(stmt)).all()

        # 同篇重锚的语料：只按出现的 arxiv_id 取，不全表扫
        arxiv_ids = {r.arxiv_id for r in rows if r.arxiv_id}
        by_paper: dict[str, list[str]] = {}
        if arxiv_ids:
            chunk_rows = (
                await session.execute(
                    select(PaperChunk.arxiv_id, PaperChunk.text).where(
                        PaperChunk.arxiv_id.in_(arxiv_ids)
                    )
                )
            ).all()
            for aid, text in chunk_rows:
                by_paper.setdefault(aid, []).append(text)

    return [
        {
            "id": r.id,
            "run_id": r.run_id,
            "arxiv_id": r.arxiv_id or "",
            "chunk_id": r.chunk_id,
            "quote": r.quote or "",
            "verified": r.verified,
            "note": r.note,
            "chunk_text": r.chunk_text,
            "same_paper": by_paper.get(r.arxiv_id or "", []),
        }
        for r in rows
    ]


def plan_changes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """重算 + 归类，产出报告数据与"允许翻转"的回写计划（纯函数，可单测）。"""
    from app.agents.nodes._common import MATCH_EXACT, residual_kind, residual_note

    stats: Counter[str] = Counter()
    flips_by_kind: Counter[str] = Counter()
    plan: list[tuple[int, bool, str | None]] = []  # (citation_id, verified_new, note_new)
    anomalies: list[int] = []
    no_note_candidates = 0

    for row in rows:
        stats[f"verified:{row['verified']}"] += 1
        match = rejudge(row["quote"], row["chunk_text"], row["same_paper"])
        stats[f"match:{match or 'not_found'}"] += 1

        if row["verified"] is True:
            # 只报告异常，不回写（见模块 docstring 的边界）
            if not match:
                anomalies.append(row["id"])
            elif match != MATCH_EXACT:
                stats["true_via_residual"] += 1
            continue

        if row["verified"] is not False:
            continue  # None = 从未校验过，翻转等于伪造软校验结论 —— 不动

        if not match:
            continue  # 找不到就是找不到：新口径也救不回来的，维持失败
        note = row["note"] or ""
        if "LLM" in note:
            # 软校验否决的行：硬校验当时其实是过的，翻转与否都与本层无关 —— 不动。
            stats["skipped_soft_reject"] += 1
            continue
        if not note.strip():
            no_note_candidates += 1
        kind = residual_kind(match) or "exact"
        flips_by_kind[kind] += 1
        plan.append((row["id"], True, None if kind == "exact" else residual_note(match)))

    return {
        "stats": stats,
        "flips": plan,
        "flips_by_kind": flips_by_kind,
        "no_note_candidates": no_note_candidates,
        "anomalies": anomalies,
        "total": len(rows),
    }


async def apply_writes(plan: list[tuple[int, bool, str | None]]) -> int:
    from sqlalchemy import select

    from app.db import session as db_session
    from app.db.models import Citation

    if not plan:
        return 0
    new_by_id = {row_id: (verified, note) for row_id, verified, note in plan}
    written = 0
    async with db_session.session_scope() as session:
        rows = (
            await session.execute(select(Citation).where(Citation.id.in_(new_by_id)))
        ).scalars().all()
        for citation in rows:
            citation.verified, citation.note = new_by_id[citation.id]
            written += 1
    return written


def report(result: dict[str, Any], write: bool) -> int:
    from app.core.config import get_settings

    stats: Counter[str] = result["stats"]
    total = result["total"]
    if total == 0:
        print("库里没有任何 citations —— 先跑一次研究（UI 或 API），再来重算。")
        return 1
    mode = "★ WRITE：将回写" if write else "dry-run（默认）：只报告，不改库"
    print(f"== 引用硬校验离线重算 · {mode} ==")
    print(f"数据库：{get_settings().sqlite_url}")
    print(f"共 {total} 条 citations")
    print(
        f"当前 verified 分布：True={stats['verified:True']} / "
        f"False={stats['verified:False']} / 未校验(None)={stats['verified:None']}"
    )
    print("新口径匹配路径分布：")
    for key in (
        "match:exact",
        "match:residual:citation_markers",
        "match:residual:punct_spacing",
        "match:residual:quotes",
        "match:residual:combined",
        "match:not_found",
    ):
        print(f"  {key.removeprefix('match:'):28s} {stats[key]}")

    flips = result["flips"]
    kinds = result["flips_by_kind"]
    print(f"\nFalse→True 候选（硬校验失败行里，新口径能对上原文的）：{len(flips)} 条")
    for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {kind:16s} {count}")
    if result["no_note_candidates"]:
        print(
            f"  ⚠️ 其中 note 为空的历史行 {result['no_note_candidates']} 条："
            "当时没落失败原因，翻转隐含\"软校验当时会通过\"的前提，建议抽检复核这些行。"
        )
    print(f"不翻转：软校验否决的行 {stats['skipped_soft_reject']} 条（字符串层无权救回）")
    if stats["true_via_residual"]:
        print(f"备注：verified=True 但新口径需容忍层才对上的行 {stats['true_via_residual']} 条"
              "（chunk 内容被重建过？只读标记，不动）。")
    if result["anomalies"]:
        print(f"异常：verified=True 但连容忍层都对不上：{len(result['anomalies'])} 条"
              f"（id 示例 {result['anomalies'][:5]}；多为 chunk 已删。只读标记，不回写）。")

    if not write:
        print("\ndry-run 结束，未改动数据库。确认数字后加 --write 落库。")
        return 0
    written = asyncio.run(apply_writes(flips))
    print(f"\n已回写 {written} 条（verified=True，note 记录归一化类别）。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="用新的两段式硬校验口径离线重算已有 citations（默认 dry-run，只读）"
    )
    parser.add_argument(
        "--write", action="store_true", help="回写允许的翻转（默认只报告，不改库）"
    )
    parser.add_argument("--run-id", default=None, help="只重算指定 run 的引用")
    args = parser.parse_args(argv)

    rows = asyncio.run(fetch_rows(args.run_id))
    return report(plan_changes(rows), write=args.write)


if __name__ == "__main__":
    sys.exit(main())

"""引用准确率**人工口径**抽检 —— 架构 §12「不要自己批改自己的作业」的落地。

背景（为什么自动口径不够）：
`citation_accuracy()` 统计的是 `citations.verified` 的占比，而 `verified` 是
Verifier（硬校验 + LLM 软校验）写回的。如果 LLM 校验是"橡皮图章"
（mock provider 就是按设计永远放行），自动口径会恒等于 1.0 —— **结构保证，不是证据**。
架构文档因此要求第二条口径：周期性从库里随机抽样引用，**由人**对照原文片段判定
"引用是否名副其实"，两个口径并列汇报，并用一致率校准 Verifier 是否虚高放行。

用法（只读现有 SQLite，**不需要任何 API key、不碰模型**）：

    # ① 抽样导出（默认 20 条，seed 固定可复现；--n/--seed/--out 可改）
    python scripts/spotcheck_citations.py
    # ② 把 CSV 里 human_verdict 列逐条填 supported / unsupported / ambiguous
    # ③ 回填统计：打印人工口径准确率 + 与自动口径的一致率
    python scripts/spotcheck_citations.py --from-csv eval_results/citation_spotcheck.csv

设计约束：
- 导出的每一行都带齐人工判定所需的**全部材料**（claim / quote / 原文 chunk 正文 /
  run 问题 / 自动校验结论），不需要再开数据库；
- 一致率把"未判定"（空 verdict）单独计数，绝不把"没标"混进分母——
  这会让准确率被样本缺失悄悄拉偏，是抽检最容易犯的统计错误。
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 默认输出与 app.eval.ragas_runner.DEFAULT_SPOTCHECK_CSV 指向同一路径，
# 报告端（run_ragas_eval）据此自动并列人工口径。
DEFAULT_OUT = PROJECT_ROOT / "eval_results" / "citation_spotcheck.csv"
# 架构 §12 原文："周期性从 golden 集里随机抽 20 条人工核对"
DEFAULT_N = 20

CSV_COLUMNS = [
    "citation_id",
    "run_id",
    "run_question",
    "arxiv_id",
    "chunk_id",
    "locator",
    "source",
    "auto_verified",   # Verifier 的自动结论（True/False/空=未校验）
    "verify_note",     # Verifier 记录的失败原因（finalize 已落进 citations.note）
    "claim",          # 待核对的论断
    "quote",          # 引用到的原文片段
    "chunk_text",     # 该 chunk 全量正文 —— 人工判定的唯一依据
    "human_verdict",  # ★ 留空供人工填写：supported / unsupported / ambiguous
    "human_note",     # 备注（可选）
]


async def fetch_pool() -> list[dict[str, Any]]:
    """取全部 citations 并 join 出人工判定所需的材料（**只读**）。"""
    from sqlalchemy import select

    from app.db import session as db_session
    from app.db.models import Citation, PaperChunk, ResearchRun

    async with db_session.session_scope() as session:
        rows = (
            await session.execute(
                select(
                    Citation.id,
                    Citation.run_id,
                    Citation.arxiv_id,
                    Citation.chunk_id,
                    Citation.locator,
                    Citation.source,
                    Citation.verified,
                    Citation.claim,
                    Citation.quote,
                    Citation.note,
                    PaperChunk.text,
                    ResearchRun.question,
                )
                .join(PaperChunk, PaperChunk.id == Citation.chunk_id, isouter=True)
                .join(ResearchRun, ResearchRun.id == Citation.run_id, isouter=True)
            )
        ).all()

    pool: list[dict[str, Any]] = []
    for row in rows:
        pool.append(
            {
                "citation_id": row[0],
                "run_id": row[1] or "",
                "run_question": row[11] or "",
                "arxiv_id": row[2] or "",
                "chunk_id": "" if row[3] is None else row[3],
                "locator": row[4] or "",
                "source": row[5] or "",
                "auto_verified": "" if row[6] is None else str(bool(row[6])),
                "claim": row[7] or "",
                "quote": row[8] or "",
                "verify_note": row[9] or "",
                "chunk_text": (row[10] or "").strip(),  # chunk 被删时为空：人工据此判 unsupported
            }
        )
    return pool


def pick_sample(pool: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """固定 seed 的随机抽样：同一库 + 同一 seed 必须抽到同一批，抽检才可复查。"""
    if n <= 0:
        raise ValueError("--n 必须为正整数")
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool))) if len(pool) > n else list(pool)


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig：Excel 直接双击也能正确显示中文（抽检大概率在 Excel 里做）
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


async def cmd_export(n: int, seed: int, out_path: Path) -> int:
    pool = await fetch_pool()
    if not pool:
        print("库里没有任何 citations —— 先跑一次研究（UI 或 API），再来抽检。")
        return 1
    sample = pick_sample(pool, n, seed)
    write_csv(sample, out_path)
    print(f"已抽样 {len(sample)} 条（库里共 {len(pool)} 条，seed={seed}）→ {out_path}")
    print("请在 human_verdict 列逐条填写 supported / unsupported / ambiguous，然后执行：")
    print(f"    python scripts/spotcheck_citations.py --from-csv {out_path}")
    if len(sample) < n:
        print(f"⚠️ 库中引用不足 {n} 条，只能全量导出 {len(sample)} 条。")
    return 0


def cmd_stats(csv_path: Path) -> int:
    from app.eval.ragas_runner import manual_accuracy

    try:
        stats = manual_accuracy(csv_path)
    except FileNotFoundError:
        print(f"文件不存在：{csv_path}（先跑不带 --from-csv 的导出模式）")
        return 1
    if stats["total_judged"] == 0:
        print(f"{csv_path} 中还没有任何 human_verdict 判定（可填 supported / unsupported / ambiguous）。")
        return 0
    print(f"== 人工口径（抽检） {csv_path} ==")
    print(f"  已判定        : {stats['total_judged']} 条"
          f"（supported={stats['supported']} / unsupported={stats['unsupported']}"
          f" / ambiguous={stats['ambiguous']}）")
    print(f"  人工准确率    : {stats['accuracy']:.4f}")
    if stats["agreement"] is not None:
        print(f"  与自动口径一致率: {stats['agreement']:.4f}"
              f"（自动口径同时报告为 {stats.get('auto_accuracy'):.4f}）")
    if stats.get("pending"):
        print(f"  未判定（不计入分母）: {stats['pending']} 条")
    if stats["total_judged"] < DEFAULT_N:
        print(f"  ⚠️ 样本量 {stats['total_judged']} < 建议的 {DEFAULT_N}，结论仅作方向参考。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="引用准确率人工口径抽检（只读，无需 key）")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="抽样条数（默认 20，架构 §12）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（同库同 seed 结果可复现）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="导出的 CSV 路径")
    parser.add_argument("--from-csv", type=Path, default=None, help="对已填写的 CSV 计算人工口径统计")
    args = parser.parse_args(argv)

    if args.from_csv:
        return cmd_stats(args.from_csv)

    import asyncio

    return asyncio.run(cmd_export(args.n, args.seed, args.out))


if __name__ == "__main__":
    sys.exit(main())

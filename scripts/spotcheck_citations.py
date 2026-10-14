"""引用准确率**人工口径**抽检 —— 架构 §12「不要自己批改自己的作业」的落地。

背景（为什么自动口径不够）：
`citation_accuracy()` 统计的是 `citations.verified` 的占比，而 `verified` 是
Verifier（硬校验 + LLM 软校验）写回的。如果 LLM 校验是"橡皮图章"
（mock provider 就是按设计永远放行），自动口径会恒等于 1.0 —— **结构保证，不是证据**。
架构文档因此要求第二条口径：周期性从库里随机抽样引用，**由人**对照原文片段判定
"引用是否名副其实"，两个口径并列汇报，并用一致率校准 Verifier 是否虚高放行。

用法（只读现有 SQLite，**不需要任何 API key、不碰模型**）：

    # ① 抽样导出（默认 20 条，seed 固定可复现；--n/--seed/--out/--provider 可改）
    python scripts/spotcheck_citations.py
    # ② 把 CSV 里 human_verdict 列逐条填 supported / unsupported / ambiguous
    # ③ 回填统计：打印人工口径准确率 + 与自动口径的一致率
    python scripts/spotcheck_citations.py --from-csv eval_results/citation_spotcheck.csv

抽样池默认剔除 mock run（--provider real）：mock 的 verifier 分支按设计永远放行，
那些行的 `verified=True` 是**结构恒真**，让人核对它们既浪费人力、又会把人工口径
系统性拉高。provider 由 `token_usage.model` 反查（finalize 落库时写入，
mock 模式下 `llm_identity()` 返回 model="mock"），**零 schema 改动**。
没跑到 finalize 的 run 没有 token_usage 行，provider 确实不可考 ——
标为 `unknown`，默认同样不抽（来源不可考的行不值得占用人力），
需要时用 `--provider unknown` / `--provider all` 显式取回，绝不悄悄猜测或丢弃。

设计约束：
- 导出的每一行都带齐人工判定所需的**全部材料**（claim / quote / 原文 chunk 正文 /
  run 问题 / 自动校验结论 / run 的 provider 来源），不需要再开数据库；
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
    "run_provider",   # 该 run 的 LLM 来源：real / mock / unknown（由 token_usage.model 反查；
                      # unknown = run 没跑到 finalize、无 token_usage 行，来源不可考）
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
    """取全部 citations 并 join 出人工判定所需的材料（**只读**）。

    每行附带 `run_provider`（real / mock / unknown）：由 `token_usage.model` 反查，
    mock 模式下 finalize 写入的是 `llm_identity()["model"] == "mock"`；
    没有任何 token_usage 行的 run（没跑到 finalize）如实标 unknown，不做猜测。
    """
    from sqlalchemy import select

    from app.db import session as db_session
    from app.db.models import Citation, PaperChunk, ResearchRun, TokenUsage

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
        # run_id → 该 run 全部 token_usage 行的 model（finalize 会写"all"+分节点多行，
        # 同一 run 内 model 相同；取 distinct 即可判定 provider）
        usage_rows = (
            await session.execute(select(TokenUsage.run_id, TokenUsage.model).distinct())
        ).all()

    models_by_run: dict[str, list[str | None]] = {}
    for rid, model in usage_rows:
        if rid:
            models_by_run.setdefault(rid, []).append(model)

    pool: list[dict[str, Any]] = []
    for row in rows:
        run_id = row[1] or ""
        pool.append(
            {
                "citation_id": row[0],
                "run_id": run_id,
                "run_provider": classify_provider(models_by_run.get(run_id, [])),
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


PROVIDERS = ("real", "mock", "unknown")


def classify_provider(models: list[str | None]) -> str:
    """由某 run 的 token_usage.model 集合判定 provider（纯函数，便于无库测试）。

    - 无任何行（或 model 全为空）→ "unknown"：run 没跑到 finalize，来源不可考；
    - 出现过非 "mock" 的 model → "real"；
    - 全部是 "mock" → "mock"（mock 模式 `llm_identity()` 固定返回 model="mock"）。
    """
    named = [m for m in models if m]
    if not named:
        return "unknown"
    return "real" if any(m != "mock" for m in named) else "mock"


def filter_pool_by_provider(
    pool: list[dict[str, Any]], provider: str = "real"
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按 provider 过滤抽样池（纯函数）。

    Returns:
        (保留的行, 各 provider 被剔除的行数)。"all" 时全保留、剔除数全 0。
        缺 `run_provider` 键的行按 "unknown" 处理——宁可显式剔除，不当成 real 混进人工样本。
    """
    if provider not in (*PROVIDERS, "all"):
        raise ValueError(f"provider 必须是 {PROVIDERS} 或 all，收到：{provider!r}")
    kept: list[dict[str, Any]] = []
    dropped: dict[str, int] = {p: 0 for p in PROVIDERS}
    for row in pool:
        prov = row.get("run_provider") or "unknown"
        if provider == "all" or prov == provider:
            kept.append(row)
        else:
            dropped[prov] += 1
    return kept, dropped


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


async def cmd_export(n: int, seed: int, out_path: Path, provider: str = "real") -> int:
    pool = await fetch_pool()
    if not pool:
        print("库里没有任何 citations —— 先跑一次研究（UI 或 API），再来抽检。")
        return 1
    candidates, dropped = filter_pool_by_provider(pool, provider)
    dropped_desc = "、".join(f"{p} {c} 条" for p, c in dropped.items() if c)
    print(f"抽样池：库里共 {len(pool)} 条引用，--provider {provider} 过滤后剩 {len(candidates)} 条"
          + (f"（剔除 {dropped_desc}）" if dropped_desc else ""))
    if provider == "real" and dropped["mock"]:
        print("  mock run 的 verified 结构恒真（verifier 按设计永远放行），核对它们会虚高人工口径。")
    if not candidates:
        print("过滤后没有任何可抽样的引用；如需查看全量（含 mock/unknown），加 --provider all。")
        return 1
    sample = pick_sample(candidates, n, seed)
    write_csv(sample, out_path)
    print(f"已抽样 {len(sample)} 条（seed={seed}）→ {out_path}")
    if len(sample) < n:
        # 有一说一：宁可明确告知样本不足，也不悄悄少给（少给会拉低人工口径的置信度）
        print(f"注意：provider={provider} 的引用只有 {len(sample)} 条，不足请求的 {n} 条，已全量导出。")
    print("请在 human_verdict 列逐条填写 supported / unsupported / ambiguous，然后执行：")
    print(f"    python scripts/spotcheck_citations.py --from-csv {out_path}")
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
        print(f"  注意：样本量 {stats['total_judged']} < 建议的 {DEFAULT_N}，结论仅作方向参考。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="引用准确率人工口径抽检（只读，无需 key）")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="抽样条数（默认 20，架构 §12）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（同库同 seed 结果可复现）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="导出的 CSV 路径")
    parser.add_argument(
        "--provider",
        choices=[*PROVIDERS, "all"],
        default="real",
        help="只抽该来源 run 的引用（默认 real：mock 的 verified 结构恒真，unknown 来源不可考）",
    )
    parser.add_argument("--from-csv", type=Path, default=None, help="对已填写的 CSV 计算人工口径统计")
    args = parser.parse_args(argv)

    if args.from_csv:
        return cmd_stats(args.from_csv)

    import asyncio

    return asyncio.run(cmd_export(args.n, args.seed, args.out, args.provider))


if __name__ == "__main__":
    sys.exit(main())

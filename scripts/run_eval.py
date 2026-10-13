"""评测统一入口 —— 一条命令跑完全部评测。

```
python scripts/run_eval.py --suite ablation    # 检索消融（只需索引，可完全离线）
python scripts/run_eval.py --suite ragas       # RAGAS 生成侧评测（必须有真实 LLM 当裁判）
python scripts/run_eval.py --suite all
```

**为什么需要一个统一入口？**

两种评测的**环境要求截然不同**：检索消融只需要一个建好的索引，
而 RAGAS 的 `faithfulness` / `answer_relevancy` 本质是"用 LLM 当裁判"，没有真实模型就没有意义。

混在一起最容易犯、后果也最严重的错，就是**在 mock 环境下跑 RAGAS，然后把没有意义的分数当成结论**。
所以本入口对每一项评测都显式汇报三件事：能不能跑、为什么不能跑、结论可信到什么程度 ——
把"不可用"当成一等公民对待，而不是静默跳过（静默跳过会让人误以为"跑过了、没问题"）。

检索消融产出的表格是**真实数字**（只要 embedding 不是 mock）；
mock 环境下仍会跑，但报告顶部会带显著警告 —— 那种情况下它只验证评测管线，不代表检索质量。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 允许 `python scripts/run_eval.py` 直接运行（把项目根加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.eval.ablation import run_and_render  # noqa: E402
from app.eval.dataset import load_golden  # noqa: E402
from app.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent


async def _run_ablation(args: argparse.Namespace, output_dir: Path) -> int:
    """检索消融：不需要 LLM，因此在本机总能跑出数字。"""
    print("=" * 78)
    print("▶ 检索消融实验（性能侧：Recall / nDCG / MRR / 耗时）")
    print("=" * 78)

    settings = get_settings()
    retriever = HybridRetriever(settings=settings)
    if retriever.is_empty:
        print("✗ 索引为空 —— 先跑 `python scripts/build_index.py` 建索引")
        return 2

    print(f"  索引：{retriever.size} chunk / {len(retriever._titles)} 篇论文")
    print(f"  embedding：{type(retriever.embedder).__name__}（维度 {retriever.embedder.dim}）")
    print(f"  reranker ：{retriever.reranker.name}")
    print()

    items = load_golden(args.golden)
    print(f"  golden 集：{len(items)} 条查询")
    print()

    out = Path(args.output) if args.output else output_dir / "ablation_latest.md"
    markdown, _results = await run_and_render(
        retriever, items, settings=settings, golden_path=args.golden, top_k=args.top_k
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    # 控制台只打表头和结果行，警告与脚注留在文件里（终端看得清才有人愿意看）
    for line in markdown.splitlines():
        if line.startswith("|") or line.startswith("## 结果表") or line.startswith("- **"):
            print(line)
    print()
    print(f"✓ 报告已写入 {out}")
    return 0


async def _run_ragas(args: argparse.Namespace, output_dir: Path) -> int:
    """RAGAS 生成侧评测。

    模块可能不存在（还没实现）或不可用（没装 ragas / 没有真实 LLM），
    这两种情况都必须给出**清晰的原因**，而不是抛栈或假装成功。
    """
    print("=" * 78)
    print("▶ RAGAS 评测（生成侧：faithfulness / answer_relevancy / ...）")
    print("=" * 78)

    try:
        from app.eval import ragas_runner
    except ImportError as exc:
        print(f"✗ 未找到 RAGAS 运行器：{exc}")
        print("  （生成侧评测是可选项，不影响检索消融与主流程）")
        return 2

    settings = get_settings()
    if settings.llm_provider == "mock":
        # 这是最需要防住的一种情况：mock 下 RAGAS 的"裁判"是假的，
        # 分数看着像模像样，实则毫无意义。显式拒绝，而不是产出一份漂亮但错误的报告。
        print("✗ 当前 LLM_PROVIDER=mock —— RAGAS 需要真实 LLM 作为裁判。")
        print("  请配置 LLM（如 LLM_PROVIDER=ollama + OLLAMA_MODEL=qwen2.5:7b）后重试。")
        print("  ⚠️ 不会产出 mock 分数：那比不跑更糟 —— 会让人误以为指标已验证。")
        return 3

    try:
        report = await ragas_runner.run_ragas_eval(run_ids=args.run_id)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ RAGAS 执行失败：{type(exc).__name__}: {exc}")
        return 1

    if not getattr(report, "available", True):
        print(f"✗ RAGAS 不可用：{getattr(report, 'reason', '未知原因')}")
        return 3

    out = output_dir / "ragas.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(str(report), encoding="utf-8")
    print(str(report))
    print(f"\n✓ 报告已写入 {out}")
    return 0


async def _main_async(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    code = 0

    if args.suite in ("ablation", "all"):
        code |= await _run_ablation(args, output_dir)
        if args.suite == "all":
            print()

    if args.suite in ("ragas", "all"):
        code |= await _run_ragas(args, output_dir)

    return code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PaperPilot 评测入口")
    parser.add_argument(
        "--suite", choices=["ablation", "ragas", "all"], default="ablation",
        help="跑哪套评测（默认 ablation，它不需要任何 API key）",
    )
    parser.add_argument("--golden", default=None, help="golden 集路径（JSONL）")
    parser.add_argument("--top-k", type=int, default=None, help="每路召回条数")
    parser.add_argument("--output", default=None, help="消融报告输出路径")
    parser.add_argument("--output-dir", default=str(_ROOT / "eval_results"), help="输出目录")
    parser.add_argument("--run-id", default=None, help="RAGAS 评测指定的 run id（默认全部已完成 run）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())

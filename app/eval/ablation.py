"""检索消融实验 —— 让「混合检索比纯向量好」这句话由数据兑现，而不是靠断言。

**这个模块为什么必须存在？**

架构 §5.3 要求给出六种检索配置的实测对比。如果不把它做成可复现的实验，
团队迟早会变成"拍脑袋决定要不要开 Reranker"：效果好的那部分增益无法归因，
延迟代价也没人量化。所以这里把它固化成一条命令：

    python -m app.eval.ablation

**三条刻意的设计约束**（每一条都对应一类会毁掉结论的错误）：

1. **只变检索策略，其它一律冻结**。同一批查询、同一份索引、同一个 top_k。
   否则"检索好了 1 分、索引换了一批"这种事会让整张表失去归因能力。
2. **必须记录耗时**。架构文档原文："只看效果不看代价会得出'Reranker 应该一直开着'的错误结论"。
   这里取**每条查询耗时的中位数**，而不是均值 —— 首条查询往往包含 jieba 词典加载等
   一次性开销，均值会被这一个离群点显著拉高，中位数则不受影响。
3. **诚实标注当前 embedding provider**。`MockEmbedder` 是哈希随机向量、不含任何语义，
   纯向量那一行的分数接近随机。若不加警告就把数字当"检索质量"呈现，比不给数字更糟 ——
   读者会拿一个假的基线去对比。所以报告顶部的警告是**根据实际 provider 动态生成**的。

⚠️ **口径**：检索/融合是 chunk 级，但本表的 Recall@5 / nDCG@10 全是 **paper 级**
（先 chunk 检索 → paper 聚合 → 再算指标），与架构 §5.2 的口径统一要求一致。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.core.config import Settings, get_settings
from app.eval.dataset import GoldenItem, load_golden
from app.eval.metrics import RetrievalMetrics, evaluate_retrieval
from app.retrieval.embedder import BaseEmbedder, MockEmbedder
from app.retrieval.hybrid_retriever import HybridRetriever, MetadataFilter, SearchConfig

# 默认输出落在项目根的 eval_results/。用 __file__ 定位而非相对 CWD ——
# 否则从任意目录执行 `python -m app.eval.ablation` 都会把文件写到别处（甚至找不到）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval_results" / "ablation.md"


# --------------------------------------------------------------------------- 配置
@dataclass(frozen=True, slots=True)
class AblationConfig:
    """一种消融配置：一个稳定 key + 人类可读标签 + 一个检索开关组合。"""

    key: str
    label: str
    rationale: str
    search: SearchConfig


# 六种配置严格对应架构 §5.3 的表格行。`SearchConfig` 的构造参数**逐位显式写出**
# （而不是只覆盖差异项）—— 因为我们希望这张表一眼就能看出每一行的完整开关状态，
# 任何一处默认值漂移都能在图里被看见。
ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        key="vector",
        label="① 纯向量（基线）",
        rationale="只开 FAISS 语义召回，作为一切增益的比较基准",
        search=SearchConfig(
            use_vector=True,
            use_bm25=False,
            fusion="none",
            use_rerank=False,
            use_metadata_filter=False,
        ),
    ),
    AblationConfig(
        key="bm25",
        label="② 纯 BM25",
        rationale="只开词面召回，用于对比语义与词面两路各自的强弱",
        search=SearchConfig(
            use_vector=False,
            use_bm25=True,
            fusion="none",
            use_rerank=False,
            use_metadata_filter=False,
        ),
    ),
    AblationConfig(
        key="weighted",
        label="③ 向量 + BM25 加权求和",
        rationale="RRF 的对照组：看归一化后的分数相加，是否真不如只看名次",
        search=SearchConfig(
            use_vector=True,
            use_bm25=True,
            fusion="weighted",
            use_rerank=False,
            use_metadata_filter=False,
        ),
    ),
    AblationConfig(
        key="rrf",
        label="④ 向量 + BM25 + RRF",
        rationale="线上生产链路（除重排/过滤外），是 ⑤⑥ 的增益基准",
        search=SearchConfig(
            use_vector=True,
            use_bm25=True,
            fusion="rrf",
            use_rerank=False,
            use_metadata_filter=False,
        ),
    ),
    AblationConfig(
        key="rerank",
        label="⑤ ④ + Reranker",
        rationale="量化交叉编码器重排带来的净增益与额外延迟",
        search=SearchConfig(
            use_vector=True,
            use_bm25=True,
            fusion="rrf",
            use_rerank=True,
            use_metadata_filter=False,
        ),
    ),
    AblationConfig(
        key="filter",
        label="⑥ ⑤ + metadata 过滤",
        rationale="量化「已知正确分类约束」带来的收益（注意其前提，见报告脚注）",
        search=SearchConfig(
            use_vector=True,
            use_bm25=True,
            fusion="rrf",
            use_rerank=True,
            use_metadata_filter=True,
        ),
    ),
)

# 允许用户用圈号写法挑选配置（`--configs ④⑤`），因为设计文档里就是这么编号的。
_CIRCLED_DIGITS = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5", "⑥": "6"}


def select_configs(spec: str | None) -> list[AblationConfig]:
    """把 `--configs` 的字符串解析成配置列表。

    支持三种写法（可混用、逗号或空格分隔）：序号 `4,5,6`、稳定 key `rrf,rerank`、
    圈号 `④⑤`。默认返回全部六种。无法识别的 token 直接报错而不是静默忽略 ——
    静默忽略会让人以为"我只跑了 ④"其实跑了全部，是最容易误读结论的一种失败模式。
    """
    if spec is None or not spec.strip() or spec.strip().lower() in {"all", "全部"}:
        return list(ABLATION_CONFIGS)

    normalized = spec
    for circled, digit in _CIRCLED_DIGITS.items():
        # 两侧补空格：`④⑤` 必须变成 `4 5` 而不是 `45`，否则会被当成一个非法 token
        normalized = normalized.replace(circled, f" {digit} ")

    tokens = [t for t in re.split(r"[,\s]+", normalized) if t]
    # 支持 `456` 这种连写（用户想表达 4、5、6）。序号只有 1~6，不存在两位数，
    # 所以按字符拆是安全的。
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(list(token) if token.isdigit() and len(token) > 1 else [token])

    chosen: list[AblationConfig] = []
    for token in expanded:
        match = next(
            (
                c
                for idx, c in enumerate(ABLATION_CONFIGS, start=1)
                if token.lower() in {str(idx), c.key, c.label.lower()}
            ),
            None,
        )
        if match is None:
            valid = " / ".join(f"{i}:{c.key}" for i, c in enumerate(ABLATION_CONFIGS, start=1))
            raise ValueError(f"无法识别的配置 {token!r}；可用写法：{valid}")
        if match not in chosen:  # 去重但保持用户给定顺序
            chosen.append(match)
    return chosen


# --------------------------------------------------------------------------- 结果
@dataclass
class AblationResult:
    """一种配置跑完所有查询后的汇总。"""

    key: str
    label: str
    metrics: RetrievalMetrics
    median_latency_ms: float
    per_query_latency_ms: list[float] = field(default_factory=list)
    # 每篇查询的 paper 级结果（arxiv_id 有序），保留它是为了能在测试里比较不同配置"确实不同"
    ranked: list[list[str]] = field(default_factory=list)


async def _evaluate_config(
    retriever: HybridRetriever,
    items: list[GoldenItem],
    config: AblationConfig,
    top_k: int,
) -> AblationResult:
    """让一条配置跑完整批查询：逐条计时，最后汇总指标。"""

    ranked: list[list[str]] = []
    latencies: list[float] = []

    for item in items:
        # 分类约束只在金标准标注了 categories 时才有；跨领域查询刻意留空，
        # 于是 ⑥ 在这些查询上会退化成 ⑤ —— 这是**预期行为**，不是 bug。
        filters = (
            MetadataFilter(categories=list(item.categories)) if item.categories else None
        )

        started = time.perf_counter()
        hits = await retriever.search(
            item.question,
            list(item.keywords),
            top_k=top_k,
            filters=filters,
            config=config.search,
        )
        latencies.append((time.perf_counter() - started) * 1000.0)
        ranked.append([h.arxiv_id for h in hits])

    metrics = evaluate_retrieval(
        ranked,
        [set(it.gold_paper_ids) for it in items],
        recall_k=5,
        precision_k=5,
        ndcg_k=10,
        query_ids=[it.id for it in items],
    )

    return AblationResult(
        key=config.key,
        label=config.label,
        metrics=metrics,
        # 中位数而非均值：首条查询常含 jieba 词典加载等一次性开销，均值会被它拉高
        median_latency_ms=statistics.median(latencies) if latencies else 0.0,
        per_query_latency_ms=latencies,
        ranked=ranked,
    )


async def run_ablation(
    retriever: HybridRetriever,
    items: list[GoldenItem],
    *,
    configs: list[AblationConfig] | None = None,
    top_k: int | None = None,
) -> list[AblationResult]:
    """把选定的配置逐个跑一遍。

    Args:
        retriever: 已加载索引的检索器（**同一份索引**贯穿所有配置，否则无法归因）。
        items: golden 查询集。
        configs: 要跑的配置，默认全部六种。
        top_k: 每路召回条数，默认取 settings.retrieve_top_k。
    """
    configs = configs if configs is not None else list(ABLATION_CONFIGS)
    top_k = top_k or retriever.settings.retrieve_top_k

    results: list[AblationResult] = []
    for config in configs:
        results.append(await _evaluate_config(retriever, items, config, top_k))
    return results


# --------------------------------------------------------------------------- 报告
def _provider_notice(
    settings: Settings, embedder: BaseEmbedder, reranker_name: str
) -> tuple[list[str], bool]:
    """根据**实际运行环境**生成报告顶部的可信度警告。

    返回 `(Markdown 行, 是否处于 mock/降级状态)`。之所以动态判断而不是写死一段文案：
    报告一旦被重定向进 README，静态文案会失真 —— 用 BGE 重跑时残留一句"本表用的是 mock"
    会让人怀疑真实数字，反之（mock 却写"已用 BGE"）则是更严重的误导。
    """
    lines: list[str] = []
    degraded = False

    if isinstance(embedder, MockEmbedder) or settings.embedding_provider == "mock":
        degraded = True
        lines.append(
            "> ⚠️ 本次运行使用 **mock embedding（哈希随机向量，无语义）**，向量路结果接近随机。"
        )
        lines.append(
            "> 本表**仅用于验证评测管线**（配置是否真的生效、指标是否真的算对），"
            "**不代表检索质量**。"
        )
        lines.append(
            '> 要得到真实数字：`pip install -e ".[local]"` → 设 `EMBEDDING_PROVIDER=local` → '
            "重建索引（`python scripts/build_index.py`）→ 重跑本实验。"
        )
    elif settings.embedding_provider == "api":
        lines.append(
            "> ℹ️ 本次运行使用远程 API embedding。其向量空间与本地 BGE 不同，"
            "数字**不可与 BGE 结果混用**（跨 embedding 对比无意义）。"
        )
    else:
        lines.append("> ✅ 本次运行使用本地 BGE embedding，向量路数字具备参考价值。")

    if reranker_name != "bge-reranker":
        degraded = True
        reason = (
            "词面重叠降级重排，非交叉编码器"
            if reranker_name == "lexical"
            else "重排未启用，NullReranker 仅保序"
        )
        lines.append(
            f"> ⚠️ 本次运行的重排器是 `{reranker_name}`（{reason}），"
            "⑤/⑥ 的「重排增益」**不是** BGE Reranker 的真实增益。"
        )

    return lines, degraded


def _delta_lines(results: list[AblationResult]) -> list[str]:
    """计算几组关键差值。

    为什么由脚本算而不是留给读者：表格只给终值，读者要自己减；而"④ 相对 ① 提升多少"
    正是这张表要回答的核心问题。把它显式列出来，能让结论无法被含糊解读。
    """
    by_key = {r.key: r for r in results}
    pairs = [
        ("④ − ①", "rrf", "vector", "混合检索(RRF) 相对纯向量基线"),
        ("② − ①", "bm25", "vector", "词面召回相对语义召回"),
        ("④ − ③", "rrf", "weighted", "RRF 相对加权求和"),
        ("⑤ − ④", "rerank", "rrf", "Reranker 的净增益"),
        ("⑥ − ⑤", "filter", "rerank", "metadata 过滤的净增益"),
    ]

    lines: list[str] = []
    for title, hi_key, lo_key, desc in pairs:
        hi, lo = by_key.get(hi_key), by_key.get(lo_key)
        if hi is None or lo is None:
            continue  # 用户没选这两个配置，就不显示对应的差值
        d_recall = hi.metrics.recall_at_5 - lo.metrics.recall_at_5
        d_ndcg = hi.metrics.ndcg_at_10 - lo.metrics.ndcg_at_10
        d_lat = hi.median_latency_ms - lo.median_latency_ms
        lines.append(
            f"- **{title}**（{desc}）：Recall@5 {d_recall:+.4f}，"
            f"nDCG@10 {d_ndcg:+.4f}，耗时 {d_lat:+.2f}ms"
        )
    return lines


def render_markdown(
    results: list[AblationResult],
    *,
    items: list[GoldenItem],
    settings: Settings,
    golden_path: Path | str | None,
    top_k: int,
    index_size: int,
    n_papers: int,
    embedder: BaseEmbedder,
    reranker_name: str,
    generated_at: datetime | None = None,
) -> str:
    """把实验结果渲染成 Markdown 报告。"""
    generated_at = generated_at or datetime.now()
    notice, degraded = _provider_notice(settings, embedder, reranker_name)

    lines: list[str] = ["# 检索消融实验（架构 §5.3）", ""]

    if degraded:
        lines.append("## ⚠️ 结果可信度警告")
        lines.append("")
    lines.extend(notice)
    lines.append("")

    # ---- 运行环境（让报告自带"可复现"所需的全部上下文）----
    lines.append("## 运行环境")
    lines.append("")
    lines.append(f"- 生成时间：{generated_at:%Y-%m-%d %H:%M:%S}")
    lines.append(f"- embedding provider：`{settings.embedding_provider}`（{embedder.__class__.__name__}，维度 {embedder.dim}）")
    lines.append(f"- reranker：`{settings.rerank_provider}`（{reranker_name}），RERANK_ENABLED={settings.rerank_enabled}")
    lines.append(f"- 检索 top_k：{top_k}")
    lines.append(f"- golden 集：`{golden_path or '默认'}`（{len(items)} 条查询）")
    lines.append(f"- 索引：{index_size} 个 chunk / {n_papers} 篇论文")
    lines.append("")

    # ---- 主表 ----
    lines.append("## 结果表")
    lines.append("")
    lines.append("> 口径：检索/融合为 chunk 级，**Recall@5 / nDCG@10 均为 paper 级**"
                 "（先 chunk 检索 → paper 聚合 → 再算指标）。耗时列为**每条查询检索耗时的中位数**。")
    lines.append("")
    lines.append("| 配置 | Recall@5 | Precision@5 | nDCG@10 | MRR | 平均耗时ms |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        row = r.metrics.as_row()
        lines.append(
            f"| {r.label} | {row['Recall@5']:.4f} | {row['Precision@5']:.4f} | "
            f"{row['nDCG@10']:.4f} | {row['MRR']:.4f} | {r.median_latency_ms:.2f} |"
        )
    lines.append("")

    # ---- 关键差值 ----
    deltas = _delta_lines(results)
    if deltas:
        lines.append("## 关键差值")
        lines.append("")
        lines.extend(deltas)
        lines.append("")

    # ---- 脚注：把⑥的前提与降级条件如实写明 ----
    # 先收集成列表再统一编号，避免"条件性脚注缺位导致序号跳号"（1,2,4 这种）看起来像漏写了内容
    no_cat = [it.id for it in items if not it.categories]
    skipped = [r for r in results if r.metrics.n_skipped]
    footnotes: list[str] = [
        "**⑥ 的优势带有前提**：其收益部分来自「分类约束已知且正确」。真实场景里分类由 "
        "Planner 推断，可能出错；一旦推错，过滤会把金标准直接挡在召回之外，收益转负。"
        "因此 ⑥ 不是可以无条件启用的配置。"
    ]
    if no_cat:
        footnotes.append(
            f"**⑥ 在无分类约束的查询上退化为 ⑤**：{len(no_cat)} 条跨领域查询"
            f"（{', '.join(no_cat)}）刻意未标注 categories，这些查询上 metadata 过滤不生效，"
            "二者结果相同 —— 属预期行为。"
        )
    if skipped:
        footnotes.append(
            f"**无金标准的查询被跳过而非计 0 分**："
            f"{skipped[0].metrics.n_skipped} 条查询因金标准为空被跳过（见 `evaluate_retrieval`）。"
        )
    footnotes.append(
        "**耗时差异在本数据规模下可能与噪声同阶**：golden 只有 20 条、索引仅数百 chunk，"
        "毫秒级差异不宜直接外推；Reranker 的真实延迟代价需在更大语料与并发下复测。"
    )

    lines.append("## 脚注")
    lines.append("")
    lines.extend(f"{i}. {text}" for i, text in enumerate(footnotes, start=1))
    lines.append("")
    return "\n".join(lines)


async def run_and_render(
    retriever: HybridRetriever,
    items: list[GoldenItem],
    *,
    settings: Settings | None = None,
    golden_path: Path | str | None = None,
    top_k: int | None = None,
    configs: list[AblationConfig] | None = None,
) -> tuple[str, list[AblationResult]]:
    """跑实验 + 渲染报告（CLI 与测试共用的入口）。"""
    settings = settings or retriever.settings
    top_k = top_k or settings.retrieve_top_k

    results = await run_ablation(retriever, items, configs=configs, top_k=top_k)
    markdown = render_markdown(
        results,
        items=items,
        settings=settings,
        golden_path=golden_path,
        top_k=top_k,
        index_size=retriever.size,
        n_papers=len(retriever._titles),
        embedder=retriever.embedder,
        reranker_name=retriever.reranker.name,
    )
    return markdown, results


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.ablation",
        description="PaperPilot 检索消融实验（架构 §5.3）：六种检索配置的效果 + 耗时对比",
    )
    parser.add_argument("--golden", default=None, help="golden 集路径（JSONL），默认 app/eval/datasets/golden_20.jsonl")
    parser.add_argument("--top-k", type=int, default=None, help="每路召回条数，默认取 settings.retrieve_top_k")
    parser.add_argument("--output", default=None, help=f"Markdown 输出路径，默认 {DEFAULT_OUTPUT}")
    parser.add_argument(
        "--configs",
        default=None,
        help="只跑指定配置：序号(4,5,6) / key(rrf,rerank) / 圈号(④⑤)，默认全部",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    golden_path = Path(args.golden) if args.golden else None
    items = load_golden(golden_path)
    if not items:
        print("⚠️  golden 集为空，实验将只产出空表（用于验证管线）。")
    try:
        configs = select_configs(args.configs)
    except ValueError as exc:
        print(f"✗ {exc}")
        return 2

    retriever = HybridRetriever(index_dir=settings.faiss_index_dir)
    if retriever.is_empty:
        print(
            "⚠️  索引为空，所有配置都会返回空结果。请先建索引：python scripts/build_index.py"
        )

    markdown, _results = asyncio.run(
        run_and_render(
            retriever,
            items,
            settings=settings,
            golden_path=golden_path,
            top_k=args.top_k,
            configs=configs,
        )
    )

    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    # 汇总打印到 stdout（含表格），方便 CI 日志里直接看到结论
    print(markdown)
    print(f"\n✓ 报告已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

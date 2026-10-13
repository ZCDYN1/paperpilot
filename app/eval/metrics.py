"""检索侧评测指标 —— **纯函数，不依赖任何模型**。

为什么这批指标才是评测主力，而不是 RAGAS？

RAGAS 的 `faithfulness` / `answer_relevancy` 本质是"用 LLM 当裁判"：
它们受**生成模型与 prompt** 的影响远大于受检索影响，而且必须有 API key。
但"混合检索到底比纯向量好在哪里"是个**检索侧**问题 ——
`Recall@K` / `nDCG@K` 只依赖两样东西：检索返回了什么、标注的相关项是哪些。
两者都是确定的，不需要模型、可复现、可对比，因此可以**离线跑出真实数字**。

架构 §12 把这条原则写得很明确：检索层优劣先用检索侧指标说话，
生成侧指标只能作为最终联合验证 —— 否则会出现"检索变好 1 分、换模型波动 10 分"
这种根本无法归因的结论。

本模块全部函数都是纯函数，输入输出只有 `list[str]` / `set[str]`，
刻意不认识 `PaperHit` 等业务对象，以便任何检索实现都能复用。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

# 检索结果里的元素标识（本项目中是 arxiv_id）。
DocId = str


def recall_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Recall@K —— 相关项里有多少被前 K 个结果捞到。

    分母是**相关项总数**（而不是 K），这是 Recall 的定义。
    在论文级评测里它回答的问题是："该找的论文，找到了没有。"

    相关项为空时返回 0.0（无法定义召回率）；聚合时应跳过这类样本，
    否则会人为拉低均值。
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    top = set(list(retrieved)[:k])
    return len(top & rel) / len(rel)


def precision_at_k(retrieved: Sequence[DocId], relevant: Iterable[DocId], k: int) -> float:
    """Precision@K —— 前 K 个结果里有多少是相关的。

    与 Recall 互补：召回高但精确低，说明结果里塞了大量噪声。

    注意分母**恒取 K**（而不是"实际返回条数"）：返回不足 K 条时缺失位置按不相关计。
    这是 IR 的标准口径；若改用实际条数，则"只返回 1 条且恰好命中"会拿到虚高的 1.0。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")
    top = set(list(retrieved)[:k])
    return len(top & set(relevant)) / k


def mrr(retrieved: Sequence[DocId], relevant: Iterable[DocId]) -> float:
    """MRR —— 第一个相关结果出现在第几名（取倒数）。

    只关心"第一个命中"，适合"用户通常只看头一条"的场景。
    一条都没命中时按 0 计。
    """
    rel = set(relevant)
    if not rel:
        return 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[DocId],
    relevant: Iterable[DocId],
    k: int,
    *,
    graded: Mapping[DocId, float] | None = None,
) -> float:
    """nDCG@K —— 考虑名次的排序质量。

    与 Recall@K 的关键区别：Recall 不管相关项排第 1 还是第 5，nDCG 会惩罚后者。
    而"把真正有用的排到前面"正是混合检索 + 重排的核心卖点，
    所以这个指标在消融表里不可省 —— 否则无法体现 Reranker 的价值。

    Args:
        graded: 可选的相关度分级 `{doc_id: gain}`。不传则按 0/1 二值处理。
    """
    if k <= 0:
        raise ValueError("k 必须为正整数")

    rel = set(relevant)

    def gain(doc: DocId) -> float:
        if graded is not None:
            return float(graded.get(doc, 0.0))
        return 1.0 if doc in rel else 0.0

    top = list(retrieved)[:k]
    # 位置 i（从 0 开始）的折损因子是 log2(i + 2)，即第 1 位折损最小
    dcg = sum(gain(doc) / math.log2(i + 2) for i, doc in enumerate(top))

    # 理想排序：把所有能拿到的增益从大到小排，取前 k 个
    pool = graded.keys() if graded is not None else rel
    ideal = sorted((gain(d) for d in pool), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    # idcg=0 表示"这篇查询在语料里根本没有正确项"（金标准为空，或分级全为 0）
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalMetrics:
    """一次消融配置在所有查询上的**汇总**指标。"""

    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    n_queries: int = 0                       # 参与统计的查询数（已剔除无金标准的）
    n_skipped: int = 0                       # 因金标准为空被跳过的查询数
    per_query: list[dict] = field(default_factory=list)   # 明细，便于定位"哪条查询拖后腿"

    def as_row(self) -> dict[str, float]:
        """转成可直接写进 Markdown 表格的一行（保留 4 位小数）。"""
        return {
            "Recall@5": round(self.recall_at_5, 4),
            "Precision@5": round(self.precision_at_5, 4),
            "nDCG@10": round(self.ndcg_at_10, 4),
            "MRR": round(self.mrr, 4),
        }


def evaluate_retrieval(
    ranked_results: Sequence[Sequence[DocId]],
    relevant_sets: Sequence[Iterable[DocId]],
    *,
    recall_k: int = 5,
    precision_k: int = 5,
    ndcg_k: int = 10,
    query_ids: Sequence[str] | None = None,
) -> RetrievalMetrics:
    """把多条查询的检索结果汇总成一套指标。

    **无金标准的查询会被跳过并单独计数**，而不是当成 0 分计入均值 ——
    标注集不完整时，把"没标注"错算成"没找到"会系统性低估检索效果，
    这正是很多评测表数字偏低的真实原因。

    Args:
        ranked_results: 每条查询的检索结果（按相关性从高到低）。
        relevant_sets: 与之一一对应的金标准相关项集合。
        query_ids: 可选，用于在明细里定位具体是哪条查询。
    """
    if len(ranked_results) != len(relevant_sets):
        raise ValueError(
            f"检索结果数({len(ranked_results)})与金标准数({len(relevant_sets)})不一致"
        )

    acc = RetrievalMetrics()
    sums: dict[str, float] = {"recall": 0.0, "precision": 0.0, "ndcg": 0.0, "mrr": 0.0}

    for i, (retrieved, relevant) in enumerate(zip(ranked_results, relevant_sets, strict=True)):
        rel = set(relevant)
        if not rel:
            acc.n_skipped += 1
            continue

        r = recall_at_k(retrieved, rel, recall_k)
        p = precision_at_k(retrieved, rel, precision_k)
        n = ndcg_at_k(retrieved, rel, ndcg_k)
        m = mrr(retrieved, rel)

        sums["recall"] += r
        sums["precision"] += p
        sums["ndcg"] += n
        sums["mrr"] += m
        acc.n_queries += 1

        acc.per_query.append(
            {
                "query_id": query_ids[i] if query_ids else str(i),
                f"recall@{recall_k}": round(r, 4),
                f"precision@{precision_k}": round(p, 4),
                f"ndcg@{ndcg_k}": round(n, 4),
                "mrr": round(m, 4),
                "n_relevant": len(rel),
                "n_retrieved": len(retrieved),
            }
        )

    if acc.n_queries:
        acc.recall_at_5 = sums["recall"] / acc.n_queries
        acc.precision_at_5 = sums["precision"] / acc.n_queries
        acc.ndcg_at_10 = sums["ndcg"] / acc.n_queries
        acc.mrr = sums["mrr"] / acc.n_queries

    return acc

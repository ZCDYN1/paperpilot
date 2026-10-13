"""RRF（Reciprocal Rank Fusion）排名融合。

**为什么不用加权求和？**

BM25 分数是无上界的正数（可能 0.3，也可能 28.7），余弦相似度落在 [-1, 1]。
直接写 `0.5 * a + 0.5 * b` 意味着 BM25 天然主导结果，除非先做归一化 ——
而归一化本身又依赖分布假设（min-max 会被离群值带偏，z-score 假设近似正态）。

RRF 只用**排名**：`Σ 1 / (k + rank)`。
量纲问题被彻底绕过，对分数分布不敏感 —— 这正是它成为混合检索默认方案的原因。

**k 的作用**：k 越大，头部名次的优势被压得越平。k=60 是论文里的经验值，
意思是"排名第 1 与第 2 的差距，远小于与第 60 名的差距"。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence

DEFAULT_RRF_K = 60


def rrf_fuse(
    rank_lists: Sequence[Sequence[Hashable]],
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """把多路召回结果按排名融合。

    Args:
        rank_lists: 每一路的排序结果（已按相关性从高到低排列）。
        k: RRF 平滑常数，越大越弱化头部名次差异。
        weights: 每路的权重；默认等权。可用于调"向量 vs BM25"的相对影响力。
        top_n: 只返回前 N 个。

    Returns:
        `[(doc_id, fused_score), ...]`，按分数降序。
    """
    if weights is not None and len(weights) != len(rank_lists):
        raise ValueError(
            f"weights 长度({len(weights)})必须与 rank_lists 数量({len(rank_lists)})一致"
        )

    scores: dict[Hashable, float] = defaultdict(float)

    for i, lst in enumerate(rank_lists):
        weight = 1.0 if weights is None else weights[i]
        seen_in_this_list: set[Hashable] = set()
        for rank, doc_id in enumerate(lst, start=1):
            # 同一路里若出现重复 id，只计一次 —— 否则重复会虚增分数
            if doc_id in seen_in_this_list:
                continue
            seen_in_this_list.add(doc_id)
            scores[doc_id] += weight / (k + rank)

    # 分数相同时用 id 字符串兜底排序，保证结果**可复现**（测试要依赖这一点）
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ranked[:top_n] if top_n is not None else ranked


def minmax_normalize(scores: dict[Hashable, float]) -> dict[Hashable, float]:
    """把一路召回的原始分线性拉到 [0, 1]。

    加权求和**必须先归一化**，这是它相比 RRF 最脆弱的地方：
    BM25 分数无上界（同一批数据里可能一篇 0.3、另一篇 28.7），
    余弦相似度却恒定落在 [-1, 1] —— 不归一化就等于让 BM25 单方面决定结果。
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        # 全部同分时不能做除法（会 ZeroDivisionError）。统一给 1.0 而非 0.0，
        # 是为了保留"这一路确实召回了它"这个信号，让另一路的名次仍能区分它们。
        return dict.fromkeys(scores, 1.0)
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_fuse(
    scored_lists: Sequence[Sequence[tuple[Hashable, float]]],
    weights: Sequence[float] | None = None,
    top_n: int | None = None,
) -> list[tuple[Hashable, float]]:
    """加权求和融合 —— **作为消融实验的对照组存在**。

    它与 `rrf_fuse` 的唯一区别：RRF 只看**名次**，这里看**归一化后的分数**。
    保留它不是为了用它，而是为了在消融表里给出"加权求和 vs RRF"的**实测差距** ——
    文档里那句"RRF 优于加权求和"必须由数据兑现，而不是靠断言。

    Args:
        scored_lists: 每一路的 `[(doc_id, raw_score), ...]`（已按分数降序）。
        weights: 每路权重；默认等权。
        top_n: 只返回前 N 个。

    Returns:
        `[(doc_id, fused_score), ...]`，按分数降序。
    """
    if weights is not None and len(weights) != len(scored_lists):
        raise ValueError(
            f"weights 长度({len(weights)})必须与 scored_lists 数量({len(scored_lists)})一致"
        )

    fused: dict[Hashable, float] = defaultdict(float)
    for i, lst in enumerate(scored_lists):
        weight = 1.0 if weights is None else weights[i]
        # dict() 顺带完成"同一路内重复 id 只保留一个"的去重
        for doc_id, score in minmax_normalize(dict(lst)).items():
            fused[doc_id] += weight * score

    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], str(kv[0])))
    return ranked[:top_n] if top_n is not None else ranked

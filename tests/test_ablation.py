"""检索消融实验的测试（mock 模式，无需模型权重 / API key）。

**测试目标不是"跑出好看的分数"，而是证明评测管线本身可信**：

- 六种配置真的构造出了预期开关；
- 开关真的改变结果（① ≠ ②），而不是一个空壳参数；
- `fusion="none"` 的单路保序性（基线的定义不能被 RRF 悄悄改写）；
- ⑥ 的分类过滤真的生效（而非只是传了个无人使用的参数）；
- 两路都关时按约定报错而不是静默返回空；
- 耗时字段真的被采集到；
- **mock provider 的警告真的出现在生成的 Markdown 里** ——
  这是本次交付最关键的一条：报告若不警告，读者会拿随机数当检索质量。

用 conftest 的受控语料（3 篇论文、跨 cs.CL/cs.IR/cs.CV），
因为它的词面重叠是刻意拉开的，断言才能稳定。
"""

from __future__ import annotations

import pytest

from app.eval.ablation import (
    ABLATION_CONFIGS,
    render_markdown,
    run_ablation,
    run_and_render,
    select_configs,
)
from app.eval.dataset import GoldenItem
from app.retrieval.hybrid_retriever import HybridRetriever, MetadataFilter, SearchConfig
from tests.conftest import (
    BROAD_KEYWORDS,
    FOCUSED_KEYWORDS,
    FOCUSED_QUERY,
    PAPER_A,
    PAPER_B,
    PAPER_C,
)


def _item(
    qid: str = "t1",
    gold: tuple[str, ...] = (PAPER_A,),
    categories: tuple[str, ...] = ("cs.CL",),
) -> GoldenItem:
    """构造一条查询样本。默认带分类约束，便于测 ⑥ 的过滤分支。"""
    return GoldenItem(
        id=qid,
        question=FOCUSED_QUERY,
        keywords=tuple(FOCUSED_KEYWORDS),
        gold_paper_ids=tuple(gold),
        categories=tuple(categories),
    )


def _cfg(key: str) -> SearchConfig:
    return next(c for c in ABLATION_CONFIGS if c.key == key).search


# --------------------------------------------------------------------- 配置面板
def test_six_configs_constructed_with_expected_switches():
    """六种配置一个不少，且每一行的开关组合与架构 §5.3 逐位一致。"""
    assert len(ABLATION_CONFIGS) == 6
    assert [c.key for c in ABLATION_CONFIGS] == [
        "vector", "bm25", "weighted", "rrf", "rerank", "filter",
    ]

    # ① 纯向量：只开向量，且必须先关融合、关重排、关过滤
    v = _cfg("vector")
    assert (v.use_vector, v.use_bm25, v.fusion, v.use_rerank, v.use_metadata_filter) == (
        True, False, "none", False, False,
    )
    # ② 纯 BM25：与 ① 镜像
    b = _cfg("bm25")
    assert (b.use_vector, b.use_bm25, b.fusion, b.use_rerank, b.use_metadata_filter) == (
        False, True, "none", False, False,
    )
    # ③ 加权求和：双路开、融合 weighted、不重排
    w = _cfg("weighted")
    assert (w.use_vector, w.use_bm25, w.fusion, w.use_rerank) == (True, True, "weighted", False)
    # ④ RRF：双路开、融合 rrf、仍不重排（是 ⑤ 的增益基准）
    r = _cfg("rrf")
    assert (r.use_vector, r.use_bm25, r.fusion, r.use_rerank, r.use_metadata_filter) == (
        True, True, "rrf", False, False,
    )
    # ⑤ 加 Reranker
    rr = _cfg("rerank")
    assert (rr.fusion, rr.use_rerank, rr.use_metadata_filter) == ("rrf", True, False)
    # ⑥ 再加 metadata 过滤（唯一开过滤的一行）
    f = _cfg("filter")
    assert (f.fusion, f.use_rerank, f.use_metadata_filter) == ("rrf", True, True)
    assert sum(c.search.use_metadata_filter for c in ABLATION_CONFIGS) == 1  # 只有 ⑥ 开过滤


async def test_pure_vector_and_pure_bm25_results_differ(retriever):
    """① 与 ② 的检索结果必须不同 —— 否则说明开关只是摆设（空壳参数）。"""
    items = [_item()]
    results = await run_ablation(retriever, items, configs=[ABLATION_CONFIGS[0], ABLATION_CONFIGS[1]])
    vec_ranked, bm25_ranked = results[0].ranked[0], results[1].ranked[0]

    assert vec_ranked != bm25_ranked, "纯向量与纯 BM25 返回了完全相同的排序，开关可能没生效"
    # 受控语料里 BM25 对 FOCUSED_KEYWORDS 完全命中论文 A，这是可断言的稳定事实
    assert bm25_ranked[0] == PAPER_A


def test_fusion_none_preserves_single_path_order():
    """`fusion="none"` 必须保序 —— 单路基线不能被 RRF 的 1/(k+rank) 重新赋分改写。"""
    single = SearchConfig(use_vector=True, use_bm25=False, fusion="none")

    out = HybridRetriever._fuse([(1, 0.9), (2, 0.8), (3, 0.7)], [], single, 60, top_n=10)
    assert [cid for cid, _ in out] == [1, 2, 3]
    # 分数被替换成"越早出现越大"的占位值，但顺序必须与原列表一致
    assert out[0][1] > out[-1][1]

    both = SearchConfig(use_vector=True, use_bm25=True, fusion="none")
    merged = HybridRetriever._fuse([(1, 0.9), (2, 0.8)], [(3, 0.5), (1, 0.4)], both, 60, top_n=10)
    # 先向量后 BM25；BM25 里重复出现的 id=1 不重复占位
    assert [cid for cid, _ in merged] == [1, 2, 3]


async def test_config_six_actually_applies_metadata_filter(retriever):
    """⑥ 必须在有分类约束的查询上真的过滤 —— 而不是只传了个没人用的参数。

    先给对照组：⑤（`use_metadata_filter=False`）即使收到同样的 filters 也必须忽略它，
    这样"⑥ 只剩一篇"才能归因于过滤生效，而不是"这篇本来就召回不到"。
    """
    filt = MetadataFilter(categories=["cs.IR"])

    hits5 = await retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, config=_cfg("rerank"), filters=filt)
    hits6 = await retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, config=_cfg("filter"), filters=filt)

    assert {h.arxiv_id for h in hits5} == {PAPER_A, PAPER_B, PAPER_C}, "⑤ 应忽略 filters"
    assert {h.arxiv_id for h in hits6} == {PAPER_B}, "⑥ 应只保留 cs.IR 的论文"


async def test_config_six_degrades_to_five_without_categories(retriever):
    """无分类约束（filters=None）时，⑥ 必须与 ⑤ 结果一致 —— 这是预期降级。"""
    hits5 = await retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, config=_cfg("rerank"))
    hits6 = await retriever.search(FOCUSED_QUERY, BROAD_KEYWORDS, top_k=10, config=_cfg("filter"))

    assert [h.arxiv_id for h in hits5] == [h.arxiv_id for h in hits6]


async def test_both_branches_disabled_raises(retriever):
    """两路召回都关时必须显式报错，而不是静默返回空（空结果会被误读为"效果为 0"）。"""
    cfg = SearchConfig(use_vector=False, use_bm25=False)
    with pytest.raises(ValueError):
        await retriever.search(FOCUSED_QUERY, FOCUSED_KEYWORDS, config=cfg)


async def test_latency_is_collected_and_positive(retriever):
    """耗时必须逐条采集且为正 —— 架构 §5.3 明确要求效果与代价一起看。"""
    items = [_item("t1"), _item("t2", gold=(PAPER_B,), categories=("cs.IR",))]
    results = await run_ablation(retriever, items, configs=[ABLATION_CONFIGS[0]])

    res = results[0]
    assert len(res.per_query_latency_ms) == len(items)
    assert all(ms > 0 for ms in res.per_query_latency_ms)
    assert res.median_latency_ms > 0


# --------------------------------------------------------------------- 报告
async def test_mock_warning_appears_in_markdown(retriever):
    """mock provider 下，报告顶部必须出现"不代表检索质量"的警告。

    这是本次交付最关键的一条断言：警告若丢失，数字会被当成真实检索效果呈现。
    """
    md, _ = await run_and_render(retriever, [_item()])

    assert "mock embedding" in md
    assert "不代表检索质量" in md
    assert "EMBEDDING_PROVIDER=local" in md  # 给出可执行的重跑指引
    assert "结果可信度警告" in md


async def test_markdown_contains_table_deltas_and_caveats(retriever):
    """报告要含：主表、关键差值、以及 ⑥ 的前提说明。"""
    items = [
        _item("t1", gold=(PAPER_A,), categories=("cs.CL",)),
        _item("t2", gold=(PAPER_B,), categories=()),  # 刻意无分类，触发"退化为 ⑤"脚注
    ]
    md, _ = await run_and_render(retriever, items)

    assert "| 配置 | Recall@5 | Precision@5 | nDCG@10 | MRR | 平均耗时ms |" in md
    assert "## 关键差值" in md
    assert "④ − ①" in md
    assert "分类约束已知且正确" in md  # ⑥ 的前提必须写明
    assert "t2" in md  # 无分类查询被点名


async def test_empty_golden_and_empty_index_are_safe(tmp_path, retriever):
    """空 golden / 空索引两类边界都要安全返回，不抛异常。"""
    # 1) 空 golden：跑了 6 种配置，但每个配置都是 0 条查询参与统计
    empty_results = await run_ablation(retriever, [])
    assert len(empty_results) == 6
    assert all(r.metrics.n_queries == 0 for r in empty_results)
    assert all(r.median_latency_ms == 0.0 for r in empty_results)

    # 空 golden 下报告仍要能渲染出来
    md = render_markdown(
        empty_results,
        items=[],
        settings=retriever.settings,
        golden_path=None,
        top_k=10,
        index_size=retriever.size,
        n_papers=3,
        embedder=retriever.embedder,
        reranker_name=retriever.reranker.name,
    )
    assert "0 条查询" in md

    # 2) 空索引：检索返回空列表，但指标照算（相关项金标准非空 → 计 0 分而非跳过）
    empty_retriever = HybridRetriever(index_dir=tmp_path / "empty_faiss")
    assert empty_retriever.is_empty
    res = await run_ablation(empty_retriever, [_item()])
    assert all(r.ranked == [[]] for r in res)
    assert res[0].metrics.n_queries == 1
    assert res[0].metrics.recall_at_5 == 0.0


# --------------------------------------------------------------------- CLI 选择
def test_select_configs_parsing():
    """`--configs` 支持序号 / key / 圈号，非法输入要报错而不是静默忽略。"""
    assert len(select_configs(None)) == 6
    assert len(select_configs("all")) == 6

    assert [c.key for c in select_configs("4,5")] == ["rrf", "rerank"]
    assert [c.key for c in select_configs("④⑤")] == ["rrf", "rerank"]
    assert [c.key for c in select_configs("vector")] == ["vector"]
    # 去重但保持给定顺序
    assert [c.key for c in select_configs("2,1,2")] == ["bm25", "vector"]

    with pytest.raises(ValueError):
        select_configs("不存在的配置")

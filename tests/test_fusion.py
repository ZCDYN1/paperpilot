"""RRF 融合的单元测试。

RRF 是纯函数，最值得先把边界条件钉死：**重复 id、并列名次、空输入、单路输入**。
这些一旦出错，错误会以"分数被虚增一点点"的形式渗进最终排序，肉眼几乎看不出来。
"""

from __future__ import annotations

import pytest

from app.retrieval.fusion import rrf_fuse

K = 60


def test_rrf_basic_order_follows_rank():
    """两路结果高度一致时，两路都靠前的文档必须排最前。"""
    fused = rrf_fuse([["a", "b", "c"], ["a", "b", "c"]], k=K)
    assert [doc for doc, _ in fused] == ["a", "b", "c"]
    # 每个 id 在两路各计一次：1/(k+1) + 1/(k+1)
    assert fused[0][1] == pytest.approx(2 / (K + 1))


def test_rrf_uses_both_lists():
    """只在一路出现的文档，分数必须低于两路都出现的文档。"""
    fused = rrf_fuse([["a", "b"], ["b", "c"]], k=K)
    scores = dict(fused)
    # b 同时出现在两路 → 分数最高
    assert fused[0][0] == "b"
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]


def test_rrf_tie_is_broken_deterministically():
    """完全并列（分数相同）时按 id 兜底排序，保证结果可复现。

    没有这个兜底，并列项的先后会取决于 dict 的遍历顺序，
    同一份数据多次运行可能给出不同排序 —— 消融实验的数字就不可信了。
    """
    fused1 = rrf_fuse([["a", "b"], ["b", "a"]], k=K)
    fused2 = rrf_fuse([["a", "b"], ["b", "a"]], k=K)
    assert [d for d, _ in fused1] == ["a", "b"]
    assert [d for d, _ in fused1] == [d for d, _ in fused2]
    assert fused1[0][1] == pytest.approx(fused1[1][1])  # 确实并列


def test_rrf_duplicate_id_in_one_list_counted_once():
    """同一路里重复出现的 id 只能计一次，否则分数被虚增。"""
    fused = rrf_fuse([["a", "a", "a"]], k=K)
    assert len(fused) == 1
    assert fused[0][0] == "a"
    # 若重复计数会得到 3/(k+1)，这里必须是 1/(k+1)
    assert fused[0][1] == pytest.approx(1 / (K + 1))


def test_rrf_empty_lists():
    assert rrf_fuse([[], []], k=K) == []
    assert rrf_fuse([], k=K) == []


def test_rrf_only_one_side_has_results():
    """一路为空（例如没有 BM25 命中）时，另一路的结果必须原样保留。"""
    fused = rrf_fuse([["x", "y", "z"], []], k=K)
    assert [d for d, _ in fused] == ["x", "y", "z"]


def test_rrf_top_n_and_weights():
    fused = rrf_fuse([["a", "b", "c"]], k=K, top_n=2)
    assert [d for d, _ in fused] == ["a", "b"]

    # 权重能改变两路的相对影响力
    weighted = rrf_fuse([["a"], ["b"]], k=K, weights=[3.0, 1.0])
    assert weighted[0][0] == "a"

    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], k=K, weights=[1.0])

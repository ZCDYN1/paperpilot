"""`scripts/recheck_citations.py` 的回写计划判定（纯函数层）。

这个脚本存在的理由：引用定位口径改过（精确比较 → 加一层抽取残留容忍），
库里那批 `verified=False` 的旧结论因此**是误杀**，但把它们翻成 True 是一次
对外发布口径的改动，必须"报告 + 可回写"两段式，而不是脚本一跑就 UPDATE。

所以这里钉的是**它敢翻哪些、不敢翻哪些**：
- 只翻"当时判 False、按新口径逐字确实找得到"的行；
- 判 None（从未校验过）不动 —— 翻它等于伪造一个软校验结论；
- note 里有 LLM 字样的不动 —— 那是软校验否决的，硬校验本来就读不出结果；
- 已 True 却找不到原文的只报异常、不回写 —— 判 False 的权力不在这一层。

回写动作本身（`--write` 的 UPDATE）靠这几个判定成立，因此不单独测 SQL。
"""

from __future__ import annotations

from typing import Any

from scripts.recheck_citations import plan_changes

# 抽取残留的两类真实形态（与 tests/test_quote_matching.py 同一事实来源）
QUOTE_MARKERS = "These versions are more factual, which may help hallucination"
CHUNK_MARKERS = "These versions [58] are more factual, which may help hallucination"


def _row(
    cid: int,
    *,
    quote: str = "some exact sentence about retrieval",
    verified: bool | None = False,
    note: str | None = "未找到原文支撑",
    chunk_text: str | None = "some exact sentence about retrieval and reranking",
    same_paper: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": cid,
        "run_id": "run-a",
        "arxiv_id": "2401.00001",
        "chunk_id": 7,
        "quote": quote,
        "verified": verified,
        "note": note,
        "chunk_text": chunk_text,
        "same_paper": same_paper if same_paper is not None else [],
    }


def test_exact_hit_flips_to_verified_without_a_note():
    """旧口径误杀、新口径逐字精确命中：翻 True，且不需要审计痕迹。"""
    out = plan_changes([_row(1)])
    assert out["flips"] == [(1, True, None)]
    assert out["flips_by_kind"]["exact"] == 1
    assert out["stats"]["verified:False"] == 1


def test_residual_hit_flips_but_carries_the_tolerance_reason():
    """靠容忍层才命中（PDF 内联引用标号）：可以翻，但 note 必须记下"放宽在哪一类"。

    没有这条痕迹，事后就无法回答"救回来的 3/20 里有多少是归一化才过的"，
    口径放宽就变成了不可审计的一次性动作。
    """
    out = plan_changes([_row(2, quote=QUOTE_MARKERS, chunk_text=CHUNK_MARKERS)])
    assert len(out["flips"]) == 1
    cid, verified, note = out["flips"][0]
    assert (cid, verified) == (2, True)
    assert note and "残留" in note, "容忍层通过却不写原因，等于伪造精确命中"
    assert out["flips_by_kind"]["citation_markers"] == 1


def test_reanchored_within_the_same_paper_still_flips():
    """chunk_id 指错但同篇别处逐字存在：与 verify 的第二条定位路径同口径，救回来。"""
    out = plan_changes([_row(3, chunk_text="完全无关的段落", same_paper=["some exact sentence about retrieval"])])
    assert [f[0] for f in out["flips"]] == [3]


def test_genuinely_absent_quote_is_not_flipped():
    """两层都找不到就是不通过：翻它就是在替模型编造证据。"""
    out = plan_changes([_row(4, quote="a sentence that appears nowhere", chunk_text="totally different text")])
    assert out["flips"] == []
    assert out["stats"]["match:not_found"] == 1


def test_never_verified_rows_are_left_alone():
    """`verified is None` = 这条从没被校验过。判它 True/False 都是凭空造结论。"""
    out = plan_changes([_row(5, verified=None)])
    assert out["flips"] == []
    assert out["stats"]["verified:None"] == 1


def test_soft_rejected_rows_are_not_rescued_by_the_hard_layer():
    """note 带 LLM 字样 = 硬校验当时过了、被软校验否决。翻转与否都不该由这层决定。"""
    out = plan_changes([_row(6, note="LLM 判定：引文虽在原文但不支撑该论断")])
    assert out["flips"] == []
    assert out["stats"]["skipped_soft_reject"] == 1


def test_verified_true_that_no_longer_matches_is_reported_not_rewritten():
    """已 True 却找不到原文：只进异常清单交人工看，脚本没有把 True 改 False 的权力。"""
    out = plan_changes(
        [
            _row(7, verified=True, note=None),
            _row(8, verified=True, quote="nowhere to be found", chunk_text="unrelated", note=None),
        ]
    )
    assert out["flips"] == []
    assert out["anomalies"] == [8]
    assert 7 not in out["anomalies"]


def test_missing_chunk_text_falls_back_to_the_same_paper():
    """chunk_id 悬空（片段已重建）时 `chunk_text` 为 None：不能因此判死，走同篇重锚。"""
    out = plan_changes([_row(9, chunk_text=None, same_paper=["some exact sentence about retrieval"])])
    assert [f[0] for f in out["flips"]] == [9]


def test_no_note_rows_are_counted_separately():
    """note 为空的 False 行：历史上是"整条被清空"而不是"某条不支撑"，值得单独报数。"""
    out = plan_changes([_row(10, note=""), _row(11, note=None)])
    assert out["no_note_candidates"] == 2
    assert out["total"] == 2

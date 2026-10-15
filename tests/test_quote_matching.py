"""「引文逐字在不在原文里」两段式判定的测试（`_common.quote_match` + verify 硬校验）。

真实背景（不是假想敌）：对 20 条抽样引用做独立裁判复核，精确比较（折叠空白 + 小写）
产生 3/20 **假阴性**，成因只有两种 PDF/HTML 抽取残留：
1. 原文带内联引用标号 `... versions [58]. These ...`，模型摘录时去掉了 `[58]`；
2. 原文标点前多一个空格 `... matches) . The ...`，模型写成 `matches). The`。

这些引用被误杀后进了导出报告的「未找到原文支撑的结论」，读者以为"没有证据"，
事实是"证据在、比对方式过严" —— 自动口径因此系统性偏低。

本文件钉住三件事：
1. 两类残留（外加弯/直引号）现在能通过对容忍层的比较 —— **且通过的痕迹必须
   区分"精确"与"归一化"**（两边都断言，否则判据放宽后无法审计救回了多少条）；
2. 精确命中仍走精确路径，不许被归一化路径污染（归因要干净）；
3. **闸门没有变松**：两层都救不回来的引文照旧判失败、照旧进
   「未找到原文支撑的结论」—— 这是放宽方向上最重要的反面用例。

隔离方式沿用 `test_verify_reanchor_and_note.py`（provider 切 mock + 打桩
`structured_invoke`），零 key、零联网。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.nodes import verifier as verifier_node
from app.agents.nodes._common import (
    MATCH_EXACT,
    normalize_text,
    quote_match,
    residual_kind,
    residual_note,
)
from app.agents.nodes.evidence import (
    REASON_QUOTE_NOT_FOUND,
    build_chunk_lookup,
    build_paper_chunks,
    resolve_citation,
)
from app.agents.schemas import VerificationItem, VerifyOutput
from app.core.config import get_settings

_A = "2401.00001"

# ---- 实测到的真实故障形态（两种抽取残留 + 弯直引号）----
CHUNK_MARKERS = (
    "these results are compared to earlier versions [58]. "
    "These state-of-the-art models dominate the leaderboard."
)
QUOTE_MARKERS = "compared to earlier versions. These state-of-the-art models"

CHUNK_PUNCT = "sparse search (capturing exact matches) . The results from both stages are fused."
QUOTE_PUNCT = "sparse search (capturing exact matches). The results from both stages"

CHUNK_QUOTES = "the authors call this \u2018robust\u2019 behaviour \u201cbaseline\u201d setup."
QUOTE_QUOTES = "the authors call this 'robust' behaviour"

# 原文里根本不存在的话 —— 两条归一化都救不回来，必须照旧判失败
QUOTE_FABRICATED = "we prove that retrieval can be skipped entirely"

RETRIEVED: list[dict[str, Any]] = [
    {
        "arxiv_id": _A,
        "title": "Hybrid Retrieval at Work",
        "chunks": [
            {"chunk_id": 101, "section": "methods", "source": "html", "text": CHUNK_MARKERS},
            {"chunk_id": 102, "section": "methods", "source": "html", "text": CHUNK_PUNCT},
            {"chunk_id": 103, "section": "intro", "source": "html", "text": CHUNK_QUOTES},
            {"chunk_id": 104, "section": "abstract", "source": "abstract", "text": "Outline-first pipelines produce section-level citations."},
        ],
    },
]


# ---------------------------------------------------------------------------
# 1) 纯函数层：匹配路径必须可区分
# ---------------------------------------------------------------------------
def test_residual_citation_markers_passes_with_its_kind_recorded():
    """`[58]` 被模型去掉 → 精确比较失败，容忍层命中且**归因到剥离标号**。"""
    # 前置：旧口径（normalize_text 包含判定）确实判不出 —— 这就是那 3/20 假阴性
    assert normalize_text(QUOTE_MARKERS) not in normalize_text(CHUNK_MARKERS)

    match = quote_match(QUOTE_MARKERS, CHUNK_MARKERS)
    assert match != MATCH_EXACT, "精确比较先跑且必须已经失败过，走到这里是第二段"
    assert match == "residual:citation_markers"
    assert residual_kind(match) == "citation_markers"


def test_residual_punct_spacing_passes_with_its_kind_recorded():
    """标点前多一个空格（`) .` vs `).`）→ 容忍层命中且**归因到折叠标点前空白**。"""
    assert normalize_text(QUOTE_PUNCT) not in normalize_text(CHUNK_PUNCT)

    match = quote_match(QUOTE_PUNCT, CHUNK_PUNCT)
    assert match == "residual:punct_spacing"
    assert residual_kind(match) == "punct_spacing"


def test_curly_vs_straight_quotes_pass_via_tolerance():
    """PDF 弯引号 vs 模型直引号 → 第三类容忍，同样要带得出归因。"""
    match = quote_match(QUOTE_QUOTES, CHUNK_QUOTES)
    assert match == "residual:quotes"


def test_exact_substring_is_not_polluted_by_the_normalized_path():
    """精确命中必须记成 exact：引文逐字取自 chunk（含 `[58]` 本身）。

    这条守的是"两段式"的第一段：归一化只兜底，不抢功。否则审计时
    "多少条靠归一化救回来"会被虚报，放宽的幅度就成了糊涂账。
    """
    assert quote_match("compared to earlier versions [58]", CHUNK_MARKERS) == MATCH_EXACT
    # 空白 / 大小写差异本来就由 normalize_text 吸收，仍算精确路径
    assert quote_match(
        "These state-of-the-art models   DOMINATE\n the leaderboard", CHUNK_MARKERS
    ) == MATCH_EXACT
    # 折叠空白/小写之后**仍**差标点前空白的（叠加空白噪声也不抢精确的功），
    # 才归因到 punct_spacing —— 两段顺序与归因干净度都不许反
    assert (
        quote_match("sparse search   (capturing\nexact matches). the RESULTS from both stages", CHUNK_PUNCT)
        == "residual:punct_spacing"
    )


def test_fabricated_quote_still_not_found():
    """★ 找不到就是找不到：两条归一化都救不回来的引文，判定口径不许变。"""
    assert quote_match(QUOTE_FABRICATED, CHUNK_MARKERS) == ""
    assert quote_match(QUOTE_FABRICATED, CHUNK_PUNCT) == ""
    assert quote_match(QUOTE_FABRICATED, CHUNK_QUOTES) == ""
    assert residual_kind("") == ""


def test_empty_quote_is_not_found():
    assert quote_match("   ", CHUNK_MARKERS) == ""


def test_quote_that_normalizes_to_empty_never_matches():
    """★ 只写了内联标号的引文（`[58]` / `[1][2]`）归一化后是空串，绝不能匹配任意原文。

    没有这道闸门时 `"" in 任意串` 恒真，`[58]` 会被判成 `residual:citation_markers`
    —— 方向恰好是最危险的那种假阳性：模型填占位值（实测它会吐 `-1` 这类噪声）时
    编造的引文反而"通过"，重锚还会把它挂到该篇第一个片段上。
    """
    unrelated = "totally unrelated content about graph neural networks"
    for quote in ("[58]", "[1][2]", "[12, 13]", "[3-5]"):
        assert quote_match(quote, unrelated) == "", f"{quote!r} 不许匹配无关原文"
    # 原文侧归一化成空也一样：引文非空就找不到，闸门不放
    assert quote_match("a real sentence from the paper", "[7]") == ""


def test_marker_number_mismatch_is_a_documented_cost():
    """钉住那个"真实的取舍"：两侧同剥标号后，引文写 [57] 而原文是 [58] 会放行。

    放宽的只有"方括号里数字标号的正确性"；句子其余文字仍必须逐字相同
    （下面第二条断言就是这个边界 —— 差一个词照样拦得住）。
    这个代价被接受的理由见 `_common.py` 的注释：标号不是论断的实质内容。
    """
    assert (
        quote_match("compared to earlier versions [57]. These state-of-the-art", CHUNK_MARKERS)
        == "residual:citation_markers"
    )
    # 没放宽的部分：剥完标号之后文字必须逐字对上，改一个词就过不了
    assert quote_match("compared to later versions. These state-of-the-art", CHUNK_MARKERS) == ""


# ---------------------------------------------------------------------------
# 2) _hard_check：两条定位路径共用同一个判定口径（不许一处特判）
# ---------------------------------------------------------------------------
def _citation(chunk_id: int, quote: str, **over: Any) -> dict[str, Any]:
    cit: dict[str, Any] = {
        "claim": "结论。",
        "paper_id": _A,
        "chunk_id": chunk_id,
        "quote": quote,
        "locator": "methods",
        "source": "html",
        "verified": False,
    }
    cit.update(over)
    return cit


def test_hard_check_exact_lookup_tolerates_residue():
    """`(paper, chunk_id)` 精确查到片段 → 引文带残留，走容忍层并记录归因。"""
    known = {( _A, 101): RETRIEVED[0]["chunks"][0]}
    papers = build_paper_chunks(RETRIEVED)
    ok, reason, anchor, how, match = verifier_node._hard_check(  # noqa: SLF001
        _citation(101, QUOTE_MARKERS), known, papers
    )
    assert ok is True and reason == ""
    assert how == "exact"
    assert match == "residual:citation_markers"
    assert anchor["resolved_chunk_id"] == 101


def test_hard_check_reanchor_path_shares_the_same_comparator():
    """编号打错 → 同篇重锚。重锚与精确查共用 quote_match：改动必须同时生效于两条路径。"""
    known: dict = {}
    papers = build_paper_chunks(RETRIEVED)
    ok, reason, anchor, how, match = verifier_node._hard_check(  # noqa: SLF001
        _citation(999, QUOTE_PUNCT), known, papers, build_chunk_lookup(RETRIEVED)
    )
    assert ok is True and reason == ""
    assert how == "quote", "定位路径照旧是重锚（定位与比对是两个正交的事实）"
    assert match == "residual:punct_spacing"
    assert anchor["resolved_chunk_id"] == 102


def test_hard_check_fabricated_quote_fails_on_both_paths():
    """编造的引文：精确查与重锚两条路都必须失败，原因口径不许变。"""
    papers = build_paper_chunks(RETRIEVED)
    for chunk_id, known in ((101, {(_A, 101): RETRIEVED[0]["chunks"][0]}), (999, {})):
        ok, reason, anchor, _how, match = verifier_node._hard_check(  # noqa: SLF001
            _citation(chunk_id, QUOTE_FABRICATED), known, papers
        )
        assert ok is False
        assert reason == "引文未能在原文中逐字匹配"
        assert anchor == {}
        assert match == ""


def test_hard_check_rejects_marker_only_quote_on_both_paths():
    """占位引文（整条就是一个内联标号）在硬校验两条路径上都必须判失败。

    这是"归一化后为空不许判包含"的端到端体现：修之前 `[99]` 会以
    `residual:citation_markers` 通过，并将引用重锚到该篇第一个片段。
    （用 `[99]` 而不是 `[58]`：后者在 CHUNK_MARKERS 里逐字存在，会走精确路径。）
    """
    papers = build_paper_chunks(RETRIEVED)
    for chunk_id, known in ((101, {(_A, 101): RETRIEVED[0]["chunks"][0]}), (999, {})):
        ok, reason, anchor, _how, match = verifier_node._hard_check(  # noqa: SLF001
            _citation(chunk_id, "[99]"), known, papers, build_chunk_lookup(RETRIEVED)
        )
        assert ok is False
        assert reason == "引文未能在原文中逐字匹配"
        assert anchor == {}
        assert match == ""


def test_resolve_citation_gate_still_rejects_fabrication():
    """evidence 层的旁证判定同步放宽后，闸门没松：假引文连重锚都锚不上。"""
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers, paper_id=_A, chunk_id=999, quote=QUOTE_FABRICATED,
        chunk_lookup=build_chunk_lookup(RETRIEVED),
    )
    assert hit.ok is False
    assert hit.reason == REASON_QUOTE_NOT_FOUND


# ---------------------------------------------------------------------------
# 3) verify_node：通过路径的痕迹经 verify_note → citations.note 可审计
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _stub_soft(monkeypatch, items: list[VerificationItem]):
    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        return VerifyOutput(items=items), {}

    monkeypatch.setattr(verifier_node, "structured_invoke", _invoke)


async def test_verify_note_distinguishes_exact_from_normalized_pass(monkeypatch):
    """三条引用：精确通过 / 残留容忍通过 / 真找不到。

    note 口径：精确通过 = 空串（落库 NULL，维持"没写"与"写了但为空"的区分）；
    容忍通过 = 非空且带归因类别；失败 = 失败原因照旧。
    「未找到原文支撑的结论」里只许出现第三条 —— 放宽比对不许把假引用放进来。
    """
    claims = ["标号残留条。", "标点残留条。", "编造条。"]
    _stub_soft(
        monkeypatch,
        [VerificationItem(claim=c, supported=True) for c in claims],
    )
    out = await verifier_node.verify_node(
        {
            "question": "混合检索怎么做融合？",
            "retrieved": RETRIEVED,
            "citations": [
                _citation(101, "compared to earlier versions [58]", claim=claims[0]),
                _citation(102, QUOTE_PUNCT, claim=claims[1]),
                _citation(104, QUOTE_FABRICATED, claim=claims[2]),
            ],
        }
    )
    by_claim = {c["claim"]: c for c in out["citations"]}

    # 精确通过：verified=True 且不带任何归一化痕迹（否则救回计数会被虚报）
    exact = by_claim[claims[0]]
    assert exact["verified"] is True
    assert exact["verify_note"] == ""

    # 归一化通过：verified=True，但 note 留下"是靠哪类容忍救回来的"
    residual = by_claim[claims[1]]
    assert residual["verified"] is True
    assert residual["verify_note"] == residual_note("residual:punct_spacing")
    assert "折叠标点前空白" in residual["verify_note"]
    assert residual["verify_note"] != "", "通过路径必须可区分：容忍通过不许伪装成精确通过"

    # 真找不到：行为一字未变
    bad = by_claim[claims[2]]
    assert bad["verified"] is False
    assert bad["verify_note"] == "引文未能在原文中逐字匹配"
    assert out["verification"]["unsupported"] == [claims[2]]
    assert out["verification"]["passed"] is False


async def test_soft_veto_note_wins_over_residual_trace(monkeypatch):
    """容忍层救回硬校验、软校验否决时，note 记的是失败原因（"为什么没过"优先于"怎么对上"）。"""
    _stub_soft(
        monkeypatch, [VerificationItem(claim="夸大了。", supported=False)]
    )
    out = await verifier_node.verify_node(
        {
            "question": "混合检索怎么做融合？",
            "retrieved": RETRIEVED,
            "citations": [_citation(101, QUOTE_MARKERS, claim="夸大了。")],
        }
    )
    cit = out["citations"][0]
    assert cit["verified"] is False
    assert "LLM" in cit["verify_note"]
    assert out["verification"]["unsupported"] == ["夸大了。"]

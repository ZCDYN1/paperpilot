"""引用重锚（`app/agents/nodes/evidence.py`）与 read/verify 两侧口径一致的测试。

保护的真实故障：模型必须逐字回吐数据库自增的 `chunk_id`，打错一个数字就被当幻觉剔除。
本地 ollama 跑真实研究时一次 revise 轮剔了 18 条，其中多数引文本身是真的。

这些用例同时守住反面：**重锚不能把真幻觉放过去**，也不允许跨论文圆场。
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from app.agents.nodes import reader as reader_node
from app.agents.nodes import verifier as verifier_node
from app.agents.nodes._common import chunk_index
from app.agents.nodes.evidence import (
    REASON_QUOTE_NOT_FOUND,
    REASON_UNKNOWN_PAPER,
    build_chunk_lookup,
    build_paper_chunks,
    normalize_arxiv_id,
    resolve_citation,
)
from app.agents.prompts import render_document
from app.agents.schemas import ExtractionItem, ReadOutput
from app.core.config import get_settings
from app.core.llm import _grounded_value, parse_untrusted_documents  # noqa: SLF001 - 契约两侧都要锁

_A = "2401.00001"
_B = "2401.00002"

RETRIEVED: list[dict[str, Any]] = [
    {
        "arxiv_id": _A,
        "title": "Hybrid Retrieval at Work",
        "chunks": [
            {
                "chunk_id": 101,
                "section": "abstract",
                "source": "abstract",
                "text": "RRF fusion combines dense and lexical rankings. BM25 scores are unbounded.",
            },
            {
                "chunk_id": 102,
                "section": "methods",
                "source": "html",
                "text": "A cross-encoder reranker rescores the fused candidate list.",
            },
        ],
    },
    {
        "arxiv_id": _B,
        "title": "Survey of Literature Review Generation",
        "chunks": [
            {
                "chunk_id": 201,
                "section": "abstract",
                "source": "abstract",
                "text": "Outline-first pipelines produce section-level citations.",
            }
        ],
    },
]


# ---------------------------------------------------------------------------
# 纯函数层
# ---------------------------------------------------------------------------
def test_normalize_arxiv_id_drops_version_suffix():
    assert normalize_arxiv_id("2411.18583v1") == "2411.18583"
    assert normalize_arxiv_id(" 2411.18583V12 ") == "2411.18583"
    assert normalize_arxiv_id(None) == ""


def test_build_paper_chunks_groups_by_paper():
    papers = build_paper_chunks(RETRIEVED)
    assert set(papers) == {_A, _B}
    assert [c["chunk_id"] for c in papers[_A].chunks] == [101, 102]
    assert papers[_A].arxiv_id == _A


def test_exact_hit_is_not_marked_recovered():
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers, paper_id=_A, chunk_id=102, quote="cross-encoder reranker rescores the fused"
    )
    assert hit.ok and hit.recovered is False
    assert hit.chunk["chunk_id"] == 102


def test_wrong_chunk_id_but_real_quote_is_reanchored():
    """★ 本模块存在的理由：编号打错、引文是真的 → 救回来并改挂到真正的 chunk。"""
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers,
        paper_id=_A,
        chunk_id=999,  # 模型打错的编号
        quote="BM25 scores are unbounded.",
    )
    assert hit.ok
    assert hit.recovered is True
    assert hit.chunk["chunk_id"] == 101


def test_version_suffix_in_paper_id_still_resolves():
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(papers, paper_id=f"{_A}v3", chunk_id=101, quote="RRF fusion")
    assert hit.ok
    assert hit.paper_id == _A  # 回填的是检索结果里的真实 id


def test_fabricated_quote_is_still_dropped():
    """闸门没有变松：论文是真的，但编号也错、引文又找不到 → 照旧剔除。"""
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers, paper_id=_A, chunk_id=999, quote="We achieved 99% accuracy on every benchmark."
    )
    assert hit.ok is False
    assert hit.reason == REASON_QUOTE_NOT_FOUND


def test_exact_chunk_id_defers_quote_check_to_verify():
    """编号精确命中时 read 不判引文 —— 引文是否逐字在原文里，是 verify 硬校验的职责。

    这条不是偷懒，是分工：两个节点用两种口径判同一件事，同一条引文会被判两次、
    失败原因也对不上。这里把这个分工钉住。
    """
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(papers, paper_id=_A, chunk_id=101, quote="完全不相关的编造引文")
    assert hit.ok and hit.recovered is False


def test_quote_from_another_paper_does_not_reanchor():
    """张冠李戴必须失败：quote 属于 B，却声称来自 A → 不能跨论文圆过去。"""
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers, paper_id=_A, chunk_id=777, quote="Outline-first pipelines produce section-level"
    )
    assert hit.ok is False
    assert hit.reason == REASON_QUOTE_NOT_FOUND


def test_unknown_paper_is_reported_as_such():
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(papers, paper_id="9999.99999", chunk_id=1, quote="whatever")
    assert hit.ok is False
    assert hit.reason == REASON_UNKNOWN_PAPER


def test_whitespace_and_case_differences_still_match():
    papers = build_paper_chunks(RETRIEVED)
    hit = resolve_citation(
        papers,
        paper_id=_A,
        chunk_id=101,
        quote="RRF   fusion\n combines dense AND lexical rankings.",
    )
    assert hit.ok and hit.recovered is False


# ---------------------------------------------------------------------------
# read 节点：误杀真的被救回来吗
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _state() -> dict[str, Any]:
    return {"question": "混合检索怎么做融合？", "retrieved": RETRIEVED, "stage": "corpus"}


@pytest.fixture
def fake_read(monkeypatch):
    def _set(claims: list[ExtractionItem]) -> None:
        async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
            return ReadOutput(claims=claims), {}

        monkeypatch.setattr(reader_node, "structured_invoke", _invoke)

    return _set


async def test_read_recovers_citation_with_wrong_chunk_id(fake_read):
    fake_read(
        [
            ExtractionItem(
                paper_id=_A,
                chunk_id=999,  # 打错的自增编号
                claim="词面得分没有上界。",
                quote="BM25 scores are unbounded.",
            )
        ]
    )

    out = await reader_node.read_node(_state())

    assert len(out["citations"]) == 1
    cit = out["citations"][0]
    assert cit["chunk_id"] == 101  # 已改挂到引文真正所在的 chunk
    assert cit["paper_id"] == _A
    # 证据强度以检索层记录为准，不接受模型自述
    assert cit["source"] == "abstract"


# ---------------------------------------------------------------------------
# 字段错位纠正（实测到的真实故障：`paper=505 chunk=None`）
# ---------------------------------------------------------------------------
def _lookups():
    papers = build_paper_chunks(RETRIEVED)
    return papers, build_chunk_lookup(RETRIEVED)


def test_swapped_chunk_id_in_paper_field_is_corrected():
    """模型把片段编号写进 paper_id、chunk_id 留空 → 认出来并挂回真正的论文。"""
    papers, lookup = _lookups()
    hit = resolve_citation(
        papers,
        paper_id="101",
        chunk_id=None,
        quote="RRF fusion combines dense and lexical rankings.",
        chunk_lookup=lookup,
        rendered_chunk_ids={101, 102},
    )
    assert hit.ok
    assert hit.recovered is True
    assert hit.how == "swapped_chunk"
    assert hit.paper_id == _A
    assert hit.chunk["chunk_id"] == 101


def test_swapped_id_without_quote_corroboration_is_rejected():
    """只凭"这个编号存在"不足以放行：引文必须也能逐字对上，否则照样拒。"""
    papers, lookup = _lookups()
    hit = resolve_citation(
        papers,
        paper_id="101",
        chunk_id=None,
        quote="completely fabricated sentence not in any chunk",
        chunk_lookup=lookup,
        rendered_chunk_ids={101},
    )
    assert hit.ok is False
    assert hit.reason == REASON_QUOTE_NOT_FOUND


def test_swapped_id_for_unrendered_chunk_is_rejected():
    """编号确实存在，但它不在渲染给模型看过的集合里 → 模型不可能引用它 → 不认。"""
    papers, lookup = _lookups()
    hit = resolve_citation(
        papers,
        paper_id="101",
        chunk_id=None,
        quote="RRF fusion combines dense and lexical rankings.",
        chunk_lookup=lookup,
        rendered_chunk_ids={102},
    )
    assert hit.ok is False
    assert hit.reason == REASON_UNKNOWN_PAPER


def test_arxiv_looking_value_is_never_treated_as_a_number():
    """`2401.00001` 含小数点，绝不能被当成片段编号 —— 那是合法 arXiv id 的形态。"""
    papers, lookup = _lookups()
    hit = resolve_citation(
        papers,
        paper_id="9999.99999",
        chunk_id=None,
        quote="RRF fusion",
        chunk_lookup=lookup,
    )
    assert hit.ok is False
    assert hit.reason == REASON_UNKNOWN_PAPER


def test_read_side_restricts_reanchor_to_rendered_chunks():
    """同一个"真编号 + 真引文"在 read 侧只认渲染过的片段；verify 侧允许退到全篇。"""
    papers, lookup = _lookups()
    strict = resolve_citation(
        papers,
        paper_id=_A,
        chunk_id=999,
        quote="A cross-encoder reranker rescores the fused candidate list.",
        rendered_chunk_ids={101},
        restrict_to_rendered=True,
        chunk_lookup=lookup,
    )
    assert strict.ok is False

    lenient = resolve_citation(
        papers,
        paper_id=_A,
        chunk_id=999,
        quote="A cross-encoder reranker rescores the fused candidate list.",
        rendered_chunk_ids={101},
        chunk_lookup=lookup,
    )
    assert lenient.ok and lenient.how == "quote"
    assert lenient.chunk["chunk_id"] == 102


async def test_read_recovers_field_swapped_citation(fake_read):
    """端到端：真实故障形态（编号写进 paper_id）现在会被救回来，而不是被剔除。

    `chunk_id=-1` 不是随手写的：字段改成必填之后，本地 qwen2.5:7b 在"不知道该填什么"时
    实测就吐 `-1` 这种占位数字（不再是留空）。所以纠正路径必须吃得下占位值。
    """
    fake_read(
        [
            ExtractionItem(
                paper_id="102",  # ← 其实是 chunk 编号
                chunk_id=-1,
                claim="融合后的候选列表由交叉编码器重新打分。",
                quote="A cross-encoder reranker rescores the fused candidate list.",
            )
        ]
    )

    out = await reader_node.read_node(_state())

    assert len(out["citations"]) == 1
    cit = out["citations"][0]
    assert cit["paper_id"] == _A          # 挂回真正的论文
    assert cit["chunk_id"] == 102
    assert cit["source"] == "html"        # 证据强度取自检索层，不信模型自述


async def test_read_still_drops_fabricated_quote(fake_read):
    fake_read(
        [
            ExtractionItem(
                paper_id=_A,
                chunk_id=999,  # 编号不存在，引文也不在任何片段里 → 真幻觉
                claim="它拿下了所有榜单。",
                quote="state-of-the-art on every single benchmark",
            )
        ]
    )

    out = await reader_node.read_node(_state())

    assert out["citations"] == []
    assert out["readings"] == []


# ---------------------------------------------------------------------------
# ★ 根因回归：渲染给模型看的属性名必须与它要填的字段名逐字同名
# ---------------------------------------------------------------------------
def test_rendered_block_exposes_chunk_id_under_its_field_name():
    """文档块里不许再出现裸 `id="…"` 属性。

    锁的是实测到的故障：块上写 `id="370" arxiv_id="2503.16581"`，而输出字段叫
    `paper_id` / `chunk_id`。本地 qwen2.5:7b 把块里唯一可见的纯数字 370 填进了
    `paper_id`、`chunk_id` 留空（日志 `paper=370 chunk=None`），查表第一步就 miss，
    整条真引用被判幻觉丢掉。改成 `chunk_id="370"` 后"看到的名字=要填的字段名"，
    模型不需要再做一次 id→chunk_id 的语义映射 —— 错位从源头消失，兜底只剩残差。
    """
    block = render_document(
        chunk_id=370,
        arxiv_id="2503.16581",
        section="abstract",
        source="abstract",
        text="RRF fusion combines dense and lexical rankings.",
    )
    assert 'chunk_id="370"' in block
    assert re.search(r'\bid="', block) is None, "裸 id 属性会重新引入 paper_id/chunk_id 错位"
    assert re.search(r'\bpaper_id="', block) is None, "块上不该出现与 arxiv_id 竞争的第二个标识名"


def test_mock_reader_still_finds_chunk_id_after_the_rename():
    """mock 与真实链路共用这份渲染格式：改名必须两侧同时对齐。

    `app/core/llm.py` 的 mock 靠解析文档块来产出"能过引用校验"的结构化结果，
    它读不到 `chunk_id` 的话，离线自测里所有引用会静默变成"定位缺失"，
    于是"零 key 跑通链路"这条卖点其实已经断了却没人报警。
    """
    block = render_document(
        chunk_id=370,
        arxiv_id="2503.16581",
        section="methods",
        source="html",
        text="RRF fusion combines dense and lexical rankings.",
    )
    docs = parse_untrusted_documents(block)
    assert docs[0]["chunk_id"] == "370"
    assert _grounded_value("chunk_id", block, 1) == 370
    assert _grounded_value("paper_id", block, 1) == "2503.16581"


# ---------------------------------------------------------------------------
# verify 节点：两侧口径必须一致（不能在 verify 里被同一个理由二次杀掉）
# ---------------------------------------------------------------------------
async def test_verify_hard_check_accepts_reanchored_citation():
    known = chunk_index(RETRIEVED)
    papers = build_paper_chunks(RETRIEVED)
    citation = {
        "paper_id": _A,
        "chunk_id": 101,
        "quote": "RRF fusion combines dense and lexical rankings.",
        "claim": "融合把两路排名合起来。",
        "verified": False,
    }
    # 第五个返回值是"逐字比较路径"（exact | residual:<类>），与定位路径正交
    ok, reason, anchor, how, match = verifier_node._hard_check(citation, known, papers)  # noqa: SLF001
    assert ok is True
    assert reason == ""
    # 第三个返回值是定位结果：verify 要靠它把真实片段带下去（见 _anchor 的键名说明）
    assert anchor["resolved_chunk_id"] == 101
    # 第四个返回值是定位路径（供 citation_rescue 事件统计）：精确命中就是 exact
    assert how == "exact"
    assert match == "exact", "逐字直接命中就不该走容忍层（否则归一化会污染精确口径）"


async def test_verify_hard_check_still_fails_fabricated_quote():
    known = chunk_index(RETRIEVED)
    papers = build_paper_chunks(RETRIEVED)
    citation = {
        "paper_id": _A,
        "chunk_id": 101,
        "quote": "we prove that retrieval is unnecessary",
        "claim": "检索其实没必要。",
        "verified": False,
    }
    ok, reason, anchor, how, match = verifier_node._hard_check(citation, known, papers)  # noqa: SLF001
    assert ok is False
    assert reason == "引文未能在原文中逐字匹配"
    assert anchor == {}, "没定位到就不该产出任何「纠正后的定位」"
    assert how == "exact", "定位路径与校验结论是两回事：定位到了（exact），但引文对不上"
    assert match == "", "编造的引文连容忍层也救不回来：找不到就是找不到"

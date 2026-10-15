"""结构化对比表抽取（产品定义 §4 产出物 A / F4）的测试。

三条必须被测试锁死的性质（它们就是"与普通 RAG 问答机器人的分界线"）：

1. **缺失留空**：证据里找不到某字段的依据时，该字段必须为空值，**而不是编造**。
2. **每行一篇论文**：行集合由检索结果决定；模型多报的 id 被丢弃、漏报的论文仍有行。
3. **权威字段不被模型污染**：`venue` / 被引数 / `code_url` 只能来自元数据与正文抽取。

新增文件，不改动任何已有测试。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.nodes import comparator
from app.agents.schemas import ComparisonRow, ComparisonTable
from app.core.config import get_settings

# 对比表一行应有的 12 个键（9 个内容字段 + arxiv_id / title / cited_at）
_ROW_KEYS = {
    "arxiv_id",
    "title",
    "method_name",
    "core_idea",
    "datasets",
    "improvement",
    "cost",
    "limitation",
    "code_url",
    "venue",
    "cited_by_count",
    "cited_at",
}

_A = "2999.00001"   # 片段里明确提到数据集与指标
_B = "2999.00002"   # 片段里**没有任何数据集信息**（关键用例）
_C = "2999.00003"   # 片段里有代码仓库链接（大小写混合，用于锁 code_url 不被小写化）


def _paper(aid: str, title: str, chunks: list[tuple[int, str, str, str]]) -> dict[str, Any]:
    return {
        "arxiv_id": aid,
        "title": title,
        "score": 1.0,
        "chunks": [
            {"chunk_id": cid, "section": sec, "source": src, "text": txt}
            for cid, sec, src, txt in chunks
        ],
    }


_RETRIEVED = [
    _paper(
        _A,
        "CrossRanker: Cross Encoder Reranking",
        [
            (
                1,
                "abstract",
                "html",
                "We propose CrossRanker, a cross encoder reranker for retrieval. "
                "On BEIR it reaches nDCG 0.62, a +3.2 point gain over the BM25 baseline.",
            ),
            (2, "method", "html", "CrossRanker scores each query document pair with a cross encoder."),
        ],
    ),
    _paper(
        _B,
        "A Lightweight Reranking Module",
        [
            (
                3,
                "abstract",
                "abstract",
                "We present a reranking module that improves ranking quality without heavy computation.",
            )
        ],
    ),
    _paper(
        _C,
        "LatteRank for Ranking",
        [
            (
                4,
                "abstract",
                "html",
                "LatteRank is evaluated on MS MARCO. "
                "Our code is available at https://github.com/Example/LatteRank.",
            )
        ],
    ),
]

_STATE: dict[str, Any] = {
    "question": "RAG 重排技术有哪些进展与代价？",
    "retrieved": _RETRIEVED,
    "readings": [
        {
            "paper_id": _A,
            "chunk_id": 1,
            "claim": "CrossRanker 在 BEIR 上超过 BM25 基线",
            "quote": "On BEIR it reaches nDCG 0.62",
            "locator": "abstract",
            "source": "html",
        }
    ],
    "citations": [],
}


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    """把 LLM 切成 mock，保证"不调网络也能跑通"且结果确定。

    `Settings` 是 `lru_cache` 单例，改了环境变量必须清缓存，否则仍用旧的 provider
    （那样会静默走真实 LLM → 测试里出现网络调用，既不稳也慢）。
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def meta(monkeypatch):
    """替换元数据读取。默认返回空（等价"库里查不到"→ venue/被引数 留空）。"""

    def _set(mapping: dict[str, dict[str, Any]] | None = None) -> None:
        mapping = mapping or {}

        async def _fake(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
            return {k: v for k, v in mapping.items() if k in set(arxiv_ids)}

        monkeypatch.setattr(comparator, "load_paper_meta", _fake)

    _set()
    return _set


@pytest.fixture
def fake_llm(monkeypatch):
    """替换结构化调用，返回受控的 ComparisonTable（或抛异常用于降级测试）。"""

    def _set(result: Any) -> None:
        async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
            if isinstance(result, Exception):
                raise result
            return result, {}

        monkeypatch.setattr(comparator, "structured_invoke", _invoke)

    return _set


def _rows_by_id(out: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r["arxiv_id"]: r for r in out["comparison"]}


# ---------------------------------------------------------------------------
# 1) mock LLM 模式下能跑通并返回合法结构
# ---------------------------------------------------------------------------
async def test_compare_node_runs_under_mock_provider(meta):
    out = await comparator.compare_node(_STATE)

    assert set(out) >= {"comparison", "events"}
    rows = out["comparison"]
    # 行集合 = 精读的论文集合，顺序与检索结果一致（每行一篇论文）
    assert [r["arxiv_id"] for r in rows] == [_A, _B, _C]
    for r in rows:
        assert set(r) == _ROW_KEYS
        assert isinstance(r["cited_by_count"], int | None)
        assert isinstance(r["venue"], str)

    # 至少要发出一条 thought 事件（前端时间线靠它显示"对比表已生成"）
    assert any(e["type"] == "thought" for e in out["events"])


# ---------------------------------------------------------------------------
# 2) ★ 关键用例：证据里没有"数据集"信息 → 字段必须留空，而不是编造
# ---------------------------------------------------------------------------
async def test_missing_dataset_evidence_yields_empty_not_fabricated(meta, fake_llm):
    # 模型对 B 报了一个"看起来很像"的数据集，但 B 的片段里根本没提数据集
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(arxiv_id=_A, datasets="BEIR", method_name="CrossRanker"),
                ComparisonRow(arxiv_id=_B, datasets="MS MARCO, BEIR", method_name="CrossRanker"),
            ]
        )
    )

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    # A 的片段里确实出现了 BEIR → 证据充分，保留
    assert rows[_A]["datasets"] == "BEIR"
    assert rows[_A]["method_name"] == "CrossRanker"
    # B 的片段里没有任何数据集信息 → 必须留空（不能接受模型给的 MS MARCO / BEIR）
    assert rows[_B]["datasets"] == ""
    # 且"CrossRanker"不是 B 的方法名（它属于 A）→ 同样应被清空
    assert rows[_B]["method_name"] == ""


# ---------------------------------------------------------------------------
# 3) 模型报的"不存在的论文"必须被丢弃（反幻觉，与 read/verify 同一规则）
# ---------------------------------------------------------------------------
async def test_hallucinated_arxiv_id_is_dropped(meta, fake_llm):
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(arxiv_id=_A, method_name="CrossRanker"),
                ComparisonRow(arxiv_id="9999.99999", method_name="PhantomRanker"),
            ]
        )
    )

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    assert set(rows) == {_A, _B, _C}          # 表里只有真实检索到的论文
    assert "9999.99999" not in rows           # 幻觉 id 被剔除
    assert all(r["method_name"] != "PhantomRanker" for r in rows.values())


# ---------------------------------------------------------------------------
# 4) 权威字段覆盖：venue / 被引数 只认元数据，不认模型自述
# ---------------------------------------------------------------------------
async def test_authoritative_metadata_overrides_model_values(meta, fake_llm):
    meta({_A: {"title": "CrossRanker", "venue": "EMNLP 2026", "cited_by_count": 3, "cited_at": "2026-10-12"}})
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(
                    arxiv_id=_A,
                    method_name="CrossRanker",
                    venue="NeurIPS 1999",        # 模型编的会议 → 必须被覆盖
                    cited_by_count=9999,          # 模型编的被引数 → 必须被覆盖
                    cited_at=None,
                )
            ]
        )
    )

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    assert rows[_A]["venue"] == "EMNLP 2026"
    assert rows[_A]["cited_by_count"] == 3
    assert rows[_A]["cited_at"] == "2026-10-12"
    # 没有元数据的论文：venue 留空、被引数为 None（而不是模型给的任何值）
    assert rows[_B]["venue"] == ""
    assert rows[_B]["cited_by_count"] is None


async def test_citation_count_without_date_is_dropped(meta, fake_llm):
    """被引数必须与"截至日期"成对：只有数字没有日期时，数字也留空（架构 §2.1 / D10）。"""
    meta({_A: {"title": "CrossRanker", "venue": "", "cited_by_count": 42, "cited_at": None}})
    fake_llm(ComparisonTable(rows=[]))

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    assert rows[_A]["cited_by_count"] is None
    assert rows[_A]["cited_at"] is None


# ---------------------------------------------------------------------------
# 5) 无依据的自由文本被清空；有依据的保留
# ---------------------------------------------------------------------------
async def test_ungrounded_free_text_fields_are_cleared(meta, fake_llm):
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(
                    arxiv_id=_C,
                    method_name="LatteRank",                       # 片段里有 → 保留
                    cost="训练成本极高，需要多卡集群",              # 纯中文，无法核对 → 清空
                    improvement="Recall +12",                      # 片段里没有 recall → 清空
                )
            ]
        )
    )

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    assert rows[_C]["method_name"] == "LatteRank"
    assert rows[_C]["cost"] == ""
    assert rows[_C]["improvement"] == ""


async def test_is_grounded_unit():
    """纯函数级校验：定义清楚"什么算有依据"。"""
    corpus = comparator.normalize_text(
        "On BEIR it reaches nDCG 0.62, a +3.2 point gain over the BM25 baseline."
    )
    assert comparator.is_grounded("BEIR", corpus) is True
    assert comparator.is_grounded("nDCG +3.2", corpus) is True
    assert comparator.is_grounded("MS MARCO", corpus) is False   # 原文没有 → 编造
    assert comparator.is_grounded("在私有数据集上效果更好", corpus) is False  # 纯中文，无法核对
    assert comparator.is_grounded("", corpus) is False


# ---------------------------------------------------------------------------
# 6) 代码链接来自正文抽取（大小写保留），拿不到就留空
# ---------------------------------------------------------------------------
async def test_code_url_extracted_from_body(meta, fake_llm):
    fake_llm(ComparisonTable(rows=[]))  # 模型不填任何东西，code_url 必须由代码抽出来

    rows = _rows_by_id(await comparator.compare_node(_STATE))

    assert rows[_C]["code_url"] == "https://github.com/Example/LatteRank"
    assert rows[_A]["code_url"] == ""          # 片段里没有代码链接 → 留空，不编造


def test_extract_code_url_strips_trailing_punctuation():
    text = "See https://github.com/foo/bar). and more"
    assert comparator._extract_code_url(text) == "https://github.com/foo/bar"
    assert comparator._extract_code_url("no link here") == ""


def test_extract_code_url_accepts_codeberg_but_bare_domain_not():
    """域名清单与入库侧共用单一事实源（CODE_REPO_HOSTS），codeberg 必须在列；
    同时保守口径不许放宽：域名后没有 `/路径` 的一律不抽（裸域名 ≠ 仓库链接）。"""
    assert (
        comparator._extract_code_url("Our code lives at https://codeberg.org/foo/bar.")
        == "https://codeberg.org/foo/bar"
    )
    assert comparator._extract_code_url("Codeberg.org 是个托管平台") == ""
    assert comparator._extract_code_url("see https://codeberg.org for more") == ""


# ---------------------------------------------------------------------------
# 7) 边界：没有论文 / LLM 失败
# ---------------------------------------------------------------------------
async def test_empty_retrieved_returns_empty_table(meta):
    out = await comparator.compare_node({"question": "q", "retrieved": []})
    assert out["comparison"] == []
    assert any(e["type"] == "thought" for e in out["events"])


async def test_llm_failure_degrades_to_empty_cells(meta, fake_llm):
    """抽取失败不能把整次研究搞挂：行还在，只是字段留空，并补一条可恢复的 error 事件。"""
    fake_llm(RuntimeError("boom"))

    out = await comparator.compare_node(_STATE)
    rows = _rows_by_id(out)

    assert [r["arxiv_id"] for r in out["comparison"]] == [_A, _B, _C]
    for r in rows.values():
        for key in ("method_name", "core_idea", "datasets", "improvement", "cost", "limitation"):
            assert r[key] == ""
    assert any(e["type"] == "error" and e["recoverable"] for e in out["events"])


async def test_notes_report_missing_columns_in_event(meta, fake_llm):
    """`notes`（表级信息）随事件传出，且如实说明哪些列缺失。"""
    fake_llm(ComparisonTable(rows=[ComparisonRow(arxiv_id=_A, datasets="BEIR")]))

    out = await comparator.compare_node(_STATE)
    thought = next(e for e in out["events"] if e["type"] == "thought")

    assert "comparison_notes" in thought
    assert "非编造" in thought["comparison_notes"]
    assert "数据集" in thought["comparison_notes"]      # B/C 的数据集缺失，应被点名
    assert thought["comparison_generated_at"]


# --------------------------------------------------------------------------- 中文值 + 英文摘录
# 这组用例锁的是 F4 曾经"整列被清空"的那个 bug：prompt 要中文描述，
# 而旧的逐字段校验要求值里含英文锚点 —— 中文散文没有，于是 8 行对比表 with_content=0。
# 现在的口径是：中文值配一段**片段里逐字存在的英文原句**当证据，校验落在摘录上。
async def test_chinese_prose_survives_when_evidence_is_verbatim(meta, fake_llm):
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(
                    arxiv_id=_A,
                    core_idea="用交叉编码器对每个查询-文档对逐对打分",
                    core_idea_evidence="CrossRanker scores each query document pair with a cross encoder.",
                )
            ]
        )
    )
    rows = _rows_by_id(await comparator.compare_node(_STATE))
    assert rows[_A]["core_idea"].startswith("用交叉编码器"), "中文值被清空 == 这个 bug 又回来了"


async def test_fabricated_evidence_clears_the_value(meta, fake_llm):
    """摘录必须是原文里真有的句子。编一句像模像样的英文，值照样要被清空。"""
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(
                    arxiv_id=_A,
                    cost="训练成本极高，需要多卡集群",
                    cost_evidence="The model was trained on 512 GPUs for three weeks.",
                )
            ]
        )
    )
    rows = _rows_by_id(await comparator.compare_node(_STATE))
    assert rows[_A]["cost"] == ""


async def test_phrase_length_evidence_is_not_accepted(meta, fake_llm):
    """三个词级别的"摘录"在任何语料里都能撞上，等于没有校验，所以有最短长度门槛。"""
    fake_llm(
        ComparisonTable(
            rows=[
                ComparisonRow(
                    arxiv_id=_A,
                    limitation="长文档场景下会掉点",
                    limitation_evidence="cross encoder",
                )
            ]
        )
    )
    rows = _rows_by_id(await comparator.compare_node(_STATE))
    assert rows[_A]["limitation"] == ""


def test_ground_prose_falls_back_to_anchor_rule():
    """值本身带英文术语时旧路径仍然有效（mock provider 不产出摘录，靠这条活着）。"""
    corpus = comparator.normalize_text(
        "We propose CrossRanker, a cross encoder reranker for retrieval."
    )
    assert comparator._ground_prose("用 CrossRanker 做重排", "", corpus) == "用 CrossRanker 做重排"
    assert comparator._ground_prose("纯中文没有锚点", "", corpus) == ""


# --------------------------------------------------------------------------- 行归位（漏填 id 不该废掉整表）
def _papers(*ids_and_titles: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"arxiv_id": a, "title": t, "score": 1.0, "chunks": []} for a, t in ids_and_titles]


def test_bind_rows_recovers_row_with_missing_id_via_title():
    """模型漏填 `arxiv_id`（7B 常见）时按标题归位，而不是当幻觉行丢掉。"""
    papers = _papers((_A, "CrossRanker: Cross Encoder Reranking"), (_B, "A Lightweight Reranking Module"))
    rows = [ComparisonRow(arxiv_id="", title="CrossRanker: cross-encoder reranking", core_idea="x")]
    bound, dropped = comparator._bind_rows(rows, papers)
    assert dropped == 0 and list(bound) == [_A]


def test_bind_rows_positional_only_when_exactly_one_left():
    """只剩一篇未认领候选、且只剩一行无主内容时才补位 —— 没有猜测空间才允许。"""
    papers = _papers((_A, "Paper A"), (_B, "Paper B"))
    rows = [
        ComparisonRow(arxiv_id=_A, title="Paper A"),
        ComparisonRow(arxiv_id="", title=""),          # 只能归到 _B
    ]
    bound, dropped = comparator._bind_rows(rows, papers)
    assert dropped == 0 and set(bound) == {_A, _B}


def test_bind_rows_refuses_to_guess_when_ambiguous():
    """两行无主、两篇未认领 → 一律不补位。把 A 的描述挂到 B 头上比空表更糟。"""
    papers = _papers((_A, "Paper A"), (_B, "Paper B"), (_C, "Paper C"))
    rows = [
        ComparisonRow(arxiv_id="", title=""),
        ComparisonRow(arxiv_id="", title=""),
    ]
    bound, dropped = comparator._bind_rows(rows, papers)
    assert bound == {} and dropped == 2


def test_bind_rows_still_drops_fabricated_id():
    """底线没松动：候选集之外的 id 仍然是丢弃，并且与"漏填"分开计数。"""
    papers = _papers((_A, "Paper A"))
    rows = [ComparisonRow(arxiv_id="9999.99999", title="Made Up")]
    bound, dropped = comparator._bind_rows(rows, papers)
    assert bound == {} and dropped == 1


def test_bind_rows_duplicate_id_keeps_first_row():
    papers = _papers((_A, "Paper A"))
    rows = [
        ComparisonRow(arxiv_id=_A, core_idea="第一行"),
        ComparisonRow(arxiv_id=_A, core_idea="重复行"),
    ]
    bound, dropped = comparator._bind_rows(rows, papers)
    assert dropped == 0 and bound[_A].core_idea == "第一行"

"""`write` 节点的 F6 剔除 + F17 语言切换测试（产品定义 §5 F5/F6/F17）。

锁死三条性质：
1. **剔除**：`verified is False` 的引用不进 prompt、不进参考文献；
   `verified` 缺失/None 的**必须保留**（三态口径，降级路径不能把报告掏空）。
2. **不重复**：被剔除的论断出现在「⚠️ 未找到原文支撑的结论」一节，
   且与 `verification.unsupported` 同源去重，同一条只列一遍。
3. **语言**：不传 `language` 时行为与改动前完全一致（中文）；传 `en` 时
   prompt 与用户消息里的语言指令都切换。

全程 mock provider + monkeypatch LLM 调用，离线、不碰网络。
"""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.nodes import comparator, writer
from app.agents.schemas import ComparisonTable
from app.core.config import get_settings

_A = "2999.00001"
_B = "2999.00002"


@pytest.fixture(autouse=True)
def _mock_provider(monkeypatch):
    """离线三件套：LLM/embedding 切 mock + 清 Settings 单例缓存（同 test_comparator）。"""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "mock")
    monkeypatch.setenv("RERANK_PROVIDER", "lexical")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _cit(paper: str, chunk: int, claim: str, **over: Any) -> dict[str, Any]:
    base = {
        "paper_id": paper,
        "chunk_id": chunk,
        "claim": claim,
        "quote": f"quote of {claim}",
        "locator": "method",
        "source": "html",
    }
    base.update(over)
    return base


_CITATIONS: list[dict[str, Any]] = [
    _cit(_A, 1, "结论甲：有支撑", verified=True),
    _cit(_A, 2, "结论乙：被判无支撑", verified=False),
    _cit(_B, 3, "结论丙：从未校验过"),                      # verified 缺失
    _cit(_B, 4, "结论丁：校验降级未写回", verified=None),      # 显式 None
]

_STATE: dict[str, Any] = {
    "question": "重排技术对比",
    "retrieved": [
        {
            "arxiv_id": _A,
            "title": "Paper One",
            "chunks": [{"chunk_id": 1, "section": "method", "source": "html", "text": "body A"}],
        },
        {
            "arxiv_id": _B,
            "title": "Paper Two",
            "chunks": [{"chunk_id": 3, "section": "abstract", "source": "abstract", "text": "body B"}],
        },
    ],
    "citations": _CITATIONS,
    # verifier 正常路径下 unsupported 就是 verified=False 的那批论断（同源）
    "verification": {"passed": False, "score": 0.75, "unsupported": ["结论乙：被判无支撑"]},
    "corpus_ready": True,
}


@pytest.fixture
def capture_writer(monkeypatch):
    """替换 text_invoke：记录 messages、返回受控正文。返回记录用的 dict。"""
    captured: dict[str, Any] = {"body": "综述正文。[1]", "usage": {}}

    async def _invoke(messages, *args, **kwargs):  # noqa: ANN001
        captured["messages"] = messages
        return captured["body"], captured["usage"]

    monkeypatch.setattr(writer, "text_invoke", _invoke)
    return captured


# ---------------------------------------------------------------------------
# 1) F6 剔除：False 被剔出 prompt 与参考文献；缺失/None 保留（三态口径）
# ---------------------------------------------------------------------------
async def test_unsupported_claim_excluded_from_prompt_and_references(capture_writer):
    out = await writer.write_node(_STATE)
    system, human = (m.content for m in capture_writer["messages"])

    # prompt（HumanMessage）里：False 的论断与原文都不许出现
    assert "结论乙：被判无支撑" not in human
    assert "quote of 结论乙：被判无支撑" not in human
    # 有支撑 / 未校验的都要在 —— "只留 True"会把后两类错杀
    assert "结论甲：有支撑" in human
    assert "结论丙：从未校验过" in human
    assert "结论丁：校验降级未写回" in human

    draft = out["draft"]
    # 参考文献恰好 3 条、编号连续，被剔条目不在其中
    refs = draft.split("## 参考文献")[1]
    entries = [line for line in refs.splitlines() if line.strip().startswith("[")]
    assert [e.split("]")[0] + "]" for e in entries] == ["[1]", "[2]", "[3]"]
    assert "结论乙" not in refs
    assert refs.count("Paper One") == 1 and refs.count("Paper Two") == 2
    assert "中文" in system  # 默认语言未被剔除逻辑波及


# ---------------------------------------------------------------------------
# 2) 被剔除论断出现在 ⚠️ 一节，且与 verification.unsupported 去重、只列一遍
# ---------------------------------------------------------------------------
async def test_dropped_claim_listed_once_in_unsupported(capture_writer):
    out = await writer.write_node(_STATE)
    draft = out["draft"]
    assert "## ⚠️ 未找到原文支撑的结论" in draft
    assert draft.count("结论乙：被判无支撑") == 1  # unsupported + dropped 合并去重


async def test_dropped_claim_survives_even_without_verification(capture_writer):
    """give_up/降级路径下 verification 可能整个缺席：剔除的论断仍要进 ⚠️ 一节。"""
    state = {**_STATE, "verification": {}}
    out = await writer.write_node(state)
    draft = out["draft"]
    assert "## ⚠️ 未找到原文支撑的结论" in draft
    assert draft.count("结论乙：被判无支撑") == 1


# ---------------------------------------------------------------------------
# 3) F17 语言：默认与改动前一致；language=en 时 prompt 与消息都切换
# ---------------------------------------------------------------------------
async def test_language_defaults_to_chinese(capture_writer):
    await writer.write_node(_STATE)
    system, human = (m.content for m in capture_writer["messages"])
    assert "{{LANGUAGE_RULE}}" not in system       # 占位符必须被渲染掉
    assert "中文" in system
    assert "撰写中文综述" in human


async def test_language_en_switches_prompt(capture_writer):
    await writer.write_node({**_STATE, "language": "en"})
    system, human = (m.content for m in capture_writer["messages"])
    assert "English" in system
    assert "撰写中文综述" not in human


# ---------------------------------------------------------------------------
# comparator 的语言切换（同一占位符机制）
# ---------------------------------------------------------------------------
@pytest.fixture
def capture_compare(monkeypatch):
    captured: dict[str, Any] = {}

    async def _invoke(schema, messages, **kwargs):  # noqa: ANN001
        captured["messages"] = messages
        return ComparisonTable(rows=[]), {}

    async def _meta(arxiv_ids):  # noqa: ANN001
        return {}

    monkeypatch.setattr(comparator, "structured_invoke", _invoke)
    monkeypatch.setattr(comparator, "load_paper_meta", _meta)
    return captured


_COMPARE_STATE: dict[str, Any] = {
    "question": "RAG reranking trade-offs",
    "retrieved": [
        {
            "arxiv_id": _A,
            "title": "Paper One",
            "chunks": [{"chunk_id": 1, "section": "abstract", "source": "html", "text": "body"}],
        }
    ],
    "readings": [],
    "citations": [],
}


async def test_comparator_language_default_and_en(capture_compare):
    await comparator.compare_node(_COMPARE_STATE)
    system = capture_compare["messages"][0].content
    assert "{{FIELD_LANGUAGE_RULE}}" not in system
    assert "字段值用**中文**描述" in system        # 与改动前逐字一致

    await comparator.compare_node({**_COMPARE_STATE, "language": "en"})
    system = capture_compare["messages"][0].content
    assert "字段值同样用**英文**描述" in system
    assert "字段值用**中文**描述" not in system
    # 锚点词保留英文的要求不因语言切换而丢（`_ground` 靠它）
    assert "方法名 / 数据集名 / 指标名 / 数值" in system


# ---------------------------------------------------------------------------
# F17 的**最后一跳**：请求字段真的进了图 state。
# 节点会读 `state["language"]` 只证明"读得到就生效"，不证明"传得进来" ——
# 本项目已经吃过两次这个亏（`top_k` / `rerank` 在请求体里躺了很久，
# `_initial_state` 从来没把它们放进 state，开关于是纯装饰）。
# ---------------------------------------------------------------------------
def test_request_language_is_forwarded_into_state():
    from app.api.v1.research import _initial_state
    from app.schemas.research import ResearchRequest

    en = _initial_state(
        ResearchRequest(question="q", language="en"), run_id="r-en", session_id="r-en", question="q"
    )
    assert en["language"] == "en"

    default = _initial_state(ResearchRequest(question="q"), run_id="r-zh", session_id="r-zh", question="q")
    assert default["language"] == "zh"


def test_request_switches_are_forwarded_into_state():
    """`top_k` / `rerank` 同理：它们必须是 state 通道，而不是节点各读一份进程级配置。

    钉的是"每次请求的配置属于这次请求"：读 `get_settings()`（lru_cache 单例）时，
    并发两个 run 会互相串配置，而且 HITL 恢复后拿到的还是进程当时的值。
    """
    from app.api.v1.research import _initial_state
    from app.schemas.research import ResearchRequest

    state = _initial_state(
        ResearchRequest(question="q", top_k=3, rerank=False), run_id="r1", session_id="r1", question="q"
    )
    assert state["top_k"] == 3
    assert state["rerank"] is False


# ---------------------------------------------------------------------------
# 2) 模型照 prompt 写了「参考文献」/「⚠️ 未支撑」时，报告里不能出现第二份
# ---------------------------------------------------------------------------
async def test_model_written_code_owned_sections_are_not_duplicated(capture_writer):
    """`prompts/writer.md` §2/§4 明确要求模型写这两节，代码又会补一份权威版本。

    真实后果（跑一次研究就能看到）：正文里 `## ⚠️ 未找到原文支撑的结论` 与
    `## 参考文献` 各出现两遍，内容还不完全一样 —— 读者无法判断哪一份是真的，
    而界面上点 [n] 跳的是代码这份的编号。页脚早就去重了，这两节是漏网。
    """
    capture_writer["body"] = (
        "综述正文。[1]\n\n"
        "## ⚠️ 未找到原文支撑的结论\n- \"模型自己列的一条\"\n\n"
        "## 参考文献\n\n[1] 模型自己数的条目 — arXiv:2501.00001\n"
    )

    draft = (await writer.write_node(_STATE))["draft"]

    assert draft.count("## 参考文献") == 1, "参考文献出现了两遍"
    assert draft.count("## ⚠️ 未找到原文支撑的结论") == 1, "未支撑小节出现了两遍"
    # 留下的是代码那份（编号与 citations 表同源），模型那份必须整段消失
    assert "模型自己数的条目" not in draft
    assert "模型自己列的一条" not in draft
    assert "结论乙：被判无支撑" in draft


async def test_stripping_a_code_owned_section_does_not_swallow_the_rest(capture_writer):
    """摘除只到下一个 `## ` 为止：模型写在它后面的正常小节不能被连带吃掉。"""
    capture_writer["body"] = (
        "综述正文。[1]\n\n"
        "## 参考文献\n\n[1] 模型自己数的条目\n\n"
        "## 我们的下一步计划\n\n这段必须留着。\n"
    )

    draft = (await writer.write_node(_STATE))["draft"]

    assert "这段必须留着。" in draft, "摘除范围越界，吃掉了模型写的后续小节"
    assert "模型自己数的条目" not in draft
    assert draft.count("## 参考文献") == 1


async def test_body_without_those_sections_is_untouched(capture_writer):
    """模型没写时不产生任何摘除动作，也不该凭空报错。"""
    capture_writer["body"] = "只有正文。[1]"
    draft = (await writer.write_node(_STATE))["draft"]
    assert "只有正文。[1]" in draft
    assert draft.count("## 参考文献") == 1

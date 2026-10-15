"""`compare` 节点 —— 结构化对比表抽取（产品定义 §4 产出物 A，F4）。

## 这个节点在做一件"普通 RAG 问答机器人做不到"的事
表格的每一格都必须能**点回原文**。所以本节点的核心不是"调一次 LLM 让它吐表格"，
而是**代码对 LLM 的每一格做后置校验**：找不到依据就清空（宁缺毋滥）。
产品定义 §8 的验收指标写得很直白：「对比表字段完整率 ≥ 80%（缺失如实留空，不允许编造）」——
它衡量的是**如实**，不是**填满**。因此这里宁可产出很多空格，也不产出一个编造的格子。

## 三类字段，三种**权威来源**（这是"可追溯"能成立的前提）
| 字段 | 来源 | 为什么不能交给模型 |
|---|---|---|
| 方法名 / 核心思路 / 数据集 / 提升 / 代价 / 局限 | LLM 读论文片段抽取 | 可以交给模型，但必须**逐字段**回原文校验（见 `_ground`） |
| 发表会议 `venue` | arXiv `comment` 字段抽取（入库时已落 `papers.venue`，ADR D9） | 让模型"回忆"会议名＝给它一次编造的机会 |
| 被引数 `cited_by_count` + `cited_at` | OpenAlex（必须带**截至日期**） | 数字一旦编造不可察觉；缺日期则跨论文不可比（架构 §2.1 / D10） |
| 代码链接 `code_url` | 正文/摘要里的**真实 URL**（正则抽取） | 它是"字面量"，抽取即可，不需要模型判断 |

因此：模型在 `venue / cited_by_count / cited_at / code_url` 上无论写什么，
都会被代码**覆盖**为权威值（或空）。`code_url` 是抽取而非生成 —— 抽到的字符串
必然是该论文文本的子串，天然可追溯。

## 行集合 = 论文集合（不是"模型报了几行就是几行"）
表的口径是「每行一篇论文」。所以行集合由**本次精读的论文**决定（与 `read` 节点同口径：
`dedup_papers(retrieved)[:rerank_top_n]`），模型只负责**填格**：
- 模型报了不在集合里的 `arxiv_id` → 丢弃（与 reader/verifier 同一条反幻觉规则）；
- 模型漏了某篇 → 该行字段留空，但**行仍然存在**。

这样"表里有几行""每行是谁"永远是可核查的事实，而不是模型的心情。

## 为什么"逐字段校验"用锚点词而不是字符串包含
`quote` 那种逐字包含判定在 reader/verifier 里成立，是因为那里要求模型**照抄英文原文**。
但对比表要求的是**中文概括**（"数据集"这一格是中文说明），中英之间做子串匹配必然失败。
可行且确定性的近似是：把字段值里的**英文/数字锚点**（方法名、数据集名、指标名、数值）
抠出来，逐个去该论文的片段里找 —— 忠实抽取一定会复用原文里的这些词，
而凭空编造的数据集名 / 指标名几乎必然找不到。
⚠️ 纯中文、不含任何英文/数字锚点的字段值**用这条规则核对不了**，而"字段值用中文描述"
又是这张表的可读性要求（产品定义 §4 产出物 A）—— 两条规则不能互相抵消。
所以三个自由文本列（核心思路 / 代价 / 局限）另走一条契约：模型必须同时给出
**片段里逐字存在的英文原句摘录**，校验落在摘录上、展示仍是中文（见 `_ground_prose`）；
抽取式字段（方法名 / 数据集 / 提升）继续用锚点规则。两条路都不满足才清空。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import (
    dedup_papers,
    merge_usage,
    node_event,
    normalize_text,
    structured_invoke,
)
from app.agents.prompts import load_prompt, render_document, render_papers_block
from app.agents.schemas import ComparisonRow, ComparisonTable
from app.agents.state import ResearchState
from app.agents.tools.paper_meta import run_get_paper_detail
from app.core.config import get_settings
from app.retrieval.hybrid_retriever import CODE_REPO_HOSTS

logger = structlog.get_logger(__name__)

# F17 输出语言切换（口径与 writer 节点一致）：只显式认识 zh / en，
# 其余取值把语言码原样写进指令；缺省 zh，与加 language 字段之前的行为完全一致。
# ⚠️ 非中文语言下 `_ground` 的锚点词校验同样成立（锚点本来就是英文/数字）。


def _output_language(state: ResearchState) -> str:
    return (state.get("language") or "zh").strip().lower() or "zh"


def _field_language_rule(lang: str) -> str:
    """渲染 comparator prompt 里的 `{{FIELD_LANGUAGE_RULE}}` 占位符。"""
    if lang == "en":
        return "片段是英文，字段值同样用**英文**描述；"
    if lang == "zh":
        return "片段是英文，字段值用**中文**描述；"
    return f"片段是英文，字段值用 {lang} 描述；"

# 需要逐字段回原文校验的**自由文本**列。
# （其余四列 `code_url / venue / cited_by_count / cited_at` 由代码权威覆盖，模型说了不算，
#   见 `_row_from` 的"权威覆盖"段。）
_GROUNDED_FIELDS = (
    "method_name",
    "core_idea",
    "datasets",
    "improvement",
    "cost",
    "limitation",
)

# 自由文本列 → 该列配套的"英文原文摘录"字段名（校验落在摘录上，见 `_ground_prose`）。
# 只有这三列需要：`method_name / datasets / improvement` 本身就是抽取式的，
# 值里天然带英文术语或数字，锚点校验对它们既有效又不误伤。
_PROSE_EVIDENCE = {
    "core_idea": "core_idea_evidence",
    "cost": "cost_evidence",
    "limitation": "limitation_evidence",
}
# 摘录至少要这么长才算证据：一个 3 个词的短语在任何语料里都能撞上，等于没校验。
_MIN_EVIDENCE_CHARS = 20

# 锚点词：① 英文词（长度 ≥3，含数字后缀，如 BM25 / nDCG）；② 数值（如 3.2 / 10）。
# 刻意**不允许连字符**：论文里 "cross-encoder" 与概括里的 "cross encoder" 必须视为同一个词，
# 否则会因一个连字符把正确的字段误清空。
_ANCHOR_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}|\d+(?:\.\d+)?")
_ALPHA_ANCHOR_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")

# 锚点命中比例阈值。取 0.5 的理由：忠实抽取会复用原文术语，命中率通常接近 1.0；
# 编造的字段值里只要混入一个原文没有的专名（数据集/指标），比例就掉到 0.5 以下。
_GROUND_MIN_RATIO = 0.5

# 代码仓库链接。域名清单引用 `hybrid_retriever.CODE_REPO_HOSTS`（单一事实源，与入库侧
# `pipeline.detect_has_code` 共用；此前两份清单各写各的，codeberg.org 只进了这一边，
# 导致"对比表认它开源、has_code 过滤却漏掉它"）。
# ⚠️ 本正则要求域名后必须跟 `/非空路径`（保守口径，与入库侧一致）：裸域名不算；
# 与入库侧的唯一差异是抽取必须带 `https?://` 前缀，因为它要产出**完整可用的 URL 字符串**。
# 只认清单内域名 —— 泛化的 URL 抽取会把参考文献链接、图片链接、arXiv 自身链接都卷进来，
# 那些不是"代码链接"。
_CODE_URL_RE = re.compile(
    rf"https?://(?:www\.)?(?:{CODE_REPO_HOSTS})/[^\s)\]}}>\"',;]+",
    re.I,
)


# ---------------------------------------------------------------------------
# 纯函数：抽取 / 校验（与网络、DB 无关，可直接单测）
# ---------------------------------------------------------------------------
def _extract_code_url(text: str) -> str:
    """从论文文本里抽出**代码仓库链接**（抽不到返回空串）。

    这是"抽取"而不是"生成"：返回值一定是原文本的子串，所以天然可追溯，
    不需要再走 `_ground` 校验。只认代码托管域名，避免把参考文献链接误当代码链接。
    """
    m = _CODE_URL_RE.search(text or "")
    if not m:
        return ""
    return m.group(0).rstrip(".,;:)]}\"")


def _anchors(value: str) -> tuple[list[str], list[str]]:
    """把字段值拆成 `(全部锚点, 英文锚点)`，统一小写便于比对。"""
    all_anchors = [a.lower() for a in _ANCHOR_RE.findall(value or "")]
    alpha = [a.lower() for a in _ALPHA_ANCHOR_RE.findall(value or "")]
    return all_anchors, alpha


def is_grounded(value: str, corpus_norm: str) -> bool:
    """字段值是否能在该论文的片段里找到依据。

    ★ 刻意**不吃** verify 硬校验那层"抽取残留容忍"（`_common.quote_match`），尽管它
    与 `_quote_in_chunk` 共用 `normalize_text`：
    1. 判定对象不同 —— 这里比的是"字段值 vs 整篇语料"的锚点命中**比例**，
       不是"引文是否完整句逐字包含"。抽取残留的两类噪声（内联标号、标点前空白）
       对锚点命中几乎无影响：标号只会给语料侧**多出**数字锚点（方向是更宽不是更严），
       标点前空白已被折叠空白吸收。残留问题发生在"整段逐字摘录"这一层，这里没有。
    2. 失败后果不同 —— 硬校验误杀会把真引用打成「未找到原文支撑的结论」（系统性压低
       自动口径，必须修）；这里误清只是对比表某格留空，不进入引用审计链路。
       反过来若在此放宽，放大的是"字段完整率"这个对外指标的注水风险，收益却是零。
    真要动，动 `_ground_prose` 里的摘录包含判定（换成 quote_match 即可）——
    但那是对比表落地口径的独立决策，不顺手改。

    判定规则（保守优先，宁可误清不可放过）：
    1. 空值视为"本来就是空的"，调用方会直接留空 —— 这里返回 False 也无所谓；
    2. 值里必须含**至少一个英文锚点**；纯中文（含只有数字）无法核对 → 判为无依据；
    3. 全部锚点里至少 `_GROUND_MIN_RATIO` 能在片段中命中（英文锚点至少命中一个）。

    `corpus_norm` 需是已 `normalize_text` 归一（小写 + 折叠空白）的片段文本。
    """
    all_anchors, alpha = _anchors(value)
    if not all_anchors or not alpha:
        return False
    hit_alpha = sum(1 for a in alpha if a in corpus_norm)
    if hit_alpha == 0:
        return False
    hit_all = sum(1 for a in all_anchors if a in corpus_norm)
    return hit_all / len(all_anchors) >= _GROUND_MIN_RATIO


def _ground(value: str, corpus_norm: str) -> str:
    """校验通过原样返回；不通过则清空。空值原样返回（保持"本来就缺"）。"""
    v = (value or "").strip()
    if not v:
        return ""
    return v if is_grounded(v, corpus_norm) else ""


# ---------------------------------------------------------------------------
# 元数据读取（venue / 被引数）—— 这几个字段不属于论文正文，不能靠模型抽取
# ---------------------------------------------------------------------------
async def load_paper_meta(arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
    """按 arxiv_id 取 `venue` / `被引数` / `截至日期`（拿不到就整体降级为空）。

    为什么要走 `run_get_paper_detail` 而不是自己查 DB：架构 §2 规定
    **Agent 层不碰数据库**，元数据读取收敛在工具层；这里复用同一入口，
    免得"工具里一套口径、节点里又一套"。

    **永不抛异常**：查询失败（库未建表 / 论文不在本地语料）只让这些列留空，
    绝不能因为一个补充字段把整张表（乃至整次研究）搞挂。
    """
    out: dict[str, dict[str, Any]] = {}
    for aid in arxiv_ids:
        try:
            # max_chunks=0：只要元数据，不要正文（正文我们已经从检索结果里拿到了）
            detail = await run_get_paper_detail(aid, max_chunks=0)
        except Exception as exc:  # noqa: BLE001 - 元数据缺失只影响几列，不影响主流程
            logger.warning("compare.meta_unavailable", arxiv_id=aid, error=str(exc)[:200])
            continue
        if not detail:
            continue
        out[aid] = {
            "title": detail.get("title") or "",
            # venue 在入库时就已从 arXiv comment 抽取（ADR D9）；这里只消费，不重抽，
            # 避免"入库一套规则、节点再一套规则"导致同一篇论文两处给出不同的会议名。
            "venue": detail.get("venue") or "",
            "cited_by_count": detail.get("cited_by_count"),
            "cited_at": detail.get("cited_at"),
        }
    return out


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def _raw_text_of(paper: dict[str, Any]) -> str:
    """某篇论文的原始文本：标题 + 全部命中 chunk 的正文（**不归一**）。"""
    parts = [paper.get("title") or ""]
    for chunk in paper.get("chunks") or []:
        parts.append(chunk.get("text") or "")
    return "\n".join(parts)


def _build_notes(rows: list[dict[str, Any]]) -> str:
    """汇总"哪些字段普遍缺失"。

    这是产出物 D（主动暴露不确定性）在对比表上的落点：空字段本身看不出是
    "原文没写"还是"系统没抓到"，用一句话说明才不会让用户误以为漏了数据。
    """
    if not rows:
        return "本次未检索到可用于对比的论文。"
    total = len(rows)

    def _is_missing(key: str) -> bool:
        # 被引数为 int|None，其余为字符串；两种"空"都算缺失
        return sum(1 for r in rows if r.get(key) in (None, "")) > 0

    columns = [
        ("方法名", "method_name"),
        ("核心思路", "core_idea"),
        ("数据集", "datasets"),
        ("相对 baseline 提升", "improvement"),
        ("代价", "cost"),
        ("局限", "limitation"),
        ("代码链接", "code_url"),
        ("发表会议", "venue"),
        ("被引数", "cited_by_count"),
    ]
    missing = [label for label, key in columns if _is_missing(key)]
    if not missing:
        return f"本表共 {total} 行，各字段均在原文或元数据中找到依据。"
    parts = []
    for label, key in columns:
        if _is_missing(key):
            n = sum(1 for r in rows if r.get(key) in (None, ""))
            parts.append(f"{label} {n}/{total}")
    return (
        f"本表共 {total} 行。以下字段存在缺失（缺失＝原文未提供或元数据不可得，**非编造**）："
        + "、".join(parts)
        + "。"
    )


def _ground_prose(value: str, evidence: str, corpus: str) -> str:
    """自由文本列（核心思路 / 代价 / 局限）的校验：**核英文摘录，不核中文译文**。

    这里修的是一条规则冲突，冲突双方各自都有道理：
    - prompt 要求"字段值用**中文**描述"（产品定义 §4 产出物 A 的可读性）；
    - `is_grounded` 要求值里**至少含一个英文锚点**（反幻觉的保守判据）。
    中文散文几乎必然不含英文锚点，于是 `_row_from` 把整列清空——实测日志
    `compare.done rows=8 with_content=0 dropped_rows=0`，且没有 `compare.llm_failed`，
    说明模型答对了、是校验把内容全杀了。

    解法是把"展示"和"证据"拆成两个字段：模型必须同时给出**片段里的英文原句逐字摘录**，
    校验落在摘录上，展示仍用中文值。这与 read/verify 那条「论断 + 逐字引文」的契约同构，
    不是新发明的口径。

    保留旧路径是为了不放松也不误伤：值本身能过锚点校验时照旧收下
    （mock provider 不产出摘录字段；模型偶尔也会直接把英文术语写进中文句子里，
    那种情况本来就有依据）。两条都不满足才清空 —— 宁缺毋滥的口径一字未改。
    """
    v = (value or "").strip()
    if not v:
        return ""
    quote = normalize_text((evidence or "").strip())
    if len(quote) >= _MIN_EVIDENCE_CHARS and quote in corpus:
        return v
    return v if is_grounded(v, corpus) else ""


def _row_from(paper: dict[str, Any], row: ComparisonRow | None, meta: dict[str, Any]) -> dict[str, Any]:
    """把「一篇论文 + 模型填的格子 + 权威元数据」合成**最终的**一行。

    ⚠️ 顺序很重要：先让模型的自由文本过校验（找不到依据清空），
    再用权威来源覆盖 `venue / cited_by_count / cited_at / code_url`，
    最后 `title` 以检索结果为准。这样任何一列都不会"因为模型编得顺口"而进入表格。
    """
    raw_text = _raw_text_of(paper)
    corpus = normalize_text(raw_text)
    final: dict[str, Any] = {
        "arxiv_id": paper.get("arxiv_id", ""),
        "title": paper.get("title") or meta.get("title") or "",
    }
    for name in _GROUNDED_FIELDS:
        raw = getattr(row, name, "") if row is not None else ""
        evidence_field = _PROSE_EVIDENCE.get(name)
        if evidence_field is None:
            final[name] = _ground(raw, corpus)
        else:
            final[name] = _ground_prose(
                raw, getattr(row, evidence_field, "") if row is not None else "", corpus
            )

    # ---- 权威覆盖 ----
    # ⚠️ code_url 必须在**未归一**的原文上抽：`normalize_text` 会把整段转小写，
    # 而 GitHub 的仓库路径是大小写敏感的，拿归一后的文本抽会把链接改坏。
    final["code_url"] = _extract_code_url(raw_text)
    final["venue"] = meta.get("venue") or ""
    final["cited_by_count"] = meta.get("cited_by_count")
    # 被引数与截至日期必须成对：少了日期，跨论文不可比（架构 §2.1 / D10）。
    # 没有日期就把数字也一并留空 —— 一个孤零零的 "0" 会被误读成"没人引用"。
    cited_at = meta.get("cited_at")
    if not cited_at:
        final["cited_by_count"] = None
    final["cited_at"] = cited_at
    return final


def _render_evidence(items: list[dict[str, Any]], allowed: set[str], *, max_items: int = 8) -> str:
    """把 `read` 已抽取的「论断 → 原句」渲染成不可信文档块，作为填表的高信号参考。

    为什么值得多喂这一段：`readings` 里的 `quote` 是**已从原文逐字摘录**的句子，
    比让模型在整段 chunk 里自己找依据命中率高得多。它同样是论文文本，
    所以照样裹 `<untrusted_document>`（防注入的规则不因"已经处理过一次"而豁免）。
    ⚠️ 它只是**参考**：真正的一票否决仍在 `_ground` —— 依据必须落在原始 chunk 上。
    """
    blocks: list[str] = []
    for item in items:
        if item.get("paper_id") not in allowed or not (item.get("quote") or "").strip():
            continue
        blocks.append(
            render_document(
                chunk_id=item.get("chunk_id") or 0,
                arxiv_id=item.get("paper_id", ""),
                section=item.get("locator") or "abstract",
                source=item.get("source") or "abstract",
                text=item.get("quote", ""),
                title=f"论断：{item.get('claim', '')}",
            )
        )
        if len(blocks) >= max_items:
            break
    return "\n\n".join(blocks)


def _title_key(title: str) -> str:
    """标题归一成一个只用于**匹配**的键（去掉大小写与所有非字母数字）。"""
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def _bind_rows(
    rows: Sequence[ComparisonRow], papers: list[dict[str, Any]]
) -> tuple[dict[str, ComparisonRow], int]:
    """把模型报出的行归位到本次候选论文上，返回 `(按 id 索引的行, 丢弃数)`。

    ★ 这里修的是"整张对比表被清空"这个真实故障：`ComparisonRow.arxiv_id` 有默认值 `""`，
    而本地小模型经常**整字段漏填 id**（实测 5 次里 4 次 `with_content=0`、
    `dropped_rows=3`，可模型填的内容本身是对的）。旧逻辑把"没填 id"和"id 是编的"
    当成同一件事一律丢弃 —— 于是 7B 模型下一个字段漏填就废掉整表。

    归位顺序如下，**任何一步都不允许造出候选集之外的 id**（这是底线，不是优化目标）：
    1. id 合法 → 直接归位（同一 id 出现多次只认第一行）；
    2. id **缺失**、且标题能**唯一**对上某篇未认领候选 → 按标题归位；
    3. 只剩一行无主内容、且只剩一篇未认领候选 → 补位。
       只做 1↔1：N↔N 的顺序对齐在小模型输出里并不可靠，猜错的代价是把 A 论文的描述
       挂到 B 头上，那比空着更糟；
    4. id 是**编的**（非空但不在候选集里）→ 始终丢弃，不参与任何归位；
    5. 仍然归不了的计入 `dropped`。
    """
    candidates = [str(p.get("arxiv_id") or "") for p in papers]
    allowed = {c for c in candidates if c}
    by_title: dict[str, str] = {}
    title_ambiguous: set[str] = set()
    for c in candidates:
        key = _title_key(str(next((p.get("title") for p in papers if p.get("arxiv_id") == c), "")))
        if not key:
            continue
        if key in by_title:
            title_ambiguous.add(key)
        else:
            by_title[key] = c

    bound: dict[str, ComparisonRow] = {}
    unbound: list[ComparisonRow] = []
    fabricated = 0
    for row in rows:
        if row.arxiv_id in allowed:
            bound.setdefault(row.arxiv_id, row)
            continue
        # ⚠️ 只救"**没填** id"的行。填了一个候选集之外的 id，是模型在指认另一篇论文，
        # 这种行即使标题碰巧对上也不归位 —— 否则"编 id"这条最该拦的行为反而被救活了，
        # 而把 A 论文的描述挂到 B 头上，比空着一格危险得多。
        if (row.arxiv_id or "").strip():
            fabricated += 1
            continue
        key = _title_key(row.title)
        if key and key not in title_ambiguous and key in by_title and by_title[key] not in bound:
            bound[by_title[key]] = row
            continue
        unbound.append(row)

    # 第 3 步：只剩一条候选、也只剩一行无主内容时才补位（一对一，不存在猜测空间）
    leftover_ids = [c for c in candidates if c and c not in bound]
    if len(leftover_ids) == 1 and len(unbound) == 1:
        bound[leftover_ids[0]] = unbound.pop()

    dropped = len(unbound) + fabricated
    if dropped:
        # 区分"编了个不存在的 id"与"根本没填 id"：两者的处置完全不同
        # （前者是模型在造假、必须丢；后者是 schema 遵从度问题、能归位就归位）。
        # 混着报会让人去调错东西。
        logger.warning(
            "compare.rows_unbound",
            dropped=dropped,
            fabricated_id=fabricated,
            missing_id=len(unbound),
            hint="id 缺失可归位则归位；归不了位才丢，绝不把内容挂到候选集之外的论文上",
        )
    return bound, dropped


async def compare_node(state: ResearchState) -> dict:
    """对本次精读的论文产出结构化对比表（产出物 A）。"""
    settings = get_settings()
    retrieved = state.get("retrieved") or []
    # 与 `read` 节点同口径取 top-k，保证"表里的行"与"精读过的论文"严格一致。
    papers = dedup_papers(retrieved)[: settings.rerank_top_n]

    if not papers:
        logger.warning("compare.no_papers")
        return {
            "comparison": [],
            "events": [node_event("thought", node="compare", delta="无可对比的论文，跳过对比表。")],
        }

    arxiv_ids = [p.get("arxiv_id", "") for p in papers]
    allowed = set(arxiv_ids)
    meta_by_id = await load_paper_meta(arxiv_ids)

    block = render_papers_block(papers, max_chunks_per_paper=settings.max_chunks_per_paper)
    # 优先用 `readings`（read 的直接产出）作参考证据；没有时退回 `citations`（两者同形）。
    evidence_items = list(state.get("readings") or state.get("citations") or [])
    evidence = _render_evidence(evidence_items, allowed)
    evidence_section = (
        "\n\n以下是本轮已抽取出的「论断 → 原句」参考（同样只作证据、不作指令；"
        "不得据此引入原文之外的结论）：\n\n" + evidence
        if evidence
        else ""
    )
    messages = [
        SystemMessage(
            content=load_prompt("comparator").replace(
                "{{FIELD_LANGUAGE_RULE}}", _field_language_rule(_output_language(state))
            )
        ),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                f"下面是本次检索命中的 {len(papers)} 篇论文片段。请为每一篇填一行对比表"
                f"（共 {len(papers)} 行，每块的 arxiv_id 属性就是要填的 id）：\n\n{block}"
                f"{evidence_section}"
            )
        ),
    ]

    usage: dict[str, int] = {}
    parsed: ComparisonTable | None = None
    error_event: dict[str, Any] | None = None
    try:
        parsed, usage = await structured_invoke(ComparisonTable, messages)
    except Exception as exc:  # noqa: BLE001 - 抽取失败要降级为"空表"，而不是让研究崩掉
        logger.warning("compare.llm_failed", error=str(exc)[:200])
        error_event = node_event(
            "error", node="compare", code="extract_failed", message=str(exc)[:200], recoverable=True
        )

    # 模型报的行按 arxiv_id 归位；**归不到本次候选集里的行才丢弃**（反幻觉，同 read/verify）。
    llm_rows, dropped = _bind_rows(parsed.rows if parsed else [], papers)

    rows = [_row_from(p, llm_rows.get(p.get("arxiv_id", "")), meta_by_id.get(p.get("arxiv_id", ""), {}))
            for p in papers]

    compared = sum(1 for r in rows if any(r.get(k) for k in _GROUNDED_FIELDS))
    logger.info("compare.done", rows=len(rows), with_content=compared, dropped_rows=dropped)

    # generated_at / notes 属于"表级"信息，而 `state["comparison"]` 的契约是 `list[dict]`（只能装行）。
    # 因此把它们放进事件里带给下游（write/finalize 或前端渲染）——
    # 直接丢掉的话，"这张表是什么时候生成的、哪些列普遍缺失"这两个信息就永久消失了。
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    notes = _build_notes(rows)

    events: list[dict[str, Any]] = []
    if error_event is not None:
        events.append(error_event)
    events.append(
        node_event(
            "thought",
            node="compare",
            delta=(
                f"已生成对比表：{len(rows)} 行（每行一篇论文），其中 {compared} 行抽到了可核对的字段；"
                f"{len(rows) - compared} 行因原文未提供而留空。{notes}"
            ),
            comparison_generated_at=generated_at,
            comparison_notes=notes,
        )
    )

    return {
        "comparison": rows,
        "token_usage": merge_usage(usage),
        "events": events,
    }


__all__ = [
    "compare_node",
    "is_grounded",
    "load_paper_meta",
]

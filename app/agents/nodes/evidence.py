"""引用定位与重锚 —— 把「模型把 id 写错了」和「模型编造了证据」分开处理。

## 它为什么长这样（两次被实测纠正的设计）
第一版这里只解决一个**假设中的**故障：模型逐字回吐数据库自增 `chunk_id`（上千的整数）时打错数字。
用本地 ollama 跑真实研究后，实测到的却是"模型把片段编号填进了 `paper_id`、`chunk_id` 留空"
（日志 `paper=505 chunk=None`）—— 于是加了 `swapped_chunk` 路径去兜。

⚠️ 兜住之后又跑了一轮（4 条真实问题 / 53 条引用），确认这是**渲染格式的锅**而不是模型的锅：
文档块上那个属性当时叫 `id="370"`，而输出字段叫 `paper_id` / `chunk_id` —— 块里唯一可见的纯数字
只有一个，字段名里带 "id" 的却有两个，小模型填反是必然结果。把属性改名为 `chunk_id`
（见 `app/agents/prompts/render_document`）之后，`swapped_chunk` 触发次数降到 **0**；
再把 schema 里这个字段从可空改成必填（见 `agents.schemas.ExtractionItem`）之后，
连 `quote` 重锚也不再需要 —— 复测 2 条问题 / 18 条引用，全部走 `exact` 一次命中、read 侧剔除 0 条。

所以现在这三条路径的定位变了：它们不再是"日常靠它撑着"的主链路，而是**残差兜底**——
模型仍然可能填错、填占位值（实测必填之后它会吐 `-1` 这类数字），兜底负责把这些噪声吃掉。
`Resolution.how` 记录到底哪条路径生效，就是为了随时能验证"根因是不是真的修掉了"，
而不是让这类判断永远靠猜。

## 边界（三条路径共用）
- 编造的引文照样落网：任何一步都要求原文能逐字对上，或编号本身能在检索结果里查到。
- 不做跨论文猜引文：引文属于 B、却声称来自 A → 判失败。张冠李戴不能为了数字好看被圆过去。
- 字段错位纠正额外要求"该编号确实是**渲染给模型看过**的片段"：模型没见过的编号不认。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.agents.nodes._common import normalize_text, quote_match

# arXiv id 的版本后缀（2411.18583v1 / v12）：模型很爱带上它，而库里存的是裸 id。
_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)

REASON_UNKNOWN_PAPER = "unknown_paper"
REASON_QUOTE_NOT_FOUND = "quote_not_found"
REASON_EMPTY_QUOTE = "empty_quote"


def normalize_arxiv_id(value: Any) -> str:
    """归一化 arXiv id：去空白、去版本后缀、统一小写。

    只用于**查表键**，不用于展示 —— 报告里仍然使用检索结果里的原始 id。
    """
    text = str(value or "").strip()
    if not text:
        return ""
    return _VERSION_SUFFIX_RE.sub("", text).lower()


@dataclass(slots=True)
class PaperChunks:
    """一篇论文在本次检索结果里的片段集合。"""

    arxiv_id: str                     # 检索结果里的**原始 id**（展示与落库都用它）
    chunks: list[dict[str, Any]]


def build_paper_chunks(retrieved: list[dict[str, Any]] | None) -> dict[str, PaperChunks]:
    """`归一化 arxiv_id → 该篇的片段集合`，保持检索层给出的顺序。

    与 `_common.chunk_index` 的差别只有两点：按篇分组（重锚要在同一篇里找 quote），
    以及键做过归一化。两者可以共存，不改后者的既有语义。
    """
    papers: dict[str, PaperChunks] = {}
    for paper in retrieved or []:
        key = normalize_arxiv_id(paper.get("arxiv_id"))
        if not key:
            continue
        entry = papers.get(key)
        if entry is None:
            entry = PaperChunks(arxiv_id=str(paper.get("arxiv_id")), chunks=[])
            papers[key] = entry
        for chunk in paper.get("chunks") or []:
            if chunk.get("chunk_id") is None:
                continue
            entry.chunks.append(chunk)
    return papers


def build_chunk_lookup(retrieved: list[dict[str, Any]] | None) -> dict[int, tuple[str, dict[str, Any]]]:
    """`chunk_id → (该 chunk 所属的 arxiv_id, chunk)` 全局索引。

    存在的理由是实测到的模型行为（本地 qwen2.5:7b）：它把片段编号填进了 `paper_id`、
    把 `chunk_id` 留空 —— 日志形态是 `paper=505 chunk=None`。
    这些数字都能在 `paper_chunks` 里查到真实论文，所以这是一个**可识别、可纠正的字段错位**，
    不是幻觉。纠正的前提见 `resolve_citation`：必须同时满足"该编号在检索结果里"
    且"引逐字出现在那个片段里"，两条都成立才放行。
    """
    lookup: dict[int, tuple[str, dict[str, Any]]] = {}
    for paper in retrieved or []:
        aid = str(paper.get("arxiv_id") or "")
        for chunk in paper.get("chunks") or []:
            cid = chunk.get("chunk_id")
            if cid is None or not aid:
                continue
            lookup[int(cid)] = (aid, chunk)
    return lookup


def _as_plain_int(value: Any) -> int | None:
    """只有"纯数字"才算编号错位；合法 arXiv id（含 `.`）一律不当数字处理。"""
    text = str(value or "").strip()
    if text.isdigit():
        return int(text)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


@dataclass(slots=True)
class Resolution:
    """一次引用定位的结果。"""

    chunk: dict[str, Any] | None
    paper_id: str | None = None      # 命中时回填**检索结果里的真实 arxiv_id**
    reason: str = ""                 # "" 表示成功
    recovered: bool = False          # True = 模型给的 id 有误，靠引文/编号纠正
    how: str = "exact"               # exact | quote | swapped_chunk —— 供日志统计哪条路径在真实分布上生效

    @property
    def ok(self) -> bool:
        return self.chunk is not None


def resolve_citation(
    papers: dict[str, PaperChunks],
    *,
    paper_id: Any,
    chunk_id: Any,
    quote: str,
    rendered_chunk_ids: set[int] | None = None,
    chunk_lookup: dict[int, tuple[str, dict[str, Any]]] | None = None,
    restrict_to_rendered: bool = False,
) -> Resolution:
    """定位一条引用应该挂在哪个 chunk 上。

    三条路径，从严到宽，**都失败才算幻觉**：

    1. `exact` —— `(论文, chunk_id)` 精确命中就直接用。**不在这一步判引文**：
       "引文是否逐字出现在原文里"是 verify 节点硬校验的职责，两个节点各管一件事，
       否则同一条引文会被两种口径判两次、失败原因也对不上；
    2. `swapped_chunk` —— 论文查不到、但 `paper_id` 是个纯数字且能在检索结果里按 chunk_id 查到
       （实测形态 `paper=505 chunk=None`：模型把片段编号填进了论文字段）。
       这条**必须同时满足引文逐字出现在那个片段里**才放行 —— 我们在猜模型的意图，
       所以要求原文作旁证；猜到了编号但引文对不上，照样拒。
    3. `quote` —— 论文是真的、编号错了：在同一篇内按逐字引文重锚。
       默认只允许锚到**渲染给模型看过的片段**（`restrict_to_rendered`，read 侧用），
       verify 侧允许退到该篇其余片段，因为重写轮里渲染集可能已经换过。

    刻意不做的事：**跨论文猜引文**（引文属于 B 却声称来自 A 必须失败，
    那是张冠李戴，为了凑数字把它圆过去等于造假）。
    """
    normalized_quote = normalize_text(quote)

    key = normalize_arxiv_id(paper_id)
    entry = papers.get(key)
    if entry is not None and entry.chunks:
        # 路径 1：精确命中
        if chunk_id is not None:
            wanted = _as_plain_int(chunk_id)
            if wanted is not None:
                for chunk in entry.chunks:
                    if int(chunk.get("chunk_id", -1)) == wanted:
                        return Resolution(chunk=chunk, paper_id=entry.arxiv_id)

        # 路径 3：同篇内按引文重锚
        # 包含判定走 `_common.quote_match`（两段式：精确 → 抽取残留容忍），与 verify
        # 硬校验共用同一份定义：这里若只用精确口径，带残留的真引文会在定位层就被拒，
        # verify 的容忍层根本轮不到执行 —— 同一个判据分叉成两种严宽度，谁都没修对。
        for chunk in _ordered_candidates(entry.chunks, rendered_chunk_ids, restrict_to_rendered):
            if normalized_quote and quote_match(quote, chunk.get("text", "")):
                return Resolution(
                    chunk=chunk, paper_id=entry.arxiv_id, recovered=True, how="quote"
                )
        return Resolution(
            chunk=None,
            paper_id=entry.arxiv_id,
            reason=REASON_EMPTY_QUOTE if not normalized_quote else REASON_QUOTE_NOT_FOUND,
        )

    # 路径 2：字段错位 —— paper_id 里装的其实是片段编号
    if chunk_lookup:
        for candidate in (paper_id, chunk_id):
            cid = _as_plain_int(candidate)
            if cid is None:
                continue
            found = chunk_lookup.get(cid)
            if found is None:
                continue
            real_paper_id, chunk = found
            if rendered_chunk_ids and cid not in rendered_chunk_ids:
                continue  # 模型没见过的片段编号，不认
            # 引文旁证与路径 3、verify 硬校验同口径（quote_match），见上面的注释。
            if normalized_quote and quote_match(quote, chunk.get("text", "")):
                return Resolution(
                    chunk=chunk, paper_id=real_paper_id, recovered=True, how="swapped_chunk"
                )
            return Resolution(
                chunk=None, paper_id=real_paper_id, reason=REASON_QUOTE_NOT_FOUND
            )

    return Resolution(chunk=None, reason=REASON_UNKNOWN_PAPER)


def _ordered_candidates(
    chunks: list[dict[str, Any]],
    rendered_chunk_ids: set[int] | None,
    restrict_to_rendered: bool,
) -> list[dict[str, Any]]:
    """重锚的候选顺序：渲染给模型看过的片段优先；read 侧只看这些。"""
    rendered = [
        c for c in chunks if rendered_chunk_ids and int(c.get("chunk_id", -1)) in rendered_chunk_ids
    ]
    if restrict_to_rendered:
        return rendered
    rest = [c for c in chunks if c not in rendered]
    return rendered + rest


__all__ = [
    "REASON_EMPTY_QUOTE",
    "REASON_QUOTE_NOT_FOUND",
    "REASON_UNKNOWN_PAPER",
    "PaperChunks",
    "Resolution",
    "build_chunk_lookup",
    "build_paper_chunks",
    "normalize_arxiv_id",
    "resolve_citation",
]

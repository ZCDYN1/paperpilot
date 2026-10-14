"""F18 —— 「相关工作」章节草稿：能直接粘进开题报告的相关工作一节。

## 与 F8 报告导出（`markdown.py`）的关系
- **取数同源**：复用 `load_run_facts`（同一个查询、同一套三态口径），不复制取数代码；
- **三态口径照抄**：引用行只有 `verified is False` 才剔除，None/缺失保留
  （理由见 `app/agents/nodes/writer.py` 与 `markdown.py` 的模块 docstring）；
- **「## 参考文献」去重口径照抄**：正文里已有这一节就不再拼第二份 —— 本文件的正文
  由我们自己组装、天然不含该标题，但规则照抄执行（防御未来正文模板变化）。
- run 不存在抛 `RunNotFoundError`（复用 `markdown.py` 里那个，不新建第二个异常）。

## 分组：按方法/主题，不按论文顺序流水账
产品定义 §5 F18 要的是"按主题组织的相关工作"。这里用**确定性锚点聚类**实现：
从每行的 `method_name + core_idea` 抠出英文/数字锚点词（规则镜像 comparator 的
`_ANCHOR_RE` —— 对比表里的自由文本天然以英文专名为锚），共享锚点的论文归为一组。
⚠️ 这是**字面依据**的聚类：锚点全部来自库里已校验的字段，不引入任何模型先验，
所以分组结果可能朴素（同主题但用词不同的论文会分开），但**绝不会把无关论文
因为"感觉像"而合到一组** —— 宁可分组偏保守，也不编造主题关系。
两个以上论文共享锚点 → 独立主题小节；落单的论文统一进「其他工作」小节。

## 缺失一律留空，不编造（产品定义 §4 / §7）
每篇的句式是"做了什么（core_idea）+ 相对谁的优势（improvement）+ 作者自述局限
（limitation）"，哪个字段是空的就**整个分句不出现** —— 不写"未提及"这类占位话，
因为空值的承诺语义就是"原文没给依据"（`ComparisonRecord` 的口径）。

## 引用编号
`[n]` 按**论文在草稿中的首次出现顺序**编号、同一篇恒同一号；
结尾引用列表的证据强度标注沿用 `markdown._reference_block` 的口径
（source 拿不到就如实标"摘要级证据"，绝不高估）。
"""

from __future__ import annotations

import re

from app.db.models import Citation, ComparisonRecord
from app.export.markdown import _FOOTER, RunNotFoundError, load_run_facts

# 锚点规则镜像 comparator（英文词 ≥3 字符）：对比表的中文概括里保留的英文专名
# 是唯一能"字面依据"地说明两篇论文属于同一主题的东西。刻意**不用纯数字锚点**分组
# （年份/数值相同不代表同主题）。
_ANCHOR_RE = re.compile(r"[A-Za-z][A-Za-z0-9]{2,}")

# 过泛的英文词：出现得太普遍，据此合并会把不相干论文拽进同一组。
_STOP_ANCHORS = frozenset(
    {
        "the", "and", "for", "with", "that", "this", "from", "their", "its",
        "based", "using", "used", "via", "our", "proposed", "method", "model",
        "approach", "paper", "new", "can", "also", "more", "than",
    }
)


def _anchors_of(row: ComparisonRecord) -> set[str]:
    """一篇论文的分组锚点：method_name + core_idea 里的英文专名（小写、去泛词）。"""
    text = f"{row.method_name or ''} {row.core_idea or ''}"
    return {a.lower() for a in _ANCHOR_RE.findall(text)} - _STOP_ANCHORS


def _cluster(rows: list[ComparisonRecord]) -> list[tuple[str, list[ComparisonRecord]]]:
    """确定性分组：共享 ≥1 锚点的论文并成一簇（union-find，纯字面依据）。

    返回 `(小节标题, 行列表)`；顺序 = 各簇首篇论文在对比表里的行序（可复现）。
    落单的论文（没和任何论文共享锚点）收进末尾的「其他工作」。
    """
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    anchors = [_anchors_of(r) for r in rows]
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if anchors[i] & anchors[j]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)

    themed: list[tuple[str, list[ComparisonRecord]]] = []
    singles: list[ComparisonRecord] = []
    for members in sorted(groups.values(), key=lambda ms: min(ms)):
        rows_g = [rows[m] for m in members]
        if len(members) == 1:
            singles.append(rows_g[0])
            continue
        # 小节标题用簇内"被最多论文共享"的锚点词 —— 它是这组论文字面共同的主题词。
        freq: dict[str, int] = {}
        for m in members:
            for a in anchors[m]:
                freq[a] = freq.get(a, 0) + 1
        label = max(sorted(freq), key=lambda a: freq[a])
        themed.append((label, rows_g))

    if singles:
        themed.append(("其他工作", singles))
    return themed


def _supported_evidence(citations: list[Citation], arxiv_id: str) -> dict[str, str] | None:
    """某篇论文在库内的引用证据（三态口径：只有 verified is False 剔除，None/缺失保留）。

    取第一条支撑引用做证据标注；一条都没有返回 None（引用列表里如实标"无逐句引用记录"，
    不高估证据强度 —— 对比表字段本身经过锚点校验，但那是"字段级"依据，不是"论断级"）。
    """
    for c in citations:
        if c.arxiv_id != arxiv_id or c.verified is False:
            continue
        strength = "摘要级证据" if (c.source or "abstract") == "abstract" else "正文原文"
        return {"locator": c.locator or "abstract", "strength": strength}
    return None


def _sentence(row: ComparisonRecord, cite: str) -> str:
    """一篇论文一行：做了什么 + 相对谁的优势 + 作者自述局限；空字段分句直接不出现。

    `cite` 是句尾引用标记（如 `"[2]"`；arxiv_id 缺失、进不了引用列表的行传空串，
    不硬造一个编号）。
    """
    display = (row.method_name or row.title or row.arxiv_id or "").strip().replace("\n", " ")
    parts: list[str] = []
    core = (row.core_idea or "").strip().replace("\n", " ")
    if core:
        parts.append(f"{display} 提出「{core}」{cite}。")
    improvement = (row.improvement or "").strip().replace("\n", " ")
    if improvement:
        parts.append(f"相对 baseline 的提升：{improvement}{cite}。")
    limitation = (row.limitation or "").strip().replace("\n", " ")
    if limitation:
        parts.append(f"作者自述局限：{limitation}{cite}。")
    if not parts:
        # 该行所有自由字段都留空（对比表的承诺语义：原文没给依据）。
        # 如实只挂名字和引用，不写"未提及"占位 —— 也不编内容。
        parts.append(f"{display}{cite}。（对比表该行的字段均因原文未提供依据而留空。）")
    return "- " + " ".join(parts)


async def render_related_work_markdown(run_id: str) -> str:
    """生成一次 run 的「相关工作」章节草稿（F18）；run 不存在抛 `RunNotFoundError`。

    结构：章节标题 + run 溯源行 → 按主题分组的小节（每篇"做了什么 + 相对优势 +
    作者自述局限"，句尾 `[n]`）→ 「## 参考文献」列表 → 固定声明。
    """
    facts = await load_run_facts(run_id)
    run = facts["run"]
    rows = facts["comparison_rows"]
    citations = facts["citations"]
    titles: dict[str, str] = facts["titles"]

    sections: list[str] = [
        "# 相关工作",
        f"> 本节由 run `{run_id}` 的结构化对比表自动生成，仅供写作参考；"
        f"研究问题：{run.question}",
    ]

    # 分组只算一次（确定性 union-find，两次调用结果本也相同，但一次更省也更不会走偏）。
    groups = _cluster(rows) if rows else []

    # 引用编号：按草稿正文中的首次出现顺序（同一篇论文恒同一号）。
    ref_no: dict[str, int] = {}
    cited_papers: list[ComparisonRecord] = []
    for _label, group in groups:
        for r in group:
            pid = r.arxiv_id or ""
            if pid and pid not in ref_no:
                ref_no[pid] = len(ref_no) + 1
                cited_papers.append(r)

    if rows:
        body_lines: list[str] = []
        for label, group in groups:
            heading = f"## {label}" if label == "其他工作" else f"## 主题：{label}"
            body_lines.append(heading)
            body_lines.append("")
            for r in group:
                no = ref_no.get(r.arxiv_id or "")
                body_lines.append(_sentence(r, f"[{no}]" if no else ""))
            body_lines.append("")
        sections.append("\n".join(body_lines).strip())
    else:
        # 没对比表就没有可组装的事实 —— 如实说明，不退回"临时生成"（同 F8 的口径）。
        sections.append("（本次 run 未生成结构化对比表，相关工作草稿为空。）")

    # 参考文献列表：标题以 papers 表官方标题为准，查不到用行内标题/id 兜底，不编标题。
    ref_lines: list[str] = []
    for r in cited_papers:
        pid = r.arxiv_id
        title = titles.get(pid) or (r.title or "").strip() or pid
        evidence = _supported_evidence(citations, pid)
        if evidence:
            ref_lines.append(
                f"[{ref_no[pid]}] {title} — arXiv:{pid}（章节：{evidence['locator']}｜{evidence['strength']}）"
            )
        else:
            ref_lines.append(
                f"[{ref_no[pid]}] {title} — arXiv:{pid}（本 run 库内无该论文的逐句引用记录；"
                "正文条目依据对比表字段的原文锚点校验）"
            )
    reference_block = "## 参考文献\n\n" + "\n".join(ref_lines) if ref_lines else ""
    # 「## 参考文献」去重口径照抄 markdown.render_run_markdown：正文里已有这一节就不再拼第二份。
    body_text = "\n\n".join(sections)
    if reference_block and "## 参考文献" not in body_text:
        sections.append(reference_block)
    if _FOOTER not in body_text:
        sections.append(_FOOTER)

    return "\n\n".join(s.strip() for s in sections if s and s.strip())


__all__ = ["RunNotFoundError", "render_related_work_markdown"]

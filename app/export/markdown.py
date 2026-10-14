"""F8 —— 把一次 run 的数据库记录渲染成完整 Markdown 报告。

## 为什么从库里读，而不是从 graph state / checkpointer 读
导出接口的用户可能是在研究结束几天后点"下载"，那时 thread 可能已被清理；
而 `research_runs / citations / comparison_rows / token_usage / messages`
这几张表正是 finalize 刻意落库的"可查询事实"（架构 §12）。
导出与它们同源，**不依赖 API 层对象**，路由只需要一行转发。

## 与 `GET /research/{run_id}` / `/{run_id}/citations` 的口径一致性
- 参考文献按 `(arxiv_id, chunk_id)` 去重、按插入顺序编号 —— 与 `write` 节点
  生成正文里 `[n]` 的编号规则同构（⚠️ 已知偏差：verify 重锚过的引用在库里
  存的是 `resolved_*`，导出编号可能与旧正文的 `[n]` 错位；这是"库里只留最终
  定位"这一落库口径的必然结果，宁错位也不编造旧定位）。
- 「未支撑结论」一节 = `verified is False` 的行（含 `note` 失败原因），
  与报告正文末尾由代码追加的那节同源。
- **三态口径照抄 writer**：`verified` 缺失/None 的行**保留**在参考文献里，
  只有显式 False 才剔除（原因见 `app/agents/nodes/writer.py` 模块 docstring）。

## 缺失一律留空，不编造（产品定义 §4 / §8 的同一条承诺）
表里任何字段拿不到就渲染成空单元格；综述正文只认 `messages` 表里
writer 落库的那份，查不到就是空 —— 不退回"临时生成"，那会引入新的 LLM 调用
和新的不可追溯内容。

## 找不到 run 的行为
抛 `RunNotFoundError`（`LookupError` 子类）。选异常而不是返回 None：
路由层"存在即 200、不存在即 404"的映射用异常表达最直白，
返回 None 会把"忘了检查"变成"导出了一份空报告"这种静默错误。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from app.db import session as db_session
from app.db.models import Citation, ComparisonRecord, Message, Paper, ResearchRun, TokenUsage

# 固定声明（产品定义 §7）：正文里已有就不重复追加（与 writer 节点同一去重约定）。
_FOOTER = "> 本文由 AI 生成，引用请以原文为准。"

# 对比表 9 列：表头 → ComparisonRecord 属性名（顺序即列顺序，产出物 A 口径）。
_TABLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("方法", "method_name"),
    ("核心思路", "core_idea"),
    ("数据集", "datasets"),
    ("相对 baseline 提升", "improvement"),
    ("代价", "cost"),
    ("局限（原文自述）", "limitation"),
    ("代码", "code_url"),
    ("发表会议", "venue"),
    ("被引", "cited_by_count"),
)


class RunNotFoundError(LookupError):
    """指定 run_id 在库里不存在。路由层应把它映射成 HTTP 404。"""

    def __init__(self, run_id: str) -> None:
        super().__init__(f"run {run_id} 不存在，无法导出报告")
        self.run_id = run_id


def _cell(value: Any) -> str:
    """单元格：None/空串 → 空；竖线转义（否则会把 Markdown 表格拦腰截断）。"""
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _comparison_block(rows: list[ComparisonRecord]) -> str:
    if not rows:
        return ""
    headers = [label for label, _ in _TABLE_COLUMNS]
    lines = [
        "## 结构化对比表",
        "",
        "| 论文 | " + " | ".join(headers) + " |",
        "|---" * (len(headers) + 1) + "|",
    ]
    for r in rows:
        # 被引数与截至日期必须成对（架构 D10）：comparator 在没有日期时已把数字清空，
        # 所以这里"有数字必有日期"；日期并进"被引"单元格，保住 9 列口径。
        cited = ""
        if r.cited_by_count is not None:
            cited = str(r.cited_by_count)
            if r.cited_at:
                cited += f"（截至 {_cell(r.cited_at)}）"
        cells = [cited if label == "被引" else _cell(getattr(r, attr)) for label, attr in _TABLE_COLUMNS]
        title = _cell(r.title or r.arxiv_id)
        lines.append(f"| {title} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _reference_block(citations: list[Citation], titles: dict[str, str]) -> tuple[str, list[str]]:
    """编号参考文献（剔除 verified=False 的行），并返回被剔除行的展示文本。

    返回 `(参考文献 Markdown, 未支撑结论列表)`；无内容的前一项为空串。
    """
    lines: list[str] = []
    unsupported: list[str] = []
    order: dict[tuple[str | None, int | None], int] = {}
    for c in citations:
        if c.verified is False:
            # note 里是 verifier 写回的失败原因；库里没有正文引用时它也要露面。
            suffix = f"（原因：{c.note}）" if c.note else ""
            unsupported.append(f'- "{c.claim}"{suffix}')
            continue
        key = (c.arxiv_id, c.chunk_id)
        if key in order:
            continue
        idx = len(order) + 1
        order[key] = idx
        title = titles.get(c.arxiv_id or "", c.arxiv_id or "")
        locator = c.locator or "abstract"
        # source 拿不到（历史行 NULL）时**如实标摘要级**，绝不高估证据强度。
        strength = "摘要级证据" if (c.source or "abstract") == "abstract" else "正文原文"
        lines.append(f"[{idx}] {title} — arXiv:{c.arxiv_id or ''}（章节：{locator}｜{strength}）")
    block = "## 参考文献\n\n" + "\n".join(lines) if lines else ""
    return block, unsupported


def _cost_block(run: ResearchRun, usages: list[TokenUsage]) -> str:
    total = next((u for u in usages if u.node_name == "all"), None)
    if total is None:
        return ""
    lines = [
        "## 成本与耗时",
        "",
        f"- Token 用量：prompt {total.prompt_tokens} / completion {total.completion_tokens}",
        f"- 成本估算：{total.cost_estimate if total.cost_estimate is not None else '—'}（粗略单价，非账单）",
    ]
    if run.started_at and run.ended_at:
        duration = run.ended_at - run.started_at
        lines.append(
            f"- 耗时：{int(duration.total_seconds() // 60)} 分 "
            f"{int(duration.total_seconds() % 60)} 秒"
        )
    else:
        lines.append("- 耗时：—（run 未正常收尾，起止时间不完整）")
    return "\n".join(lines)


async def load_run_facts(run_id: str) -> dict[str, Any]:
    """一次 run 的全部"事实行"读取 —— `render_run_markdown` 与 F18 相关工作草稿共用。

    ⚠️ 这就是原来内联在 `render_run_markdown` 里的那段查询的**原样搬家**：
    谁要改"取哪些行、三态口径、标题兜底"，改这里一处，两个消费者同步生效
    （F18 不许复制一份取数代码，见 `app/export/related_work.py` 模块 docstring）。

    与搬家前唯一的差异：`titles` 现在同时覆盖 citations **和**对比表里出现的 arxiv_id
    （F18 需要对比表论文的官方标题）。对 `render_run_markdown` 行为零影响 ——
    它只按引用行的 id 去查这个 dict，多出来的键从不被读取。

    run 不存在时抛 `RunNotFoundError`（异常约定与导出接口一致）。
    """
    async with db_session.session_scope() as session:
        run = await session.get(ResearchRun, run_id)
        if run is None:
            raise RunNotFoundError(run_id)

        citations = list(
            (
                await session.execute(
                    select(Citation).where(Citation.run_id == run_id).order_by(Citation.id)
                )
            )
            .scalars()
            .all()
        )
        comparison_rows = list(
            (
                await session.execute(
                    select(ComparisonRecord)
                    .where(ComparisonRecord.run_id == run_id)
                    .order_by(ComparisonRecord.row_index)
                )
            )
            .scalars()
            .all()
        )
        usages = list(
            (
                await session.execute(select(TokenUsage).where(TokenUsage.run_id == run_id))
            )
            .scalars()
            .all()
        )
        # 综述正文只认 DB：finalize 把 writer 的 draft 以 node_name="writer" 落进 messages，
        # 取最新一条（revise 重跑后可能有多条，最后一条才对应最终报告）。
        draft = (
            await session.execute(
                select(Message.content)
                .where(Message.run_id == run_id, Message.node_name == "writer")
                .order_by(Message.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none() or ""
        # 引用与对比表涉及的论文标题：库里查不到的（历史数据 / 未入库）就用 arxiv_id 兜底，不编标题。
        paper_ids = {c.arxiv_id for c in citations if c.arxiv_id} | {
            r.arxiv_id for r in comparison_rows if r.arxiv_id
        }
        titles: dict[str, str] = {}
        if paper_ids:
            for pid, title in (
                await session.execute(select(Paper.arxiv_id, Paper.title).where(Paper.arxiv_id.in_(paper_ids)))
            ).all():
                titles[pid] = title

    return {
        "run": run,
        "citations": citations,
        "comparison_rows": comparison_rows,
        "usages": usages,
        "draft": draft,
        "titles": titles,
    }


async def render_run_markdown(run_id: str) -> str:
    """渲染一次运行的完整 Markdown 报告；run 不存在时抛 `RunNotFoundError`。

    内容顺序（对齐产品定义 §4 产出物 A~E）：
    标题与研究问题 → 对比表 → 综述正文 → 参考文献 → ⚠️ 未支撑结论 → 成本与耗时 → 固定声明。
    """
    facts = await load_run_facts(run_id)
    run = facts["run"]
    citations = facts["citations"]
    comparison_rows = facts["comparison_rows"]
    usages = facts["usages"]
    draft = facts["draft"]
    titles = facts["titles"]

    sections: list[str] = [
        f"# 文献研究报告：{run.question}".strip(),
        f"> run_id: `{run.id}` ｜ session: `{run.session_id}` ｜ 状态: {run.status}",
    ]
    comparison_block = _comparison_block(comparison_rows)
    if comparison_block:
        sections.append(comparison_block)
    if draft.strip():
        sections.append(f"## 综述正文\n\n{draft.strip()}")
    reference_block, unsupported = _reference_block(citations, titles)
    # ⚠️ 去重：`draft` 本身就是 writer 节点产出的**完整报告**，它内部已经带了
    # 「## 参考文献」与「⚠️ 未找到原文支撑的结论」两节（见 `nodes/writer.py` 的 sections 装配）。
    # 不加这个判断，导出文件里同一份参考文献会出现两遍，而且两份编号可能不一致
    # （库里存的是 verify 重锚后的定位，正文里的 `[n]` 是按当时的引用顺序编的）。
    # 正文里已有就以正文为准 —— 它和正文的 `[n]` 是同一套编号，才是对读者负责的那一份。
    # 只有老数据 / 正文缺失的 run 才落到"用库里的行自己拼一节"。
    if reference_block and "## 参考文献" not in draft:
        sections.append(reference_block)
    if unsupported and "未找到原文支撑的结论" not in draft:
        sections.append(
            "## ⚠️ 未找到原文支撑的结论\n\n"
            "以下结论未在检索到的文献中找到直接原文支撑，请勿直接引用：\n\n"
            + "\n".join(unsupported)
        )
    cost_block = _cost_block(run, usages)
    if cost_block:
        sections.append(cost_block)
    if _FOOTER not in draft:
        sections.append(_FOOTER)

    return "\n\n".join(s.strip() for s in sections if s and s.strip())


__all__ = ["RunNotFoundError", "load_run_facts", "render_run_markdown"]

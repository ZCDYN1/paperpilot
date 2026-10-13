"""节点结构化输出的 Pydantic 模型（架构 §4.2 的工具/结构化输出侧）。

为什么这些 schema 不放进 `app/schemas/`：
`app/schemas/` 是**对外契约**（API 出入参、Tool 入参），改动会波及前端与其他模块；
而下面这些是**图内部的节点间契约**，只有 orchestrator 和节点自己关心。
放在一起会让"哪些字段改了要通知前端"变得看不出来。

`SubQuestion` 直接复用 `app.schemas.research.SubQuestion`：它是 Planner 的输出，
也确实是 API 的对外模型（用户编辑子问题后回传），本身就是同一个东西，
再定义一份会导致两边的校验规则漂移（尤其"关键词必须英文"这条）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.research import SubQuestion


class PlanOutput(BaseModel):
    """`plan` 节点的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    rationale: str = Field(default="", description="为什么这样拆解（≤100 字）")
    sub_questions: list[SubQuestion] = Field(
        min_length=3,
        max_length=5,
        description=(
            "3~5 个互补子问题。每个子问题必须给出**英文** keywords"
            "（含同义/缩写扩展），否则 BM25 侧等于失效。"
        ),
    )


class ExtractionItem(BaseModel):
    """一条「论断 → 证据片段」。

    ⚠️ `quote` 必须是**从原文逐字摘录**的片段，不允许改写/翻译/概括。
    Verifier 会用字符串包含关系去原文里核对它 —— 一旦模型"润色"了引文，
    校验就会失败，这正是我们想要的行为（宁缺毋滥）。
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field(description="arXiv id，必须来自本次检索结果")
    chunk_id: int | None = Field(default=None, description="证据所在的 chunk_id")
    claim: str = Field(min_length=1, description="该片段支持的中文论断（一句话）")
    quote: str = Field(min_length=1, description="原文片段，逐字摘录，越长越好但不超过 300 字")
    locator: str = Field(default="abstract", description="章节名 / 页码，便于用户回原文核对")
    source: Literal["html", "abstract"] = Field(
        default="abstract", description="证据强度：html=正文原文，abstract=摘要级降级证据"
    )


class ReadOutput(BaseModel):
    """`read` 节点的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    claims: list[ExtractionItem] = Field(
        default_factory=list, description="从这些论文中抽取出的「论断 → 证据」对"
    )


class VerificationItem(BaseModel):
    """单条论断的校验结论。"""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="被校验的论断原文")
    supported: bool = Field(description="该论断是否能在给出的原文片段中找到直接支撑")
    reason: str = Field(default="", description="判定理由（≤80 字）")


class VerifyOutput(BaseModel):
    """`verify` 节点的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    items: list[VerificationItem] = Field(default_factory=list)


class ReviseOutput(BaseModel):
    """`revise` 节点的结构化输出：针对未通过的论断生成补充检索子问题。"""

    model_config = ConfigDict(extra="forbid")

    sub_questions: list[SubQuestion] = Field(
        default_factory=list,
        description="补充检索用的子问题（仍然要求英文 keywords）",
    )


# ---------------------------------------------------------------------------
# 结构化对比表（产品定义 §4 产出物 A）—— `compare` / `comparator` 节点的输出
# ---------------------------------------------------------------------------
# 为什么把"缺失留空、不许编造"逐字段写进 Field(description)：
# 这段 description 会随 schema 一起送进模型的 structured output 约束里，
# 是模型**在生成现场**能看到的唯一硬性规则。写在 prompt 里当然也要写，
# 但字段级的说明离"要填的那个格子"最近，最不容易被模型在大段上下文里忽略。
# 真正的兜底在 `comparator.py` 的后置校验（找不到依据就清空），
# description 只是把"为什么必须留空"提前告诉模型，减少无效生成。
class ComparisonRow(BaseModel):
    """对比表的一行 = **一篇论文**。

    ⚠️ 三条不可协商的规则（与"普通 RAG 问答机器人"的分界线）：
    1. 字段值必须能在该论文的片段里找到依据，**找不到就留空**；
    2. 不允许用领域常识/其他论文的知识/模型先验"补全"任何字段；
    3. 宁可整行留空，也不允许写"未提及""多种数据集"这类占位式猜测 ——
       因为那看起来像结论，会污染用户的判断。
    """

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str = Field(
        default="", description="arXiv id。必须是本次给定论文里真实存在的 id，不允许凭记忆编造。"
    )
    title: str = Field(
        default="", description="论文标题。由系统从检索结果回填；缺失留空，不编造。"
    )

    method_name: str = Field(
        default="",
        description=(
            "论文提出的方法/系统名。必须**在片段里出现过**该名称（英文原名照抄）；"
            "片段里没有明确的方法名就填 \"\"，不要拿标题或作者名充当方法名。"
        ),
    )
    core_idea: str = Field(
        default="",
        description=(
            "一句话（≤60 字）说明方法怎么做。只允许概括片段里**说过**的内容；"
            "方法名/组件名/指标名保留英文原文。片段信息不足就填 \"\"。"
        ),
    )
    datasets: str = Field(
        default="",
        description=(
            "实验所用数据集名称（英文原名）。片段里必须**出现**该数据集名；"
            "没有提到数据集就填 \"\"，绝不允许写常见数据集或\"未提及\"之类的内容。"
        ),
    )
    improvement: str = Field(
        default="",
        description=(
            "相对 baseline 的提升，尽量带上指标名与数值（如 nDCG@10 +3.2）。"
            "片段里必须出现对应的 baseline / 指标 / 数值；否则填 \"\"。"
        ),
    )
    cost: str = Field(
        default="",
        description=(
            "该方法付出的代价（延迟 / 显存 / 索引体积 / 训练或推理成本）。"
            "必须是片段里**明说**的代价；作者没提就填 \"\"，不要外推。"
        ),
    )
    limitation: str = Field(
        default="",
        description=(
            "**作者自己承认**的局限或失败场景（原文自述）。"
            "必须是片段里作者的自述，不能是推断、不能是他人评价；没有就填 \"\"。"
        ),
    )
    code_url: str = Field(
        default="",
        description="代码仓库链接。由系统从正文/摘要里抽取真实 URL；模型请一律留空。",
    )
    venue: str = Field(
        default="",
        description="发表会议/期刊。由系统从 arXiv comment 字段抽取；模型请一律留空。",
    )
    cited_by_count: int | None = Field(
        default=None, description="被引数。由系统从 OpenAlex 补充；模型请一律留空（None）。"
    )
    cited_at: str | None = Field(
        default=None, description="被引数的截至日期（ISO 日期字符串）。必须与被引数同时存在。"
    )


class ComparisonTable(BaseModel):
    """完整对比表（产出物 A）。

    `notes` 刻意由**代码**在后置校验之后生成，用来如实说明"哪些列普遍缺失"——
    这是产品"主动暴露不确定性"（产出物 D）在对比表上的落点：
    一行行空字段本身看不出是"作者没写"还是"系统没抓到"，用文字说清才不会误导。
    """

    model_config = ConfigDict(extra="forbid")

    rows: list[ComparisonRow] = Field(
        default_factory=list,
        description="每行一篇论文；不允许出现本次检索结果之外的 arxiv_id。",
    )
    generated_at: str = Field(default="", description="生成时间（ISO 8601）。由系统回填。")
    notes: str = Field(
        default="",
        description=(
            "对字段缺失情况的说明（哪些字段在多篇论文中普遍缺失，以及缺失原因是原文未提供）。"
            "由系统生成，模型可留空。"
        ),
    )


__all__ = [
    "ComparisonRow",
    "ComparisonTable",
    "ExtractionItem",
    "PlanOutput",
    "ReadOutput",
    "ReviseOutput",
    "VerificationItem",
    "VerifyOutput",
]

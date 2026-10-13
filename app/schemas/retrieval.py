"""检索类工具的入参 / 出参模型（Pydantic 强校验）。

**为什么检索工具一定要用 Pydantic 把入参卡死？**（对应架构文档 §10「工具侧」）

LLM 不是可靠的参数生产者：它会把 `top_k` 传成 500（一次拉爆上下文与延迟）、
把 `year_gte` 传成 1900（arXiv 1991 年才成立），把 `category` 传成它编出来的
`"cs.AI.LLM"` 这种不存在的分类。这些错误如果放任到检索层，表现是"结果莫名其妙变少"，
极难定位。把校验放在**工具边界**，错误会在第一时间以明确的报错暴露出来。

这里刻意不做「容忍式纠偏」（比如把 500 悄悄截成 50）：静默纠偏会让 LLM
永远学不会正确调用，而明确的报错能借助重试把参数修正回来。
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# arXiv 分类白名单。**只收录与本产品（AI/ML/IR/相关学科）真正相关的分类** ——
# 白名单宁窄勿宽：宽到"随便什么都能过"就等于没有校验。
# 需要新增分类时改这一处即可（例如做物理/生物方向的综述）。
ARXIV_CATEGORIES: frozenset[str] = frozenset(
    {
        # 计算机科学
        "cs.AI", "cs.CL", "cs.CV", "cs.IR", "cs.LG", "cs.NE", "cs.RO", "cs.SE",
        "cs.DB", "cs.DC", "cs.DS", "cs.IT", "cs.MA", "cs.MM", "cs.SD", "cs.CR",
        "cs.HC", "cs.GR",
        # 统计 / 数学
        "stat.ML", "stat.AP", "stat.ME", "math.OC", "math.ST", "math.NA",
        # 电气工程 / 信号
        "eess.AS", "eess.IV", "eess.SP", "eess.SY",
        # 交叉与其他常被 AI 综述引用的方向
        "q-bio.QM", "q-fin.CP", "econ.EM", "physics.data-an", "nlin.AO",
    }
)

# 检索工具的入参里，年份的下界是 arXiv 的诞生年；上界给一点余量以容忍预印本超前发布。
_MIN_YEAR = 1991
_MAX_YEAR = 2100

# 中文（CJK）字符：用来拦截"用中文关键词去打英文 BM25 索引"这一高频错误
_CJK_RE = re.compile(r"[\u3000-\u9fff]")


def normalize_english_keywords(values: list[str]) -> list[str]:
    """清洗并校验 BM25 关键词：去空白 + 拒绝中文。

    `HybridSearchArgs`（工具边界）与 `SubQuestion`（Planner 输出）共用这一条规则，
    保证「BM25 关键词必须是英文」这个不变量在两个入口都不会被绕过（架构 §4.1 注①）。
    """
    cleaned = [k.strip() for k in values if k and k.strip()]
    bad = [k for k in cleaned if _CJK_RE.search(k)]
    if bad:
        raise ValueError(
            f"BM25 关键词必须是英文术语，检测到中文：{bad}。"
            "请由 Planner 产出英文术语（含同义/缩写扩展），例如 重排 → "
            "['reranking', 'reranker', 'cross-encoder']。"
        )
    return cleaned


class HybridSearchArgs(BaseModel):
    """混合检索工具 `hybrid_search` 的入参。

    这是 Agent 与检索层之间唯一的契约，字段顺序与 `HybridRetriever.search` 对齐。
    """

    model_config = ConfigDict(extra="forbid")  # 多传字段直接报错，避免 LLM 幻觉出无用参数

    query: str = Field(
        min_length=1,
        max_length=2000,
        description="用户的自然语言查询（中文亦可，向量侧支持跨语言）",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="**英文**术语列表，供 BM25 使用（中文关键词打英文索引会失效，见架构 §4.1 注①）",
    )
    top_k: int = Field(default=8, ge=1, le=50, description="返回的论文数上限")
    year_gte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR, description="起始年份")
    year_lte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR, description="结束年份")
    category: str | None = Field(
        default=None, description="arXiv 分类，如 cs.CL；必须在白名单内"
    )
    sub_question_id: str | None = Field(
        default=None, description="所属子问题 id，便于把检索结果回填到 SSE retrieval 事件"
    )

    @field_validator("category")
    @classmethod
    def _check_category(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v not in ARXIV_CATEGORIES:
            raise ValueError(
                f"不支持的 arXiv 分类 {v!r}；请在白名单内选择，例如 cs.CL / cs.LG / cs.IR"
            )
        return v

    @field_validator("keywords")
    @classmethod
    def _reject_cjk_keywords(cls, v: list[str]) -> list[str]:
        return normalize_english_keywords(v)

    @model_validator(mode="after")
    def _check_year_range(self) -> HybridSearchArgs:
        if self.year_gte is not None and self.year_lte is not None and self.year_gte > self.year_lte:
            raise ValueError(f"year_gte({self.year_gte}) 不能大于 year_lte({self.year_lte})")
        return self

    def to_filter_kwargs(self) -> dict:
        """转成 `MetadataFilter` 的构造参数（不直接依赖检索层，保持 schemas 无副作用）。"""
        return {
            "categories": [self.category] if self.category else None,
            "year_gte": self.year_gte,
            "year_lte": self.year_lte,
        }

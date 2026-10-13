"""研究请求 / 结果相关的 Pydantic 模型。

对齐关系（便于交叉核对）：
- `ResearchRequest.filters` ↔ 架构 §7 的 `POST /research/stream` 入参；
- `SubQuestion` ↔ 架构 §4.1 的 `SubQuestion` TypedDict（Planner 的结构化输出）；
- `CitationOut` ↔ §4.1 的 `Evidence` + §10 的"引用必须能对应到真实 chunk"；
- `PaperOut` / `ChunkOut` ↔ §7 的 `GET /papers` 与引用卡片。

约定：**凡是对外暴露的模型都显式声明字段**，不使用 `extra="allow"` ——
引用卡片是要给用户看的证据，多出来的字段宁可报错也不要悄悄透给前端。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.retrieval import ARXIV_CATEGORIES, normalize_english_keywords

_MIN_YEAR = 1991
_MAX_YEAR = 2100


class ResearchFilters(BaseModel):
    """研究请求的 metadata 过滤条件（对应 `MetadataFilter`）。"""

    model_config = ConfigDict(extra="forbid")

    categories: list[str] = Field(
        default_factory=list, description="arXiv 分类白名单，如 ['cs.CL', 'cs.LG']"
    )
    year_gte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR)
    year_lte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR)

    @field_validator("categories")
    @classmethod
    def _check_categories(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if c not in ARXIV_CATEGORIES]
        if bad:
            raise ValueError(f"不支持的 arXiv 分类：{bad}")
        return v

    @model_validator(mode="after")
    def _check_year_range(self) -> ResearchFilters:
        if self.year_gte is not None and self.year_lte is not None and self.year_gte > self.year_lte:
            raise ValueError(f"year_gte({self.year_gte}) 不能大于 year_lte({self.year_lte})")
        return self


class ResearchRequest(BaseModel):
    """`POST /api/v1/research/stream` 的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, description="研究问题")
    filters: ResearchFilters | None = Field(default=None, description="语料 metadata 过滤")
    top_k: int = Field(default=8, ge=1, le=50, description="每个子问题召回的论文数上限")
    rerank: bool = Field(
        default=True,
        description="是否启用交叉编码器重排；延迟敏感场景可关闭（消融表需要量化这个权衡）",
    )


class SubQuestion(BaseModel):
    """Planner 拆解出的子问题（结构化输出，见架构 §4.1）。"""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="子问题 id，用于 SSE retrieval 事件回填")
    text: str = Field(min_length=1, description="子问题的自然语言描述")
    keywords: list[str] = Field(
        default_factory=list,
        description=(
            "★ 必须是**英文**术语（含同义/缩写扩展），供 BM25 使用。"
            "主用法是「中文提问 → 检索英文论文」，BM25 是词面匹配、中英词表不通，"
            "中文关键词会让 BM25 召回接近随机（架构 §4.1 注①）。"
        ),
    )
    category: str | None = Field(default=None, description="arXiv 分类；必须在白名单内")

    @field_validator("keywords")
    @classmethod
    def _reject_cjk_keywords(cls, v: list[str]) -> list[str]:
        """同 `HybridSearchArgs`：keywords 是给 BM25 用的，必须是英文术语。"""
        return normalize_english_keywords(v)

    @field_validator("category")
    @classmethod
    def _check_category(cls, v: str | None) -> str | None:
        if v is not None and v not in ARXIV_CATEGORIES:
            raise ValueError(f"不支持的 arXiv 分类：{v!r}")
        return v


class CitationOut(BaseModel):
    """引用证据（输出侧渲染溯源卡片用）。"""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="被支撑的论断")
    paper_id: str = Field(description="arXiv id")
    chunk_id: int = Field(description="证据所在 chunk；必须能在本次检索结果中找到（§10 反幻觉）")
    quote: str = Field(description="原文片段")
    locator: str = Field(description="章节 / 页码，便于用户回原文核对")
    source: Literal["html", "abstract"] = Field(
        default="abstract",
        description="证据强度：html=正文原文，abstract=摘要级降级证据（必须在 UI 上如实标注）",
    )
    verified: bool = Field(default=True, description="是否通过引用校验；False 表示已标记为幻觉")


class ChunkOut(BaseModel):
    """chunk 级输出（检索命中片段 / 引用卡片）。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    arxiv_id: str
    section: str = "abstract"
    chunk_type: Literal["text", "formula", "table", "figure"] = "text"
    source: Literal["html", "abstract"] = "abstract"
    page: int | None = None
    text: str
    token_count: int = 0
    score: float | None = Field(default=None, description="检索/重排分数（无则为 None）")


class PaperOut(BaseModel):
    """paper 级输出（`GET /api/v1/papers`、对比表、检索结果列表）。"""

    model_config = ConfigDict(extra="forbid")

    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    published: date | None = None
    year: int | None = None  # 便于前端直接展示与按年过滤
    primary_category: str | None = None
    venue: str | None = None
    # ⚠️ 被引数必须带"截至日期"，否则数字会随抓取时间漂移、跨论文不可比（§2.1）
    cited_by_count: int | None = None
    cited_at: date | None = None
    pdf_url: str | None = None
    has_code: bool | None = None
    score: float | None = Field(default=None, description="检索分（仅在检索结果中出现）")

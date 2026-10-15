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

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.retrieval import ARXIV_CATEGORIES, normalize_english_keywords

_MIN_YEAR = 1991
_MAX_YEAR = 2100

# F14「指定论文对比」的 id 形状：新式短 id（2305.15334），容忍 `v2` 版本后缀。
# 之所以强制短 id 而不是整条 URL：`papers` 表主键就是无版本短 id
# （理由见 arxiv_client.parse_arxiv_id），过滤最终要落到它上面。
_ARXIV_ID_RE = re.compile(r"^(\d{4}\.\d{4,5})(?:v\d+)?$")


def normalize_arxiv_ids(values: list[str]) -> list[str]:
    """校验论文 id 形状、剥离版本后缀、按序去重。

    `ResearchFilters.arxiv_ids` 与 `ResearchRequest.arxiv_ids` 两个入口共用，
    与 `normalize_english_keywords` 同一个思路：同一条不变量不允许在两处校出两种口径。
    """
    out: list[str] = []
    bad: list[str] = []
    for raw in values:
        m = _ARXIV_ID_RE.match((raw or "").strip())
        if not m:
            bad.append(raw)
            continue
        if m.group(1) not in out:
            out.append(m.group(1))
    if bad:
        raise ValueError(f"非法的 arXiv 论文 id：{bad}。形如 2305.15334（可带 v2 版本后缀）")
    return out


class ResearchFilters(BaseModel):
    """研究请求的 metadata 过滤条件（对应 `MetadataFilter`）。"""

    model_config = ConfigDict(extra="forbid")

    categories: list[str] = Field(
        default_factory=list, description="arXiv 分类白名单，如 ['cs.CL', 'cs.LG']"
    )
    year_gte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR)
    year_lte: int | None = Field(default=None, ge=_MIN_YEAR, le=_MAX_YEAR)
    has_code: bool | None = Field(
        default=None,
        description=(
            "是否有开源仓库（F2 第三类过滤）：true=只召回 arXiv 元数据里**确认**有仓库链接的论文；"
            "false=只召回未确认有链接的论文；不传=不过滤。口径见 `HybridSearchArgs.has_code`"
        ),
    )
    arxiv_ids: list[str] = Field(
        default_factory=list,
        description="F14：把研究限定在这几篇论文里（形如 2305.15334，可带版本后缀，会被剥离）",
    )

    @field_validator("categories")
    @classmethod
    def _check_categories(cls, v: list[str]) -> list[str]:
        bad = [c for c in v if c not in ARXIV_CATEGORIES]
        if bad:
            raise ValueError(f"不支持的 arXiv 分类：{bad}")
        return v

    @field_validator("arxiv_ids")
    @classmethod
    def _check_arxiv_ids(cls, v: list[str]) -> list[str]:
        return normalize_arxiv_ids(v)

    @model_validator(mode="after")
    def _check_year_range(self) -> ResearchFilters:
        if self.year_gte is not None and self.year_lte is not None and self.year_gte > self.year_lte:
            raise ValueError(f"year_gte({self.year_gte}) 不能大于 year_lte({self.year_lte})")
        return self


class ResearchRequest(BaseModel):
    """`POST /api/v1/research/stream` 的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, description="研究问题")
    # ★ F9「会话内追问」：可选的会话加入凭证。
    # 传了 → 本次提问挂进这个会话，能读到该会话此前的问答（短期记忆）；
    # 不传 → 与改动前完全一致（`session_id = run_id`，一次提问即一个会话）。
    # 字符集与 `guards.RUN_ID_RE` 同口径：这个值会直接当 `sessions.id` 主键写库，
    # 裸 `/`、空白、控制字符进主键和外键是给自己找麻烦（路径穿越式输入尤其）。
    session_id: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description="要加入的会话 id（追问时传上一轮 `run_started` 所在会话的 id）；留空表示新开会话",
    )
    filters: ResearchFilters | None = Field(default=None, description="语料 metadata 过滤")
    # F14「指定 2~3 篇论文只做对比」：顶层便利入口，值会自动并进 `filters.arxiv_ids`。
    # 为什么要并进 filters 而不是单独再传一条通道：API 层只把 `filters.model_dump()`
    # 喂进图 state（`_initial_state`），检索/抓取两侧都从那里取 —— 并进既有的
    # 过滤通道，两处下游无需感知新顶层字段。
    arxiv_ids: list[str] = Field(
        default_factory=list,
        description="限定研究的论文 id 列表（形如 2305.15334）；非空时检索只在这些论文里做",
    )
    language: Literal["zh", "en"] = Field(
        default="zh",
        description="研究报告的输出语言。zh=中文（默认，主用法「中文提问→检索英文论文」），en=英文",
    )
    top_k: int = Field(default=8, ge=1, le=50, description="每个子问题召回的论文数上限")
    rerank: bool = Field(
        default=True,
        description="是否启用交叉编码器重排；延迟敏感场景可关闭（消融表需要量化这个权衡）",
    )

    @field_validator("session_id", mode="before")
    @classmethod
    def _blank_session_is_new_session(cls, v: object) -> object:
        """`""` / 全空白 = "不加入任何会话"，而不是形状错误。

        面板类客户端常常是"有会话就带上、没有就传空串"，把空串判成 422 会让
        首次提问（新会话）直接失败，而这个请求本身完全合法。
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("arxiv_ids")
    @classmethod
    def _check_arxiv_ids(cls, v: list[str]) -> list[str]:
        return normalize_arxiv_ids(v)

    @model_validator(mode="after")
    def _merge_arxiv_ids_into_filters(self) -> ResearchRequest:
        """把顶层 `arxiv_ids` 并进 `filters.arxiv_ids`（两处都给时取并集，保序）。

        没有这一步，顶层字段会在 `_initial_state` 的 `filters.model_dump()` 处丢失 ——
        请求侧承诺的 F14 就又回到"检索层支持但传不进去"的原状。
        """
        if self.arxiv_ids:
            if self.filters is None:
                self.filters = ResearchFilters()
            merged = normalize_arxiv_ids([*self.filters.arxiv_ids, *self.arxiv_ids])
            self.filters.arxiv_ids = merged
        return self


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
    source: Literal["html", "abstract", "pdf"] = Field(
        default="abstract",
        description=(
            "证据强度：html=正文原文，abstract=摘要级降级证据（必须在 UI 上如实标注），"
            "pdf=用户上传的全文原文（`POST /papers/upload` 产出的片段，见 `app/ingest/pdf.py`）"
        ),
    )
    verified: bool = Field(default=True, description="是否通过引用校验；False 表示已标记为幻觉")


class ChunkOut(BaseModel):
    """chunk 级输出（检索命中片段 / 引用卡片）。"""

    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    arxiv_id: str
    section: str = "abstract"
    chunk_type: Literal["text", "formula", "table", "figure"] = "text"
    source: Literal["html", "abstract", "pdf"] = "abstract"
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

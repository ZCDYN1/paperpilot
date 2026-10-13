"""ORM 模型：12 张表。

几个刻意的设计：

- **`citations` 单独建表**：让「引用可溯源」变成*可查询、可校验的事实*，
  而不只是 prompt 里的一句约束。Verifier 的校验结果直接写回 `verified` 字段，
  于是"引用准确率"这个指标可以零成本自动统计。
- **`paper_chunks.source`**：值只有 `html` / `abstract` 两种。arXiv 官方 HTML 取不到的旧论文
  只能给到摘要级证据，引用卡片上必须能一眼看出证据强度，不能让用户误以为是正文原文。
- **`research_runs.langgraph_state` 只存轻量摘要**：全量 state 里含 retrieved / readings 全文，
  整包塞进 SQLite 会让库迅速膨胀。完整状态由 LangGraph 的 checkpointer 负责。
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin


# =============================================================================
# 语料层
# =============================================================================
class Paper(Base):
    """论文元数据。

    文本字段（标题 / 摘要 / 作者 / 分类）**只采信 arXiv 官方**；
    第三方源（OpenAlex）只允许补数字字段，且必须交叉校验后才写入。
    """

    __tablename__ = "papers"

    arxiv_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str] = mapped_column(Text, default="", nullable=False)
    authors: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    published: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    updated: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    primary_category: Mapped[str | None] = mapped_column(String(32), index=True)
    categories: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)          # arXiv comment 字段（抽 venue 用）
    journal_ref: Mapped[str | None] = mapped_column(Text)
    # 以下字段来自 comment 抽取 / 第三方补充
    venue: Mapped[str | None] = mapped_column(String(128))
    cited_by_count: Mapped[int | None] = mapped_column(Integer)
    cited_at: Mapped[date | None] = mapped_column(Date)        # ⚠️ 被引数必须带"截至日期"
    # OpenAlex 补充的其余数字/状态字段（同样只补数字，不补文本）
    referenced_works_count: Mapped[int | None] = mapped_column(Integer)
    is_open_access: Mapped[bool | None] = mapped_column(Boolean)
    oa_status: Mapped[str | None] = mapped_column(String(32))
    pdf_url: Mapped[str | None] = mapped_column(Text)
    html_url: Mapped[str | None] = mapped_column(Text)
    has_code: Mapped[bool | None] = mapped_column(Boolean)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[PaperChunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", lazy="selectin"
    )


class PaperChunk(Base):
    """分块正文（检索的最小单位）。

    检索与融合在 **chunk 级**进行，展示与评测在 **paper 级**进行，
    两者之间需要一次显式的聚合（见 retrieval/hybrid_retriever.py）。
    """

    __tablename__ = "paper_chunks"
    __table_args__ = (
        UniqueConstraint("arxiv_id", "chunk_index", name="uq_paper_chunks_arxiv_chunk"),
        Index("ix_paper_chunks_section", "section"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(
        ForeignKey("papers.arxiv_id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section: Mapped[str] = mapped_column(String(128), default="abstract", nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    chunk_type: Mapped[str] = mapped_column(String(32), default="text", nullable=False)  # text|formula|table|figure
    source: Mapped[str] = mapped_column(String(16), default="abstract", nullable=False)  # html|abstract
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class Category(Base):
    """arXiv 分类树（用于前端过滤下拉）。"""

    __tablename__ = "categories"

    code: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    parent_code: Mapped[str | None] = mapped_column(String(32))


# =============================================================================
# 会话与运行
# =============================================================================
class ResearchSession(Base, TimestampMixin):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(256))
    user_fingerprint: Mapped[str | None] = mapped_column(String(128), index=True)

    runs: Mapped[list[ResearchRun]] = relationship(back_populates="session")


class ResearchRun(Base):
    """一次研究运行，与 LangGraph 的 thread 一一对应。"""

    __tablename__ = "research_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    thread_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True, nullable=False)
    langgraph_state: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)  # 轻量摘要
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    session: Mapped[ResearchSession] = relationship(back_populates="runs")


class Message(Base):
    """短期记忆：会话内的消息流水。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)      # user|assistant|system|tool
    content: Mapped[str] = mapped_column(Text, nullable=False)
    node_name: Mapped[str | None] = mapped_column(String(32))          # 由哪个节点产生
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class LongTermMemory(Base):
    """长期记忆：向量化的用户偏好 / 已读论文清单。

    必须有 importance + last_used_at 并做淘汰，否则越用越慢、越用越吵。
    """

    __tablename__ = "long_term_memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str | None] = mapped_column(String(64), index=True)
    memory_type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)  # direction|read_paper|preference
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# =============================================================================
# 证据与观测
# =============================================================================
class Citation(Base):
    """引用证据链 —— 「引用可溯源」这一核心承诺的落地载体。"""

    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    claim: Mapped[str] = mapped_column(Text, nullable=False)            # 待验证的论断
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("paper_chunks.id", ondelete="SET NULL"), index=True
    )
    arxiv_id: Mapped[str | None] = mapped_column(String(64), index=True)
    quote: Mapped[str] = mapped_column(Text, default="", nullable=False)   # 原文片段
    locator: Mapped[str | None] = mapped_column(String(128))               # 章节/页码
    source: Mapped[str | None] = mapped_column(String(16))                 # html|abstract
    verified: Mapped[bool | None] = mapped_column(Boolean, index=True)     # None=未校验
    note: Mapped[str | None] = mapped_column(Text)                         # 校验说明
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ToolCallLog(Base):
    """工具调用审计。"""

    __tablename__ = "tool_call_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    node_name: Mapped[str | None] = mapped_column(String(32))
    tool_name: Mapped[str] = mapped_column(String(64), nullable=False)
    args: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="ok", nullable=False)  # ok|error|timeout
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class TokenUsage(Base):
    """Token 用量与成本，按节点聚合。"""

    __tablename__ = "token_usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    node_name: Mapped[str | None] = mapped_column(String(32), index=True)
    model: Mapped[str | None] = mapped_column(String(64))
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_estimate: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# =============================================================================
# 评测
# =============================================================================
class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    dataset: Mapped[str | None] = mapped_column(String(128))
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    samples: Mapped[list[EvalSample]] = relationship(
        back_populates="eval_run", cascade="all, delete-orphan"
    )


class EvalSample(Base):
    __tablename__ = "eval_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    eval_run_id: Mapped[int] = mapped_column(
        ForeignKey("eval_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    contexts: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    scores: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    eval_run: Mapped[EvalRun] = relationship(back_populates="samples")


__all__ = [
    "Paper",
    "PaperChunk",
    "Category",
    "ResearchSession",
    "ResearchRun",
    "Message",
    "LongTermMemory",
    "Citation",
    "ToolCallLog",
    "TokenUsage",
    "EvalRun",
    "EvalSample",
]

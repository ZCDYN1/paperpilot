"""ORM 模型：15 张表。

几个刻意的设计：

⚠️ **改这张文件前先读：只允许新建表，不允许给已有表加列/改列。**
   SQLite 下 `Base.metadata.create_all` 只补建**缺失的表**，不会 ALTER 已存在的表 ——
   所以给老表加一列的后果是：新 clone 的库正常，任何已有 `data/sqlite/paperpilot.db`
   的机器上那一列永远不存在，而 ORM 读它时才炸。本项目已经真实踩过一次
   （`comparison_rows` 表建得比代码晚，老库里没有它，`finalize` 的落库整段被
   `try/except` 吞掉，表现成"对比表静默消失"）。要加字段就**新建一张附表**。

- **`citations` 单独建表**：让「引用可溯源」变成*可查询、可校验的事实*，
  而不只是 prompt 里的一句约束。Verifier 的校验结果直接写回 `verified` 字段，
  于是"引用准确率"这个指标可以零成本自动统计。
- **`comparison_rows` 单独建表**：对比表此前只活在 graph state 与 SSE 帧里，
  导致「对比表字段完整率 ≥ 80%」（产品定义 §8）没有可测的数据，
  旧 run 的表格也取不回来（`GET /research/{run_id}` 读的是 checkpointer，thread 一清就没了）。
  **每行一篇论文、与 `ComparisonRow` 字段一一对应**（而不是 paper×field 长表）：
  产品契约的字段集是固定的 12 列（`extra="forbid"`），宽表取回整表无需 pivot，
  完整率就是逐列一次 COUNT —— 长表的两项好处（字段可增删、按字段自由聚合）
  在这里都用不上，反而把"schema 与 pydantic 模型必须同步"这条约束藏起来。
  空值语义照抄产品承诺：**空串 / NULL = 「没找到依据」的刻意留空，不是待补的数据**。
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
class ComparisonRecord(Base):
    """结构化对比表的一行 = **一篇论文**（与 `app/agents/schemas.ComparisonRow` 同形）。

    ⚠️ 空值的产品语义（`ComparisonRow` 三条不可协商的规则）：
    **空串 / None 表示「原文里没找到依据」的刻意留空，不是待补全的缺失数据**。
    因此这里的统计口径只数"填上了多少"，绝不给空值兜底默认 ——
    任何"把空字段填个占位值再算完整率"的做法都在推翻这张表的可信度承诺。

    `run_id` 不建外键（与 `citations` 同口径）：收尾写入发生在 run 登记之后、
    且 run 行可能被其它路径改写，FK 只会把"落库失败"的爆炸半径扩大，没有额外约束收益。
    也不建 `(run_id, arxiv_id)` 唯一约束：arxiv_id 理论可为空串，两行空串会撞约束，
    而约束冲突抛错的代价是**整张表丢失**（finalize 的每步副作用都单独 try/except）。
    幂等性由写入方保证：`finalize._persist` 先删本 run 旧行再插新行。
    """

    __tablename__ = "comparison_rows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    # 表内行序（compare 节点按精读顺序产出）。恢复旧表需要稳定顺序；
    # 用自增 id 排序也行，但 id 会随"删旧插新"的幂等重写重新分配，row_index 不会。
    row_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    arxiv_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False, default="")
    title: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # ---- 逐字段回原文校验的自由文本列（找不到依据即留空）----
    method_name: Mapped[str] = mapped_column(Text, default="", nullable=False)
    core_idea: Mapped[str] = mapped_column(Text, default="", nullable=False)
    datasets: Mapped[str] = mapped_column(Text, default="", nullable=False)
    improvement: Mapped[str] = mapped_column(Text, default="", nullable=False)
    cost: Mapped[str] = mapped_column(Text, default="", nullable=False)
    limitation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # ---- 代码权威覆盖的列（模型说了不算，见 comparator._row_from）----
    code_url: Mapped[str] = mapped_column(Text, default="", nullable=False)
    venue: Mapped[str] = mapped_column(Text, default="", nullable=False)
    # 被引数与截至日期**成对出现**： comparator 在没有 cited_at 时把数字也清空
    # （孤零零的 "0" 会被误读成"没人引用"），所以这两列可空，None = 留空。
    cited_by_count: Mapped[int | None] = mapped_column(Integer)
    cited_at: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


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
    # 引用定位的可观测面（此前只存在于后端日志 read.done 的 reanchored/recovered_by 里）：
    # resolution_how = read 侧走通的路径（exact | quote | swapped_chunk），
    # verify_reanchored = verify 硬校验时是否又重锚过一次。
    # 两列都可空：迁移前的历史行保持 NULL（= 当时没记录），与"记录了且为 exact/False"区分开。
    resolution_how: Mapped[str | None] = mapped_column(String(16))
    verify_reanchored: Mapped[bool | None] = mapped_column(Boolean)
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


# =============================================================================
# 引用关系与被引趋势（产品定义 F16 / F19）
# =============================================================================
class CitationEdge(Base):
    """「谁引用了谁」的一条边。**只存证据，不存推断。**

    来源（`source`）目前只有一种真实可用的：OpenAlex work 的 `referenced_works`
    列表（论文自己参考文献的机器可读版）。⚠️ 刻意**不允许**用"年份先后 + 主题相似"
    造边 —— F12 的"谁补了谁的坑"一旦建立在推断上，就重新踩回产品定义 §7
    承诺不做的事：把没有依据的叙事写成看起来通顺的结论。
    库里查不到的被引方也存（`in_corpus=False`），它至少说明"这篇引了谁"；
    但画图和推演进关系时前端必须区分：非库内节点没有正文证据可点。
    """

    __tablename__ = "citation_edges"
    __table_args__ = (
        # 同一对论文从同一来源重复抓取是常态（每次 enrich 都会再拿到一遍），
        # 靠唯一约束做幂等，而不是靠"先查再插"（并发下会双插）。
        UniqueConstraint(
            "citing_arxiv_id", "cited_arxiv_id", "source", name="uq_citation_edge"
        ),
        Index("ix_citation_edges_cited", "cited_arxiv_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    citing_arxiv_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    cited_arxiv_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="openalex", nullable=False)
    # 两端的 OpenAlex id：OpenAlex 的记录会被合并/重定向，只留 arXiv id 就没法复查
    citing_openalex_id: Mapped[str | None] = mapped_column(String(32))
    cited_openalex_id: Mapped[str | None] = mapped_column(String(32))
    in_corpus: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CitationSnapshot(Base):
    """某篇论文在**某一天**的被引数。F19「被引趋势」的唯一诚实来源。

    为什么单独建表而不是给 `papers` 加列：一列只能存最新值，历史就没了；
    而"趋势"的定义本身就是多个采样点。`papers.cited_at` 已经存了最近一次的日期，
    这张表补的是它之前的整条序列。
    ⚠️ 只有 1 个采样点时**不许**报趋势（代码里判的就是 `len(points) >= 2`），
    报出来只会是一个被随机波动带偏的斜率。
    """

    __tablename__ = "citation_snapshots"
    __table_args__ = (
        # 一天一条：同一天重复抓只保留最新的那次数字
        UniqueConstraint("arxiv_id", "taken_on", name="uq_citation_snapshot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    arxiv_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    cited_by_count: Mapped[int] = mapped_column(Integer, nullable=False)
    taken_on: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    source: Mapped[str] = mapped_column(String(32), default="openalex", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


__all__ = [
    "Paper",
    "PaperChunk",
    "Category",
    "ResearchSession",
    "ResearchRun",
    "Message",
    "LongTermMemory",
    "Citation",
    "CitationEdge",
    "CitationSnapshot",
    "ToolCallLog",
    "TokenUsage",
    "EvalRun",
    "EvalSample",
]

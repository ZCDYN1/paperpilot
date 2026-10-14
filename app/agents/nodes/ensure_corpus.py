"""`ensure_corpus` 节点 —— 语料保障（架构 §2.2 的①阶段）。

**这个节点存在的理由**：用户问的是任意研究方向，本地索引不可能预先覆盖。
没有它，第一次问某个方向时检索必然返回空，Agent 会在空索引上"硬编答案" ——
那正是产品最反对的行为。所以检索之前必须先把语料补齐。

三条不可混淆的边界：

1. **串行限速**：抓 arXiv 必须 ≥3s/请求。这个节点是**串行**的，
   与②阶段（本地索引并行检索）是两套节奏（ADR D11）。
2. **缓存优先**：`papers` 表按 `arxiv_id` 去重，同一方向第二次提问基本命中缓存、
   秒级完成。这是"第二次体验远好于第一次"的全部实现，也是省限速配额的关键。
3. **不足则如实降级**：命中数 < `min_corpus` 时把 `corpus_ready=False` 交出去，
   由 `route_after_corpus` 走"语料不足"分支，**显式声明检索覆盖度**，
   而不是假装检索到了东西。

4. **离线可关**：`INGEST_ENABLED=false` 时本节点完全不联网，只用库里已有语料，
   并把覆盖度如实交出去 —— 供离线演示、CI 与"我只想要本地索引"的部署使用。
"""

from __future__ import annotations

import structlog
from sqlalchemy import select

from app.agents.nodes._common import node_event
from app.agents.state import STAGE_CORPUS, ResearchState
from app.agents.tools.hybrid_search import get_shared_retriever, reset_shared_retriever
from app.core.config import get_settings
from app.db.models import Paper, PaperChunk
from app.db.session import init_db, session_scope
from app.retrieval.hybrid_retriever import ChunkRecord

logger = structlog.get_logger(__name__)


async def load_corpus() -> tuple[list[ChunkRecord], dict[str, str]]:
    """把库里全部 chunk 读成检索层视图（与 `scripts/build_index.py` 同一口径）。

    这里刻意与 `scripts/build_index.py` 保持同构：索引是 DB 的**派生物**，
    两条重建路径必须得到一模一样的结果，否则"手动重建能修好、自动重建修不好"
    这种问题会浪费掉一整个下午。
    """
    async with session_scope() as session:
        papers = (await session.execute(select(Paper))).scalars().all()
        titles = {p.arxiv_id: p.title for p in papers}
        years = {p.arxiv_id: (p.published.year if p.published else None) for p in papers}
        categories = {p.arxiv_id: p.primary_category for p in papers}
        has_codes = {p.arxiv_id: p.has_code for p in papers}
        rows = (
            (await session.execute(select(PaperChunk).order_by(PaperChunk.id))).scalars().all()
        )

    records = [
        ChunkRecord(
            chunk_id=row.id,
            arxiv_id=row.arxiv_id,
            section=row.section,
            text=row.text,
            source=row.source,
            chunk_type=row.chunk_type,
            year=years.get(row.arxiv_id),
            primary_category=categories.get(row.arxiv_id),
            page=row.page,
            # 没有这一行，has_code 过滤在"节点重建的索引"上永远查不到真阳性 ——
            # DB 里有值但没进索引元数据，又是一类"字段存在却没接通"的静默缺口
            has_code=has_codes.get(row.arxiv_id),
        )
        for row in rows
    ]
    return records, titles


def build_ingest_query(state: ResearchState) -> str:
    """用子问题的英文关键词拼一个 arXiv 检索式。

    为什么用关键词而不是把中文问题直接丢给 arXiv：arXiv 的检索是**词面匹配**，
    中文 query 在它的索引里命不中英文论文（与 BM25 同一个道理，架构 §4.1 注①）。
    关键词是 Planner 产出的英文术语，正好可以直接用。
    """
    kws: list[str] = []
    for sq in state.get("sub_questions") or []:
        for kw in sq.get("keywords") or []:
            if kw and kw not in kws:
                kws.append(kw)
    if not kws:
        return state.get("question", "").strip() or "retrieval augmented generation"
    # arXiv API 用 `OR` 连接同义扩展，避免用 AND 把召回卡死（同义词之间通常互斥）
    return " OR ".join(kws[:8])


def collect_categories(state: ResearchState) -> list[str] | None:
    """分类约束：子问题分类 ∪ API 请求 filters 里的全局分类（去重）。

    并集是安全的：arXiv 查询侧的 `cat:` 子句是 OR 集合（见 arxiv_client._build_query），
    两个来源的分类不互相排斥。此前只取子问题分类，请求侧 filters.categories
    在**抓取环节**被整个忽略 —— 抓回来的语料与过滤条件无关，白耗限速配额。
    `None` 表示不限制分类。
    """
    cats = {sq.get("category") for sq in (state.get("sub_questions") or []) if sq.get("category")}
    cats |= {c for c in (state.get("filters") or {}).get("categories") or [] if c}
    return sorted(cats) or None


def collect_ingest_params(state: ResearchState) -> dict:
    """把请求 filters 翻成 `ingest()` 的抓取参数（F2 承诺在抓取侧落地）。

    - 年份区间走 arXiv 的 `submittedDate:[...]`，能**真正减少抓取量**（限速环境下最值钱的优化）；
    - `has_code` 不在这张表上：它不是 arXiv 查询接口的可过滤字段，只能在入库时检测元数据、
      在检索时由 `_prefilter` 兑现 —— 为此把抓取改成"多抓再筛"反而更烧配额。
    仍不引入任何并发：抓取全程串行（§2.2 / D11），这里只是收窄同一个串行请求的条件。
    """
    filters = state.get("filters") or {}
    return {
        "categories": collect_categories(state),
        "year_from": filters.get("year_gte"),
        "year_to": filters.get("year_lte"),
    }


async def ensure_corpus_node(state: ResearchState) -> dict:
    """保障本地语料齐备，并把索引刷新到与 DB 一致。"""
    settings = get_settings()
    await init_db()

    records, titles = await load_corpus()
    fetched = cached = 0
    corpus_error: str | None = None

    # F14「定向对比模式」：请求点名了具体论文（filters.arxiv_ids 由 ResearchRequest
    # 的顶层同名字段并进，见 app/schemas/research.py 的 `_merge_arxiv_ids_into_filters`）。
    # 这种模式下没有"再按关键词抓一批"的道理：抓回来的语料既进不了对比表
    # （检索侧已被 arxiv_ids 限死），又要吃 arXiv 的串行限速配额、把"3 分钟"拖长。
    # 点名的论文不在库里就如实检索不到 —— 留空，不猜、不拿相近的论文顶替。
    target_ids = list((state.get("filters") or {}).get("arxiv_ids") or [])
    if target_ids:
        logger.info(
            "ensure_corpus.directed_compare",
            targeted=len(target_ids),
            local=len(titles),
            hint="只在这几篇已入库的论文里做对比，跳过关键词抓取",
        )
    elif len(titles) < settings.min_corpus and not settings.ingest_enabled:
        # INGEST_ENABLED=false：离线/CI 模式。不联网，只用库里已有的东西，
        # 并按同一口径如实声明覆盖度（corpus_ready 由下面的阈值判断给出）。
        logger.info("ensure_corpus.ingest_disabled", local=len(titles), need=settings.min_corpus)
    elif len(titles) < settings.min_corpus:
        # 语料不足 → 联网抓取。失败**不中断**：本地已有多少就用多少，
        # 只是把 corpus_ready 置 False，让下游如实声明覆盖度（架构 §2.2 兜底）。
        query = build_ingest_query(state)
        logger.info("ensure_corpus.ingest_start", query=query[:80], local=len(titles))
        try:
            from app.ingest.pipeline import ingest

            result = await ingest(
                query,
                **collect_ingest_params(state),
                max_results=settings.arxiv_max_results,
                settings=settings,
            )
            records, titles = result.records, result.titles
            fetched = result.stats.fetched
            cached = result.stats.cached
        except Exception as exc:  # noqa: BLE001 - 抓取失败必须降级而不是中断全流程
            corpus_error = str(exc)[:200]
            logger.warning("ensure_corpus.ingest_failed", error=corpus_error)

    # 索引刷新：**内容指纹没变就复用，变了必然重建**。
    # 为什么不是"索引非空就跳过"：embedding 模型、分块策略、chunk 内容任一变化，
    # 旧索引都会静默失真（不报错，只是结果变差）—— 这类"静默陈旧"是本项目反复出现的风险。
    # 为什么也不是"无条件重建"（本节点此前的做法）：那等于每次提问都把全部 chunk
    # 重新 encode 一遍，本地 BGE 下以分钟计，直接把验收标准"单次研究 < 3 分钟"打掉。
    # 指纹（corpus_digest）把两种失败模式同时堵住：内容没变 → 秒级跳过；内容变了 → 必然重建。
    retriever = get_shared_retriever()
    if retriever.index_is_current(records, titles):
        logger.info("ensure_corpus.index_reused", chunks=len(records))
        index_action = "reused"
    else:
        await retriever.build_index(records, titles)
        reset_shared_retriever()  # 索引已换，丢弃可能持有旧索引的引用
        index_action = "rebuilt"

    corpus_ready = len(titles) >= settings.min_corpus
    stats = {
        "fetched": fetched,
        "cached": cached,
        "total_papers": len(titles),
        "chunks": len(records),
        "corpus_ready": corpus_ready,
        "threshold": settings.min_corpus,
        "index": index_action,
        "error": corpus_error,
    }
    if not corpus_ready:
        logger.warning("ensure_corpus.insufficient", found=len(titles), need=settings.min_corpus)

    return {
        "corpus_ready": corpus_ready,
        "corpus_stats": stats,
        "stage": STAGE_CORPUS,
        "events": [
            node_event(
                "corpus",
                fetched=fetched,
                cached=cached,
                total=len(titles),
                corpus_ready=corpus_ready,
                chunks=len(records),
                threshold=settings.min_corpus,
                index=index_action,
            )
        ],
    }


__all__ = [
    "build_ingest_query",
    "collect_categories",
    "collect_ingest_params",
    "ensure_corpus_node",
    "load_corpus",
]

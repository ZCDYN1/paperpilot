"""入库编排：检索 arXiv → 抓 HTML 正文 → 分块 → 落库 → 产出可建索引的数据。

这一层是架构文档 §2.2「语料供给链路」的①阶段：**串行、受 arXiv 限速约束**。
它跟②阶段（本地索引检索）的并行是两套节奏，边界就画在这里 —— pipeline 内部
对 arXiv 只有串行请求，绝不并发。

几个刻意的设计：

- **缓存优先**：`papers` 主键是短 arxiv_id，已入库且有 chunk 的论文直接读库、
  一个网络请求都不发。这是"第二次问同一方向秒级完成"的全部实现，也是省限速配额的关键。
  注意这里返回的是**全量** records（缓存 + 新增），否则 `build_index` 重建索引时
  会把缓存论文从索引里抹掉 —— 那是个"第一次好好的、第二次越用越少"的隐蔽 bug。
- **单篇失败不中断**：抓 HTML / 分块 / 落库任一步出错，只把该篇计入 failed 继续跑。
  一篇论文挂了就放弃整次检索，产品体验上等于不可用。
- **网络阶段与数据库事务分离**：先抓完再一次性写库，避免在长事务里等网络，
  否则会长时间持有 SQLite 写锁，与 LangGraph checkpointer / 其他分支互相阻塞。
- **落库字段的来源口径**：文本字段来自 arXiv 官方，被引数来自 OpenAlex
  （且已交叉校验），发表会议与「是否有代码」从 `comment`/`abstract` 正则抽取（ADR D9/D7）。
"""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

import httpx
import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.db.models import Paper, PaperChunk
from app.db.session import init_db, session_scope
from app.ingest.arxiv_client import ArxivClient, ArxivPaper
from app.ingest.html_extractor import fetch_sections
from app.ingest.openalex_client import OpenAlexClient, OpenAlexMetrics
from app.retrieval.chunking import Chunk, chunk_paper, chunks_from_abstract
from app.retrieval.hybrid_retriever import CODE_REPO_HOSTS, ChunkRecord

logger = structlog.get_logger(__name__)

# 从 arXiv comment 里抽发表会议（ADR D9）。comment 是作者自填的自由文本，
# 只认两种高置信写法，其余一律留空 —— 抽错会议名比留空更坑，
# 因为它会被写进对比表当成"已核实的事实"。
#   ① 有动词引导："Accepted at NeurIPS 2020"、"To appear in ACL 2024"
#   ② 无动词但有"全大写缩写 + 年份"："EMNLP 2026 Findings"
_VENUE_VERB_RE = re.compile(
    r"\b(?:accepted|published|to\s+appear|forthcoming)\b[^,;.]{0,20}?\s*(?:at|in|to)\s+(?:the\s+)?"
    r"([A-Z0-9][\w&'\-]*(?:\s+(?:of|on|for|and|&|[A-Z0-9][\w&'\-]*)){0,5})",
    re.I,  # comment 里 "Accepted"/"ACCEPTED"/"accepted" 混用，必须忽略大小写
)
_VENUE_ACRONYM_RE = re.compile(
    r"\b([A-Z]{2,8}\s+(?:19|20)\d{2}(?:\s+(?:Findings|Workshop|Track|Proceedings|Conference|Symposium|Journal))?)\b",
)


def extract_venue(comment: str | None) -> str | None:
    """从 comment 抽取发表会议。抽不到就返回 None（留空，绝不编造）。"""
    if not comment:
        return None
    for pattern in (_VENUE_VERB_RE, _VENUE_ACRONYM_RE):
        m = pattern.search(comment)
        if not m:
            continue
        venue = re.sub(r"\s+", " ", m.group(1)).strip(" .,;")
        # 太长基本是正则跑偏吞进了整句话，宁可不要
        if 2 <= len(venue) <= 100:
            return venue
    return None


# 「是否有代码」（F2 第三类过滤）只从 arXiv 官方元数据判定（ADR D7：文本字段只采信
# arXiv 官方；D9 同款思路——comment/abstract 是作者自填、arXiv 托管的自由文本）。
# 域名清单引用 `hybrid_retriever.CODE_REPO_HOSTS`（单一事实源，与 comparator 的
# code_url 抽取共用；此前两边各写一份清单，codeberg.org 只进了一边就开始漂移）。
# 这里保留更严格的判定口径：**域名后必须跟 `/路径`** —— 裸域名（如行文里的
# "GitHub.com 是托管平台"）不算仓库链接，误判成 True 会被"仅有代码"过滤当成真阳性，
# 比漏判更坑。与 comparator 的唯一差异是检测侧不要求 `https?://` 前缀：comment 里
# 作者常裸写 "gitlab.com/a/b"（既有测试钉住这一形态），命中的域名仍走同一份清单。
# 识别不到返回 None 而不是 False：元数据里没有链接 ≠ 论文没有代码，不许编造；
# "未见链接"这一保守口径由检索侧 `_prefilter` 负责解释（见 MetadataFilter.has_code）。
_CODE_URL_RE = re.compile(
    rf"(?:{CODE_REPO_HOSTS})/",
    re.I,  # comment 里链接大小写混用（GitHub/GITHUB）
)


def detect_has_code(comment: str | None, abstract: str | None) -> bool | None:
    """从 arXiv 官方元数据识别开源仓库链接。检测到返回 True，否则 None。"""
    for text in (comment, abstract):
        if text and _CODE_URL_RE.search(text):
            return True
    return None


@dataclass
class IngestStats:
    """一次入库的统计。命令行与上层（`ensure_corpus` 节点）都用它做状态展示。"""

    fetched: int = 0        # 本轮新抓取并入库存
    cached: int = 0         # 命中缓存、未发网络请求
    failed: int = 0
    html_papers: int = 0    # 拿到官方 HTML 正文（章节级证据）
    abstract_papers: int = 0  # 降级为摘要块（摘要级证据）
    chunks: int = 0
    elapsed_s: float = 0.0
    errors: list[dict] = field(default_factory=list)


@dataclass
class IngestResult:
    """入库结果 —— 直接喂给 `HybridRetriever.build_index(records, titles)`。"""

    records: list[ChunkRecord]
    titles: dict[str, str]
    corpus_ready: bool
    stats: IngestStats

    @property
    def paper_count(self) -> int:
        return len(self.titles)

    def build_index_args(self) -> tuple[list[ChunkRecord], dict[str, str]]:
        """按 build_index 的入参顺序拆包，省得调用方每次都写一遍。"""
        return self.records, self.titles


# =============================================================================
# 缓存读取
# =============================================================================
async def _load_cache(
    arxiv_ids: list[str],
) -> tuple[list[ChunkRecord], dict[str, str], set[str]]:
    """读出已入库且有 chunk 的论文。

    返回 `(records, titles, cached_ids)`。**只有确实有 chunk 的才算缓存命中** ——
    上次跑到一半失败留下的空 Paper 行必须重新抓，否则那条记录会永久没有正文。
    """
    if not arxiv_ids:
        return [], {}, set()

    async with session_scope() as session:
        papers = (
            await session.execute(select(Paper).where(Paper.arxiv_id.in_(arxiv_ids)))
        ).scalars().all()
        paper_map = {p.arxiv_id: p for p in papers}
        if not paper_map:
            return [], {}, set()

        chunks = (
            await session.execute(
                select(PaperChunk)
                .where(PaperChunk.arxiv_id.in_(list(paper_map)))
                .order_by(PaperChunk.arxiv_id, PaperChunk.chunk_index)
            )
        ).scalars().all()

    by_paper: dict[str, list[PaperChunk]] = defaultdict(list)
    for row in chunks:
        by_paper[row.arxiv_id].append(row)

    records: list[ChunkRecord] = []
    titles: dict[str, str] = {}
    cached_ids: set[str] = set()
    for arxiv_id, rows in by_paper.items():
        paper = paper_map[arxiv_id]
        year = paper.published.year if paper.published else None
        for row in rows:
            records.append(
                ChunkRecord(
                    chunk_id=row.id,
                    arxiv_id=arxiv_id,
                    section=row.section,
                    text=row.text,
                    source=row.source,
                    chunk_type=row.chunk_type,
                    year=year,
                    primary_category=paper.primary_category,
                    page=row.page,
                    # 缓存论文沿用入库时已判定的 has_code（同一口径，不在读侧重新检测）
                    has_code=paper.has_code,
                )
            )
        titles[arxiv_id] = paper.title
        cached_ids.add(arxiv_id)

    logger.info("ingest.cache_hit", cached=len(cached_ids), chunks=len(records))
    return records, titles, cached_ids


# =============================================================================
# 抓正文 + 分块
# =============================================================================
async def _prepare_chunks(
    paper: ArxivPaper,
    *,
    use_html: bool,
    client: httpx.AsyncClient | None,
    settings: Settings,
) -> list[Chunk]:
    """产出这篇论文的 chunk：优先 HTML 章节，取不到才降级为摘要。

    分块**必须复用** `retrieval/chunking.py`：检索层、Reader、Verifier 都依赖
    `chunk.source` 与 `[标题][章节]` 头部这套约定，这里另写一套分块逻辑
    就等于把"证据强度标注"这条产品承诺做废了。
    """
    if use_html:
        sections = await fetch_sections(paper.arxiv_id, settings=settings, client=client)
        if sections:
            return chunk_paper(paper.title, sections, source="html")
    return chunks_from_abstract(paper.title, paper.abstract)


# =============================================================================
# 落库
# =============================================================================
def _paper_row(paper: ArxivPaper, metrics: OpenAlexMetrics | None) -> Paper:
    return Paper(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        abstract=paper.abstract,
        authors=list(paper.authors),
        published=paper.published,
        updated=paper.updated,
        primary_category=paper.primary_category,
        categories=list(paper.categories),
        comment=paper.comment,
        journal_ref=paper.journal_ref,
        venue=extract_venue(paper.comment),
        # 数字字段来自交叉校验过的补充源；拿不到就留 None（对比表留空，不编造）
        cited_by_count=metrics.cited_by_count if metrics else None,
        cited_at=metrics.cited_at if metrics else None,
        pdf_url=paper.pdf_url,
        html_url=paper.html_url,
        # F2「是否有代码」：从 arXiv 官方元数据（comment/abstract）检测仓库链接（D7）。
        # 检测不到留 None —— 不写 False，那是编造。
        has_code=detect_has_code(paper.comment, paper.abstract),
    )


async def _persist(
    session: AsyncSession,
    prepared: list[tuple[ArxivPaper, list[Chunk]]],
    metrics: dict[str, OpenAlexMetrics],
) -> tuple[list[ChunkRecord], dict[str, str]]:
    """把新论文与 chunk 写库，并回填数据库自增的 `chunk_id`。

    按 `arxiv_id` upsert：先 merge Paper（主键冲突即更新），再删掉该论文的旧 chunk
    后重插 —— 这样"改了分块策略后重跑"不会残留旧块，而 chunk_index 又是
    `(arxiv_id, chunk_index)` 唯一约束，不先删会直接撞约束报错。
    """
    records: list[ChunkRecord] = []
    titles: dict[str, str] = {}

    for paper, chunks in prepared:
        await session.merge(_paper_row(paper, metrics.get(paper.arxiv_id)))
        await session.execute(delete(PaperChunk).where(PaperChunk.arxiv_id == paper.arxiv_id))
        await session.flush()

        rows = [
            PaperChunk(
                arxiv_id=paper.arxiv_id,
                chunk_index=c.chunk_index,
                section=c.section,
                page=c.page,
                chunk_type=c.chunk_type,
                source=c.source,
                text=c.text,
                token_count=c.token_count,
            )
            for c in chunks
        ]
        session.add_all(rows)
        await session.flush()  # flush 后才有自增 id，ChunkRecord 必须有它才能建索引

        for chunk, row in zip(chunks, rows, strict=True):
            records.append(
                ChunkRecord(
                    chunk_id=row.id,
                    arxiv_id=paper.arxiv_id,
                    section=chunk.section,
                    text=chunk.text,
                    source=chunk.source,
                    chunk_type=chunk.chunk_type,
                    year=paper.year,
                    primary_category=paper.primary_category,
                    page=chunk.page,
                    # 与 _paper_row 写入 DB 的同一纯函数结果（口径必须一致）
                    has_code=detect_has_code(paper.comment, paper.abstract),
                )
            )
        titles[paper.arxiv_id] = paper.title

    return records, titles


# =============================================================================
# 主流程
# =============================================================================
async def ingest(
    query: str,
    *,
    categories: list[str] | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    max_results: int | None = None,
    fetch_html_content: bool | None = None,
    enrich_openalex: bool | None = None,
    settings: Settings | None = None,
) -> IngestResult:
    """完整入库流程。返回 `IngestResult`，供 `HybridRetriever.build_index()` 使用。"""
    s = settings or get_settings()
    started = time.perf_counter()
    stats = IngestStats()

    await init_db()
    use_html = s.fetch_html if fetch_html_content is None else fetch_html_content
    use_openalex = s.openalex_enabled if enrich_openalex is None else enrich_openalex
    limit = int(max_results if max_results is not None else s.arxiv_max_results)

    # ① 检索 arXiv（客户端内部串行限速，这里不做任何并发）
    async with ArxivClient(s) as arxiv:
        papers = await arxiv.search(
            query,
            categories=categories,
            year_from=year_from,
            year_to=year_to,
            max_results=limit,
        )
    # 保序去重：同一篇论文可能在多个分类下重复命中
    dedup: dict[str, ArxivPaper] = {}
    for p in papers:
        dedup.setdefault(p.arxiv_id, p)
    papers = list(dedup.values())
    logger.info("ingest.search_done", query=query[:60], hits=len(papers))

    if not papers:
        stats.elapsed_s = round(time.perf_counter() - started, 2)
        return IngestResult([], {}, False, stats)

    # ② 缓存优先（纯本地，不发网络请求）
    cached_records, cached_titles, cached_ids = await _load_cache([p.arxiv_id for p in papers])
    pending = [p for p in papers if p.arxiv_id not in cached_ids]

    # ③ 新论文：抓正文 + 分块（网络阶段，刻意不持有数据库事务）
    prepared: list[tuple[ArxivPaper, list[Chunk]]] = []
    html_client = (
        httpx.AsyncClient(
            headers={"User-Agent": s.arxiv_user_agent, "Accept": "text/html"},
            timeout=30.0,
            follow_redirects=True,
        )
        if use_html
        else None
    )
    try:
        for paper in pending:
            try:
                chunks = await _prepare_chunks(
                    paper, use_html=use_html, client=html_client, settings=s
                )
                if not chunks:
                    # 没摘要也没正文（极罕见）：记为失败，但不中断
                    raise ValueError("no chunks produced (empty abstract & no html)")
                prepared.append((paper, chunks))
                if chunks[0].source == "html":
                    stats.html_papers += 1
                else:
                    stats.abstract_papers += 1
            except Exception as exc:  # noqa: BLE001 - 单篇失败不拖垮全流程
                stats.failed += 1
                stats.errors.append({"arxiv_id": paper.arxiv_id, "error": str(exc)[:200]})
                logger.warning(
                    "ingest.paper_failed", arxiv_id=paper.arxiv_id, error=str(exc)[:200]
                )
    finally:
        if html_client is not None:
            await html_client.aclose()

    # ④ OpenAlex 补数字字段（失败只影响这一列，不影响主流程）
    metrics: dict[str, OpenAlexMetrics] = {}
    if use_openalex and prepared:
        async with OpenAlexClient(s) as oa:
            for paper, _ in prepared:
                m = await oa.enrich(paper.title, paper.authors, arxiv_id=paper.arxiv_id)
                if m:
                    metrics[paper.arxiv_id] = m

    # ⑤ 落库（单个短事务）
    if prepared:
        async with session_scope() as session:
            new_records, new_titles = await _persist(session, prepared, metrics)
    else:
        new_records, new_titles = [], {}

    stats.fetched = len(new_titles)
    stats.cached = len(cached_titles)
    stats.chunks = len(new_records) + len(cached_records)

    records = cached_records + new_records
    titles = {**cached_titles, **new_titles}
    # min_corpus 用「论文数」而不是 chunk 数：chunk 数受分块策略影响，
    # 会随 overlap / 章节粒度抖动，用它当阈值不稳定
    corpus_ready = len(titles) >= s.min_corpus
    if not corpus_ready:
        logger.warning("ingest.corpus_insufficient", found=len(titles), threshold=s.min_corpus)

    stats.elapsed_s = round(time.perf_counter() - started, 2)
    logger.info(
        "ingest.done",
        fetched=stats.fetched,
        cached=stats.cached,
        failed=stats.failed,
        chunks=stats.chunks,
        corpus_ready=corpus_ready,
        elapsed_s=stats.elapsed_s,
    )
    return IngestResult(records, titles, corpus_ready, stats)


# =============================================================================
# 命令行入口
# =============================================================================
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.ingest.pipeline",
        description="PaperPilot 入库：检索 arXiv → 抓正文 → 分块 → 落库",
    )
    p.add_argument("--query", required=True, help="检索关键词（英文效果最好）")
    p.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="最多取多少篇；不填则用 settings.arxiv_max_results",
    )
    p.add_argument(
        "--category",
        action="append",
        default=None,
        help='arXiv 分类过滤，可重复，如 --category cs.CL --category cs.LG',
    )
    p.add_argument("--year-from", type=int, default=None, help="起始年份（按提交时间）")
    p.add_argument("--year-to", type=int, default=None, help="结束年份")
    p.add_argument("--no-html", action="store_true", help="不抓 HTML 正文，全部走摘要降级")
    p.add_argument("--no-openalex", action="store_true", help="不调 OpenAlex 补被引数")
    return p


def _print_report(result: IngestResult, args: argparse.Namespace, settings: Settings) -> None:
    st = result.stats
    print("=" * 64)
    print(f"查询          : {args.query}")
    print(f"新抓取并入库  : {st.fetched} 篇")
    print(f"命中缓存      : {st.cached} 篇（未发网络请求）")
    print(f"失败          : {st.failed} 篇")
    print(
        f"正文来源      : HTML 章节 {st.html_papers} 篇 / 摘要降级 {st.abstract_papers} 篇"
    )
    print(f"分块总数      : {st.chunks} 个 chunk（覆盖 {result.paper_count} 篇论文）")
    print(
        f"语料是否达标  : {result.corpus_ready}"
        f"（阈值 min_corpus={settings.min_corpus}，实得 {result.paper_count} 篇）"
    )
    print(f"耗时          : {st.elapsed_s}s")
    if result.records:
        sample = result.records[0]
        print(f"首个 chunk    : [{sample.arxiv_id}] section={sample.section!r} source={sample.source}")
        print(f"                {sample.text[:90]!r}")
    for err in st.errors:
        print(f"  ! 失败 {err['arxiv_id']}: {err['error']}")
    print("=" * 64)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    setup_logging()
    settings = get_settings()

    result = asyncio.run(
        ingest(
            args.query,
            categories=args.category,
            year_from=args.year_from,
            year_to=args.year_to,
            max_results=args.max_results,
            fetch_html_content=False if args.no_html else None,
            enrich_openalex=False if args.no_openalex else None,
            settings=settings,
        )
    )
    _print_report(result, args, settings)
    return 0 if result.paper_count else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IngestResult",
    "IngestStats",
    "detect_has_code",
    "extract_venue",
    "ingest",
    "main",
]

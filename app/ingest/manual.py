"""手动入库服务层：撑起 `POST /api/v1/papers/ingest`（按 id 列表 / 按关键词批量入库）。

为什么存在：架构 §7 里这条接口长期标着「⏳ 待实现」，实际由 `ensure_corpus` 节点与
`scripts/build_index.py` 顶班 —— 那两条一个是"系统自动抓"、一个是"开发者跑脚本"，
**"用户点名几篇论文要入库"从来没有入口**。本模块只做编排：抓取、分块、落库
全部复用 `pipeline.py` 主流程所调用的同一组函数，因此 venue 抽取、`has_code`
检测、chunk 头部口径在全仓库仍只有一处实现。

三条边界：

1. **仍是串行限速**：id 模式对每篇"未命中缓存"的论文发一次 `get_paper`
   （arXiv 要求 ~1 req/3s，§2.2 / D11），不提供任何并发入口。
2. **缓存优先**：已入库且有 chunk 的论文走 `_load_cache` 直接计数，一个网络
   请求都不发 —— 与 pipeline 的"第二次秒级"是同一条实现，不是第二份。
3. **索引刷新走同一通道**：入库后复用 `ensure_corpus` 的 `load_corpus` +
   `corpus_digest` / `index_is_current` 判复用还是重建（`pdf.py` 的上传路径
   也走这里）。不另造第二条刷新通道 —— 两条通道一旦分叉，就会回到
   "手动重建能修好、自动重建修不好"那个老问题。刷新失败**只降级不抛**：
   DB 才是事实来源，下一次 `ensure_corpus` 按同一指纹必然把索引补上。

关于 import pipeline 的下划线私有函数：本次改动约束是**不许动 `pipeline.py`**，
把这三个函数提成公开 API 得先改那个文件。这里 import 的是"ingest() 主流程
自己调用的同一份实现"，而不是照着重写 —— 真正危险的从来不是访问私有名字，
而是第二份实现带来的口径分叉（本项目反复记录过的那类"静默失真"）。
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from app.core.config import get_settings
from app.db.session import init_db, session_scope
from app.ingest.arxiv_client import ArxivClient, ArxivPaper, parse_arxiv_id
from app.ingest.openalex_client import OpenAlexClient
from app.ingest.pipeline import (
    IngestResult,
    _load_cache,
    _persist,
    _prepare_chunks,
    ingest,
)
from app.retrieval.chunking import Chunk

logger = structlog.get_logger(__name__)

# 单篇失败原因统一截到 200 字符 —— 与 pipeline 的 errors 口径一致（不放大段堆栈进响应体）
_ERROR_CHARS = 200


def _elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 2)


def _failure(arxiv_id: str, error: str) -> dict[str, str]:
    return {"arxiv_id": arxiv_id, "error": error[:_ERROR_CHARS]}


async def refresh_search_index() -> dict[str, Any]:
    """入库后让检索层立刻看到新语料 —— 与 `ensure_corpus` / 上传路径同一套指纹通道。

    用函数内 import 的理由与 `pdf.py::_refresh_index` 相同：`app.agents.nodes` 包会
    连带加载全部节点，不该让 ingest 路径在模块导入期就依赖编排层。
    """
    try:
        from app.agents.nodes.ensure_corpus import load_corpus
        from app.agents.tools.hybrid_search import get_shared_retriever, reset_shared_retriever

        records, titles = await load_corpus()
        retriever = get_shared_retriever()
        if retriever.index_is_current(records, titles):
            action = "reused"
        else:
            await retriever.build_index(records, titles)
            reset_shared_retriever()  # 索引已换，丢弃持旧索引的单例（同 ensure_corpus 的收尾）
            action = "rebuilt"
        return {
            "refreshed": True,
            "action": action,
            "total_papers": len(titles),
            "chunks": len(records),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - 刷新失败不能毁掉已经成功的入库（见模块 docstring 边界 3）
        logger.warning("ingest.manual.index_refresh_failed", error=str(exc)[:_ERROR_CHARS])
        return {
            "refreshed": False,
            "action": "unknown",
            "total_papers": None,
            "chunks": None,
            "error": str(exc)[:_ERROR_CHARS],
        }


def _papers_view(titles: dict[str, str], new_ids: set[str] | None) -> list[dict[str, Any]]:
    """把"这次涉及的论文"列成响应体视图。

    `new_ids` 给定时逐项带 `status: fetched|cached`（id 模式分流是本地算出来的，
    知道谁是谁）；给 None 时**不带 status** —— 关键词模式下 pipeline 只汇总计数、
    不标注哪篇来自缓存，硬编一个就是编造（宁缺毋滥，逐篇状态请前端用 id 模式或列表接口查）。
    """
    if new_ids is None:
        return [{"arxiv_id": aid, "title": title} for aid, title in sorted(titles.items())]
    return [
        {"arxiv_id": aid, "title": title, "status": "fetched" if aid in new_ids else "cached"}
        for aid, title in sorted(titles.items())
    ]


# =============================================================================
# id 列表模式
# =============================================================================
async def ingest_by_ids(arxiv_ids: list[str]) -> dict[str, Any]:
    """按 arxiv_id 列表批量入库。返回可直接进 API 响应体的 dict。

    流程是 pipeline 主流程的同一批零件，只换了"论文从哪来"这一步：
    ① 归一化 + 去重（`parse_arxiv_id`，短 id 是 papers 主键，vN 后缀必须去掉）
    ② `_load_cache` 分流：已入库有 chunk 的记 cached，一篇都不发网络请求
    ③ 未命中的逐篇 `ArxivClient.get_paper`（串行限速；查无此篇记为失败，不中断）
    ④ `_prepare_chunks`（HTML 优先、摘要兜底）→ OpenAlex 补数字 → `_persist` 落库
    ⑤ `refresh_search_index` 按 corpus_digest 决定复用/重建

    **为什么 50 篇 ≈ 至少 150 秒的长请求仍然同步返回 200，而不是 202 + 任务 id**：
    - 202 需要"任务状态存哪、怎么轮询"的第三处设施；引入任务队列是新依赖（禁止），
      而把状态塞进 LangGraph checkpointer 会把"入库"这件 ingest 侧的事耦合到编排层。
    - 处理函数是 async 的，逐篇 `await` 期间不占用事件循环 —— 长的是这一条连接，
      不是整个服务；其他请求（含检索）照常并发。
    - 串行是 arXiv 的外部硬约束（§2.2 / D11）：挪进后台不会变快，只会把"等"变成"轮询"。
    - 时间预算由 `MAX_IDS_PER_REQUEST = 50` 封顶；且入库天然幂等（缓存优先 + 按
      arxiv_id upsert），客户端超时后**原样重试就是最优恢复手段**。
    """
    s = get_settings()
    started = time.perf_counter()
    await init_db()

    # ① 归一化短 id + 保序去重（多篇用户粘贴的列表里混 vN 后缀与重复项是常态）
    ids = list(dict.fromkeys(filter(None, (parse_arxiv_id(i) for i in arxiv_ids))))
    failures: list[dict[str, str]] = [
        _failure(raw, "无法解析成 arXiv id（形如 2305.15334；旧式 id 含 `/`，不被接受）")
        for raw in dict.fromkeys(arxiv_ids)
        if not parse_arxiv_id(raw)
    ]

    # ② 缓存优先（纯本地）
    _, cached_titles, cached_ids = await _load_cache(ids)
    pending = [i for i in ids if i not in cached_ids]

    # ③ 逐篇抓元数据。客户端内部串行限速，这里刻意不并发。
    papers: list[ArxivPaper] = []
    if pending:
        async with ArxivClient(s) as arxiv:
            for aid in pending:
                try:
                    paper = await arxiv.get_paper(aid)
                except Exception as exc:  # noqa: BLE001 - 单篇失败不连坐（同 pipeline）
                    failures.append(_failure(aid, f"arXiv 抓取失败: {exc}"))
                    logger.warning("ingest.manual.fetch_failed", arxiv_id=aid, error=str(exc)[:_ERROR_CHARS])
                    continue
                if paper is None:
                    failures.append(_failure(aid, "arXiv 查无此篇（id 不存在或已被撤下）"))
                    continue
                papers.append(paper)

    # ④ 抓正文 + 分块（网络阶段，不持有数据库事务 —— 与 pipeline 同一取舍）
    prepared: list[tuple[ArxivPaper, list[Chunk]]] = []
    html_client = (
        httpx.AsyncClient(
            headers={"User-Agent": s.arxiv_user_agent, "Accept": "text/html"},
            timeout=30.0,
            follow_redirects=True,
        )
        if s.fetch_html
        else None
    )
    try:
        for paper in papers:
            try:
                chunks = await _prepare_chunks(paper, use_html=s.fetch_html, client=html_client, settings=s)
                if not chunks:
                    raise ValueError("no chunks produced (empty abstract & no html)")
                prepared.append((paper, chunks))
            except Exception as exc:  # noqa: BLE001 - 单篇失败不连坐
                failures.append(_failure(paper.arxiv_id, f"正文抽取失败: {exc}"))
                logger.warning("ingest.manual.chunk_failed", arxiv_id=paper.arxiv_id, error=str(exc)[:_ERROR_CHARS])
    finally:
        if html_client is not None:
            await html_client.aclose()

    # OpenAlex 补数字字段（失败只影响那两列，不中断；口径与 pipeline ④ 相同）
    metrics = {}
    if s.openalex_enabled and prepared:
        async with OpenAlexClient(s) as oa:
            for paper, _ in prepared:
                m = await oa.enrich(paper.title, paper.authors, arxiv_id=paper.arxiv_id)
                if m:
                    metrics[paper.arxiv_id] = m

    # 落库（单个短事务；按 arxiv_id upsert，重复入库天然幂等）
    if prepared:
        async with session_scope() as session:
            _, new_titles = await _persist(session, prepared, metrics)
    else:
        new_titles = {}

    # ⑤ 索引跟上：同一套 corpus_digest 判定
    index = await refresh_search_index()

    titles = {**cached_titles, **new_titles}
    logger.info(
        "ingest.manual.done",
        requested=len(ids),
        fetched=len(new_titles),
        cached=len(cached_ids),
        failed=len(failures),
    )
    return {
        "mode": "ids",
        "requested": len(ids),
        "fetched": len(new_titles),
        "cached": len(cached_ids),
        "failed": len(failures),
        "failures": failures,
        "papers": _papers_view(titles, set(new_titles)),
        "index": index,
        "elapsed_s": _elapsed(started),
    }


# =============================================================================
# 关键词模式
# =============================================================================
async def ingest_by_query(
    query: str,
    *,
    categories: list[str] | None = None,
    year_gte: int | None = None,
    year_lte: int | None = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """按关键词检索并入库 —— 抓取/落库整段直接复用 `pipeline.ingest()`，不重写第二份。

    年份参数在本模块就叫 `year_gte/year_lte`（与 API 请求体、F2 filters 同口径），
    在这里翻回 pipeline 的 `year_from/year_to`：译名只发生在这一次转调，
    往上一层（路由）和往下一层（pipeline）各自保持自己已有的命名，不改任何人。

    抓不到任何论文返回 200 + `requested: 0`，不是 404：这是一次**合法执行了的检索**，
    "没有命中"是结果而不是错误（状态码口径与 papers.py 模块 docstring 一致）。
    单篇失败由 pipeline 记进 `stats.errors`，这里原样透出 —— 不许静默吞掉。
    """
    s = get_settings()
    started = time.perf_counter()

    result: IngestResult = await ingest(
        query,
        categories=categories,
        year_from=year_gte,
        year_to=year_lte,
        max_results=max_results,
        settings=s,
    )
    st = result.stats
    index = await refresh_search_index()

    # pipeline 的 titles 是"缓存 + 新增"的并集，且不带逐项来源标注；
    # 逐项 status 就不编造（见 _papers_view），新增/缓存数在 fetched/cached 汇总字段里。
    logger.info(
        "ingest.manual.query_done",
        query=query[:60],
        requested=result.paper_count,
        fetched=st.fetched,
        cached=st.cached,
        failed=st.failed,
    )
    return {
        "mode": "query",
        "query": query,
        "requested": result.paper_count,
        "fetched": st.fetched,
        "cached": st.cached,
        "failed": st.failed,
        "failures": [dict(e) for e in st.errors],
        "papers": _papers_view(result.titles, None),
        "index": index,
        "elapsed_s": _elapsed(started),
        # pipeline 自己的耗时（含检索阶段），比这里重新计一遍更贴近事实
        "ingest_elapsed_s": st.elapsed_s,
        "corpus_ready": result.corpus_ready,
    }


__all__ = ["ingest_by_ids", "ingest_by_query", "refresh_search_index"]

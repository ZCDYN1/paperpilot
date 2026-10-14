"""语料侧接口：单篇深挖（F13）、上传入库（F15）、引用关系（F16）、被引趋势（F19）。

这一层刻意薄：判定与取数全在 `app/papers/`、`app/references/`、`app/ingest/pdf.py`，
路由只做四件事 —— 输入闸门、把服务层异常映射成状态码、把结果原样吐出去、
以及在文件上传时**不把整份文件读进日志**。

## 状态码口径（与 `research.py` 一致）
- `422` 形状不合法（id 里有 `/`、空白、控制字符，列表为空/超长，问题为空）；
- `404` 指定的论文不在库里；
- `415` 上传了不支持的格式；`501` 缺可选依赖（PDF 解析器）—— 两者都要和 500 分开，
  客户端对前者的正确反应是"换个文件"，对后者是"装依赖"，对 500 才是"重试或报 bug"；
- `200` + `status` 字段 = "请求合法但这一步没做成"。引用边与被引数来自第三方补充源，
  **取不到是常态而不是错误**，抛 500 会让调用方以为服务坏了。

## 用户可控文本一律先过闸门
`ask` 的问题、上传的 `title_hint`、批量 id 都是用户输入，会进 LLM prompt 或 DB。
`guard_text` 负责注入判定与清洗，**清洗后的那份**才继续往下走
（判的是一份、用的是另一份，是本项目记录过的老坑）。
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select

from app.analysis.venue_tiers import lookup as lookup_tier
from app.api.guards import guard_run_id, guard_text
from app.db.models import Paper, PaperChunk
from app.db.session import session_scope
from app.ingest.pdf import (
    EmptyDocumentError,
    EncryptedDocumentError,
    PdfLibraryMissingError,
    UnsupportedFileFormatError,
    UploadIngestError,
    ingest_uploaded_document,
)
from app.papers import PaperNotFoundError, ask_about_paper, build_paper_detail
from app.references import (
    build_reference_edges,
    citation_trend,
    get_reference_graph,
    record_citation_snapshots,
)

router = APIRouter(prefix="/papers", tags=["papers"])
logger = structlog.get_logger(__name__)

# 一次请求最多处理多少篇。抓引用边是**串行 + 限速**的外部调用（架构 §2.2 / D11），
# 没有这个上限，一个 5000 篇的列表就足以把请求挂到超时并占住连接。
MAX_IDS_PER_REQUEST = 50
# 上传体积上限。刻意在路由层挡住：`UploadFile.read()` 是一次性读进内存的，
# 没有上限时"上传一个 2GB 的 PDF"就是一次免费的 DoS。
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class IdList(BaseModel):
    """`{arxiv_ids: [...]}` 形状的请求体。"""

    model_config = {"extra": "forbid"}

    arxiv_ids: list[str] = Field(min_length=1, max_length=MAX_IDS_PER_REQUEST)


class AskBody(BaseModel):
    """F13 单篇问答的请求体。"""

    model_config = {"extra": "forbid"}

    question: str = Field(min_length=1, max_length=2000)


def _guard_ids(ids: list[str]) -> list[str]:
    """论文 id / 上传文档 id 复用与 `run_id` 同一套字符集闸门。

    不另写一份正则：这些值都是直接进 DB 查询的参数，两处校验一旦分叉，
    弱的那一处就是入口。
    """
    out: list[str] = []
    for raw in ids:
        value = (raw or "").strip()
        if not value or len(value) > 64:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_input",
                    "field": "arxiv_ids",
                    "reason": "id 为空或超过 64 个字符",
                },
            )
        try:
            out.append(guard_run_id(value))
        except HTTPException as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_input",
                    "field": "arxiv_ids",
                    "reason": "论文 id 只允许字母、数字与 . _ -，长度 1~64",
                    "hint": "形如 2305.15334；上传文档的 id 形如 upload-1a2b3c4d5e6f",
                },
            ) from exc
    return out


# --------------------------------------------------------------------------- F19 会议分级
# ⚠️ 这条**必须声明在 `/{arxiv_id}` 之前**：FastAPI 按声明顺序匹配，
# 静态段 `/venue-tier` 若排在动态路由后面，会被 `/{arxiv_id}` 吃掉并返回"论文不存在"的 404。
@router.get("/venue-tier", summary="会议/期刊分档（F19，独立字段、不进 9 列对比表）")
async def venue_tier(venue: str, comment: str | None = None) -> dict[str, Any]:
    """`{tier, label, evidence}`，tier ∈ top | core | preprint | unknown。

    分档只作为**独立字段**随接口返回，不写进 9 列对比表 —— 那 9 列是 F4 的对外口径，
    README、导出、测试都锁着，加一列等于同时改三处契约。
    """
    return lookup_tier(venue, comment=comment)


# --------------------------------------------------------------------------- 论文列表
ESCAPE = "\\"


def _like_pattern(q: str) -> str:
    """把用户输入包成 LIKE 模式，并转义 `%` `_` 与转义符本身。

    不转义的话输入一个 `%` 就静默变成"匹配全部"，看起来像搜索坏了。
    """
    escaped = q.replace(ESCAPE, ESCAPE * 2).replace("%", ESCAPE + "%").replace("_", ESCAPE + "_")
    return f"%{escaped}%"


@router.get("", summary="语料库论文列表（分页 + 关键词过滤）")
async def list_papers(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(default=None, max_length=200, description="标题/摘要关键词"),
    category: str | None = Query(default=None, max_length=16),
) -> dict[str, Any]:
    """前端「论文库」页的数据源。

    之前只有 `GET /papers/{arxiv_id}`，等于要求调用方**先知道 id 才能查到东西** ——
    列表页、定向对比、上传后的回显全都没有数据来源。

    `q` 走 SQL `LIKE` 而不是全文检索：这一层只是"在已入库的几十篇里找一篇"，
    真正的语义检索在 `app/retrieval/`，把两套东西混在一个接口里迟早互相拖累。
    """
    conds = []
    if q:
        conds.append(or_(Paper.title.ilike(_like_pattern(q), escape=ESCAPE), Paper.abstract.ilike(_like_pattern(q), escape=ESCAPE)))
    if category:
        conds.append(Paper.primary_category == category)

    async with session_scope() as db:
        total = (
            await db.execute(select(func.count()).select_from(Paper).where(*conds))
        ).scalar_one()

        papers = (
            (
                await db.execute(
                    select(Paper).where(*conds).order_by(Paper.arxiv_id.desc()).limit(limit).offset(offset)
                )
            )
            .scalars()
            .all()
        )

        ids = [p.arxiv_id for p in papers]
        chunk_counts: dict[str, int] = {}
        if ids:
            agg = await db.execute(
                select(PaperChunk.arxiv_id, func.count())
                .where(PaperChunk.arxiv_id.in_(ids))
                .group_by(PaperChunk.arxiv_id)
            )
            chunk_counts = {row[0]: int(row[1]) for row in agg}

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "arxiv_id": p.arxiv_id,
                "title": p.title,
                "published": p.published.isoformat() if p.published else None,
                "primary_category": p.primary_category,
                "categories": p.categories or [],
                "venue": p.venue,
                # 分档只作为独立字段返回，不进 F4 的九列口径（见 venue_tier 的注释）。
                "venue_tier": lookup_tier(p.venue or "") if p.venue else None,
                "cited_by_count": p.cited_by_count,
                # 被引数必须带截至日期，否则读者会把它当成实时值比较。
                "cited_at": p.cited_at.isoformat() if p.cited_at else None,
                "has_code": p.has_code,
                "chunk_count": chunk_counts.get(p.arxiv_id, 0),
            }
            for p in papers
        ],
    }


# --------------------------------------------------------------------------- F13 单篇深挖
@router.get("/{arxiv_id}", summary="单篇详情（F13）")
async def paper_detail(arxiv_id: str) -> dict[str, Any]:
    """元数据 + 章节级 chunk 列表 + 该篇历史产生的引用与对比行。

    chunk 上的 `source` 必须原样透出：`html` 才是正文原文，
    `abstract` / `pdf` 是另外两种证据强度，前端不能画成同一种卡片。
    """
    try:
        return await build_paper_detail(_guard_ids([arxiv_id])[0])
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{arxiv_id}/ask", summary="只在这一篇的内容上回答（F13）")
async def paper_ask(arxiv_id: str, body: AskBody) -> dict[str, Any]:
    """检索范围硬钉在这一篇的 chunks 上，不做全库检索。

    答案里带不进 `[n]` 的部分由服务层列进 `ungrounded`；取不到支撑时服务层会返回
    固定的"内容不足以回答"，而不是让模型凭参数记忆编（产品定义 §7 的边界）。
    """
    doc_id = _guard_ids([arxiv_id])[0]
    question = guard_text(body.question, field="question")
    try:
        return await ask_about_paper(doc_id, question)
    except PaperNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "invalid_input", "reason": str(exc)}) from exc


# --------------------------------------------------------------------------- F15 上传入库
@router.post("/upload", summary="上传自己的 PDF / TXT / MD 入库（F15）")
async def upload_document(
    file: UploadFile = File(...),
    title_hint: str | None = Form(default=None, max_length=300),
) -> dict[str, Any]:
    """把用户自己的文档切块入库，之后就能被检索与引用溯源。

    三条刻意的限制：
    - **只走文本抽取，不碰版面**：与架构 §7「不解析 PDF 的表格/公式」一致，
      上传件同样只得到文本 chunk，因此引用卡片会如实显示它是 `pdf` 级证据；
    - 主键用 `upload-<内容 sha1[:12]>`，不改任何表结构就避开主键冲突，
      而且前缀本身就是"这不是 arXiv 论文"的诚实标注；同一份文件重复上传幂等；
    - 体积上限 `MAX_UPLOAD_BYTES`：`read()` 是一次性进内存的，没有上限就是免费的 DoS。
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_input",
                "field": "file",
                "reason": f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB 上限",
                "hint": "大文件请先拆分，或用 scripts/build_index.py 那套离线流程入库。",
            },
        )
    hint = guard_text(title_hint, field="title_hint") if title_hint else None
    try:
        result = await ingest_uploaded_document(file.filename or "upload.bin", data, title_hint=hint)
    except UnsupportedFileFormatError as exc:
        raise HTTPException(status_code=415, detail={"code": "unsupported_format", "reason": str(exc)}) from exc
    except PdfLibraryMissingError as exc:
        # 501 而不是 500：这是"少装了一个可选 extra"，修复动作是照抄命令，不是重试
        raise HTTPException(status_code=501, detail={"code": "pdf_extra_missing", "reason": str(exc)}) from exc
    except (EncryptedDocumentError, EmptyDocumentError) as exc:
        raise HTTPException(status_code=422, detail={"code": "undigestible_document", "reason": str(exc)}) from exc
    except UploadIngestError as exc:
        raise HTTPException(status_code=400, detail={"code": "upload_failed", "reason": str(exc)}) from exc

    # 只记 id 与块数：文件名可能带用户信息，正文更不该进日志
    logger.info("papers.uploaded", doc_id=result.get("arxiv_id"), chunks=result.get("chunks"))
    return result


# --------------------------------------------------------------------------- F16 引用关系
@router.post("/references/build", summary="抓取并落库引用边（F16）")
async def build_references(body: IdList) -> dict[str, Any]:
    """按 arXiv DOI 精确取 OpenAlex 的 `referenced_works`，落成 `citation_edges`。

    幂等：同一对论文 + 同一来源只有一条边（唯一约束），重复调用只补新增。
    """
    return await build_reference_edges(_guard_ids(body.arxiv_ids))


@router.get("/references/graph", summary="取一层的引用关系图（F16）")
async def references_graph(ids: str) -> dict[str, Any]:
    """`?ids=2305.15334,2401.00001` —— 以这些论文为中心取一层邻居。

    只取一层是刻意的：递归展开等于把整张 OpenAlex 引文网拉进内存，
    而产品要回答的是"这几篇之间谁补了谁的坑"。
    """
    parts = [p for p in (ids or "").split(",") if p.strip()]
    if not parts or len(parts) > MAX_IDS_PER_REQUEST:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_input",
                "field": "ids",
                "reason": f"ids 不能为空，且一次最多 {MAX_IDS_PER_REQUEST} 篇",
            },
        )
    return await get_reference_graph(_guard_ids(parts))


# --------------------------------------------------------------------------- F19 被引趋势
@router.post("/citations/snapshot", summary="记录今天的被引数（F19 趋势的数据源）")
async def snapshot_citations(body: IdList) -> dict[str, Any]:
    """把这几篇论文当前的被引数写进 `citation_snapshots`。

    趋势需要**多天**的采样点，所以这是一个要定期调用的动作（cron / CI 皆可），
    不是一个能凭空算出趋势的接口。
    """
    return await record_citation_snapshots(_guard_ids(body.arxiv_ids))


@router.get("/{arxiv_id}/citation-trend", summary="单篇被引趋势（F19）")
async def get_citation_trend(arxiv_id: str) -> dict[str, Any]:
    """少于 2 个采样点时 `verdict="insufficient_data"`，不给斜率。"""
    return await citation_trend(_guard_ids([arxiv_id])[0])


__all__ = ["router"]

"""F15 上传自有文档入库（产品定义 §5，P2）。

## 这一层存在的理由
语料层此前只有 arXiv 一条供给链路（`pipeline.ingest`），用户自己的笔记/非 arXiv 论文
进不了库。本模块把「上传一份文档」变成与 arXiv 入库**同一套下游约定**：
抽纯文本 → `retrieval/chunking.py` 切块 → 落 `papers` / `paper_chunks` → 刷新索引。
分块绝不自写一套 —— `chunk.source` 与 `[标题][章节]` 头部是 Reader / Verifier /
F13 深挖共同依赖的证据契约（见 chunking.py 模块 docstring 里同样的话）。

## 主键约定：不改表结构就绕开冲突
`papers.arxiv_id` 是主键，而上传文档没有 arXiv id。这里用
`upload-<内容 sha1 前 12 位>`：
- 不改任何表结构（加列/新表都会被别人本机的已有 SQLite 库背刺，见项目硬约束）；
- 内容寻址天然幂等：同一份文件重复上传命中同一行，直接返回、不重复入库；
- 前缀本身就是「这不是 arXiv 论文」的诚实标注，展示层与统计都能一眼区分。

## source 用字符串值 "pdf"
`paper_chunks.source` 是自由字符串列，写 "pdf" 不是 schema 改动。
它同样进入 F13 的 `source` 语义：前端据此区分「正文原文 / 摘要级 / 用户上传」证据。

## pypdf 是可选依赖，默认安装必须仍然能跑
`.txt` / `.md` 路径零依赖（别人 clone 下来不装 extras 也要能用 F15）；
`.pdf` 才 lazy import pypdf，没装时抛 `PdfLibraryMissingError` 并直接告诉用户
该跑 `pip install -e ".[pdf]"`。

## 落库后检索层立刻可见
复用 `ensure_corpus` 的 corpus_digest 机制（同一入口 `load_corpus` +
`index_is_current` / `build_index`），不是另造一条刷新通道：内容变了必然重建，
没变则秒级跳过。刷新失败**不让入库失败** —— DB 是事实来源，下一次
`ensure_corpus` 会按同一指纹自动补上，返回体里如实带 `index_refreshed=False`。
"""

from __future__ import annotations

import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import delete, func, select

from app.db.models import Paper, PaperChunk
from app.db.session import init_db, session_scope
from app.ingest.pipeline import detect_has_code
from app.retrieval.chunking import Section, chunk_paper

logger = structlog.get_logger(__name__)

# 上传文档的 arxiv_id 前缀（见模块 docstring「主键约定」）。
# F13 的 `is_upload` 判定与路由层的"别拿去调 arXiv API"检查都引用这一个常量。
UPLOAD_ID_PREFIX = "upload-"

# 无依赖的纯文本通道。.markdown 是 .md 的常见全写，一并收。
_TEXT_EXTS = frozenset({".txt", ".md", ".markdown"})

# PDF 魔数：扩展名会说谎（浏览器另存常丢后缀），内容开头不会。
_PDF_MAGIC = b"%PDF"

# 上传文档没有 arXiv 式的章节结构，统一切进一个 body 章节。
# （解析 markdown 标题分节是"锦上添花"，会引入一类新的切错风险，v1 刻意不做）
_UPLOAD_SECTION = "body"

# Paper.abstract 列在这里存纯文本开头，只为详情卡片有一小段可展示正文摘要。
_ABSTRACT_CHARS = 1000
_TITLE_CHARS = 200


# ---------------------------------------------------------------------------
# 错误类型：路由按 isinstance 映射状态码（基类兜住全部上传入库失败）
# ---------------------------------------------------------------------------
class UploadIngestError(Exception):
    """上传入库失败的基类。消息必须直接可读给用户，不含栈信息。"""


class UnsupportedFileFormatError(UploadIngestError):
    """扩展名不在支持列表（.pdf / .txt / .md）。"""


class PdfLibraryMissingError(UploadIngestError):
    """pypdf 未安装。消息必须给出安装命令，否则用户无从自助修复。"""


class EncryptedDocumentError(UploadIngestError):
    """加密 PDF（含构造期就抛 FileNotDecrypted 的 pypdf 版本）。"""


class EmptyDocumentError(UploadIngestError):
    """空文本 / 扫描件 PDF（抽不出字）。扫描件提示 OCR，不抛裸栈。"""


def _import_pypdf():
    """可选 import pypdf。没装时给出**可照抄**的修复命令，而不是 ImportError 栈。

    硬约束：默认安装（不装 extras）必须仍然能跑通全部主链路，
    所以 pypdf 只能在真正要读 PDF 的这一刻才 import。
    """
    try:
        import pypdf
    except ImportError as exc:
        raise PdfLibraryMissingError(
            '解析 PDF 需要可选依赖 pypdf，当前环境未安装。'
            '安装命令：pip install -e ".[pdf]"'
            "（不想装依赖的话，把正文另存为 .txt / .md 上传，走零依赖通道。）"
        ) from exc
    return pypdf


def _extract_pdf_text(data: bytes) -> str:
    """pypdf 抽文本。加密 / 损坏 / 扫描件都在这里翻成可读错误，不冒裸栈。"""
    pypdf = _import_pypdf()
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if getattr(reader, "is_encrypted", False):
            # 空口令能解开的是"只设了权限没设打开口令"那一类；解不开才如实报加密
            if not reader.decrypt(""):
                raise EncryptedDocumentError("PDF 已加密，无法抽取文本：请先解除加密后再上传。")
        parts = [(page.extract_text() or "") for page in reader.pages]
    except EncryptedDocumentError:
        raise
    except Exception as exc:  # noqa: BLE001
        # pypdf 各版本对加密文件的抛错时机不同（构造期 vs 访问 pages），
        # 这里按异常类名兜住"构造期就抛 FileNotDecryptedError"的形态。
        name = type(exc).__name__.lower()
        if "encrypt" in name or "decrypt" in name:
            raise EncryptedDocumentError(f"PDF 已加密，无法读取（{type(exc).__name__}）。") from exc
        raise UploadIngestError(f"PDF 解析失败，文件可能已损坏：{str(exc)[:200]}") from exc
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def _decode_text_bytes(data: bytes) -> str:
    """按 BOM → utf-8 → gb18030 → latin-1 的顺序解码。

    为什么 utf-16 只认带 BOM 的：无 BOM 的 UTF-16 中文文件会被 gb18030 **静默解成乱码**
    （两个字节组恰好是合法 GBK 序列），错得不报。带 BOM 可以零成本判对，
    无 BOM 的 UTF-16 在中文 Windows 环境里几乎不存在，不值得为它冒乱码风险。
    最后兜底 latin-1 永不抛错：宁可字符显示偏、也不让整条上传链路炸栈。
    """
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    # 内容魔数优先于扩展名：真实事故是"浏览器另存把 .pdf 存成 .txt"，
    # 按扩展名走文本通道会把 PDF 二进制当文本切块入库。
    if ext == ".pdf" or data.startswith(_PDF_MAGIC):
        return _extract_pdf_text(data)
    if ext in _TEXT_EXTS:
        return _decode_text_bytes(data)
    raise UnsupportedFileFormatError(
        f"不支持的文件类型 {ext or filename!r}：目前接受 .txt / .md（零依赖）"
        '与 .pdf（需可选依赖，pip install -e ".[pdf]"）。'
    )


def _title_from(text: str, filename: str, title_hint: str | None) -> str:
    """标题优先级：显式 hint > 正文第一行 > 文件名。绝不编造。"""
    if title_hint and title_hint.strip():
        return title_hint.strip()[:_TITLE_CHARS]
    for line in text.splitlines():
        candidate = line.lstrip("#").strip()  # markdown 标题行去掉 # 号
        if candidate:
            return candidate[:_TITLE_CHARS]
    stem = Path(filename).stem.strip()
    return (stem or "上传文档")[:_TITLE_CHARS]


async def _refresh_index() -> tuple[bool, str | None]:
    """让检索层立刻看到新语料：与 `ensure_corpus` 节点同一套 corpus_digest 机制。

    用函数内 import 的原因：`app.agents.nodes` 包会连带加载全部节点，
    让"上传入库"这条 ingest 路径在模块导入期就依赖编排层是不必要的耦合
    （`ensure_corpus` 反过来 import `app.ingest.pipeline` 也是同样写法）。

    刷新失败只降级不抛：DB 才是事实来源，下一次 ensure_corpus 按指纹必然补建。
    """
    try:
        from app.agents.nodes.ensure_corpus import load_corpus
        from app.agents.tools.hybrid_search import get_shared_retriever, reset_shared_retriever

        await init_db()
        records, titles = await load_corpus()
        retriever = get_shared_retriever()
        if retriever.index_is_current(records, titles):
            return True, None
        await retriever.build_index(records, titles)
        reset_shared_retriever()  # 索引已换，丢弃持旧索引的单例（同 ensure_corpus 的收尾）
        return True, None
    except Exception as exc:  # noqa: BLE001 - 见上：刷新失败不能毁掉已经成功的入库
        logger.warning("ingest.upload.index_refresh_failed", error=str(exc)[:200])
        return False, str(exc)[:200]


async def ingest_uploaded_document(
    filename: str,
    data: bytes,
    *,
    title_hint: str | None = None,
) -> dict[str, Any]:
    """上传文档入库（F15）。返回可直接进 API 响应的 dict。

    Args:
        filename: 原始文件名（只用于取扩展名和标题兜底，不做路径拼接）。
        data: 文件字节。内容 sha1 是主键来源，所以重复上传天然幂等。
        title_hint: 用户在上传表单里手填的标题；优先于从正文猜的第一行。

    Returns:
        ``{"arxiv_id", "title", "status": created|existed, "duplicate", "chunks",
        "source": "pdf", "has_code", "index_refreshed", "index_error"}``

    Raises:
        UnsupportedFileFormatError / PdfLibraryMissingError / EncryptedDocumentError /
        EmptyDocumentError / UploadIngestError（均继承 UploadIngestError）。
    """
    if not data:
        raise EmptyDocumentError("上传的文件是空的（0 字节）。")

    text = _extract_text(filename, data).strip()
    if not text:
        # 扫描件 PDF 走的就是这条：pypdf 对图片页只能抽出空白，给方向而不是抛栈
        raise EmptyDocumentError(
            "未能从文件中抽取到任何文本：若是扫描件/图片版 PDF，请先做 OCR "
            "转成文字，或直接把正文粘贴进 .txt 上传。"
        )

    sha1 = hashlib.sha1(data).hexdigest()[:12]
    arxiv_id = f"{UPLOAD_ID_PREFIX}{sha1}"
    title = _title_from(text, filename, title_hint)
    # has_code 与 arXiv 论文同一纯函数、同一保守口径（只认带 /路径 的仓库链接）。
    # 检测对象从官方元数据换成抽取全文：上传文档没有 comment，"正文里有链接"
    # 同样是实证，不算编造。
    has_code = detect_has_code(None, text)

    existing_summary: dict[str, Any] | None = None
    chunk_total = 0
    async with session_scope() as session:
        existing = await session.get(Paper, arxiv_id)
        if existing is not None:
            chunk_count = int(
                (
                    await session.execute(
                        select(func.count())
                        .select_from(PaperChunk)
                        .where(PaperChunk.arxiv_id == arxiv_id)
                    )
                ).scalar_one()
            )
            if chunk_count > 0:
                # 幂等命中：同一份字节已经入过库，一个 chunk 都不重插、一行都不改写。
                # 出会话后再刷新索引 —— 指纹没变时 _refresh_index 只是比对后即返回，
                # 但它能兜住"上次入库后索引又被换掉"的场景，让本函数每次成功返回都满足
                # 「调用成功 ⇒ 检索层可见」这条承诺。
                existing_summary = {
                    "arxiv_id": arxiv_id,
                    "title": existing.title,
                    "status": "existed",
                    "duplicate": True,
                    "chunks": chunk_count,
                    "source": "pdf",
                    "has_code": existing.has_code,
                }
            # 否则：行存在但没有 chunk = 上次入库中途失败留下的空壳，往下重灌
            # （同 pipeline._load_cache 对"空壳必须重抓"的口径）
            else:
                existing_summary = None

        if existing_summary is None:
            chunks = chunk_paper(
                title, [Section(name=_UPLOAD_SECTION, text=text)], source="pdf"
            )
            if not chunks:
                raise EmptyDocumentError("文本非空但切不出任何 chunk（全是空白字符？）。")

            await session.merge(
                Paper(
                    arxiv_id=arxiv_id,
                    title=title,
                    abstract=text[:_ABSTRACT_CHARS],
                    # 上传者没提供作者就留空列表：对比表与详情页的空值语义是
                    # 「没找到依据」，不是待补数据（见 app/db/models.py 同一条承诺）
                    authors=[],
                    published=None,  # 没有可靠发表日期，留 None，不拿 ingested_at 冒充
                    updated=datetime.now(UTC),
                    primary_category=None,
                    categories=[],
                    venue=None,
                    cited_by_count=None,
                    cited_at=None,
                    pdf_url=None,
                    html_url=None,
                    has_code=has_code,
                )
            )
            # 先删旧块再插：(arxiv_id, chunk_index) 有唯一约束，重灌不先删会直接撞约束
            # （同 pipeline._persist 的写法）。空壳行走到这里删 0 条，无害。
            await session.execute(delete(PaperChunk).where(PaperChunk.arxiv_id == arxiv_id))
            await session.flush()
            session.add_all(
                [
                    PaperChunk(
                        arxiv_id=arxiv_id,
                        chunk_index=c.chunk_index,
                        section=c.section,
                        page=c.page,
                        chunk_type=c.chunk_type,
                        source=c.source,  # "pdf"（见模块 docstring：纯字符串值，不是 schema 改动）
                        text=c.text,
                        token_count=c.token_count,
                    )
                    for c in chunks
                ]
            )
            chunk_total = len(chunks)

    refreshed, refresh_error = await _refresh_index()
    if existing_summary is not None:
        return {**existing_summary, "index_refreshed": refreshed, "index_error": refresh_error}

    logger.info(
        "ingest.upload.done",
        arxiv_id=arxiv_id,
        chunks=chunk_total,
        source_format=Path(filename).suffix.lower() or "unknown",
        index_refreshed=refreshed,
    )
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "status": "created",
        "duplicate": False,
        "chunks": chunk_total,
        "source": "pdf",
        "has_code": has_code,
        "index_refreshed": refreshed,
        "index_error": refresh_error,
    }


__all__ = [
    "UPLOAD_ID_PREFIX",
    "EmptyDocumentError",
    "EncryptedDocumentError",
    "PdfLibraryMissingError",
    "UnsupportedFileFormatError",
    "UploadIngestError",
    "ingest_uploaded_document",
]

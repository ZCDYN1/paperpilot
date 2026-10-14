"""F15 上传自有文档入库的离线单测。

全离线：临时 SQLite + 临时 FAISS 目录，不碰 data/；pypdf 本机**没装**也要全绿
—— PDF 路径用假模块（monkeypatch sys.modules）测，"默认安装能跑"由
「.txt 路径真跑通 + pypdf 缺席时给出安装命令」两条用例分别钉死。
"""

from __future__ import annotations

import re
import sys
import types

import pytest
from sqlalchemy import func, select

from app.agents.tools.hybrid_search import get_shared_retriever, reset_shared_retriever
from app.core.config import get_settings
from app.db.models import Paper, PaperChunk
from app.db.session import dispose_engine, init_db, session_scope
from app.ingest.pdf import (
    UPLOAD_ID_PREFIX,
    EmptyDocumentError,
    EncryptedDocumentError,
    PdfLibraryMissingError,
    UnsupportedFileFormatError,
    UploadIngestError,
    ingest_uploaded_document,
)
from app.papers.detail import build_paper_detail
from tests.api_support import BASE_ENV

_ID_RE = re.compile(r"^upload-[0-9a-f]{12}$")


@pytest.fixture
async def offline_env(monkeypatch, tmp_path):
    for k, v in BASE_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SQLITE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'f15.db').as_posix()}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SQLITE_DIR", str(tmp_path / "data" / "sqlite"))
    monkeypatch.setenv("FAISS_INDEX_DIR", str(tmp_path / "data" / "faiss"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "data" / "sqlite" / "checkpoints.db"))
    get_settings.cache_clear()
    await dispose_engine()
    await init_db()
    yield tmp_path
    reset_shared_retriever()
    await dispose_engine()
    get_settings.cache_clear()


def _install_fake_pypdf(monkeypatch, *, pages: tuple[str, ...] = (), encrypted: bool = False,
                        broken: bool = False) -> None:
    """造一个够用的 pypdf 假模块：只实现被测的 PdfReader/extract_text/is_encrypted/decrypt。"""

    mod = types.ModuleType("pypdf")

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def extract_text(self) -> str:
            return self._text

    class _Reader:
        def __init__(self, stream) -> None:
            if broken:
                raise ValueError("corrupt pdf stream")
            self.is_encrypted = encrypted
            self.pages = [_Page(t) for t in pages]

        def decrypt(self, password: str) -> int:
            return 0  # 恒失败：测试里"能解开的加密"不存在，解不开才如实报错

    mod.PdfReader = _Reader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", mod)


async def _chunk_count(arxiv_id: str) -> int:
    async with session_scope() as session:
        return int(
            (
                await session.execute(
                    select(func.count())
                    .select_from(PaperChunk)
                    .where(PaperChunk.arxiv_id == arxiv_id)
                )
            ).scalar_one()
        )


# --------------------------------------------------------------------------- 纯文本通道（零依赖）
async def test_upload_txt_creates_paper_and_chunks(offline_env):
    data = "混合检索笔记\n\n第一段：hybrid retrieval 用向量加 BM25 双路召回。\n\n第二段：RRF 融合后重排。".encode()
    res = await ingest_uploaded_document("notes.txt", data)

    assert _ID_RE.fullmatch(res["arxiv_id"]), res["arxiv_id"]
    assert res["status"] == "created" and res["duplicate"] is False
    assert res["chunks"] >= 1 and res["source"] == "pdf"
    assert res["index_refreshed"] is True
    # 标题兜底：没给 hint 时取正文第一行（绝不编造）
    assert res["title"] == "混合检索笔记"

    async with session_scope() as session:
        paper = await session.get(Paper, res["arxiv_id"])
        assert paper is not None
        rows = (
            (
                await session.execute(
                    select(PaperChunk).where(PaperChunk.arxiv_id == res["arxiv_id"])
                )
            )
            .scalars()
            .all()
        )
    # source 写 "pdf" 是纯字符串值（不是 schema 改动）：F13 前端据此区分证据来源
    assert {r.source for r in rows} == {"pdf"}
    assert any("hybrid retrieval" in r.text for r in rows)


async def test_upload_is_idempotent_by_content_hash(offline_env):
    text = "# 我的论文\n\n重复上传同一份内容不应该把 chunk 切两遍。"
    data = text.encode("utf-8")
    first = await ingest_uploaded_document("a.md", data)
    # 换文件名、字节完全相同 → 内容寻址主键相同 → 幂等命中，不重复入库
    second = await ingest_uploaded_document("改名了.txt", data)

    assert second["arxiv_id"] == first["arxiv_id"]
    assert second["status"] == "existed" and second["duplicate"] is True
    assert second["chunks"] == first["chunks"]
    assert await _chunk_count(first["arxiv_id"]) == first["chunks"]

    # 内容变了就是另一篇：sha1 前缀不同，互不覆盖
    third = await ingest_uploaded_document("a.md", (text + "\n新增一段。").encode("utf-8"))
    assert third["arxiv_id"] != first["arxiv_id"]


async def test_upload_title_hint_and_gbk_decoding(offline_env):
    text = "中文笔记：这是一段用 GBK 保存的正文，混检索与重排。"
    res = await ingest_uploaded_document("old.txt", text.encode("gb18030"), title_hint=" 我的旧笔记 ")
    assert res["title"] == "我的旧笔记"  # hint 优先且被 strip
    d = await build_paper_detail(res["arxiv_id"])
    assert "GBK 保存的正文" in d["chunks"][0]["text"]  # 没解码错就不可能匹配这句
    assert d["is_upload"] is True


async def test_upload_detects_repo_links_same_conservative_rule(offline_env):
    with_link = "方法说明见 https://github.com/foo/bar 仓库。".encode()
    res = await ingest_uploaded_document("with_code.txt", with_link)
    assert res["has_code"] is True
    # 裸域名不算（同 pipeline.detect_has_code 的口径：误判 True 比漏判更坑）
    no_link = "方法说明见 GitHub.com 这个网站。".encode()
    res2 = await ingest_uploaded_document("no_code.txt", no_link)
    assert res2["has_code"] is None


# --------------------------------------------------------------------------- PDF 通道（可选依赖）
async def test_pdf_path_without_pypdf_gives_install_command(offline_env, monkeypatch):
    """没装 pypdf：错误消息必须给出可照抄的 pip 命令（默认安装不能因此崩掉）。"""
    monkeypatch.setitem(sys.modules, "pypdf", None)  # import 时对 None 条目抛 ImportError
    with pytest.raises(PdfLibraryMissingError) as ei:
        await ingest_uploaded_document("paper.pdf", b"%PDF-1.4 fake bytes")
    assert '.[pdf]' in str(ei.value)


async def test_txt_path_still_works_without_pypdf(offline_env, monkeypatch):
    """底线用例：pypdf 缺席时 .txt 通道照常入库（"别人 clone 就能跑"的 F15 部分）。"""
    monkeypatch.setitem(sys.modules, "pypdf", None)
    res = await ingest_uploaded_document("note.txt", "零依赖通道的一段正文。".encode())
    assert res["status"] == "created"


async def test_upload_pdf_with_fake_pypdf(offline_env, monkeypatch):
    _install_fake_pypdf(monkeypatch, pages=("PDF 正文第一段：模型结构。", "第二段：实验结论。"))
    res = await ingest_uploaded_document("paper.pdf", b"%PDF-1.4 fake\nstream")
    assert res["status"] == "created" and res["chunks"] >= 1
    d = await build_paper_detail(res["arxiv_id"])
    assert {c["source"] for c in d["chunks"]} == {"pdf"}
    assert "PDF 正文第一段" in d["chunks"][0]["text"]


async def test_pdf_magic_beats_extension(offline_env, monkeypatch):
    """浏览器另存常丢后缀：内容是 %PDF 就必须走 PDF 通道，
    否则 PDF 二进制会被当文本切块入库（垃圾进语料库还不报错）。"""
    _install_fake_pypdf(monkeypatch, pages=("按魔数走 pypdf 抽取的正文。",))
    res = await ingest_uploaded_document("downloaded-file.txt", b"%PDF-1.7 fake")
    d = await build_paper_detail(res["arxiv_id"])
    assert "按魔数走 pypdf 抽取的正文" in d["chunks"][0]["text"]


async def test_encrypted_pdf_readable_error(offline_env, monkeypatch):
    _install_fake_pypdf(monkeypatch, pages=("不该被读到。",), encrypted=True)
    with pytest.raises(EncryptedDocumentError) as ei:
        await ingest_uploaded_document("lock.pdf", b"%PDF-1.4 fake")
    assert "加密" in str(ei.value)


async def test_scanned_pdf_readable_error(offline_env, monkeypatch):
    """扫图版：pypdf 只抽得出空白 → EmptyDocumentError + OCR 指引，而不是抛栈。"""
    _install_fake_pypdf(monkeypatch, pages=("", "   "))
    with pytest.raises(EmptyDocumentError) as ei:
        await ingest_uploaded_document("scan.pdf", b"%PDF-1.4 fake")
    assert "OCR" in str(ei.value)


async def test_corrupt_pdf_readable_error(offline_env, monkeypatch):
    _install_fake_pypdf(monkeypatch, broken=True)
    with pytest.raises(UploadIngestError) as ei:
        await ingest_uploaded_document("bad.pdf", b"%PDF-1.4 garbage")
    assert "解析失败" in str(ei.value)


async def test_unsupported_extension_rejected(offline_env):
    with pytest.raises(UnsupportedFileFormatError):
        await ingest_uploaded_document("thesis.docx", b"whatever bytes")


async def test_empty_uploads_rejected(offline_env):
    with pytest.raises(EmptyDocumentError):
        await ingest_uploaded_document("empty.txt", b"")
    with pytest.raises(EmptyDocumentError):
        await ingest_uploaded_document("blank.txt", b"   \n\t \n")


# --------------------------------------------------------------------------- 检索层立刻可见
async def test_upload_visible_to_retrieval_immediately(offline_env):
    """落库后必须马上能被定向检索命中（corpus_digest 机制，见 pdf._refresh_index）。"""
    res = await ingest_uploaded_document("a.txt", "cross encoder reranking 的课堂笔记正文。".encode())
    # id 形状本身就是契约：`upload-` 前缀让"这不是 arXiv 论文"在任何一个列表里都看得出来
    assert res["arxiv_id"].startswith(UPLOAD_ID_PREFIX)
    retriever = get_shared_retriever()  # 单例已被刷新流程重置，这里从磁盘重建
    assert retriever.size >= res["chunks"]

    from app.retrieval.hybrid_retriever import MetadataFilter

    hits = await retriever.search(
        "reranking note",
        ["reranking", "cross", "encoder"],
        top_k=10,
        filters=MetadataFilter(arxiv_ids=[res["arxiv_id"]]),
    )
    assert hits and hits[0].arxiv_id == res["arxiv_id"]

    # 第二份上传 → 内容变了指纹必变 → 重建后两份都在（不出现"新的进了旧的没了"）
    res2 = await ingest_uploaded_document("b.txt", "late interaction multi-vector retrieval 笔记。".encode())
    r2 = get_shared_retriever()
    ids = {rec.arxiv_id for rec in r2._chunks.values()}
    assert {res["arxiv_id"], res2["arxiv_id"]} <= ids

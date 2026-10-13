"""从 arXiv 官方 HTML 正文抽取章节结构。

为什么是 HTML 而不是 PDF（架构文档 §5.1 / ADR D8）：
「不做全文 PDF 解析」不等于「只能用摘要」。arxiv.org 自己发布的 HTML 渲染版
（`/html/{id}`）既合规、又自带 `<h2>/<h3>` 章节锚点，正好满足"引用要定位到具体章节"
这条产品承诺。取不到时才退回摘要块，并在 `chunk.source` 里如实标记。

**只用标准库**（`re` + `html.parser`）：
解析目标不是"任意网页"，而是 LaTeXML 生成的、结构高度规律的单一模板。
为它引入 bs4/lxml 会平白增加部署体积与解析不确定性 —— 收益远小于成本。

模板事实（实测 arXiv LaTeXML 输出）：
- 正文章节：`<section class="ltx_section|ltx_subsection|ltx_appendix|ltx_bibliography">`
- 标题：`<h2 class="ltx_title ltx_title_section">`（往下依次 h3/h4）
- 摘要：`<div id="abstract1" class="ltx_abstract"><h6 ...>Abstract</h6>`
- 噪声：`<nav class="ltx_page_navbar">`（目录）、`ltx_page_footer`、
  `<span class="ltx_note ltx_role_footnote">`（脚注）、`ltx_bibliography`（参考文献）

**降级是第一原则**：老论文可能没有 HTML、网络可能超时、模板可能改版。
任何一种情况都必须安静地返回空列表，由 pipeline 改用摘要成块 ——
抓正文失败绝不能变成整个 run 失败。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import httpx
import structlog

from app.core.config import Settings, get_settings
from app.retrieval.chunking import Section

logger = structlog.get_logger(__name__)

ARXIV_HTML_URL = "https://arxiv.org/html/{arxiv_id}"

# ---- 噪声过滤规则 -----------------------------------------------------------
# 整块丢弃的标签
_NOISE_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "button",
    "svg",
    # 公式：LaTeXML 会把 `<annotation encoding="application/x-tex">` 的 LaTeX 也塞进来，
    # 直接抽会得到一堆错乱的符号，噪音远大于信息量。宁可不要，也不污染 chunk。
    "math",
}
# 按 class 丢弃。**必须按 token 精确匹配，不能做子串匹配** ——
# 实测 `<article class="ltx_document ltx_authors_1line">` 包住了整个文档，
# 子串匹配 `ltx_authors` 会把整篇正文判成噪声、抽出一片空白，
# 而且不报任何错（最隐蔽的一类 bug）。
_NOISE_CLASS_TOKENS = {
    "ltx_page_navbar",   # 目录导航
    "ltx_page_footer",
    "ltx_page_header",
    "ltx_page_banner",
    "ltx_toc",
    "ltx_authors",
    "ltx_contact",
    "ltx_role_author",
    "ltx_bibliography",  # 参考文献列表
    "ltx_biblist",
    "ltx_listing",
}
# 前缀匹配只用于「一个 class 派生多个」的家族，避免误伤同前缀的正文 class
_NOISE_CLASS_PREFIXES = ("ltx_note", "ltx_cite")
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PARA_TAGS = {"p", "figcaption", "blockquote"}
_PARA_CLASS_TOKENS = {"ltx_para", "ltx_caption"}
_VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "area", "base", "col", "embed", "param", "track", "wbr"}

# 命中即"到此为止"：References 之后通常是参考文献 + 附录，
# 对"某个方向的方法/结论"综述没有价值，还会把 chunk 塞满编号噪声。
_STOP_HEADING_RE = re.compile(r"^(references|bibliography|references and notes)\b", re.I)
# 命中即跳过该章节内容（但不停止解析后续章节）。
# 注意这里**不能加尾随 \b**：`acknowledg\b` 匹配不了 "Acknowledgment"，
# 结果致谢章节会漏进语料 —— 这类"词干后接字母"的坑只能靠实测发现。
_SKIP_HEADING_RE = re.compile(
    r"^(?:acknowledg|appendix|supplement|footnote|author contributions)", re.I
)

# 太短的"章节"通常是没内容的装饰性标题（如空的 "3.2.1 Chunking strategies"），
# 单独成 Section 只会污染分块结果
_MIN_SECTION_CHARS = 40

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """折叠空白。HTML 源码里大量缩进换行，不归一化会让 chunk 里出现整屏空白。"""
    return _WS_RE.sub(" ", text or "").strip()


class _SectionParser(HTMLParser):
    """把 HTML 拉平成一串 `(标签, 文本)` 块。

    刻意不做 DOM 树：只需要"标题"和"段落"两类块按出现顺序排列，
    上层再据此切成 Section。避免建树也避免了 HTML 不闭合时的递归麻烦。

    用 `_stack` 记录每个开始标签是否为噪声容器：只要栈里**任意一层**是噪声，
    内部的文本与标题就全部忽略 —— 这样 `<nav>` 里的伪标题不会混进正文。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, bool]] = []
        self._blocks: list[tuple[str, str]] = []  # (标签名 或 "para", 文本)
        self._cap_tag: str | None = None
        self._cap_kind: str | None = None
        self._buf: list[str] = []

    @property
    def _skipping(self) -> bool:
        return any(noise for _, noise in self._stack)

    # ------------------------------------------------------------------ 解析钩子
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS:
            if tag == "br" and self._cap_kind is not None and not self._skipping:
                self._buf.append("\n")  # 换行是段落内的有效分隔，不能丢
            return

        cls = dict(attrs).get("class") or ""
        tokens = {t.lower() for t in cls.split()}
        noise = (
            tag in _NOISE_TAGS
            or bool(tokens & _NOISE_CLASS_TOKENS)
            or any(t.startswith(p) for t in tokens for p in _NOISE_CLASS_PREFIXES)
        )
        self._stack.append((tag, noise))
        if noise:
            return

        if tag in _HEADING_TAGS:
            self._start_capture(tag, tag)
        elif tag in _PARA_TAGS or bool(tokens & _PARA_CLASS_TOKENS):
            self._start_capture(tag, "para")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_data(self, data: str) -> None:
        if self._skipping or self._cap_kind is None:
            return
        self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._cap_kind is not None and tag == self._cap_tag:
            self._flush_block()
        # 出栈时按名字找最近的一层，而不是假设与栈顶严格配对 ——
        # HTML 常有未闭合标签，严格配对会一步错步步错
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break

    # ------------------------------------------------------------------ 内部
    def _start_capture(self, tag: str, kind: str) -> None:
        # 进入新块时若上一块还没收尾（标签没闭合），先落盘，避免文本粘连
        if self._cap_kind is not None:
            self._flush_block()
        self._cap_tag = tag
        self._cap_kind = kind
        self._buf = []

    def _flush_block(self) -> None:
        text = _normalize(" ".join(self._buf))
        if len(text) >= 2:
            self._blocks.append((self._cap_kind or "para", text))
        self._cap_tag = None
        self._cap_kind = None
        self._buf = []


def _blocks_to_sections(blocks: list[tuple[str, str]]) -> list[Section]:
    """把拉平的块序列切成 Section：标题开一章，段落归属最近的那章。"""
    sections: list[Section] = []
    name: str | None = None
    parts: list[str] = []

    def flush() -> None:
        nonlocal name, parts
        if name is not None:
            text = "\n\n".join(parts).strip()
            if len(text) >= _MIN_SECTION_CHARS:
                sections.append(Section(name=name, text=text))
        name, parts = None, []

    for kind, text in blocks:
        if kind in _HEADING_TAGS:
            if kind == "h1":
                # h1 是论文标题。标题的权威版本来自 arXiv API（文本字段只信官方），
                # 这里再收一份只会造成两个来源不一致。
                continue
            heading = _normalize(text)
            if not heading:
                continue
            if _STOP_HEADING_RE.match(heading):
                break  # 到参考文献即停
            flush()
            if _SKIP_HEADING_RE.match(heading):
                continue  # 跳过致谢/附录，但继续往后找正文
            name = heading
        else:
            if name is None:
                continue  # 首个标题之前的作者/机构等杂项，丢弃
            parts.append(text)

    flush()
    return sections


def extract_sections(html: str) -> list[Section]:
    """纯函数：HTML 文本 → 章节列表。解析中断也返回已解析出的部分。

    单独拆出来是为了可单测（喂一段 HTML 断言章节切分），不必发网络请求。
    """
    if not html or not html.strip():
        return []
    parser = _SectionParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # noqa: BLE001 - 模板千奇百怪，宁可要半份也不要 0 份
        logger.warning("html.parse_failed", error=str(exc)[:200])
    return _blocks_to_sections(parser._blocks)


async def fetch_sections(
    arxiv_id: str,
    *,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
    timeout: float = 30.0,
) -> list[Section]:
    """抓 `https://arxiv.org/html/{id}` 并抽取章节。

    **任何失败都返回空列表**（404 / 没有 HTML 的老论文 / 超时 / 模板改版），
    由 pipeline 自动降级为摘要成块。这是调用方唯一需要知道的契约，
    所以这里不抛异常 —— 让"抓正文失败"和"这篇论文没正文"在调用方看来是同一种情况。
    """
    settings = settings or get_settings()
    url = ARXIV_HTML_URL.format(arxiv_id=arxiv_id)
    owns = client is None
    http = client or httpx.AsyncClient(
        headers={"User-Agent": settings.arxiv_user_agent, "Accept": "text/html"},
        timeout=timeout,
        follow_redirects=True,
    )
    try:
        resp = await http.get(url)
        if resp.status_code != 200:
            logger.warning("html.not_available", arxiv_id=arxiv_id, status=resp.status_code)
            return []
        html = resp.text
        # 老论文会返回一个"HTML 不可用"的提示页（有时仍是 200），
        # 不识别它就会把提示页当成正文切出一堆废块
        if "No HTML available" in html or "HTML is not available" in html:
            logger.info("html.not_rendered", arxiv_id=arxiv_id)
            return []
        return extract_sections(html)
    except Exception as exc:  # noqa: BLE001 - 网络/超时/解析全都要降级
        logger.warning("html.fetch_failed", arxiv_id=arxiv_id, error=str(exc)[:200])
        return []
    finally:
        if owns:
            await http.aclose()


__all__ = ["ARXIV_HTML_URL", "extract_sections", "fetch_sections"]

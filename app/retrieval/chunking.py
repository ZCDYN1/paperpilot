"""章节感知分块。

几个刻意的取舍（对应架构文档 §5.1）：

- **按「章节 + 段落」切，而不是固定字符数**：论文有天然结构
  （Abstract / Intro / Method / Exp / Conclusion），按结构切语义才完整。
- **每个块前置 `[论文标题] [章节名]`**：小块脱离全文后就没有上下文了，
  这是低成本高收益的一步。
- **公式 / 表格 / 图注单独成块**并打 `chunk_type`，避免被切碎后失去语义。
- **`source` 只有两种取值**：`html`（拿得到 arXiv 官方 HTML 正文）
  或 `abstract`（降级）。这个值会一路传到引用卡片，用来如实标注证据强度 ——
  不能让用户误以为摘要级证据是正文原文。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 粗略的 token 估算：中日韩字符按 1 token/字，其余按 1 token/4 字符。
# 不引入 tiktoken 这类依赖，估算误差对分块场景完全可接受。
_CJK_RE = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

DEFAULT_TARGET_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 96


def estimate_tokens(text: str) -> int:
    """估算 token 数（无需分词器依赖）。"""
    cjk = len(_CJK_RE.findall(text))
    others = len(text) - cjk
    return cjk + max(1, others // 4)


@dataclass(slots=True)
class Section:
    """论文的一个章节（或摘要）。

    `text` 应当已经是**纯文本**——解析 HTML / 抽取章节的活由 ingest 层负责，
    分块层只处理结构化输入，这样可以独立测试。
    """

    name: str
    text: str
    page: int | None = None


@dataclass(slots=True)
class Chunk:
    chunk_index: int
    section: str
    text: str
    token_count: int
    chunk_type: str = "text"  # text | formula | table | figure
    source: str = "abstract"  # html | abstract
    page: int | None = None


def _split_paragraphs(text: str) -> list[str]:
    """按空行切段，并丢掉纯空白段。"""
    parts = re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    return [p.strip() for p in parts if p.strip()]


def _is_special_block(paragraph: str) -> str | None:
    """粗判段落是否为公式 / 表格 / 图注，返回 chunk_type 或 None。

    这里用启发式而非严格解析：目的是**别把公式切碎**，
    而不是精确分类。误判的代价很小。
    """
    stripped = paragraph.strip()
    if re.match(r"^(Table|表)\s*\d+", stripped):
        return "table"
    if re.match(r"^(Figure|Fig\.?|图)\s*\d+", stripped):
        return "figure"
    if stripped.startswith("$$") or stripped.startswith("\\[") or stripped.startswith("\\begin{equation"):
        return "formula"
    return None


def _with_header(title: str, section: str, body: str) -> str:
    """给块注入标题上下文。"""
    header = f"[{title}] [{section}]"
    return f"{header}\n{body}".strip()


def chunk_section(
    title: str,
    section: Section,
    start_index: int,
    source: str = "html",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """把单个章节切成若干 chunk。"""
    chunks: list[Chunk] = []
    idx = start_index
    buf: list[str] = []
    buf_tokens = 0

    def flush() -> None:
        nonlocal idx, buf, buf_tokens
        if not buf:
            return
        body = "\n\n".join(buf).strip()
        if not body:
            buf, buf_tokens = [], 0
            return
        text = _with_header(title, section.name, body)
        chunks.append(
            Chunk(
                chunk_index=idx,
                section=section.name,
                text=text,
                token_count=estimate_tokens(text),
                source=source,
                page=section.page,
            )
        )
        idx += 1
        # 保留尾部 overlap，防止结论被硬切断
        if overlap_tokens > 0 and buf:
            tail = buf[-1]
            kept = tail[-overlap_tokens * 4 :] if len(tail) > overlap_tokens * 4 else tail
            buf = [kept]
            buf_tokens = estimate_tokens(kept)
        else:
            buf, buf_tokens = [], 0

    for para in _split_paragraphs(section.text):
        special = _is_special_block(para)
        para_tokens = estimate_tokens(para)

        # 公式 / 表格 / 图注：先冲刷缓冲区，再单独成块
        if special:
            flush()
            buf, buf_tokens = [], 0
            text = _with_header(title, section.name, para)
            chunks.append(
                Chunk(
                    chunk_index=idx,
                    section=section.name,
                    text=text,
                    token_count=estimate_tokens(text),
                    chunk_type=special,
                    source=source,
                    page=section.page,
                )
            )
            idx += 1
            continue

        # 单个段落就超长：再按句子硬切
        if para_tokens > target_tokens * 1.5:
            flush()
            buf, buf_tokens = [], 0
            for piece in re.split(r"(?<=[。！？.!?])\s+", para):
                piece = piece.strip()
                if not piece:
                    continue
                buf.append(piece)
                buf_tokens += estimate_tokens(piece)
                if buf_tokens >= target_tokens:
                    flush()
            continue

        buf.append(para)
        buf_tokens += para_tokens
        if buf_tokens >= target_tokens:
            flush()

    flush()
    return chunks


def chunk_paper(
    title: str,
    sections: list[Section],
    source: str = "html",
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """把一篇论文的所有章节切成 chunk 列表（chunk_index 全局连续）。"""
    all_chunks: list[Chunk] = []
    for section in sections:
        produced = chunk_section(
            title=title,
            section=section,
            start_index=len(all_chunks),
            source=source,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        all_chunks.extend(produced)
    return all_chunks


def chunks_from_abstract(title: str, abstract: str) -> list[Chunk]:
    """降级路径：只有摘要时（arXiv HTML 取不到的老论文），摘要单独成块。"""
    if not abstract.strip():
        return []
    return chunk_paper(
        title=title,
        sections=[Section(name="abstract", text=abstract)],
        source="abstract",
    )

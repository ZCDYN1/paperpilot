"""章节感知分块的单元测试。

分块是整条链路的地基：它的 `chunk_type` / `source` 会一路传到引用卡片与消融统计。
这里重点锁三件事：**超长段落必须切、特殊块必须独立、overlap 不能被吃掉**。
"""

from __future__ import annotations

from app.retrieval.chunking import (
    Chunk,
    Section,
    chunk_paper,
    chunks_from_abstract,
    estimate_tokens,
)


def _body(chunk: Chunk) -> str:
    """去掉 `[标题][章节]` 头部，只留正文，方便断言重叠。"""
    return chunk.text.split("\n", 1)[1]


def test_long_paragraph_is_split():
    """单个超长段落（无空行可分）也必须被硬切成多块，不能整段塞进去。"""
    giant = " ".join(["retrieval"] * 400)  # ~1000 tokens，远超 512 的块目标
    chunks = chunk_paper("T", [Section(name="intro", text=giant)])
    assert len(chunks) >= 2
    # chunk_index 必须是从 0 开始的连续整数
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_type == "text" for c in chunks)


def test_special_blocks_are_isolated_with_correct_type():
    """公式 / 表格 / 图注必须各自单独成块，并打上正确的 chunk_type。"""
    text = "\n\n".join(
        [
            "A normal paragraph of prose.",
            "$$ E = mc^2 $$",
            "Table 1: results on the benchmark.",
            "Figure 2: the overall architecture.",
            "Another normal paragraph of prose.",
        ]
    )
    chunks = chunk_paper("Paper X", [Section(name="method", text=text)])
    assert [c.chunk_type for c in chunks] == ["text", "formula", "table", "figure", "text"]
    # 特殊块不能被并进相邻正文块
    assert "E = mc^2" in chunks[1].text
    assert chunks[1].token_count == estimate_tokens(chunks[1].text)


def test_overlap_prevents_hard_cut():
    """相邻块之间必须保留尾部 overlap，防止结论被硬切断。"""
    # 每段 ~50 tokens（大于目标 40，但小于硬切阈值 1.5×），保证按段落边界冲刷
    paras = [f"para{i} " + " ".join(["word"] * 40) + f" tail{i}" for i in range(4)]
    chunks = chunk_paper(
        "T", [Section(name="intro", text="\n\n".join(paras))], target_tokens=40, overlap_tokens=10
    )
    assert len(chunks) >= 3
    bodies = [_body(c) for c in chunks]
    # 下一块的开头应当是上一块结尾 overlap 的重复（overlap_tokens=10 → 40 字符）
    for prev, nxt in zip(bodies, bodies[1:], strict=False):
        assert nxt.startswith(prev[-40:]), "overlap 丢失：结论被硬切断"

    # 相邻块共享的前缀不能是整块（否则说明根本没切开）
    assert len(bodies[1]) > 40


def test_abstract_fallback_source_and_metadata():
    """降级路径：只有摘要时 `source` 必须是 abstract，且标记在正确的字段上。"""
    chunks = chunks_from_abstract("Paper Y", "We study hybrid retrieval in depth.")
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source == "abstract"
    assert chunk.section == "abstract"
    assert chunk.chunk_type == "text"
    assert chunk.chunk_index == 0
    assert chunk.text.startswith("[Paper Y] [abstract]")

    # 空白摘要不产生任何块，而不是产生一个空块
    assert chunks_from_abstract("Paper Z", "   \n  ") == []


def test_html_source_propagates_to_chunks():
    """正文路径的 `source=html` 必须逐块保留 —— 前端靠它标注证据强度。"""
    chunks = chunk_paper("T", [Section(name="intro", text="Body text.")], source="html")
    assert chunks and all(c.source == "html" for c in chunks)

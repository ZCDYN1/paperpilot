"""对外契约模型（Pydantic）的统一出口。

集中导出，是为了让调用方只记一个导入路径：`from app.schemas import ResearchRequest`。
这样日后拆分 / 移动模块时，调用方不需要跟着改。
"""

from __future__ import annotations

from app.schemas.research import (
    ChunkOut,
    CitationOut,
    PaperOut,
    ResearchFilters,
    ResearchRequest,
    SubQuestion,
)
from app.schemas.retrieval import ARXIV_CATEGORIES, HybridSearchArgs

__all__ = [
    "ARXIV_CATEGORIES",
    "ChunkOut",
    "CitationOut",
    "HybridSearchArgs",
    "PaperOut",
    "ResearchFilters",
    "ResearchRequest",
    "SubQuestion",
]

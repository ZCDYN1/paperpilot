"""检索层测试的公共夹具。

**为什么要在 conftest 顶部就设环境变量？**
`Settings` 是 `lru_cache` 的单例，一旦被构造就固定下来了。若测试采用默认的
`EMBEDDING_PROVIDER=local`，pytest 一跑就会去下载 3GB 的 BGE 权重。
所以必须在**任何 `app.*` 模块被导入之前**就把环境切成 mock ——
pytest 保证 conftest 先于测试模块导入，这里是最合适的落点。
"""

from __future__ import annotations

import os

os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["RERANK_PROVIDER"] = "lexical"
os.environ["RERANK_ENABLED"] = "true"
# ★ 同一条理由还要覆盖 LLM：`.env` 里配的是真实付费端点（openai + key），而测试只靠
# 逐个 patch 节点层的 `get_llm` 来避免出网。只要有一条路径绕开那个 patch，pytest 就会
# 真的去打 DeepSeek —— 已经发生过一次：断言"抛 TransientError"的用例拿到真模型的正常
# 返回，报 DID NOT RAISE，看起来像容错坏了，实际是**测试在花钱**。
# 默认切 mock 把这条路彻底堵死；确实要验证 openai 构造的用例自己 setenv（成本局部、可审计）。
os.environ["LLM_PROVIDER"] = "mock"
os.environ["OPENAI_API_KEY"] = ""

import pytest  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever  # noqa: E402

get_settings.cache_clear()


# 受控语料：刻意让"与查询词的词面重叠"在论文之间拉开明显差距，
# 这样重排（LexicalReranker 用 Jaccard）之后的顺序是可断言的，而不是随机的。
# chunk_id 手工指定（而非顺序生成），便于在聚合测试里精确引用。
_PAPER_A = "2401.00001"  # 与查询词高度重叠，应当排第一
_PAPER_B = "2401.00002"
_PAPER_C = "2401.00003"

_TITLES = {
    _PAPER_A: "Hybrid Retrieval with BM25 and Reranker Fusion",
    _PAPER_B: "Sparse and Dense Indexing Survey",
    _PAPER_C: "Vision Transformer Image Classification",
}

# (arxiv_id, chunk_id, section, body, source, year, category)
_SPEC = [
    (_PAPER_A, 1, "abstract",
     "We propose hybrid retrieval combining dense embedding and sparse bm25 with a reranker fusion.",
     "html", 2024, "cs.CL"),
    (_PAPER_A, 2, "method",
     "The hybrid retrieval pipeline uses bm25 retrieval and a cross encoder reranker fusion module.",
     "html", 2024, "cs.CL"),
    (_PAPER_A, 3, "results",
     "Our reranker fusion lifts hybrid retrieval recall over bm25 retrieval baseline.",
     "html", 2024, "cs.CL"),
    (_PAPER_A, 4, "conclusion",
     "Hybrid retrieval with a reranker fusion is robust and effective.",
     "html", 2024, "cs.CL"),
    (_PAPER_B, 11, "abstract",
     "We survey sparse and dense indexing techniques for information lookup systems.",
     "abstract", 2021, "cs.IR"),
    (_PAPER_B, 12, "introduction",
     "Sparse indexing relies on lexical matching while dense indexing uses learned vectors.",
     "abstract", 2021, "cs.IR"),
    (_PAPER_B, 13, "conclusion",
     "Indexing practice favors a combination of sparse and dense approaches.",
     "abstract", 2021, "cs.IR"),
    (_PAPER_C, 21, "abstract",
     "We train vision transformer models for large scale image classification tasks.",
     "html", 2023, "cs.CV"),
    (_PAPER_C, 22, "method",
     "The vision transformer patch embedding is followed by attention layers.",
     "html", 2023, "cs.CV"),
    (_PAPER_C, 23, "results",
     "Image classification accuracy improves with a deeper vision transformer.",
     "html", 2023, "cs.CV"),
]

# 跨三篇论文都能命中的关键词，用于 metadata 过滤测试
BROAD_KEYWORDS = [
    "hybrid", "retrieval", "bm25", "reranker",
    "sparse", "dense", "indexing", "vision", "transformer", "classification",
]
# 只在论文 A 上高度命中的关键词，用于端到端排序测试
FOCUSED_KEYWORDS = ["hybrid", "retrieval", "bm25", "reranker", "fusion"]
FOCUSED_QUERY = "hybrid retrieval bm25 reranker fusion"


def make_corpus() -> tuple[list[ChunkRecord], dict[str, str]]:
    """构造受控假语料（3 篇论文、每篇 3~4 个 chunk）。"""
    records = [
        ChunkRecord(
            chunk_id=cid,
            arxiv_id=aid,
            section=section,
            # 前缀 [标题][章节] 与 chunking.py 的输出保持一致，
            # 这样重排/词面重叠的测试贴近真实数据形态
            text=f"[{_TITLES[aid]}] [{section}]\n{body}",
            source=source,
            chunk_type="text",
            year=year,
            primary_category=category,
        )
        for aid, cid, section, body, source, year, category in _SPEC
    ]
    return records, dict(_TITLES)


@pytest.fixture
def corpus() -> tuple[list[ChunkRecord], dict[str, str]]:
    return make_corpus()


@pytest.fixture
async def retriever(tmp_path, corpus) -> HybridRetriever:
    """建好索引的检索器（索引文件落在 pytest 的临时目录，不污染 data/）。"""
    records, titles = corpus
    r = HybridRetriever(index_dir=tmp_path / "faiss")
    await r.build_index(records, titles)
    return r


PAPER_A = _PAPER_A
PAPER_B = _PAPER_B
PAPER_C = _PAPER_C
TITLES = _TITLES

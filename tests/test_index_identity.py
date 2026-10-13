"""索引指纹与兼容性检查的测试。

这块逻辑保护的是使用者最容易踩、也最难自查的坑：**换了 embedding 模型却没重建索引**。

它必须被测试覆盖，因为一旦这个保护失效，症状仅仅是"检索结果不太相关" ——
没有任何报错，用户在线上几乎不可能定位到问题出在索引上。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.retrieval.embedder import ApiEmbedder, BaseEmbedder, BGEEmbedder, MockEmbedder
from app.retrieval.hybrid_retriever import (
    ChunkRecord,
    HybridRetriever,
    IndexCompatibilityError,
    SearchConfig,
)

# 纯 BM25 配置：完全不依赖 embedding，用于验证"不误伤"
BM25_ONLY = SearchConfig(
    use_vector=False, use_bm25=True, fusion="none", use_rerank=False, use_metadata_filter=False
)


def _records() -> list[ChunkRecord]:
    """测试语料：**6 篇**，不是 2 篇。

    这里踩过一个坑值得记下来：`rank_bm25` 的 IDF 是 `log((N-df+0.5)/(df+0.5))`。
    当语料只有 2 篇、某词只出现 1 次时，IDF = log(1.5/1.5) = **0**，
    而 `BM25Store.search` 会把 0 分结果当噪声过滤掉 —— 于是"纯 BM25 检索返回空",
    看起来像功能坏了，其实只是语料太小。语料扩到 6 篇后 IDF 为正，行为才符合直觉。
    """
    spec = [
        (1, "2401.00001", "hybrid retrieval with bm25 and dense embeddings for reranking"),
        (2, "2401.00001", "the cross encoder reranker rescores the fused candidate list"),
        (3, "2401.00002", "sparse indexing survey for information retrieval systems"),
        (4, "2401.00003", "vision transformer image classification with patch embeddings"),
        (5, "2401.00004", "graph neural networks for molecular property prediction"),
        (6, "2401.00005", "reinforcement learning from human feedback for alignment"),
    ]
    return [
        ChunkRecord(chunk_id=cid, arxiv_id=aid, section="abstract", text=text)
        for cid, aid, text in spec
    ]


def _titles() -> dict[str, str]:
    return {
        "2401.00001": "Hybrid Retrieval and Reranking",
        "2401.00002": "Sparse Indexing Survey",
        "2401.00003": "Vision Transformer Classification",
        "2401.00004": "Graph Neural Networks",
        "2401.00005": "RLHF Alignment",
    }


def _api(model: str, dim: int = 1024, base: str = "https://api.example.com/v1") -> ApiEmbedder:
    # ApiEmbedder 构造时不会联网（只有 encode 才发请求），所以测试里可以安全实例化
    return ApiEmbedder(base, "test-key", model, dim)


async def _build(index_dir: Path, embedder: BaseEmbedder | None = None) -> HybridRetriever:
    r = HybridRetriever(embedder=embedder or MockEmbedder(1024), index_dir=index_dir)
    await r.build_index(_records(), _titles())
    return r


# ------------------------------------------------------------------ identity
def test_identity_differs_across_providers() -> None:
    """三种 provider 的指纹必须互不相同，否则换 provider 检测不出来。"""
    mock = MockEmbedder(1024).identity
    api = _api("text-embedding-v3").identity
    local = BGEEmbedder("BAAI/bge-m3", 1024).identity  # 懒加载，不会触发下载
    assert len({mock, api, local}) == 3


def test_api_identity_includes_base_url() -> None:
    """同一个模型名在不同服务商那里未必是同一个模型 —— 指纹必须能区分。"""
    a = _api("text-embedding-v3", base="https://a.example.com/v1")
    b = _api("text-embedding-v3", base="https://b.example.com/v1")
    assert a.identity != b.identity


def test_api_identity_includes_model_name_not_just_dim() -> None:
    """维度相同、模型不同 —— 必须能被区分（这正是最危险的情形）。"""
    v3 = _api("text-embedding-v3", dim=1024)
    other = _api("some-other-model", dim=1024)
    assert v3.identity != other.identity


# ------------------------------------------------------------------ 写入与恢复
async def test_build_index_writes_fingerprint(tmp_path: Path) -> None:
    r = HybridRetriever(embedder=MockEmbedder(512), index_dir=tmp_path / "idx")
    assert r.index_identity is None, "空检索器的指纹应为未知"
    await r.build_index(_records(), {"2401.00001": "A"})
    assert r.index_identity == "mock:512"
    assert r.index_built_at, "应记录构建时间，便于排查'索引是什么时候建的'"


async def test_fingerprint_survives_reload(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    await _build(idx)
    # 重新构造（模拟进程重启）应从磁盘恢复指纹，而不是丢成 None
    r2 = HybridRetriever(embedder=MockEmbedder(1024), index_dir=idx)
    assert r2.index_identity == "mock:1024"


# ------------------------------------------------------------------ 兼容性
async def test_same_embedder_is_compatible(tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    await _build(idx)
    r = HybridRetriever(embedder=MockEmbedder(1024), index_dir=idx)
    assert r.check_index_compatibility() is None
    assert await r.search("hybrid retrieval"), "同配置下检索应正常工作"


async def test_same_dim_different_model_is_blocked(tmp_path: Path) -> None:
    """★ 核心用例：维度相同、模型不同。

    维度不同时 FAISS 至少会抛错；维度相同时**什么都不会发生** ——
    检索悄悄退化成噪声。这种"不报错、只是变差"的情形才是真正要防的。
    """
    idx = tmp_path / "idx"
    await _build(idx)
    r = HybridRetriever(embedder=_api("text-embedding-v3", dim=1024), index_dir=idx)

    msg = r.check_index_compatibility()
    assert msg is not None
    # 光说"不匹配"没用，必须给出**可执行的**修复命令
    assert "build_index.py" in msg

    with pytest.raises(IndexCompatibilityError):
        await r.search("hybrid retrieval")


async def test_bm25_only_not_blocked_on_mismatch(tmp_path: Path) -> None:
    """纯 BM25 检索不依赖 embedding —— 不该被向量索引的问题误伤。"""
    idx = tmp_path / "idx"
    await _build(idx)
    r = HybridRetriever(embedder=_api("text-embedding-v3"), index_dir=idx)
    assert r.check_index_compatibility() is not None
    assert await r.search("hybrid retrieval", config=BM25_ONLY)


async def test_empty_index_is_not_blocked(tmp_path: Path) -> None:
    """索引为空时没有可比对的对象，放行（否则首次部署会被自己拦住）。"""
    r = HybridRetriever(embedder=_api("anything"), index_dir=tmp_path / "empty")
    assert r.is_empty
    assert r.check_index_compatibility() is None


async def test_legacy_index_without_fingerprint_is_tolerated(tmp_path: Path) -> None:
    """老版本产出的索引没有指纹字段。

    这种情况必须放行 —— 让用户"升级代码后立刻完全不可用"，
    比索引不匹配本身更糟（而且用户根本没有办法自己恢复）。
    """
    idx = tmp_path / "idx"
    r = await _build(idx)

    # 手工剥掉指纹字段，模拟旧版本产出的索引文件
    data = np.load(r.meta_path, allow_pickle=True)
    keep = {k: data[k] for k in data.files if k != "embedding_identity"}
    np.savez_compressed(r.meta_path, **keep)

    r2 = HybridRetriever(embedder=_api("text-embedding-v3"), index_dir=idx)
    assert r2.index_identity is None, "缺少指纹字段时应视为'未知'"
    assert r2.check_index_compatibility() is None, "未知来源不应阻塞检索"

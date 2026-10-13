"""索引复用判定（`corpus_digest` / `index_is_current`）的测试。

保护的行为：**内容没变就不要重算 embedding，内容变了就必须重建。**

这两侧都曾经是真实故障：
- 只按"索引非空就跳过" → 换模型/改分块后旧索引静默失真，症状只是"结果不太相关"；
- 无条件每次重建 → 每次提问都重算全部 chunk 的向量，本地 BGE 下以分钟计。

判定输入是**语料内容与指纹**，与向量本身无关，所以这些测试用 conftest 的 mock embedding 即可离线跑。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever, corpus_digest


def _fresh(index_dir) -> HybridRetriever:
    return HybridRetriever(index_dir=index_dir)


def test_no_index_is_not_current(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    assert r.index_is_current(records, titles) is False


async def test_reuse_after_build_same_process_and_new_process(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)
    assert r.index_is_current(records, titles) is True

    # 模拟"进程重启后从磁盘恢复"：这才是每次提问走 ensure_corpus 的真实形态
    revived = _fresh(tmp_path / "faiss")
    assert revived.index_corpus_digest == r.index_corpus_digest
    assert revived.index_is_current(records, titles) is True


async def test_text_change_invalidates(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)

    changed = [dataclasses.replace(records[0], text=records[0].text + " one more word")] + list(
        records[1:]
    )
    assert r.index_is_current(changed, titles) is False


async def test_title_change_invalidates(tmp_path, corpus):
    """标题也在索引元数据里（paper 级展示用它），只改标题同样必须重建。"""
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)

    aid = records[0].arxiv_id
    assert r.index_is_current(records, {**titles, aid: titles[aid] + " (updated)"}) is False


async def test_metadata_only_change_invalidates(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)

    changed = [dataclasses.replace(records[0], section="methods")] + list(records[1:])
    assert r.index_is_current(changed, titles) is False


async def test_embedding_identity_change_invalidates(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)

    r.index_identity = "mock:768"  # 等价于"用户换了 EMBEDDING_MODEL 却没重建"
    assert r.index_is_current(records, titles) is False


async def test_old_index_without_digest_is_rebuilt(tmp_path, corpus):
    """升级前建的老索引没写 corpus_digest → 一律当"过期"重建，绝不猜。"""
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)

    path = r.meta_path
    data = dict(np.load(path, allow_pickle=True))
    data.pop("corpus_digest")
    np.savez_compressed(path, **data)

    legacy = _fresh(tmp_path / "faiss")
    assert legacy.index_corpus_digest is None
    assert legacy.index_is_current(records, titles) is False


async def test_missing_index_file_forces_rebuild(tmp_path, corpus):
    records, titles = corpus
    r = _fresh(tmp_path / "faiss")
    await r.build_index(records, titles)
    r.bm25_path.unlink()  # 三件套缺一个（半截索引）就不能复用

    r2 = _fresh(tmp_path / "faiss")
    assert r2.index_is_current(records, titles) is False


def test_digest_is_deterministic_and_order_insensitive(corpus):
    records, titles = corpus
    assert corpus_digest(records, titles) == corpus_digest(list(reversed(records)), titles)
    assert corpus_digest(records, titles) != corpus_digest(records[:-1], titles)
    assert len(corpus_digest([records[0]], {})) == 16


def test_digest_distinguishes_chunk_id_reuse(corpus):
    """同一个 chunk_id 换了文本必须被识别出来 —— 只按 id 集合比对会漏。"""
    records, _ = corpus
    a: list[ChunkRecord] = [records[0]]
    b = [dataclasses.replace(records[0], text="completely different body")]
    assert corpus_digest(a, {}) != corpus_digest(b, {})

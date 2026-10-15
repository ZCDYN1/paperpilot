"""检索门面的端到端测试（mock 模式，无需任何模型权重 / API key）。

覆盖两条最容易出错的路径：
① chunk 级融合 → **paper 级聚合** 的口径（代表分取最高、每篇限 2 块）；
② 索引落盘 → 进程重启后重新加载（这是"检索永远返回空"那类静默 bug 的现场）。
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from app.retrieval.hybrid_retriever import ChunkRecord, HybridRetriever
from tests.conftest import (
    FOCUSED_KEYWORDS,
    FOCUSED_QUERY,
    PAPER_A,
    PAPER_B,
    PAPER_C,
)


async def test_end_to_end_mock_returns_paper_level_results(retriever):
    """建索引 → 双路召回 → RRF → 聚合 → 重排，最终返回 paper 级结果且顺序符合预期。"""
    hits = await retriever.search(FOCUSED_QUERY, FOCUSED_KEYWORDS, top_k=10)

    assert {h.arxiv_id for h in hits} == {PAPER_A, PAPER_B, PAPER_C}
    # 与查询词高度重叠的论文 A 必须排第一（BM25 + 重排都指向它）
    assert hits[0].arxiv_id == PAPER_A
    assert "bm25" in hits[0].matched_by

    for hit in hits:
        assert hit.title  # 标题注入没有丢
        assert 1 <= len(hit.chunks) <= 2  # 每篇最多 2 个 chunk 进入展示
        assert hit.rep_chunk.chunk_id in {c.chunk_id for c in hit.chunks}


async def test_paper_aggregation_keeps_best_and_limits_chunks(retriever):
    """聚合规则：同一论文取最高分 chunk 作代表分，最多保留 2 个 chunk。

    直接调 `_aggregate_to_papers` 是因为重排会覆盖 `hit.score`，
    只有在这一步之前才能精确验证"代表分 = 最高分"这一口径。
    """
    fused = [
        (1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6),   # 论文 A 四个 chunk
        (11, 0.5), (12, 0.4),                        # 论文 B 两个
        (21, 0.3),                                   # 论文 C 一个
    ]
    matched = {1: {"vector"}, 2: {"bm25"}}

    hits = retriever._aggregate_to_papers(fused, matched, max_chunks_per_paper=2)

    by_id = {h.arxiv_id: h for h in hits}
    assert by_id[PAPER_A].rep_chunk.chunk_id == 1        # 代表块 = 最高分块
    assert by_id[PAPER_A].score == 0.9                    # 代表分 = 最高分
    assert [c.chunk_id for c in by_id[PAPER_A].chunks] == [1, 2]  # 第 3、4 块被截掉
    assert by_id[PAPER_A].matched_by == ["bm25", "vector"]        # 双路命中被合并去重
    assert [c.chunk_id for c in by_id[PAPER_B].chunks] == [11, 12]
    assert [c.chunk_id for c in by_id[PAPER_C].chunks] == [21]

    # 论文间按代表分降序
    assert [h.arxiv_id for h in hits] == [PAPER_A, PAPER_B, PAPER_C]


async def test_index_persists_to_disk(tmp_path, corpus):
    """索引三件套（faiss / bm25 / meta）都必须真的写进磁盘。"""
    records, titles = corpus
    index_dir = tmp_path / "faiss"
    r = HybridRetriever(index_dir=index_dir)
    await r.build_index(records, titles)

    assert (index_dir / "paperpilot.faiss").exists()
    assert (index_dir / "paperpilot.bm25.pkl").exists(), "BM25 索引从未落盘"
    assert (index_dir / "paperpilot.meta.npz").exists()


async def test_reload_restores_metadata_and_hybrid_retrieval(tmp_path, corpus):
    """模拟进程重启：一个新的 retriever 指向同一索引目录，仍须能正常检索。

    这里同时回归两个静默 bug：不恢复 meta 会导致 `is_empty` 恒为真（永远返回空），
    BM25 不落盘会导致混合检索悄悄退化成纯向量检索。
    """
    records, titles = corpus
    index_dir = tmp_path / "faiss"

    first = HybridRetriever(index_dir=index_dir)
    await first.build_index(records, titles)

    reloaded = HybridRetriever(index_dir=Path(index_dir))
    assert reloaded.size == len(records)
    assert not reloaded.is_empty

    hits = await reloaded.search(FOCUSED_QUERY, FOCUSED_KEYWORDS, top_k=10)
    assert hits, "重启后检索返回为空：meta 没有被加载"
    assert any("bm25" in h.matched_by for h in hits), "BM25 索引没有被恢复"


async def test_empty_index_returns_empty(tmp_path):
    """空索引检索要安全返回空列表，而不是抛异常。"""
    r = HybridRetriever(index_dir=tmp_path / "faiss")
    assert r.is_empty
    assert await r.search("anything", ["term"]) == []
    # 显式建空索引也不应报错
    assert await r.build_index([], {}) == 0
    assert isinstance(r.faiss.size, int) and r.faiss.size == 0


async def test_chunk_record_defaults_are_conservative():
    """ChunkRecord 的默认 source 必须是 abstract（宁可低估证据强度，不可高估）。"""
    rec = ChunkRecord(chunk_id=1, arxiv_id="x", section="abstract", text="t")
    assert rec.source == "abstract"
    assert rec.chunk_type == "text"


async def test_faiss_and_bm25_search_run_off_the_event_loop(retriever):
    """faiss(C++) / rank_bm25(纯 Python) 是同步 CPU 密集调用，必须经 `asyncio.to_thread`
    离开事件循环线程 —— 并发请求下否则会卡死整个服务。

    证明方式不靠 sleep 猜时序：假 store 的 search 阻塞在 `threading.Event` 上，
    而这个事件只能由**事件循环上的协程**置位。
    - 若 search 仍跑在循环线程：循环被同步调用冻结，置位协程永远得不到执行，
      `Event.wait` 超时 → 断言失败；
    - 若 search 已进线程池：循环照常调度置位协程，毫秒级通过。
    """
    loop_ident = threading.get_ident()
    release = threading.Event()
    worker_idents: list[int] = []

    class _GateStore:
        """包住真 store：先自证"我在哪个线程、有没有把循环卡死"，再委托原实现。"""

        def __init__(self, inner) -> None:
            self._inner = inner

        def search(self, *args, **kwargs):
            worker_idents.append(threading.get_ident())
            if not release.wait(timeout=5.0):
                raise AssertionError("事件循环被阻塞：置位协程没机会运行（检索没被丢到线程）")
            return self._inner.search(*args, **kwargs)

    retriever.faiss = _GateStore(retriever.faiss)
    retriever.bm25 = _GateStore(retriever.bm25)

    task = asyncio.create_task(retriever.search(FOCUSED_QUERY, FOCUSED_KEYWORDS, top_k=10))
    # 轮询等两路检索真正进入工作线程（不猜固定时长）；循环在 sleep 间正常调度，
    # 若假 store 跑在循环线程上，这里会一直空转直到 _GateStore 的 5s 超时先报错。
    while not worker_idents:
        await asyncio.sleep(0.01)
    assert not task.done(), "闸门还没放行，检索不应完成"

    release.set()
    hits = await asyncio.wait_for(task, timeout=5.0)

    # 两路都被线程化：各留下一条"不在事件循环线程上"的执行记录
    assert len(worker_idents) == 2
    assert all(ident != loop_ident for ident in worker_idents)
    # 线程化不能改变结果语义：双路照常参与融合，排序与既有用例一致
    assert hits and hits[0].arxiv_id == PAPER_A
    assert {PAPER_A, PAPER_B, PAPER_C} == {h.arxiv_id for h in hits}
    assert "vector" in hits[0].matched_by and "bm25" in hits[0].matched_by

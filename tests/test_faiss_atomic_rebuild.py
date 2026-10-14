"""FAISS 索引「重建 + 原子切换」（`faiss_store.rebuild_atomic`）的行为锁。

架构 §7 把这条列为硬性要求，`faiss_store.py` 的模块注释也写明了理由：
ingest（写）与 research（读）可能同时发生，**原地覆盖**会读到半截索引文件，
表现是"随机返回错结果 / 直接解析失败"—— 最难查的一类 bug（不报错，只是结果不对）。

现有测试只证明了"正常路径下能存能读"（`test_retrieval.py`），
没有任何用例检验**构建中途失败**时的状态。而这恰恰是原子切换唯一要买到的东西：
失败时旧索引必须仍然完整可读，既不能变成空索引，也不能变成半截文件。

全程本地临时目录，零网络、零模型权重。
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pytest

from app.retrieval.faiss_store import FaissStore

DIM = 8
OLD_IDS = [11, 12, 13]
NEW_IDS = [21, 22]

# 用标准正交基当向量：内积=余弦，命中谁完全确定，断言不必依赖随机性。
OLD_VECTORS = np.eye(DIM, dtype="float32")[:3]
NEW_VECTORS = np.eye(DIM, dtype="float32")[3:5]


def _query(which: int) -> np.ndarray:
    return np.eye(DIM, dtype="float32")[which]


def _seeded(path: Path) -> FaissStore:
    store = FaissStore(dim=DIM, index_path=path)
    store.rebuild_atomic(OLD_IDS, OLD_VECTORS)
    return store


# ---------------------------------------------------------------------------
def test_crash_while_writing_leaves_the_old_index_readable(tmp_path):
    """★ 写新索引的途中挂掉：磁盘上那份旧索引必须照旧可用（这正是原子切换的卖点）。

    若这里是 `index.tofile(target)` 式的原地覆盖，中断后 target 就是一堆残缺字节：
    读侧要么解析失败、要么读到半截向量返回错结果，而两者都**不会**告诉你是谁干的。
    """
    path = tmp_path / "paperpilot.faiss"
    store = _seeded(path)
    old_bytes = path.read_bytes()

    original_write = faiss.write_index

    def half_write(index, where) -> None:  # 模拟进程在写 .tmp 途中被杀
        # 不去读 `where`：faiss 原生的第二参是路径，而 Unicode 安全的写法
        # （见 `faiss_store._write_index`）传的是 PyCallbackIOWriter。
        # 这个用例要模拟的是"中途被杀"，落在哪个文件上是 `rebuild_atomic` 的约定，
        # 这里直接按同一个约定写坏 .tmp，两种 IO 写法下断言都成立。
        path.with_name(path.name + ".tmp").write_bytes(b"\x00\x01\x02 truncated mid-write")
        raise OSError("索引写入中断")

    faiss.write_index = half_write
    try:
        with pytest.raises(OSError):
            store.rebuild_atomic(NEW_IDS, NEW_VECTORS)
    finally:
        faiss.write_index = original_write

    # ① 正式文件一个字节都没被动过（原子切换 = 只改 .tmp，没走到 os.replace）
    assert path.read_bytes() == old_bytes
    # ② 内存里也还是旧索引：`self.index = fresh` 排在文件切换之后，不能被半截替换污染
    assert store.size == len(OLD_IDS)
    # ③ 新读者加载的是完整旧索引，而不是空索引或损坏文件（内积 0 的并列项顺序不固定，故比集合）
    reader = FaissStore.load(path, DIM)
    assert reader.size == len(OLD_IDS)
    assert {cid for cid, _ in reader.search(_query(0), 3)} == set(OLD_IDS)

    # ④ 残留的 .tmp 不会挡住下一次重建：`os.replace` 是直接覆盖同名文件
    store.rebuild_atomic(NEW_IDS, NEW_VECTORS)
    reader_after = FaissStore.load(path, DIM)
    assert {cid for cid, _ in reader_after.search(_query(3), 2)} == set(NEW_IDS)


def test_rebuild_failing_before_any_write_swaps_nothing(tmp_path):
    """构建阶段就抛错（向量维度不符）：磁盘与内存两侧都不该出现"新索引"的痕迹。"""
    path = tmp_path / "paperpilot.faiss"
    store = _seeded(path)
    old_bytes = path.read_bytes()

    with pytest.raises(Exception):  # noqa: B017, PT011 - faiss 的维度不符异常类型不作契约
        store.rebuild_atomic([31, 32], np.ones((2, DIM + 4), dtype="float32"))

    assert path.read_bytes() == old_bytes, "构建失败却改了正式索引文件 = 原地覆盖那类故障"
    assert store.size == len(OLD_IDS)


def test_successful_swap_publishes_whole_index_and_leaves_no_tmp(tmp_path):
    """成功路径：切换后读侧看到的是**完整的新索引**，且目录里不残留 `.tmp`。

    残留 `.tmp` 意味着某条路径用了原地写、或者 replace 之前就把文件留在了读者视野里 ——
    下一次重建读到它（同名覆盖）虽然不会坏，但它同时是"这套原子约定被绕过"的物证。
    """
    path = tmp_path / "paperpilot.faiss"
    store = _seeded(path)

    store.rebuild_atomic(NEW_IDS, NEW_VECTORS)

    assert store.size == len(NEW_IDS)
    reader = FaissStore.load(path, DIM)
    assert reader.size == len(NEW_IDS), "读侧拿到的是残缺/混合索引"
    assert [cid for cid, _ in reader.search(_query(3), 2)] == NEW_IDS
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_rebuild_with_empty_corpus_swaps_to_a_valid_empty_index(tmp_path):
    """清库重建（空语料）也必须走同一条原子路径，并留下一个"能读的空索引"。

    这是 ingest 侧的真实场景：语料清空后如果留下一个 0 字节的 faiss 文件，
    重启后 `read_index` 直接抛异常，整个检索接口会打不开 —— 比返回空结果糟得多。
    """
    path = tmp_path / "paperpilot.faiss"
    store = _seeded(path)

    store.rebuild_atomic([], np.zeros((0, DIM), dtype="float32"))

    assert path.exists()
    reader = FaissStore.load(path, DIM)
    assert reader.size == 0
    assert reader.search(_query(0), 5) == []

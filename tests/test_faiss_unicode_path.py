"""FAISS 读写必须扛得住非 ASCII 路径（中文目录名）。

为什么单独钉这一条：faiss 原生的 `read_index(path)` / `write_index(index, path)`
在 Windows 上把路径的 UTF-8 字节交给 CRT 的 `fopen`，CRT 按当前 ANSI 代码页
（中文 Windows 是 GBK）解释那串字节 —— 目录名带中文时文件根本打不开，
报的是 `could not open ...`，长得像"索引损坏"，实际是编码问题。

这个失败**只在非 ASCII 路径上出现**，而 CI 的 runner 永远是英文路径，
所以单靠 CI 永远测不出来；而中文用户把项目放在桌面/中文目录下是常态。
本文件用 `tmp_path` 造一个中文目录名，把读写跑满，钉住 `faiss_store` 的
`_write_index` / `_read_index`（走 Python `open()` + PyCallbackIO*）。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.retrieval.faiss_store import FaissStore

DIM = 8
# 中文 + 空格 + 一个不在 GBK 单字节范围内的字符：覆盖真实会遇到的路径形状
WEIRD_DIR = "arXiv论文研究Agent 中文目录"


def _vectors(n: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    vecs = rng.random((n, DIM), dtype="float32")
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def weird_dir(tmp_path: Path) -> Path:
    d = tmp_path / WEIRD_DIR / "data" / "faiss"
    d.mkdir(parents=True)
    return d


def test_save_and_load_round_trip_under_non_ascii_path(weird_dir):
    target = weird_dir / "paperpilot.faiss"
    store = FaissStore(dim=DIM, index_path=target)
    vecs = _vectors(5)
    store.add([11, 12, 13, 14, 15], vecs)
    store.save()

    assert target.exists(), "save() 在非 ASCII 路径下没写出文件"
    loaded = FaissStore.load(target, DIM)
    assert loaded.size == 5

    hits = loaded.search(vecs[2], 2)
    assert hits and hits[0][0] == 13, "读回来的索引检索结果不对（等于静默用了一份坏索引）"


def test_rebuild_atomic_writes_under_non_ascii_path(weird_dir):
    """`rebuild_atomic` 走的是另一条写路径（先写 .tmp 再 os.replace），也要一起钉住。"""
    target = weird_dir / "rebuilt.faiss"
    store = FaissStore(dim=DIM)
    vecs = _vectors(4)
    store.rebuild_atomic([21, 22, 23, 24], vecs, path=target)

    assert target.exists()
    assert not (weird_dir / "rebuilt.faiss.tmp").exists(), "临时文件没被原子切换掉"
    reloaded = FaissStore.load(target, DIM)
    assert reloaded.size == 4
    assert reloaded.search(vecs[1], 1)[0][0] == 22


def test_plain_faiss_path_api_is_not_used(tmp_path):
    """回归哨兵：如果哪天有人把 `_write_index/_read_index` 换回 `faiss.*(str(path))`，
    这条会在**英文路径**下照样通过 —— 所以它防不住 bug。

    真正防得住的是上面两条（中文目录）。这里只钉一件事：**ASCII 路径仍然正常**，
    免得为了修中文路径把普通路径写坏（比如 file object 没关、写了一半就返回）。
    """
    target = tmp_path / "plain" / "idx.faiss"
    target.parent.mkdir()
    store = FaissStore(dim=DIM, index_path=target)
    vecs = _vectors(3)
    store.add([31, 32, 33], vecs)
    store.save()
    again = FaissStore.load(target, DIM)
    assert [i for i, _ in again.search(vecs[0], 1)] == [31]

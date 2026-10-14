"""FAISS 向量索引。

三个关键决策：

1. **`IndexIDMap2(IndexFlatIP)`** —— 用内积模拟余弦相似度（前提是向量已 L2 归一化，见 embedder）。
   用 IDMap 是为了自持 id（chunk_id），后续能按 id 删除 / 重建。
2. **metadata 过滤用「超采样 + 白名单过滤」**：`IndexFlat` 不支持高效的 pre-filter
   （FAISS 的 IDSelector 主要作用于 IVF 系列索引）。这里先取 `top_k × 4` 再按白名单过滤。
   数据量上万之后应换成 `IndexIVFFlat + IDSelector`，这个取舍是**知情的**，不是疏忽。
3. **写索引必须「重建 + 原子切换」**：ingest（写）与 research（读）可能同时发生，
   原地覆盖会读到半截索引文件，表现为"随机返回错结果"——最难查的一类 bug。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

OVER_FETCH_FACTOR = 4


def _write_index(faiss_module, index, target: Path) -> None:
    """写索引：把路径交给 Python 的 `open()`，不交给 faiss 的 C++ 层。

    faiss 自带的 `write_index(index, path)` 在 Windows 上是把路径的 UTF-8 字节丢给 CRT 的
    `fopen`，而 CRT 按**当前 ANSI 代码页**（中文 Windows 是 GBK）解释那串字节 ——
    只要目录名里有非 ASCII 字符就会报 `could not open ...`，
    看起来像"索引文件损坏"，实际是编码问题，而且**只在中文/日文等路径上复现**，
    所以 CI（英文路径）永远测不出来。
    改走 `PyCallbackIOWriter` + Python 文件对象后，路径由 Python 的宽字符接口处理。
    旧版 faiss 没有这两个回调类时退回原生写法：那时退回的是原生限制，不会新增失败模式。
    """
    writer_cls = getattr(faiss_module, "PyCallbackIOWriter", None)
    if writer_cls is None:
        faiss_module.write_index(index, str(target))
        return
    with open(target, "wb") as fh:
        faiss_module.write_index(index, writer_cls(fh.write))


def _read_index(faiss_module, path: Path):
    """`_write_index` 的读侧对称实现（编码原因见其 docstring）。"""
    reader_cls = getattr(faiss_module, "PyCallbackIOReader", None)
    if reader_cls is None:
        return faiss_module.read_index(str(path))
    with open(path, "rb") as fh:
        return faiss_module.read_index(reader_cls(fh.read))


class FaissStore:
    """向量索引的极薄封装，只做「增 / 查 / 存 / 原子重建」。"""

    def __init__(self, dim: int, index_path: Path | str | None = None):
        import faiss  # 延迟导入：mock 模式下也不需要它

        self._faiss = faiss
        self.dim = dim
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.index_path = Path(index_path) if index_path else None

    # ------------------------------------------------------------------ 读
    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ 写
    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        if len(ids) == 0:
            return
        vecs = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        if vecs.shape[1] != self.dim:
            raise ValueError(f"向量维度 {vecs.shape[1]} 与索引维度 {self.dim} 不一致")
        self.index.add_with_ids(vecs, np.asarray(ids, dtype="int64"))

    def remove(self, ids: Sequence[int]) -> None:
        if len(ids) == 0:
            return
        self.index.remove_ids(np.asarray(ids, dtype="int64"))

    # ------------------------------------------------------------------ 查
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        allow_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """返回 `[(chunk_id, score), ...]`，按分数降序。"""
        if self.size == 0 or top_k <= 0:
            return []

        query = np.ascontiguousarray(np.asarray([query_vector], dtype="float32"))

        if allow_ids is None:
            scores, ids = self.index.search(query, min(top_k, self.size))
            return [
                (int(i), float(s))
                for i, s in zip(ids[0], scores[0], strict=True)
                if i != -1
            ]

        allow = {int(i) for i in allow_ids}
        if not allow:
            return []

        fetch_k = min(max(top_k * OVER_FETCH_FACTOR, top_k + 32), self.size)
        scores, ids = self.index.search(query, fetch_k)

        hits: list[tuple[int, float]] = []
        for i, s in zip(ids[0], scores[0], strict=True):
            if i == -1:
                continue
            if int(i) in allow:
                hits.append((int(i), float(s)))
            if len(hits) >= top_k:
                break
        return hits

    # ------------------------------------------------------------- 持久化
    def save(self, path: Path | str | None = None) -> None:
        target = Path(path or self.index_path)
        if target is None:
            raise ValueError("未指定索引保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_index(self._faiss, self.index, target)

    @classmethod
    def load(cls, path: Path | str, dim: int) -> FaissStore:
        store = cls(dim=dim, index_path=path)
        p = Path(path)
        if p.exists():
            store.index = _read_index(store._faiss, p)
        return store

    def rebuild_atomic(
        self,
        ids: Sequence[int],
        vectors: np.ndarray,
        path: Path | str | None = None,
    ) -> None:
        """用新数据整体重建索引，并**原子替换**磁盘上的旧文件。

        先写 `.tmp` 再 `os.replace()` —— 后者在 POSIX 与 Windows 上都是原子操作，
        保证读侧任何时刻看到的都是一份完整索引。
        """
        import faiss

        fresh = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))
        if len(ids):
            fresh.add_with_ids(
                np.ascontiguousarray(np.asarray(vectors, dtype="float32")),
                np.asarray(ids, dtype="int64"),
            )

        target = Path(path or self.index_path) if (path or self.index_path) else None
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            _write_index(faiss, fresh, tmp)
            os.replace(tmp, target)  # 原子切换

        self.index = fresh

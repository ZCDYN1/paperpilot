"""BM25 稀疏索引。

⚠️ **跨语言约束（本项目最容易踩的一条）**

主用法是「中文提问 → 检索英文论文」。BM25 是**词面匹配**，中英词表不通 ——
用中文关键词去打英文索引，召回接近随机，等于花了一半工程成本买了个噪声源。

所以查询侧必须传**英文术语**（含同义/缩写扩展），例如
`"重排" → ["reranking", "re-ranker", "cross-encoder", "rerank"]`。
**这一层不做翻译** —— 翻译与术语扩展是 Planner 节点的职责（见架构文档 §4.1 注①）。
如果把翻译塞进这一层，就会得出"BM25 无用"的消融结论，而真实原因只是关键词语言错了。

其他：
- 中文走 jieba 分词，英文按空白切并小写化
- 用 pickle 持久化（BM25Okapi 没有通用交换格式）
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

# 保留：字母数字、下划线、CJK、以及术语里常见的 + - .
_TOKEN_KEEP = re.compile(r"^[\w\u4e00-\u9fff+\-.]+$")

_STOPWORDS = frozenset(
    """a an the of to in on for and or with without by from as at is are was were be been
    this that these those it its we our you your they their he she his her""".split()
)


def tokenize(text: str) -> list[str]:
    """中英混合分词：jieba 切中文，英文小写化并剔除停用词与标点。"""
    import jieba

    tokens: list[str] = []
    for raw in jieba.lcut(text):
        token = raw.strip().lower()
        if not token or not _TOKEN_KEEP.match(token):
            continue
        if token in _STOPWORDS:
            continue
        tokens.append(token)
    return tokens


class BM25Store:
    """BM25 索引的封装，接口与 FaissStore 保持对称（build / search / save / load）。"""

    def __init__(self, path: Path | str | None = None):
        self.ids: list[int] = []
        self.texts: list[str] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25 = None
        self.path = Path(path) if path else None

    @property
    def size(self) -> int:
        return len(self.ids)

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ 建
    def build(self, ids: Sequence[int], texts: Sequence[str]) -> None:
        """整体重建（BM25 是全局统计量，新增文档会改变 IDF，所以只能整体重建）。"""
        from rank_bm25 import BM25Okapi

        self.ids = [int(i) for i in ids]
        self.texts = list(texts)
        self._corpus_tokens = [tokenize(t) for t in self.texts]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    # ------------------------------------------------------------------ 查
    def search(
        self,
        query: str | Sequence[str],
        top_k: int,
        allow_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        """返回 `[(chunk_id, score), ...]`，按分数降序。

        `query` 可以是字符串，也可以是**已切好的英文关键词列表** ——
        推荐后者（Planner 产出英文术语后直接传入，避免二次分词把术语切碎）。
        """
        if self._bm25 is None or top_k <= 0:
            return []

        if isinstance(query, str):
            query_tokens = tokenize(query)
        else:
            query_tokens = [t.strip().lower() for t in query if t and t.strip()]

        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        allow = None if allow_ids is None else {int(i) for i in allow_ids}

        hits: list[tuple[int, float]] = []
        for pos, score in enumerate(scores):
            if score <= 0:  # 一个词都没命中，不如不返回（避免噪声进融合）
                continue
            chunk_id = self.ids[pos]
            if allow is not None and chunk_id not in allow:
                continue
            hits.append((chunk_id, float(score)))

        hits.sort(key=lambda kv: (-kv[1], kv[0]))
        return hits[:top_k]

    # -------------------------------------------------------------- 持久化
    def save(self, path: Path | str | None = None) -> None:
        target = Path(path or self.path)
        if target is None:
            raise ValueError("未指定 BM25 保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self.ids,
            "texts": self.texts,
            "corpus_tokens": self._corpus_tokens,
        }
        with target.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path | str) -> BM25Store:
        from rank_bm25 import BM25Okapi

        p = Path(path)
        store = cls(path=p)
        if not p.exists():
            return store
        with p.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - 本地自产文件，非外部输入
        store.ids = payload["ids"]
        store.texts = payload["texts"]
        store._corpus_tokens = payload["corpus_tokens"]
        store._bm25 = BM25Okapi(store._corpus_tokens) if store._corpus_tokens else None
        return store

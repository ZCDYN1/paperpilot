"""BGE 交叉编码器重排。

**为什么必须有重排？**
初筛 top-50 里总有大量"摘要读起来相似、结论却相反"的论文。
向量检索和 BM25 都只看相似度，不判断"这段是否真的回答了这个问题"。
交叉编码器把 `(query, document)` 拼在一起过一遍模型，精度明显更高 ——
代价是慢，所以只对 top-30 左右的重排候选做，而不是全量。

**降级原则（架构文档 §5.4）**：模型没装好时打 warning 并退化为保持原序，
**绝不因此让整个流程崩掉**。别人 clone 下来大概率不会先装 3GB 权重，
如果这时直接抛异常，第一印象就毁了。
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class BaseReranker(ABC):
    """统一接口：输入 query + 文档列表，返回 `[(原始下标, 分数), ...]` 降序。"""

    name: str = "base"

    @abstractmethod
    def rerank_sync(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]: ...

    async def rerank(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        # 交叉编码器是重计算，必须离开事件循环（同 embedder 的理由）
        return await asyncio.to_thread(self.rerank_sync, query, documents, top_n)


class BGEReranker(BaseReranker):
    name = "bge-reranker"

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    'RERANK_ENABLED=true 且 RERANK_PROVIDER=local 需要 FlagEmbedding。\n'
                    '请执行 pip install -e ".[local]"，或把 RERANK_ENABLED 设为 false。'
                ) from exc
            self._model = FlagReranker(self.model_name, use_fp16=False)
        return self._model

    def rerank_sync(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        model = self._ensure_model()
        pairs = [[query, doc] for doc in documents]
        raw = model.compute_score(pairs, normalize=True)
        # compute_score 在单条时返回 float，多条时返回 list
        scores = [float(raw)] if isinstance(raw, (int, float)) else [float(s) for s in raw]
        ranked = sorted(enumerate(scores), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_n] if top_n else ranked


class LexicalReranker(BaseReranker):
    """无模型时的降级重排：用词面重叠度（Jaccard）打分。

    它远不如交叉编码器，但**确定性**且有一定区分度，
    足以让 mock 模式下的流程与测试跑通、且结果可断言。
    """

    name = "lexical"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re

        return {t for t in re.split(r"[\s,;:()\[\]/]+", text.lower()) if len(t) > 1}

    def rerank_sync(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        q = self._tokens(query)
        scored: list[tuple[int, float]] = []
        for idx, doc in enumerate(documents):
            d = self._tokens(doc)
            if not q or not d:
                scored.append((idx, 0.0))
                continue
            inter = len(q & d)
            union = len(q | d)
            scored.append((idx, inter / union if union else 0.0))
        ranked = sorted(scored, key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_n] if top_n else ranked


class NullReranker(BaseReranker):
    """重排关闭时使用：保留原顺序，分数沿用位置倒数，便于统一接口。"""

    name = "none"

    def rerank_sync(
        self, query: str, documents: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        ranked = [(i, 1.0 / (i + 1)) for i in range(len(documents))]
        return ranked[:top_n] if top_n else ranked


def get_reranker(settings: Settings | None = None, *, strict: bool = False) -> BaseReranker:
    """按配置构造 reranker。

    Args:
        strict: True 时模型缺失直接抛错；默认 False（降级为词面重排并打 warning）。
    """
    s = settings or get_settings()
    if not s.rerank_enabled:
        logger.info("reranker.disabled", reason="RERANK_ENABLED=false")
        return NullReranker()

    if s.rerank_provider == "lexical":
        return LexicalReranker()

    reranker = BGEReranker(s.rerank_model)
    if strict:
        return reranker

    # 预检：装不上就降级，而不是等第一个请求进来才炸
    try:
        import FlagEmbedding  # noqa: F401
    except ImportError:
        logger.warning(
            "reranker.fallback",
            reason="FlagEmbedding 未安装，已降级为词面重排",
            hint='想启用真正的 BGE 重排请执行 pip install -e ".[local]"',
        )
        return LexicalReranker()
    return reranker

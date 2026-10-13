"""Embedding 封装：三种 provider 可切换。

| provider | 说明 |
|---|---|
| `local` | 本地 BGE（FlagEmbedding），检索质量最好，首次需下载权重 |
| `api`   | OpenAI 兼容的 `/v1/embeddings`，无需下模型，但需要 key |
| `mock`  | 确定性伪向量，用于**完全离线**跑通全流程与单元测试 |

三条硬约束：

1. **必须 L2 归一化**：FAISS 侧用内积（IndexFlatIP）来模拟余弦相似度，
   前提就是向量已归一化。忘了这步，相似度排序会默默出错。
2. **编码是 CPU 密集操作**，必须丢到线程池 —— 否则它会卡死 FastAPI 的事件循环，
   表现为"单个请求跑的时候，所有其他请求都无响应"。这是异步服务里最容易被忽略的坑。
3. **mock 模式必须确定性**：同样的文本永远得到同样的向量。
   否则缓存、幂等性、单测断言全部失效。
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import Settings, get_settings


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


class BaseEmbedder(ABC):
    """所有 embedder 的统一接口。"""

    dim: int

    @property
    def identity(self) -> str:
        """**索引指纹** —— 唯一标识"这一版向量空间"。

        为什么不能只用维度判断索引能不能复用？

        不同 embedding 模型产出的向量落在完全不同的空间里，**但维度经常撞车**：
        1024 就是极常见的取值（bge-m3、text-embedding-v3 都是）。
        如果只比维度，用户换掉 embedding 服务商后不会收到任何警告 ——
        检索结果会悄悄退化成噪声，而外在表现仅仅是"结果不太相关"。

        这类"不报错、只是变差"的故障，用户几乎不可能靠自己定位到索引上。
        所以指纹必须包含 **provider + 模型名 + 维度**；远端 API 还要带上 `base_url`，
        因为同一个模型名在不同服务商那里未必是同一个东西。
        """
        return f"{type(self).__name__}:{self.dim}"

    @abstractmethod
    def encode_sync(self, texts: list[str]) -> np.ndarray:
        """同步编码，返回 shape=(n, dim) 的 **已归一化** float32 矩阵。"""

    def check_readiness(self) -> str | None:
        """自检：返回 `None` 表示可用，否则返回一句**可直接展示给用户**的说明。

        默认实现认为一切就绪 —— 只有依赖外部凭据的 provider 才需要覆写。
        存在的意义是让 `/healthz` 能替用户回答"我到底配好了没有"，
        而不是等到第一次提问才炸。
        """
        return None

    async def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        # 丢到线程池：避免阻塞事件循环（见模块 docstring 第 2 条）
        return await asyncio.to_thread(self.encode_sync, texts)


class BGEEmbedder(BaseEmbedder):
    """本地 BGE（默认 bge-m3，1024 维，支持多语言 —— 正对"中文提问检索英文论文"的场景）。"""

    def __init__(self, model_name: str, dim: int, batch_size: int = 16):
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self._model = None  # 懒加载：不要在建对象时就去下载 3GB 权重

    @property
    def identity(self) -> str:
        return f"local:{self.model_name}:{self.dim}"

    def check_readiness(self) -> str | None:
        # 用 find_spec 探测而不是直接 import：后者会把 torch 整个拉起来（数秒 + 数百 MB 内存），
        # 而健康检查是会被高频调用的接口。
        import importlib.util

        if importlib.util.find_spec("FlagEmbedding") is None:
            return 'EMBEDDING_PROVIDER=local 依赖未安装：pip install -e ".[local]"（或改用 api / mock）'
        return None

    def _ensure_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:  # pragma: no cover - 取决于是否装了 [local]
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=local 需要 FlagEmbedding。\n"
                    '请执行：pip install -e ".[local]"\n'
                    "或改用 EMBEDDING_PROVIDER=api / mock。"
                ) from exc
            self._model = BGEM3FlagModel(self.model_name, use_fp16=False)
        return self._model

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        out = model.encode(texts, batch_size=self.batch_size)
        vecs = out["dense_vecs"] if isinstance(out, dict) else out
        return l2_normalize(np.asarray(vecs, dtype="float32"))


class ApiEmbedder(BaseEmbedder):
    """走 OpenAI 兼容的 /v1/embeddings。用户不想下 3GB 模型时的退路。"""

    def __init__(self, base_url: str, api_key: str, model: str, dim: int):
        # 刻意**不在构造时抛错**。缺配置应该体现为"健康检查里明确标出来 + 首次检索时给出
        # 可操作的提示"，而不是让整个应用起不来 —— 让使用者去排查"服务为什么启动失败"，
        # 远比让他补一个环境变量困难得多。
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.dim = dim

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def check_readiness(self) -> str | None:
        if self.is_configured:
            return None
        missing = [
            name
            for name, value in (("EMBEDDING_API_BASE_URL", self.base_url), ("EMBEDDING_API_KEY", self.api_key))
            if not value
        ]
        return f"EMBEDDING_PROVIDER=api 但缺少配置：{', '.join(missing)}"

    @property
    def identity(self) -> str:
        # 带上 base_url：同一个模型名在不同服务商那里未必是同一个模型
        return f"api:{self.base_url}|{self.model}:{self.dim}"

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        import httpx

        ready = self.check_readiness()
        if ready:
            raise RuntimeError(ready)

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        # 按 index 排序，避免服务端乱序返回导致向量与文本错位
        data = sorted(data, key=lambda item: item["index"])
        vecs = np.asarray([item["embedding"] for item in data], dtype="float32")
        if vecs.shape[1] != self.dim:
            # 实际维度与配置不符时以实际为准，避免下游 FAISS 维度冲突
            self.dim = int(vecs.shape[1])
        return l2_normalize(vecs)


class MockEmbedder(BaseEmbedder):
    """确定性伪向量：哈希文本 → 固定随机种子 → 向量。

    它不携带任何语义，但**同一文本永远得到同一向量**，
    因此缓存/幂等/单测都能正常工作，全流程可以离线跑通。
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    @property
    def identity(self) -> str:
        return f"mock:{self.dim}"

    def encode_sync(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return l2_normalize(np.asarray(rows, dtype="float32"))


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """按配置构造 embedder。"""
    s = settings or get_settings()
    if s.embedding_provider == "local":
        return BGEEmbedder(s.embedding_model, s.embedding_dim, s.embedding_batch_size)
    if s.embedding_provider == "api":
        return ApiEmbedder(
            base_url=s.embedding_api_base_url,
            api_key=s.embedding_api_key,
            model=s.embedding_api_model,
            dim=s.embedding_dim,
        )
    return MockEmbedder(s.embedding_dim)

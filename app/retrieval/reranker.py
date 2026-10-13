"""BGE 交叉编码器重排。

**为什么必须有重排？**
初筛 top-50 里总有大量"摘要读起来相似、结论却相反"的论文。
向量检索和 BM25 都只看相似度，不判断"这段是否真的回答了这个问题"。
交叉编码器把 `(query, document)` 拼在一起过一遍模型，精度明显更高 ——
代价是慢，所以只对 top-30 左右的重排候选做，而不是全量。

**降级原则（架构文档 §5.4）**：模型没装好/权重没下时打 warning 并退化为词面重排，
**绝不因此让整个流程崩掉**。别人 clone 下来大概率不会先装 2.3GB 权重，
如果这时直接抛异常，第一印象就毁了 —— 但降级必须**带可照抄的获取命令**（含 hf-mirror 镜像），
否则用户连"怎么把它变成真重排"都不知道。

`RERANK_MODEL` 支持两种写法：**HF 仓库名**（首次加载自动下载）或**本地目录**
（`FlagReranker` 首参即 `model_name_or_path`，与 `BGEM3FlagModel` 一致）。
"""

from __future__ import annotations

import asyncio
import importlib.util
from abc import ABC, abstractmethod
from pathlib import Path

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- 就绪性探测
def _flagembedding_installed() -> bool:
    """FlagEmbedding 是否可 import —— 用 find_spec 探测而不是直接 import。

    和 embedder 侧同理：直接 import 会把 torch 整个拉起来（数秒 + 数百 MB 内存），
    而这段预检在**每次构造检索器**时都会跑，必须便宜。
    """
    return importlib.util.find_spec("FlagEmbedding") is not None


def _looks_like_local_dir(model: str) -> bool:
    """`RERANK_MODEL` 允许两种写法：HF 仓库名（`BAAI/bge-reranker-v2-m3`）**或本地目录**。

    本地目录是"国内网络下用户自己下好权重"的唯一现实路径，
    而 `FlagReranker` 的第一个参数本来就是 `model_name_or_path`
    （与 `embedder.py` 里 `BGEM3FlagModel(self.model_name, ...)` 的用法一致），
    传目录即从该目录加载 —— 所以这里只需在**就绪性检查**里把目录形态识别出来。
    """
    if not model:
        return False
    p = Path(model).expanduser()
    return p.is_dir() and (p / "config.json").exists()


def _cached_on_hf(model: str) -> bool:
    """HF 缓存里是否已有该仓库的权重（不联网、不触发下载）。"""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001 - 没装 huggingface_hub 就当"没缓存"，走降级指引
        return False
    try:
        found = try_to_load_from_cache(model, "config.json")
    except Exception:  # noqa: BLE001 - 非法仓库名等一律视为不可用
        return False
    return isinstance(found, str)


def model_locally_available(model: str) -> bool:
    """权重是否已在本地（本地目录 或 HF 缓存）。**只探测，绝不触发下载。**"""
    return _looks_like_local_dir(model) or _cached_on_hf(model)


def download_instructions(model: str) -> str:
    """权重缺失时给用户**可照抄**的获取命令（含国内镜像）。

    为什么把指引写死在这里而不是只写在 README：clone 者未必读 README，
    但一定看得到启动时的 warning / 报错。降级可以，**无声降级不行**。
    """
    local_dir = f"models/{model.rsplit('/', 1)[-1]}" if "/" in model else f"models/{model}"
    return (
        f"BGE Reranker 权重不可用（RERANK_MODEL={model}，约 2.3GB，不随仓库分发）。\n"
        "方式①（推荐）：先设 HuggingFace 镜像（实测国内直连 huggingface.co 不通），再让首次使用时自动下载：\n"
        '    Windows PowerShell:  $env:HF_ENDPOINT="https://hf-mirror.com"\n'
        "    Git Bash / Linux:    export HF_ENDPOINT=https://hf-mirror.com\n"
        f"方式②：手动下载到本地目录，然后把 RERANK_MODEL 指向该目录（支持本地路径）：\n"
        f"    HF_ENDPOINT=https://hf-mirror.com python -c \"from huggingface_hub import snapshot_download; snapshot_download('{model}', local_dir='{local_dir}')\"\n"
        f"    RERANK_MODEL={local_dir}\n"
        '依赖缺失时另需：pip install -e ".[local]"'
    )


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
        # model_name 既可以是 HF 仓库名，也可以是**本地目录**（见 _looks_like_local_dir）
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    'RERANK_ENABLED=true 且 RERANK_PROVIDER=local 需要 FlagEmbedding。\n'
                    '请执行 pip install -e ".[local]"，或把 RERANK_ENABLED 设为 false。\n'
                    + download_instructions(self.model_name)
                ) from exc
            try:
                # FlagReranker 首参是 model_name_or_path：本地目录直接透传即可
                self._model = FlagReranker(self.model_name, use_fp16=False)
            except Exception as exc:  # noqa: BLE001 - 网络/权重缺失等加载失败都要转成"照抄可执行"的指引
                raise RuntimeError(
                    f"加载 BGE Reranker 失败（{type(exc).__name__}: {str(exc)[:200]}）。\n"
                    + download_instructions(self.model_name)
                ) from exc
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
        strict: True 时依赖/权重缺失**当场抛错**（评测与消融用，宁可失败也不要拿降级结果冒充 BGE）；
            默认 False —— 降级为词面重排并打 warning，warning 里带可照抄的获取命令。
    """
    s = settings or get_settings()
    if not s.rerank_enabled:
        logger.info("reranker.disabled", reason="RERANK_ENABLED=false")
        return NullReranker()

    if s.rerank_provider == "lexical":
        return LexicalReranker()

    # 预检分两步，都只做"探测"，**绝不触发权重下载**：
    # 1) FlagEmbedding 是否装得上（find_spec，不 import、不拉 torch）；
    # 2) 权重是否已在本地（RERANK_MODEL 指向的目录 或 HF 缓存）。
    # 缺任何一步：strict 就抛，否则降级 lexical —— 等第一个请求进来才炸，
    # 用户看到的是"检索变慢/报错"，而不是"我该装什么"。
    if not _flagembedding_installed():
        reason = '未安装 FlagEmbedding：pip install -e ".[local]"'
        if strict:
            raise RuntimeError(reason + "\n" + download_instructions(s.rerank_model))
        logger.warning(
            "reranker.fallback",
            provider="local→lexical",
            reason=reason + "，已降级为词面重排",
            guide=download_instructions(s.rerank_model),
        )
        return LexicalReranker()

    if not model_locally_available(s.rerank_model):
        reason = download_instructions(s.rerank_model)
        if strict:
            raise RuntimeError(reason)
        logger.warning(
            "reranker.fallback",
            provider="local→lexical",
            reason=f"本地未找到权重（RERANK_MODEL={s.rerank_model}），已降级为词面重排",
            guide=reason,
        )
        return LexicalReranker()

    return BGEReranker(s.rerank_model)


__all__ = [
    "BaseReranker",
    "BGEReranker",
    "LexicalReranker",
    "NullReranker",
    "download_instructions",
    "get_reranker",
    "model_locally_available",
]

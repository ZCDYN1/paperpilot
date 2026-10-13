"""健康检查。

除了给容器 healthcheck 用，它还有一个更重要的用途：
**让使用者一眼看出「我到底配好了没有」** —— 没配好会明确标出来，
而不是等第一次提问时才炸。

对"clone 下来接自己的 API"这条路径来说，这里回答的是三个递进的问题：

1. **LLM 配好了吗** —— `llm_configured`
2. **Embedding 配好了吗** —— `embedding.ready`（缺依赖 / 缺 key 都会说清是哪一种）
3. ⭐ **索引和我现在的 embedding 配置匹配吗** —— `index.compatible`
4. **容错层现在是什么状态** —— `resilience.breakers`（LLM / HTTP 两个熔断器的三态与拦截计数）
5. **追踪到底生效了没** —— `tracing.enabled`（配了 key ≠ 在上报，这里给确定答案）

第 3 条最容易被忽略，却最致命：换了 embedding 模型却不重建索引时，
检索会静默退化成噪声（维度相同的话连报错都没有）。
所以这里直接把"该跑哪条命令"告诉你。

第 4、5 条是"工程化承诺的可核查面"：README 说"重试 / 超时 / 熔断"和"LangSmith 全链路追踪"，
运维不需要读源码、也不需要等出事，打开 /healthz 就能看到熔断器叫什么、现在是不是打开着、
以及 trace 是否真的在上报。
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app import __version__
from app.core.config import Settings, get_settings
from app.core.resilience import breaker_snapshot
from app.observability.tracing import tracing_status
from app.retrieval.embedder import get_embedder

router = APIRouter(tags=["health"])

MANIFEST_NAME = "paperpilot.manifest.json"


def _embedding_status(settings: Settings) -> dict:
    """当前 embedding 配置是否可用（不联网、不加载模型）。"""
    try:
        embedder = get_embedder(settings)
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "hint": f"embedding 初始化失败：{exc}"}

    problem = embedder.check_readiness()
    return {
        "ready": problem is None,
        "provider": settings.embedding_provider,
        "identity": embedder.identity,
        "hint": problem,
    }


def _index_status(settings: Settings, embedding_identity: str | None) -> dict:
    """索引是否就绪、以及是否与当前 embedding 配置匹配。

    刻意只读那份极简清单（不加载 FAISS / npz）——
    健康检查是高频接口，每次都反序列化几百条 chunk 文本是没必要开销。
    """
    manifest_path = Path(settings.faiss_index_dir) / MANIFEST_NAME
    if not manifest_path.exists():
        return {
            "ready": False,
            "compatible": False,
            "hint": "索引尚未构建：python scripts/build_index.py",
        }

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "compatible": False, "hint": f"索引清单损坏：{exc}"}

    built_with = data.get("embedding_identity")
    compatible = built_with is None or built_with == embedding_identity
    return {
        "ready": True,
        "compatible": compatible,
        "chunks": data.get("chunks"),
        "papers": data.get("papers"),
        "built_at": data.get("built_at"),
        "built_with": built_with,
        "hint": None
        if compatible
        else (
            f"索引是用 {built_with} 建立的，与当前 embedding 配置 {embedding_identity} 不匹配。\n"
            "请重建：python scripts/build_index.py"
        ),
    }


@router.get("/healthz", summary="健康检查")
async def healthz() -> dict:
    settings = get_settings()
    embedding = _embedding_status(settings)
    return {
        "status": "ok",
        "version": __version__,
        **settings.summary(),
        "embedding": embedding,
        "index": _index_status(settings, embedding.get("identity")),
        # 容错与追踪的"开关面板"（脱敏：只有状态与阈值，绝无 key）
        "resilience": {
            "enabled": settings.resilience_enabled,
            "input_guard": settings.input_guard_enabled,
            "breakers": breaker_snapshot(),
        },
        "tracing": tracing_status(settings),
    }


@router.get("/resilience", summary="熔断器实时状态（运维排查用）")
async def resilience_state() -> dict:
    """单独开一个只读端点：/healthz 是容器 healthcheck 在高频打的，
    而"哪些下游正被熔断、拦掉了多少次"是人工排查时才看的东西。

    `GET /healthz` 里也有同一份数据；这里只是给一个更窄、更适合 `watch` 的入口。
    """
    settings = get_settings()
    return {
        "enabled": settings.resilience_enabled,
        "timeout_budget": settings.resilience_timeout,
        "attempts": settings.resilience_attempts,
        "breakers": breaker_snapshot(),
    }

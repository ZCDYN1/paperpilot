"""LangSmith 追踪接线：让 `.env` 里的 `LANGCHAIN_*` 真的生效。

## 为什么需要这个模块（而不是"langchain 自己会读环境变量"）
`Settings` 从 `.env` 读到的值是 **Pydantic 实例字段**，它**不会**回流进 `os.environ`。
而 langchain / langsmith 的自动追踪是靠**读进程环境变量**决定的
（`LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT`）。
于是以前出现过很别扭的状态：`.env` 里 `LANGCHAIN_API_KEY` 明明填了、
`settings.langchain_api_key` 也确实拿到了值，但 trace 一条都不上报 ——
因为那份配置只活在 Settings 对象里，从来没进过环境。README 却写着"LangSmith 全链路追踪"。

本模块干的就是这一件事：**在应用启动时把三个配置项同步进 `os.environ`**，
把"配了却没生效"这条静默路径彻底堵掉。

## 三条硬性要求
1. **没配就安静跳过**：不抛异常、不发网络请求、不打警告（新用户第一次跑不该被
   "追踪未配置"刷屏）。`LANGCHAIN_TRACING_V2` 会显式写成 `false`，
   这样即便宿主环境里残留了一个 true（CI、别人机器上的 shell profile）也不会偷偷上传数据；
2. **绝不打印 key**：状态字典里只有 `has_api_key` 这样的布尔值；
3. **幂等**：`configure_tracing()` 可以被 lifespan、脚本、测试反复调用。

## 与"新 clone 就能跑"的关系
追踪是**可选**能力：`pip install -e ".[tracing]"` 与否都不影响主链路
（langsmith 只是回调上报，没装就不开 callback），所以这里对接不上 langsmith 的情况
也只降级、不失败。
"""

from __future__ import annotations

import os
from typing import Any

import structlog

from app.core.config import Settings, get_settings

logger = structlog.get_logger(__name__)

ENV_TRACING = "LANGCHAIN_TRACING_V2"
ENV_API_KEY = "LANGCHAIN_API_KEY"
ENV_PROJECT = "LANGCHAIN_PROJECT"
ENV_ENDPOINT = "LANGCHAIN_ENDPOINT"


def _langsmith_available() -> tuple[bool, str | None]:
    """langsmith 包是否可用（可用≠已启用，只是"具备启用的条件"）。"""
    try:
        import langsmith  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - 没装 / 装坏了都只该降级
        return False, str(exc)
    return True, None


def configure_tracing(settings: Settings | None = None, *, environ: Any = None) -> dict[str, Any]:
    """把 `LANGCHAIN_*` 配置同步进进程环境变量，返回**脱敏**状态字典。

    返回的字典可以安全地打进日志、也可以直接端给 `/healthz`：
    它只有布尔与项目名，**永不含 key 的值**。

    Args:
        settings: 覆盖配置（测试用）。默认 `get_settings()`。
        environ: 目标环境映射，默认 `os.environ`。留出这个注入点是给单测的 ——
                 测"配置是否真的进了环境"必须能断言一个假字典，
                 而不能依赖本机进程环境（否则这测试在别人机器上必红）。
    """
    s = settings or get_settings()
    env = os.environ if environ is None else environ

    want = bool(s.langchain_tracing_v2)
    has_key = bool(s.langchain_api_key)
    enabled = want and has_key

    # 三态写清：开就写真值，关就显式写 false（不是"删掉变量"）。
    # 为什么显式写 false：宿主机 shell 里残留的 LANGCHAIN_TRACING_V2=true
    # 会让 langsmith 在没有 key 的情况下反复告警、甚至尝试上报。
    env[ENV_TRACING] = "true" if enabled else "false"
    env[ENV_PROJECT] = s.langchain_project or "paperpilot"
    if has_key:
        env[ENV_API_KEY] = s.langchain_api_key
    if s.langchain_endpoint:
        env[ENV_ENDPOINT] = s.langchain_endpoint
    else:
        env.pop(ENV_ENDPOINT, None)  # 留空 = 用 langsmith 默认端点，不该塞一个空串进去

    # langsmith 用 `@lru_cache` 缓存 `get_env_var` 的读取结果 —— 也就是说**第一次读取之后
    # 再改环境变量就没人看了**。服务启动路径（lifespan 在建客户端之前 configure）通常没事，
    # 但脚本/评测里"先跑图、后配追踪"就会退化成"配了 key 却永远不上报"，而且无声无息。
    # 这里显式失效一次缓存，把"配了就该生效"变成确定行为，而不是依赖调用顺序。
    if _langsmith_available()[0]:
        try:
            from langsmith.utils import get_env_var as _ls_getenv

            _ls_getenv.cache_clear()
        except Exception as exc:  # noqa: BLE001 - 版本差异不该影响主流程，降级即可
            logger.debug("tracing.env_cache_clear_failed", error=str(exc)[:120])

    installed, import_error = _langsmith_available()

    if enabled and not installed:
        # 配了 key 也开了开关，但包没装：降级并说清怎么修（而不是静默不追踪，
        # 用户会以为"追踪坏了"，实际是少了 `pip install -e ".[tracing]"`）。
        env[ENV_TRACING] = "false"
        enabled = False
        logger.warning(
            "tracing.disabled_missing_dependency",
            hint='langsmith 不可用，追踪已跳过。启用：pip install -e ".[tracing]"',
            error=(import_error or "")[:200],
        )
    elif enabled:
        logger.info("tracing.enabled", project=env[ENV_PROJECT], endpoint=env.get(ENV_ENDPOINT) or "default")
    else:
        # 没配就安静：只在**明确开了开关却没 key**时提示一次（这是最常见的配错方式）
        if want and not has_key:
            logger.info("tracing.skipped_no_api_key", hint="LANGCHAIN_TRACING_V2=true 但 LANGCHAIN_API_KEY 为空")
        else:
            logger.debug("tracing.disabled")

    return tracing_status(s, enabled=enabled, installed=installed)


def tracing_status(settings: Settings | None = None, *, enabled: bool | None = None, installed: bool | None = None) -> dict[str, Any]:
    """脱敏的追踪状态（`/healthz` 与启动日志共用一个口径）。"""
    s = settings or get_settings()
    if enabled is None:
        enabled = bool(s.langchain_tracing_v2 and s.langchain_api_key)
    if installed is None:
        installed, _ = _langsmith_available()
    return {
        # 用户最关心的三个问题，一眼能答：装了吗 / 配了吗 / 现在真的在上报吗
        "langsmith_installed": installed,
        "requested": bool(s.langchain_tracing_v2),
        "has_api_key": bool(s.langchain_api_key),
        "enabled": bool(enabled),
        "project": s.langchain_project,
        "hint": (
            None
            if enabled
            else (
                '已请求开启但缺 LANGCHAIN_API_KEY：填上它，或把 LANGCHAIN_TRACING_V2 设为 false'
                if s.langchain_tracing_v2
                else "未开启（可选能力）。想在全链路看 trace：LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY=ks-…"
            )
        ),
    }


__all__ = ["configure_tracing", "tracing_status"]

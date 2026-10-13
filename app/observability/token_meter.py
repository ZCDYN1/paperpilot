"""按「节点」归集 LLM token 用量（架构 §12「Token 统计」）。

## 为什么需要单独一个计量器
`token_usage` 现在只能落一条 `node_name="all"` 的总量，看不出**成本瓶颈在哪个节点**——
而「让使用者判断这个任务是否值得跑、成本瓶颈在哪」正是架构 §12 给的验收口径。
所以要在每次 LLM 调用结束后把用量**按当前节点分桶**，最终得到
`{"plan": {...}, "read": {...}, "write": {...}}` 这样的分列。

## 节点名从哪来（优先级从高到低）
1. **LangGraph 注入的 `metadata["langgraph_node"]`**：回调的 `on_chat_model_start`
   能直接拿到它（LangGraph 在超级步里把节点名写进 config.metadata，再随 config
   透传给内层 runnable）。这是**唯一"不用各节点配合、也不改 graph.py"**就能拿到的来源，
   是接线能真正跑通的关键。
2. **`contextvars` 里的当前节点**（`node_scope` / `set_current_node`）：
   给"不经由图的直接调用"（单测、脚本、未来的工具调用）兜底。

## 为什么节点名走 contextvars 而不是 handler 实例属性
handler 是**进程内单例**（必须单例，才能把并行的 retrieve 分支聚合成一份全局账）。
若把"当前节点"存成 handler 的实例属性，两个并行分支会互相覆盖 —— 分桶必然串。
`contextvars` 天然是**每个 asyncio 任务一份副本**，且能穿透 LangChain 的线程池
（已实测：mock 的同步 `_generate` 在 executor 线程里跑，仍能读到调用方 task 的值）。

## 为什么 `on_llm_end` 不直接用 metadata
`on_llm_end` 的签名里**没有 `metadata`**（只有 run_id / tags），metadata 只在
`on_*_start` 给。所以这里在 start 时用 run_id 记下节点名，end 时凭 run_id 取回。
用 run_id 作 key 也顺带让"并发调用"天然隔离：每条 run 各自知道自己属于哪个节点。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# 拿不到节点名时的兜底桶。**不丢弃**这类用量：宁可归到 unknown，
# 也好过让总量对不上（对不上会让人怀疑整个计量器都在骗人）。
UNKNOWN_NODE = "unknown"

# 三个标准口径的 key：与 `app.agents.nodes._common.usage_of` 的返回保持一致，
# 这样 finalize 的 `_estimate_cost` 与 `merge_usage` 不需要为分桶结果做二次适配。
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")

_current_node: ContextVar[str | None] = ContextVar("paperpilot_current_node", default=None)


# ---------------------------------------------------------------------------
# 当前节点：contextvars 接口
# ---------------------------------------------------------------------------
def current_node() -> str | None:
    """当前上下文里的节点名（未设置时为 None）。"""
    return _current_node.get()


def set_current_node(name: str) -> Token:
    """设置当前节点，返回可用于 `reset_current_node` 的 token。"""
    return _current_node.set(name)


def reset_current_node(token: Token) -> None:
    """还原到 `set_current_node` 之前的值。"""
    _current_node.reset(token)


@contextmanager
def node_scope(name: str) -> Iterator[None]:
    """`with node_scope("read"):` 把作用域内的 LLM 调用都算到 read 账上。

    用 contextmanager 是为了保证**异常路径也会还原** —— 漏还原会让后续代码
    把用量记到错误的节点上，且这种串桶在正常路径下完全不报错。
    """
    token = _current_node.set(name)
    try:
        yield
    finally:
        _current_node.reset(token)


def resolve_node(metadata: dict[str, Any] | None = None) -> str:
    """按优先级决定这次调用属于哪个节点。

    先看 LangGraph 的 metadata（准确），再看 contextvars（兜底），最后 unknown。
    """
    if metadata:
        node = metadata.get("langgraph_node")
        if node:
            return str(node)
    return current_node() or UNKNOWN_NODE


# ---------------------------------------------------------------------------
# 从 LLMResult 里抠用量
# ---------------------------------------------------------------------------
def _normalize(raw: dict[str, Any]) -> dict[str, int]:
    """把各家 provider 的不同字段名归一成三个标准 key。

    兼容 `input_tokens/output_tokens`（langchain 1.x 的 usage_metadata）
    与 `prompt_tokens/completion_tokens`（多数 provider 的原始字段）。
    全 0 时返回空 dict —— 空用量不该占一个桶、也不该让 `calls` 计数虚高。
    """
    prompt = int(raw.get("prompt_tokens", raw.get("input_tokens", 0)) or 0)
    completion = int(raw.get("completion_tokens", raw.get("output_tokens", 0)) or 0)
    total = int(raw.get("total_tokens", 0) or 0) or (prompt + completion)
    if not (prompt or completion or total):
        return {}
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def usage_from_result(response: Any) -> dict[str, int]:
    """从 `LLMResult` 抽出 token 用量。

    两条来源都试：`llm_output.token_usage`（mock / 部分 provider）优先，
    再退回 `generations[..].message.usage_metadata`（langchain 1.x 标准位）。
    任一取到就返回，避免同一份用量被算两遍。
    """
    llm_output = getattr(response, "llm_output", None) or {}
    for key in ("token_usage", "usage"):
        raw = llm_output.get(key)
        if isinstance(raw, dict):
            usage = _normalize(raw)
            if usage:
                return usage

    for generation_list in getattr(response, "generations", None) or []:
        for generation in generation_list:
            message = getattr(generation, "message", None)
            meta = getattr(message, "usage_metadata", None)
            if isinstance(meta, dict):
                usage = _normalize(meta)
                if usage:
                    return usage
    return {}


# ---------------------------------------------------------------------------
# 计量器
# ---------------------------------------------------------------------------
class TokenMeter(BaseCallbackHandler):
    """LLM 回调式的按节点 token 计量器。

    使用姿势：把**同一个实例**挂到所有 ChatModel 上（见 `app.core.llm.get_llm`），
    它就会在每次调用结束时自动归集。图跑完后用 `snapshot()` 取分桶结果。
    """

    def __init__(self) -> None:
        super().__init__()
        self._buckets: dict[str, dict[str, int]] = {}
        # run_id → 节点名。回调可能来自不同线程（mock 的同步 _generate 在 executor 里跑），
        # 所以这个映射与账本都用锁保护。锁只护"字典写入"这么短的临界区，不构成性能瓶颈。
        self._run_node: dict[Any, str] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- 回调
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """ChatModel 走这个回调（而不是 `on_llm_start`），metadata 在此刻可用。"""
        with self._lock:
            self._run_node[run_id] = resolve_node(metadata)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """非 chat 的 LLM（纯文本模型）走这里；一并支持，免得将来换模型时静默漏计。"""
        with self._lock:
            self._run_node[run_id] = resolve_node(metadata)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """一次调用结束：把用量记到它所属的节点桶。"""
        with self._lock:
            node = self._run_node.pop(run_id, None)
        # run_id 映射缺失（例如只挂了 callback 没挂 start 事件）时退回当前上下文，
        # 而不是丢弃这笔用量。
        self.record(node or resolve_node(), usage_from_result(response))

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any = None,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """失败的调用不记账，但要清掉 run_id 映射，避免它一直挂在字典里。"""
        with self._lock:
            self._run_node.pop(run_id, None)

    # ---------------------------------------------------------------- 记账
    def record(self, node: str, usage: dict[str, int]) -> None:
        """把一笔用量并入某个节点的桶。

        公开方法：mock 的**结构化输出**路径不经过 `_generate`（因此不触发回调），
        由 `app/core/llm.py` 直接调它补记 —— 否则离线模式下 plan/read/verify/revise
        四个节点的用量会在 token_by_node 里整条缺失。
        """
        if not usage:
            return
        with self._lock:
            bucket = self._buckets.setdefault(
                node, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
            )
            for key in _USAGE_KEYS:
                bucket[key] += int(usage.get(key, 0) or 0)
            bucket["calls"] += 1

    # ---------------------------------------------------------------- 取数
    def snapshot(self) -> dict[str, dict[str, int]]:
        """返回 `{节点名: {prompt_tokens, completion_tokens, total_tokens, calls}}` 的副本。

        返回副本（而不是内部字典）是为了防止调用方无意中改动账本 ——
        那会让"跑完一次图后数字却变了"变成灵异事件。
        """
        with self._lock:
            return {name: dict(bucket) for name, bucket in self._buckets.items()}

    def totals(self) -> dict[str, int]:
        """把各节点桶加总，得到与旧口径 `token_usage` 对齐的总量。"""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
        for bucket in self.snapshot().values():
            for key in total:
                total[key] += int(bucket.get(key, 0) or 0)
        return total

    def reset(self) -> None:
        """清空账本（每次研究/每个测试用例开始前调用，避免被上一次污染）。"""
        with self._lock:
            self._buckets.clear()
            self._run_node.clear()


# ---------------------------------------------------------------------------
# 进程内默认实例
# ---------------------------------------------------------------------------
_default: TokenMeter | None = None


def get_token_meter() -> TokenMeter:
    """进程内单例。

    必须单例：`get_llm()` 每次调用都会新建 ChatModel，若计量器也跟着新建，
    各节点就各记各的，永远拼不出一张完整的账。
    """
    global _default
    if _default is None:
        _default = TokenMeter()
    return _default


def reset_token_meter() -> None:
    """清空单例账本（测试友好入口）。"""
    get_token_meter().reset()


__all__ = [
    "UNKNOWN_NODE",
    "TokenMeter",
    "current_node",
    "get_token_meter",
    "node_scope",
    "reset_current_node",
    "reset_token_meter",
    "resolve_node",
    "set_current_node",
    "usage_from_result",
]

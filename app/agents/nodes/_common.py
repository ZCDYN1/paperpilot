r"""节点共用的小工具：LLM 调用 + token 记账 + 检索结果整形。

## 为什么要把 token 记账放在这里
`finalize` 要产出"本次研究消耗 42,180 tokens / 约 ¥0.63"（产品定义 §4 产出物 E），
这要求**每个节点的消耗都能被归集**。做法是让统一的调用封装顺手把
`AIMessage.usage_metadata` 抠出来返回，节点把它并进 `state["token_usage"]`
（该 channel 的 reducer 按 key 求和）。

为什么不自己按字符数估：不同分词器差异可达 ±30%，拿它算钱会算错，
而"成本可量化"是硬指标。真实 provider 一律以服务端返回的 usage 为准。

## 为什么用 `include_raw=True`
`with_structured_output` 默认只返回解析后的对象，**把原始 AIMessage 丢了** ——
丢掉的正是 usage。打开 `include_raw` 才能既拿到结构化结果、又拿到 token 计数。

## ★ 为什么"重试 → 超时 → 熔断"接在这里（以及只接在这里）
README 承诺的是"tenacity 重试 / 超时 / 熔断"一条完整链路，而不是三个各自为政的工具函数。
接线的落点选择依据只有一条：**必须是所有 LLM 调用都要经过的公共入口**。
本模块的 `structured_invoke` / `text_invoke` 正是这个入口
（plan / read / verify / revise / compare / write 六个节点全部走它们，
见 `grep -n "structured_invoke\|text_invoke" app/agents/nodes/*.py`），
所以在这里包一层 = 全项目的所有 LLM 调用都被保护，
而**不需要**改任何节点（节点里也没有重试参数可传，容易漏接）。

顺序由 `app/core/resilience.resilient()` 保证：重试在最内、超时在中间、熔断在最外
（一次调用只记一次失败，详见该模块 docstring 与 `tests/test_resilience.py` 的顺序用例）。

## 三条"不许把链路堵死"的护栏
1. `RESILIENCE_ENABLED=false` 时 `guarded()` 直接透传 —— 保护逻辑自己出问题时一条环境变量就能退场；
2. mock provider 永不失败 → 断路器永远停在 closed，离线链路行为与接线前完全一致
   （`tests/test_resilience_wiring.py` 用「mock 下连打 N 次、断言熔断计数为零」把这条钉住）；
3. 熔断打开时**快速失败并携带明确 code**（`llm_circuit_open`），由 API 层转成 SSE `error` 事件 ——
   对用户是"立刻知道下游挂了"，而不是"卡 3 分钟后超时"。LLM 没有合理的降级产物
   （伪造一份报告会违反反幻觉承诺），所以这里的"降级"= 快速失败 + 如实上报。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from app.core.llm import get_llm
from app.core.resilience import BREAKER_LLM, CircuitOpenError, guarded
from app.observability.events import emit_out_of_band
from app.observability.resilience_events import RESILIENCE_NODE, sse_observer

# 引用校验时用的空白归一化：PDF/HTML 抽取出来的文本里有大量换行与多余空格，
# 直接做子串匹配会因为"少一个空格"而误判为不通过 —— 那会把好引用错杀。
_WS_RE = re.compile(r"\s+")

# 熔断打开时上报给前端的错误码：与 API 层的异常→错误码映射同一口径，
# 前端按 code 分支（`llm_circuit_open` 可提示"稍后重试"，`internal_error` 只报错）。
CIRCUIT_OPEN_CODE = "llm_circuit_open"


def normalize_text(text: str) -> str:
    """折叠空白并统一小写，供"引文是否真在原文里"的包含判定使用。"""
    return _WS_RE.sub(" ", (text or "")).strip().lower()


def node_event(event_type: str, **data: Any) -> dict[str, Any]:
    """构造一条放进 `state["events"]` 的事件（由 SSEEmitter 转成 SSE 帧）。"""
    return {"type": event_type, **data}


def usage_of(message: Any) -> dict[str, int]:
    """从 AIMessage 里抠出 token 用量，归一成 `prompt_tokens` / `completion_tokens`。

    兼容两条来源：`usage_metadata`（langchain-core 1.x 标准位）与
    `response_metadata.token_usage`（部分 provider 只给这个）。
    """
    um = getattr(message, "usage_metadata", None) or {}
    if um:
        return {
            "prompt_tokens": int(um.get("input_tokens", 0) or 0),
            "completion_tokens": int(um.get("output_tokens", 0) or 0),
            "total_tokens": int(um.get("total_tokens", 0) or 0),
        }
    rm = (getattr(message, "response_metadata", None) or {}).get("token_usage") or {}
    if rm:
        return {
            "prompt_tokens": int(rm.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(rm.get("completion_tokens", 0) or 0),
            "total_tokens": int(rm.get("total_tokens", 0) or 0),
        }
    return {}


def merge_usage(*usages: dict[str, int]) -> dict[str, int]:
    """把若干次调用的用量并成一个（同 key 相加）。"""
    out: dict[str, int] = {}
    for u in usages:
        for k, v in (u or {}).items():
            out[k] = out.get(k, 0) + int(v or 0)
    return out


async def _structured_call(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    temperature: float | None,
) -> tuple[Any, dict[str, int]]:
    """**裸**的结构化调用（不带任何容错）。抽出来只为让"被保护的东西"边界清楚。"""
    runnable = get_llm(temperature).with_structured_output(schema, include_raw=True)
    result = await runnable.ainvoke(messages)
    parsed = result.get("parsed")
    raw = result.get("raw")
    return parsed, usage_of(raw)


async def _text_call(messages: list[BaseMessage], temperature: float | None) -> tuple[str, dict[str, int]]:
    """**裸**的自由文本调用。"""
    message: AIMessage = await get_llm(temperature).ainvoke(messages)
    content = message.content
    text = content if isinstance(content, str) else str(content)
    return text, usage_of(message)


def _on_circuit_open(breaker_name: str, exc: CircuitOpenError) -> None:
    """熔断拒绝请求时：给前端一条明确的 SSE `error`（带着可识别的 code），然后照原样抛。

    为什么"抛"也算降级：LLM 是**必需**依赖，没有合法的降级产物
    （伪造一份报告会直接违反"每条引用都能回到原文"的产品承诺）。
    这里降的是**失败的速度与可解释性** —— 用户立刻看到"下游已被熔断"，
    而不是等三个 60s 超时后收到一句含糊的 internal_error。
    """
    emit_out_of_band(
        "error",
        node=RESILIENCE_NODE,
        code=CIRCUIT_OPEN_CODE,
        message=str(exc)[:300],
        recoverable=True,  # 冷却结束后半开试探可能成功，稍后重试是有意义的
        breaker=breaker_name,
    )


async def structured_invoke(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> tuple[Any, dict[str, int]]:
    """调 LLM 并返回 `(结构化结果, token 用量)`，全程走「重试 → 超时 → 熔断」。"""
    try:
        return await guarded(
            _structured_call,
            schema,
            messages,
            temperature,
            breaker_name=BREAKER_LLM,
            observer=sse_observer,
        )
    except CircuitOpenError as exc:
        _on_circuit_open(BREAKER_LLM, exc)
        raise


async def text_invoke(
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> tuple[str, dict[str, int]]:
    """调 LLM 并返回 `(文本, token 用量)`，保护口径与 `structured_invoke` 一致。"""
    try:
        return await guarded(
            _text_call,
            messages,
            temperature,
            breaker_name=BREAKER_LLM,
            observer=sse_observer,
        )
    except CircuitOpenError as exc:
        _on_circuit_open(BREAKER_LLM, exc)
        raise


# ---------------------------------------------------------------------------
# 检索结果整形
# ---------------------------------------------------------------------------
def dedup_papers(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """跨子问题去重（paper 级，口径见架构 §5.2 的"3.5 步"）。

    同一篇论文会被多个子问题命中，直接喂给 Reader 会让它的 token 消耗
    随子问题数线性膨胀，而信息量几乎不增。去重时**保留最高分**，
    并把各分支命中的 chunk 合并（不同子问题命中的 chunk 往往互补）。
    """
    buckets: dict[str, dict[str, Any]] = {}
    for paper in retrieved or []:
        aid = paper.get("arxiv_id")
        if not aid:
            continue
        cur = buckets.get(aid)
        if cur is None:
            buckets[aid] = {
                "arxiv_id": aid,
                "title": paper.get("title", ""),
                "score": float(paper.get("score", 0.0)),
                "chunks": list(paper.get("chunks") or []),
            }
            continue
        if float(paper.get("score", 0.0)) > cur["score"]:
            cur["score"] = float(paper.get("score", 0.0))
        seen = {c.get("chunk_id") for c in cur["chunks"]}
        for chunk in paper.get("chunks") or []:
            if chunk.get("chunk_id") not in seen:
                cur["chunks"].append(chunk)
                seen.add(chunk.get("chunk_id"))
    return sorted(buckets.values(), key=lambda p: (-p["score"], p["arxiv_id"]))


def prioritize_unread(
    papers: list[dict[str, Any]], read_ids: set[str] | None
) -> list[dict[str, Any]]:
    """把长期记忆里"已精读过"的论文排到队尾（产品定义 F10「下次不重复推荐」）。

    **稳定分区**：组内保持 `dedup_papers` 给出的重排序不变，只是未读整体前移。
    不重排是因为重排分是唯一有数据支撑的顺序（架构 §5.3 的消融表），
    在这里引入任何新的打分口径都会让"为什么这篇排第 6"变得无法解释。

    **不剔除**：已读论文只是让出候选位，未读不足 `read_top_k`（精读篇数）时仍会补位进来。
    "读过"不等于"与新问题无关"，把它彻底踢出去等于主动丢掉合法证据。
    """
    if not read_ids:
        return list(papers)
    unread: list[dict[str, Any]] = []
    already: list[dict[str, Any]] = []
    for paper in papers:
        target = already if str(paper.get("arxiv_id")) in read_ids else unread
        target.append(paper)
    return unread + already


def chunk_index(retrieved: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """建 `(arxiv_id, chunk_id) → chunk` 索引。

    这是**反幻觉的落点之一**（架构 §10 输出侧）：校验与写作时，只有能在
    本次检索结果里找到对应 chunk 的引用才被允许保留。
    """
    idx: dict[tuple[str, int], dict[str, Any]] = {}
    for paper in retrieved or []:
        aid = paper.get("arxiv_id")
        for chunk in paper.get("chunks") or []:
            cid = chunk.get("chunk_id")
            if aid is None or cid is None:
                continue
            idx[(aid, int(cid))] = chunk
    return idx


__all__ = [
    "CIRCUIT_OPEN_CODE",
    "chunk_index",
    "dedup_papers",
    "merge_usage",
    "node_event",
    "normalize_text",
    "prioritize_unread",
    "structured_invoke",
    "text_invoke",
    "usage_of",
]

"""节点共用的小工具：LLM 调用 + token 记账 + 检索结果整形。

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
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from app.core.llm import get_llm

# 引用校验时用的空白归一化：PDF/HTML 抽取出来的文本里有大量换行与多余空格，
# 直接做子串匹配会因为"少一个空格"而误判为不通过 —— 那会把好引用错杀。
_WS_RE = re.compile(r"\s+")


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


async def structured_invoke(
    schema: type[BaseModel],
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> tuple[Any, dict[str, int]]:
    """调 LLM 并返回 `(结构化结果, token 用量)`。"""
    runnable = get_llm(temperature).with_structured_output(schema, include_raw=True)
    result = await runnable.ainvoke(messages)
    parsed = result.get("parsed")
    raw = result.get("raw")
    return parsed, usage_of(raw)


async def text_invoke(
    messages: list[BaseMessage],
    *,
    temperature: float | None = None,
) -> tuple[str, dict[str, int]]:
    """调 LLM 并返回 `(文本, token 用量)`。"""
    message: AIMessage = await get_llm(temperature).ainvoke(messages)
    content = message.content
    text = content if isinstance(content, str) else str(content)
    return text, usage_of(message)


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
    "chunk_index",
    "dedup_papers",
    "merge_usage",
    "node_event",
    "normalize_text",
    "structured_invoke",
    "text_invoke",
    "usage_of",
]

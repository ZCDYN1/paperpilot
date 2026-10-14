"""F11「术语解释」接口。

## 为什么词表是写死的，而不是让模型现场解释
这个产品对外承诺的是"没有依据的话就不说"。术语解释恰恰最容易编：模型对一个没见过的
缩写会给出读起来很顺、但是错的定义，而读者没有怀疑的理由。所以：

- 词表命中 → 给解释；
- 词表没有 → `known=false`，**不猜**。前端此时应当显示"词表内暂无该词条"，
  而不是退回一个 LLM 生成的通用解释。

代价是覆盖面受限于人工维护的条目数。这是刻意换取的那部分可信度。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.terms import annotate, lookup, unknown_terms

router = APIRouter(prefix="/terms", tags=["terms"])

_MAX_ANNOTATE_CHARS = 20000


class TermList(BaseModel):
    model_config = {"extra": "forbid"}

    terms: list[str] = Field(min_length=1, max_length=200)


@router.get("", summary="标出文本里出现的术语（F11）")
async def terms_annotate(text: str) -> dict[str, Any]:
    """返回 `{terms: [{term, matched, start, end, explanation, category}]}`。

    `start/end` 是字符偏移，前端据此加下划线；重叠的匹配只保留更长的那个
    （`cross-encoder reranking` 不会被拆成两个短词）。
    """
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_input", "field": "text", "reason": "text 不能为空"},
        )
    if len(text) > _MAX_ANNOTATE_CHARS:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_input",
                "field": "text",
                "reason": f"一次最多标注 {_MAX_ANNOTATE_CHARS} 个字符",
            },
        )
    return {"terms": annotate(text)}


@router.post("/unknown", summary="批量查哪些术语词表里没有（F11）")
async def terms_unknown(body: TermList) -> dict[str, Any]:
    """前端拿到 `unknown` 列表后才知道该显示"暂无词条"而不是空白。"""
    return {"requested": len(body.terms), "unknown": unknown_terms(body.terms)}


@router.get("/{term}", summary="查单个术语（F11）")
async def terms_lookup(term: str) -> dict[str, Any]:
    """⚠️ 未命中是 `200 + known=false`，不是 404。

    "这个词词表里没有"是一个**有效的查询结果**，不是请求出错 ——
    用 404 会让前端把它当成接口故障去重试，而正确的反应是显示"暂无词条"。
    """
    entry = lookup(term)
    if entry is None:
        return {"term": term, "known": False, "explanation": None}
    return {"term": entry["term"], "known": True, **entry}


__all__ = ["router"]

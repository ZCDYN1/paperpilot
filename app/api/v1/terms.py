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
from pydantic import BaseModel, Field, field_validator

from app.terms import annotate, lookup, unknown_terms

router = APIRouter(prefix="/terms", tags=["terms"])

_MAX_ANNOTATE_CHARS = 20000


class TermList(BaseModel):
    model_config = {"extra": "forbid"}

    terms: list[str] = Field(min_length=1, max_length=200)


class TermAnnotateBody(BaseModel):
    """`POST /terms` 的请求体。

    为什么要单独有 POST 这条路：要标注的是**整篇报告正文**（实测 4700+ 字符，中文
    URL 编码后按 3 字节/字算会到 14KB 以上），放进 query 就撞上请求行长度限制。
    GET 版本保留给命令行与测试用。
    """

    model_config = {"extra": "forbid"}

    text: str = Field(default="")

    @field_validator("text")
    @classmethod
    def _check_text(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text 不能为空")
        if len(v) > _MAX_ANNOTATE_CHARS:
            raise ValueError(f"一次最多标注 {_MAX_ANNOTATE_CHARS} 个字符")
        return v


def _annotate_payload(text: str) -> dict[str, Any]:
    """返回 `{matches: [{term, matched, start, end, explanation, category}]}`。

    `start/end` 是字符偏移，前端据此加下划线；重叠的匹配只保留更长的那个
    （`cross-encoder reranking` 不会被拆成两个短词）。

    键名用 `matches` 而不是 `terms`：`POST /terms/unknown` 的请求体里 `terms` 已经是
    "术语字符串列表"，同一份 API 里让 `terms` 一会儿是 list[str] 一会儿是 list[obj]
    会逼着调用方按端点区分类型。
    """
    return {"matches": annotate(text)}


@router.get("", summary="标出文本里出现的术语（F11）")
async def terms_annotate(text: str) -> dict[str, Any]:
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
    return _annotate_payload(text)


@router.post("", summary="标出文本里出现的术语（F11，正文较长时用这条）")
async def terms_annotate_post(body: TermAnnotateBody) -> dict[str, Any]:
    """与 GET 同一份结果，只是文本走请求体。校验失败由 pydantic 直接给 422。"""
    return _annotate_payload(body.text)


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

"""`revise` 节点 —— 把"没有原文支撑的论断"转成**新的补充检索子问题**。

## 它在反思回路里的位置
```
verify --(不通过)--> revise --Send 扇出--> retrieve --> read --> verify --> ...
                          ↑ 增加 revision_count，受 MAX_REVISION 约束（give_up 出口）
```
反思能解决的是"**偶尔漏检**"——即文献里有答案，但第一轮的关键词没命中。
它解决不了"**文献里根本没有答案**"，那种情况必须靠 `give_up` 如实暴露，
而不是靠重试掩盖（架构 §4.3-③）。

## 为什么用 Send 扇出而不是普通边
架构 §4.2 点名的坑：`retrieve` 依赖 Send 注入的 `sub_question`。
若 `revise → retrieve` 用普通边，`retrieve` 拿到的是**全局 state**，
而那里根本没有 `sub_question`（Send 注入的字段不落在全局 state 上），
于是它会带着空子问题去检索 —— 空 keyworeds + 空 query，检索结果随机且无声。
所以这里必须与 plan 之后同构，用 `Send`。

## 为什么把 stage 改成 revising
`read` 节点有一道幂等哨兵（`stage == "read"` 就跳过）。补充检索之后必须**重新抽取**，
所以 revise 负责把哨兵解除 —— 这样"读一次就够"和"回退时必须再读"两个需求
由同一个 `stage` 字段同时满足，不需要额外的布尔开关。
"""

from __future__ import annotations

import re

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import merge_usage, node_event, structured_invoke
from app.agents.prompts import load_prompt
from app.agents.schemas import ReviseOutput
from app.agents.state import STAGE_REVISING, ResearchState, SubQuestion

logger = structlog.get_logger(__name__)


def _fallback_keywords(state: ResearchState) -> list[str]:
    """补充子问题的关键词兜底。

    用**已检索到的论文标题**里的英文词：标题是英文、且与当前方向相关，
    比用原问题的中文去猜要可靠得多。
    """
    words: list[str] = []
    for paper in state.get("retrieved") or []:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", paper.get("title", "")):
            lw = w.lower()
            if lw not in words and lw not in {"the", "and", "for", "with", "from", "using"}:
                words.append(lw)
        if len(words) >= 6:
            break
    return words[:6] or ["retrieval", "benchmark"]


async def revise_node(state: ResearchState) -> dict:
    """针对未通过的论断生成补充检索子问题，并推进 revision_count。"""
    verification = state.get("verification") or {}
    unsupported: list[str] = list(verification.get("unsupported") or [])
    revision_count = int(state.get("revision_count", 0)) + 1

    fallback_kws = _fallback_keywords(state)
    messages = [
        SystemMessage(content=load_prompt("planner")),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                "以下论断在本轮检索结果中**没有找到原文支撑**，请为它们各设计一个"
                "更精准的补充检索子问题（关键词必须英文，含同义/缩写扩展）：\n\n"
                + "\n".join(f"- {c}" for c in unsupported)
            )
        ),
    ]

    revision_sub_questions: list[SubQuestion] = []
    usage: dict[str, int] = {}
    try:
        parsed, usage = await structured_invoke(ReviseOutput, messages)
        for i, sq in enumerate(parsed.sub_questions, start=1):
            kws = [k for k in (sq.keywords or []) if k] or fallback_kws
            revision_sub_questions.append(
                {
                    "id": f"rev-{revision_count}-{i}",
                    "text": sq.text,
                    "keywords": kws,
                    "category": sq.category,
                }
            )
    except Exception as exc:  # noqa: BLE001 - 生成失败就退化为确定性兜底，别让回路断在这
        logger.warning("revise.llm_failed", error=str(exc)[:200])

    if not revision_sub_questions:
        # 确定性兜底：模型不可用/输出为空时，用不支持论断的原文 + 英文关键词构造一个子问题。
        revision_sub_questions = [
            {
                "id": f"rev-{revision_count}-1",
                "text": "补充检索：" + ("；".join(unsupported[:3]) or "未支持的论断"),
                "keywords": fallback_kws,
                "category": None,
            }
        ]

    logger.info(
        "revise.done",
        revision_count=revision_count,
        unsupported=len(unsupported),
        new_sub_questions=len(revision_sub_questions),
    )

    return {
        "revision_sub_questions": revision_sub_questions,
        "revision_count": revision_count,
        # 解除 read 的幂等哨兵，使补充检索后能重新抽取
        "stage": STAGE_REVISING,
        "token_usage": merge_usage(usage),
        "events": [
            node_event(
                "node_started",
                node="revise",
                label=f"第 {revision_count} 次补充检索",
                unsupported=len(unsupported),
            )
        ],
    }


__all__ = ["revise_node"]

"""会话历史注入块的裁剪口径（`short_term.render_history_block`）。

这块代码是 F9 唯一的实现处：`plan` 拆子问题与 `followup` 答追问都读它，
所以这里钉住的不是"渲染好不好看"，而是**追问到底接不接得住指代**。

真实触发场景（不是假想敌）：上一轮报告 6666 字、预算 400 字，朴素截前 400 字
只能保住"总体概览 + 第一节"，用户问的"第二种方法的代价"所在的那节整个没了 ——
表现为"历史确实传进去了，追问却仍然答非所问"，从日志上完全看不出问题。
所以长回答必须**先保小标题、再保正文开头**。
"""

from __future__ import annotations

import pytest

from app.memory.short_term import (
    HISTORY_BLOCK_MAX_CHARS,
    HISTORY_MSG_MAX_CHARS,
    render_history_block,
)

# 一份有代表性的长报告：结构在前 400 字之外，还有编号子项
_REPORT = (
    "# 检索增强生成抑制幻觉：主流方法与代价\n\n"
    "## 总体概览\n\n" + "概述正文。" * 60 + "\n\n"
    "## 一、基础机制：以检索锚定生成\n\n" + "机制正文。" * 60 + "\n\n"
    "## 二、忠实性增强的两条技术路线\n\n"
    "**（1）微调式自我批判（SELF-RAG）。** " + "微调路线正文。" * 40 + "\n\n"
    "**（2）免训练的结构化证据评估（FAIR-RAG）。** " + "免训练路线正文。" * 40 + "\n\n"
    "## 三、代价：算力、容量与工程约束\n\n" + "代价正文。" * 40 + "\n\n"
    "## 参考文献\n\n[1] A — arXiv:2005.11401（章节：Abstract）\n[2] B — arXiv:2310.11511\n"
)


def test_report_is_long_enough_to_make_the_test_meaningful():
    """前置条件：这份夹具必须真的超预算，否则下面所有断言都是在测截断没发生。"""
    assert len(_REPORT) > HISTORY_MSG_MAX_CHARS


def test_ordinal_reference_survives_the_clip():
    """"第二种方法"所在的小标题必须活过截断 —— 这条块存在的唯一理由。"""
    block = render_history_block(
        [
            {"role": "user", "content": "RAG 去幻觉有哪些方法？", "is_summary": False},
            {"role": "assistant", "content": _REPORT, "is_summary": False},
        ]
    )
    assert "二、忠实性增强的两条技术路线" in block
    assert "（2）免训练的结构化证据评估" in block, "序数指代的对象被裁掉了，追问必然答非所问"
    assert "三、代价" in block


def test_clip_still_respects_the_per_message_budget():
    """保住结构不等于放弃预算：真正进 prompt 的那条历史行仍受同一个上限约束。

    量的是 `<session_history>` **里面**的内容，不含固定引导语 —— 引导语是这一块的
    固定开销，把它算进每条消息的预算会得出错误结论。
    """
    block = render_history_block([{"role": "assistant", "content": _REPORT, "is_summary": False}])
    inner = block.split("<session_history>")[1].split("</session_history>")[0].strip()
    assert len(inner) <= HISTORY_MSG_MAX_CHARS + 60


def test_plain_prose_without_headings_falls_back_to_head_truncation():
    """没有可识别结构时退回朴素截前缀：至少保住开头的信息，不制造空骨架。"""
    prose = "这段回答没有任何标题。" * 200
    block = render_history_block([{"role": "assistant", "content": prose, "is_summary": False}])
    assert "结构" not in block.split("<session_history>")[1]
    assert "助手：这段回答没有任何标题。" in block
    inner = block.split("<session_history>")[1].split("</session_history>")[0].strip()
    assert len(inner) <= HISTORY_MSG_MAX_CHARS + 10


def test_outline_and_head_are_both_kept_when_the_outline_fits():
    """骨架放得下时不该只给骨架：开头正文还要留着，模型才看得到论证从哪开始。"""
    report = "## 第一节\n\n" + "正文。" * 200 + "\n\n## 第二节\n\n" + "正文。" * 200
    block = render_history_block([{"role": "assistant", "content": report, "is_summary": False}])
    inner = block.split("<session_history>")[1].split("</session_history>")[0]
    assert "## 第二节" in inner, "只列第一节 == 序数指代仍然接不住"
    assert "（开头正文）" in inner and "正文。" in inner, "只剩骨架没有正文，模型无从判断每节说了什么"


def test_summary_rows_are_still_marked_apart_from_user_words():
    """滚动摘要与用户原话同表存储，渲染时必须区分，否则模型把压缩骨架当成用户本轮说的话。"""
    block = render_history_block(
        [
            {"role": "system", "content": "此前讨论过 RAG 的三种范式……", "is_summary": True},
            {"role": "user", "content": "只看 2024 年之后的", "is_summary": False},
        ]
    )
    assert "（更早对话的摘要）" in block
    assert "用户：只看 2024 年之后的" in block


def test_whole_block_is_capped_and_keeps_the_recent_turns():
    """多轮累积超总预算时从**头部**截：越近的轮次对指代越有用。"""
    history = [
        {"role": "assistant", "content": f"第 {i} 轮回答正文。" * 40, "is_summary": False}
        for i in range(12)
    ]
    block = render_history_block(history)
    body = block.split("<session_history>")[1]
    assert len(body) <= HISTORY_BLOCK_MAX_CHARS + 40
    assert "第 11 轮" in body, "最近的轮次被牺牲掉了，指代会指向错误的上下文"


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_short_messages_are_passed_through_untouched(role):
    """没超预算就不该出现任何改写痕迹（省略号、骨架提示）——历史是用户说过的原话。"""
    block = render_history_block([{"role": role, "content": "一句短话。", "is_summary": False}])
    assert "一句短话。" in block and "…" not in block and "结构" not in block

"""短期记忆：会话窗口 + **滚动摘要压缩**（架构 §9）。

## 要解决的问题
一次研究会话里，用户会连续追问。如果把全部历史都塞进 prompt：
1. **越用越贵**：多轮对话的 token 随轮次线性增长，第 20 轮光是历史就撑爆 context；
2. **越用越噪**：早期轮次里有大量已经"过时"的细节，反而稀释了当前问题的信号。

所以策略是**滚动压缩**：保留最近 N 轮的**原文**（保证指代、细节不丢），
更早的轮次压成一段**摘要**（保留"聊过什么、结论是什么"的骨架，丢掉冗余措辞）。

## 为什么摘要要单独存一行 `messages`（`node_name="rolling_summary"`）
而不是放内存：会话是跨进程、跨 run 的（用户关掉浏览器再来，或换个 worker 起服务）。
放内存里一重启就丢，而摘要恰恰是"最贵"的那部分（它是历史信息的浓缩物）。
落到 `messages` 表，天然跟着会话走，也和架构 §9「短期记忆落 messages 表」一致。

## 两个入口现在各是谁在用（F9 接线之后才成立的事实）
- 写：`append_message` ← `finalize` 节点，每轮落"用户提问 + 最终报告"两条；
- 读：`load_context` ← `plan` 节点（`app/agents/nodes/planner.py` 的
  `_recall_session_history`），把本会话此前的问答注入拆解子问题的 prompt。
读之前这里**只有单测在调**，所以"会话内追问"（产品定义 F9）实际不可用：数据一直在写，
却没有人把它读回来。改这块时请记住**写与读是同一个窗口的两端**，
窗口/摘要标记（`SUMMARY_NODE`）只能有一处定义。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from sqlalchemy import delete, select

from app.db.models import Message, ResearchSession
from app.db.session import session_scope

logger = structlog.get_logger(__name__)

# 摘要行的标记：用 role=system + 固定 node_name 双重标识，避免和真正的 system 消息混淆
SUMMARY_NODE = "rolling_summary"
SUMMARY_ROLE = "system"

# 一轮对话 ≈ 用户问 + 助手答两条消息
MESSAGES_PER_TURN = 2

# 摘要生成函数签名：接收（待压缩的旧消息, 已有摘要）→ 返回新摘要文本
Summarizer = Callable[[list[Message], str], Awaitable[str]]


def _clip(text: str, limit: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def _default_summarize(older: list[Message], prior_summary: str) -> str:
    """默认摘要器：**不调模型**的抽取式压缩，离线/单测都能跑。

    它做的是"保留骨架"：把每条旧消息的要点截断拼接，而不是让模型重写。
    为什么不默认用 LLM 摘要：摘要质量对成本敏感（老轮次会反复被再摘要），
    而且离线环境下根本调不通。真实链路可通过构造参数注入 LLM 版摘要器（见 `ShortTermMemory`）。
    """
    parts: list[str] = []
    if prior_summary:
        parts.append(prior_summary.strip())
    for msg in older:
        parts.append(f"{msg.role}:{_clip(msg.content, 80)}")
    return "[历史摘要] " + " | ".join(parts)


class ShortTermMemory:
    """会话短期记忆的读写门面。

    `scope` 可注入（默认 `app.db.session.session_scope`），以便单测用临时库隔离，
    不污染真实的 `data/sqlite/paperpilot.db`。
    """

    def __init__(
        self,
        *,
        max_turns: int = 6,
        scope: Callable[[], Any] | None = None,
        summarizer: Summarizer | None = None,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns 必须 >= 1")
        self.max_turns = max_turns
        self._scope = scope or session_scope
        self._summarize = summarizer or _default_summarize

    @property
    def window(self) -> int:
        """保留原文的消息条数（最近 `max_turns` 轮）。"""
        return self.max_turns * MESSAGES_PER_TURN

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        run_id: str | None = None,
        node_name: str | None = None,
    ) -> None:
        """追加一条消息；超过窗口即触发滚动压缩。

        `node_name` 用于区分消息来源（planner / writer …），
        排查"这段历史到底是谁写的"时非常有用。
        """
        async with self._scope() as session:
            await self._ensure_session(session, session_id)
            session.add(
                Message(
                    session_id=session_id,
                    run_id=run_id,
                    role=role,
                    content=content,
                    node_name=node_name,
                )
            )
        # 压缩放在事务外、单独一次会话：append 是热路径，压缩是低频操作，
        # 把它们解耦可以避免"因为压缩里的模型调用失败，把刚写入的消息也回滚掉"。
        await self.compact(session_id)

    async def compact(self, session_id: str) -> bool:
        """把超出窗口的旧消息压缩进摘要行；返回是否发生了压缩。

        保留最近 `window` 条原文，其余全部折叠进摘要。
        已有摘要会作为"前情提要"传给摘要器，保证压缩是**累加**的而不是每轮从头来。
        """
        async with self._scope() as session:
            rows = (
                (await session.execute(
                    select(Message).where(Message.session_id == session_id).order_by(Message.id)
                ))
                .scalars()
                .all()
            )
            summary_rows = [r for r in rows if r.node_name == SUMMARY_NODE]
            normal = [r for r in rows if r.node_name != SUMMARY_NODE]

            if len(normal) <= self.window:
                return False

            older = normal[: -self.window] or []
            prior = summary_rows[0].content if summary_rows else ""
            new_summary = await self._summarize(older, prior)

            # 删旧 + 去重旧摘要，再写新摘要（保证永远只有一行摘要）
            for row in older:
                await session.delete(row)
            for row in summary_rows:
                await session.delete(row)
            session.add(
                Message(session_id=session_id, role=SUMMARY_ROLE, content=new_summary, node_name=SUMMARY_NODE)
            )
            logger.info(
                "short_term.compacted",
                session_id=session_id,
                compacted=len(older),
                kept=len(normal) - len(older),
            )
            return True

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    async def load_context(self, session_id: str, max_turns: int | None = None) -> list[dict[str, Any]]:
        """载入可用上下文：摘要（若有）+ 最近 N 轮原文。

        返回的 `role` 序列可直接拼成消息列表喂给 LLM。
        摘要固定排在**最前**，因为它代表的是"更早的、已被压缩的"历史。
        """
        window = (max_turns or self.max_turns) * MESSAGES_PER_TURN
        async with self._scope() as session:
            rows = (
                (await session.execute(
                    select(Message).where(Message.session_id == session_id).order_by(Message.id)
                ))
                .scalars()
                .all()
            )

        summary = next((r for r in rows if r.node_name == SUMMARY_NODE), None)
        normal = [r for r in rows if r.node_name != SUMMARY_NODE]
        recent = normal[-window:] if window > 0 else []

        context: list[dict[str, Any]] = []
        if summary is not None:
            context.append({"role": summary.role, "content": summary.content, "node_name": SUMMARY_NODE})
        context.extend(
            {"role": r.role, "content": r.content, "node_name": r.node_name} for r in recent
        )
        return context

    async def clear(self, session_id: str) -> None:
        """清空会话消息（含摘要）。"""
        async with self._scope() as session:
            await session.execute(delete(Message).where(Message.session_id == session_id))

    async def _ensure_session(self, session, session_id: str) -> None:
        """messages.session_id 有外键约束（SQLite 已开 foreign_keys=ON），会话行必须先存在。"""
        if await session.get(ResearchSession, session_id) is None:
            session.add(ResearchSession(id=session_id, title=None))


# ----------------------------------------------------------------------
# 模块级便捷接口（默认实例）
# ----------------------------------------------------------------------
_default: ShortTermMemory | None = None


def get_short_term() -> ShortTermMemory:
    """进程内默认实例。"""
    global _default
    if _default is None:
        _default = ShortTermMemory()
    return _default


async def append_message(
    session_id: str,
    role: str,
    content: str,
    *,
    run_id: str | None = None,
    node_name: str | None = None,
) -> None:
    """模块级快捷写入口。"""
    await get_short_term().append_message(
        session_id, role, content, run_id=run_id, node_name=node_name
    )


async def load_context(session_id: str, max_turns: int = 6) -> list[dict[str, Any]]:
    """模块级快捷读入口。"""
    return await get_short_term().load_context(session_id, max_turns=max_turns)


__all__ = [
    "MESSAGES_PER_TURN",
    "SUMMARY_NODE",
    "ShortTermMemory",
    "append_message",
    "get_short_term",
    "load_context",
]

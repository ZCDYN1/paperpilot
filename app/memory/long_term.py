"""长期记忆：向量化的用户偏好 / 已读论文 / 关注方向（架构 §9）。

## 它解决什么
同一个用户第二次来提问时，不该从零开始。长期记忆让系统记住三件事：
- **关注的子方向**（"长期关注 RAG 与 Agent 相关论文"）→ 注入 `plan` 的 prompt，帮助拆解
- **已读论文清单** → 避免重复精读同一批论文（既省 token 又省时间）
- **输出偏好**（"偏好表格对比""要求标注代价"）→ 影响 `write` 的成稿形态

## ★ 为什么必须有 importance + last_used_at 并做淘汰
架构 §9 的原话是"否则越用越慢、越用越吵"。具体机理：
- **越用越慢**：召回是"取全部记忆向量逐个算相似度"的线性扫描。
  记忆只增不减，扫描集合无限膨胀，召回延迟随之线性上升。
- **越用越吵**：一次性、过时的记忆（比如三个月前问过的一个边角问题）会和
  当前真正相关的记忆一起被召回，把 prompt 噪声化 —— 召回数量没变，信噪比却在下降。

所以本模块在**写入时**就带 `importance`（重要度）与 `last_used_at`（最近一次被用到的时刻），
并在超出容量时按 `importance + 时间衰减` 的组合分淘汰最差的那批。
"用过的记忆"会因为 `last_used_at` 被刷新而更不容易被淘汰 —— 这形成了正向反馈：
**越有用的记忆越容易留下来**。

## 存储形态
向量以 `np.float32` 通过 `tobytes()` 存进 `long_term_memories.embedding`（BLOB），
召回时 `np.frombuffer(..., dtype="float32")` 还原。
embedder 已做 L2 归一化，所以**余弦相似度 == 点积**，不需要再算模长。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import numpy as np
import structlog
from sqlalchemy import select

from app.db.models import LongTermMemory
from app.db.session import session_scope
from app.retrieval.embedder import get_embedder

logger = structlog.get_logger(__name__)

# 记忆类型（与 models.LongTermMemory.memory_type 的取值一致）
TYPE_DIRECTION = "direction"      # 关注的子方向
TYPE_READ_PAPER = "read_paper"    # 已读论文
TYPE_PREFERENCE = "preference"    # 输出偏好

# 每个会话最多保留多少条记忆；超出即淘汰。定小一点是有意的：
# 召回成本与噪声都随条数上升，而真正有用的长期记忆其实很少。
DEFAULT_MAX_ITEMS = 200

# 淘汰打分的权重：重要度为主，新鲜度为辅。
# 为什么不让新鲜度占主导：用户三年前明确表达的"必须标注代价"偏好，
# 不该因为"很久没用"就被丢掉 —— 那正是长期记忆存在的意义。
_IMPORTANCE_WEIGHT = 0.7
_RECENCY_WEIGHT = 0.3


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _as_aware(dt: datetime | None) -> datetime:
    """SQLite 取回的 datetime 往往是 naive 的（丢失 tzinfo），补成 UTC 再比较，避免 TypeError。"""
    if dt is None:
        return _utcnow()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


class LongTermMemoryStore:
    """长期记忆读写门面。

    `scope` / `embedder` 均可注入，便于单测用临时库 + mock embedder 隔离。
    """

    def __init__(
        self,
        *,
        scope: Callable[[], Any] | None = None,
        embedder: Any | None = None,
        max_items: int = DEFAULT_MAX_ITEMS,
    ) -> None:
        self._scope = scope or session_scope
        self._embedder = embedder or get_embedder()
        self.max_items = max_items

    async def _embed_one(self, text: str) -> np.ndarray:
        vectors = await self._embedder.encode([text])
        return np.asarray(vectors[0], dtype="float32")

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    async def remember(
        self,
        content: str,
        *,
        session_id: str | None = None,
        memory_type: str = TYPE_DIRECTION,
        importance: float = 0.5,
        run_id: str | None = None,
    ) -> int:
        """写入一条长期记忆并返回其 id；写入后触发淘汰。

        `importance` 由调用方给出（0~1）：例如"用户明确的输出偏好"给 0.9，
        "本次读过的某一篇论文"给 0.4 —— 前者比后者更该被长期保留。
        """
        if not content or not content.strip():
            raise ValueError("记忆内容不能为空")
        importance = min(1.0, max(0.0, float(importance)))
        vector = await self._embed_one(content)

        async with self._scope() as session:
            row = LongTermMemory(
                session_id=session_id,
                memory_type=memory_type,
                content=content,
                embedding=vector.tobytes(),
                importance=importance,
            )
            session.add(row)
            await session.flush()  # 拿到自增 id
            mem_id = int(row.id)

        await self.evict(session_id=session_id)
        logger.info(
            "long_term.remembered",
            memory_type=memory_type,
            importance=importance,
            session_id=session_id,
            run_id=run_id,
        )
        return mem_id

    async def remember_run(
        self,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
        question: str | None = None,
        paper_ids: list[str] | None = None,
        summary: str | None = None,
    ) -> list[int]:
        """一次研究结束后批量写入长期记忆（`finalize` 节点调用）。

        写三类：
        1. **关注方向**：本次研究的问题本身（重要度 0.6）
        2. **已读论文**：本次精读过的论文（重要度 0.4，供"避免重复召回"）
        3. **一次性的研究结论摘要**（重要度 0.5，含问题以便召回时有语义锚点）

        ⚠️ 这个函数**不能抛异常**：它是 `finalize` 的副作用之一，
        失败绝不该拖垮整次研究（架构 §4.2「收尾失败不能毁掉整次研究」）。
        所以内部对每一类记忆单独 try/except。
        """
        written: list[int] = []
        jobs: list[tuple[str, str, float]] = []
        if question:
            jobs.append((TYPE_DIRECTION, question, 0.6))
            if summary:
                jobs.append((TYPE_PREFERENCE, f"问题：{question}\n结论摘要：{summary}", 0.5))
        for pid in paper_ids or []:
            if pid:
                jobs.append((TYPE_READ_PAPER, f"已读论文 arxiv_id={pid}", 0.4))

        for memory_type, content, importance in jobs:
            try:
                written.append(
                    await self.remember(
                        content,
                        session_id=session_id,
                        memory_type=memory_type,
                        importance=importance,
                        run_id=run_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - 单条记忆失败不拖垮收尾
                logger.warning("long_term.remember_run_failed", error=str(exc)[:200], type=memory_type)
        return written

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    async def recall(
        self,
        query: str,
        top_k: int = 5,
        *,
        session_id: str | None = None,
        memory_type: str | None = None,
    ) -> list[str]:
        """召回与 `query` 最相关的若干条记忆（返回内容文本，按相关度降序）。

        之所以返回 `list[str]` 而不是完整记录：调用方（`plan` 节点）要的是
        "把相关记忆拼进 prompt"，它不关心里程碑字段 —— 窄接口降低耦合。
        被召回的记录会刷新 `last_used_at`，从而在淘汰打分里获得"续命"。
        """
        if not query or not query.strip():
            return []
        query_vec = await self._embed_one(query)

        async with self._scope() as session:
            stmt = select(LongTermMemory)
            if session_id is not None:
                stmt = stmt.where(LongTermMemory.session_id == session_id)
            if memory_type is not None:
                stmt = stmt.where(LongTermMemory.memory_type == memory_type)
            rows = (await session.execute(stmt)).scalars().all()

            scored: list[tuple[float, LongTermMemory]] = []
            for row in rows:
                if not row.embedding:
                    continue
                vec = np.frombuffer(row.embedding, dtype="float32")
                # 维度不一致（例如换了 embedding 模型）就跳过，避免广播报错；
                # 这类记忆实际已失效，跳过比让它把整次召回搞崩更合适。
                if vec.shape != query_vec.shape:
                    continue
                score = float(np.dot(query_vec, vec))  # 已归一化 → 点积即余弦
                scored.append((score, row))

            scored.sort(key=lambda pair: -pair[0])
            top = scored[: max(0, top_k)]
            now = _utcnow()
            for _, row in top:
                row.last_used_at = now  # 刷新使用时间 → 淘汰时更不容易被清掉
            results = [row.content for _, row in top]

        logger.info("long_term.recall", session_id=session_id, hits=len(results), top_k=top_k)
        return results

    # ------------------------------------------------------------------
    # 淘汰
    # ------------------------------------------------------------------
    def _retention_score(self, row: LongTermMemory, now: datetime) -> float:
        """保留分 = 重要度 * w1 + 新鲜度 * w2。

        新鲜度用 `1/(1+天数)` 的对数式衰减（而不是线性/指数），
        这样"昨天用过"与"一周前用过"之间仍有区分，但不会悬殊到让旧记忆毫无机会。
        """
        last = _as_aware(row.last_used_at or row.created_at)
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
        recency = 1.0 / (1.0 + age_days)
        return row.importance * _IMPORTANCE_WEIGHT + recency * _RECENCY_WEIGHT

    async def evict(self, *, session_id: str | None = None, keep: int | None = None) -> int:
        """把超出容量的记忆按保留分从低到高删掉，返回删除条数。

        防的正是架构 §9 点名的"越用越慢、越用越吵"（见模块 docstring）。
        """
        keep = self.max_items if keep is None else keep
        now = _utcnow()
        async with self._scope() as session:
            stmt = select(LongTermMemory)
            if session_id is not None:
                stmt = stmt.where(LongTermMemory.session_id == session_id)
            rows = (await session.execute(stmt)).scalars().all()
            if len(rows) <= keep:
                return 0

            ordered = sorted(rows, key=lambda r: -self._retention_score(r, now))
            victims = ordered[keep:]
            for row in victims:
                await session.delete(row)
            logger.info("long_term.evicted", removed=len(victims), remaining=keep, session_id=session_id)
            return len(victims)

    async def count(self, *, session_id: str | None = None) -> int:
        """当前记忆条数（供测试与看板使用）。"""
        async with self._scope() as session:
            stmt = select(LongTermMemory)
            if session_id is not None:
                stmt = stmt.where(LongTermMemory.session_id == session_id)
            return len((await session.execute(stmt)).scalars().all())


# ----------------------------------------------------------------------
# 模块级便捷接口（默认实例）—— `finalize` 节点通过 `remember_run` 接入
# ----------------------------------------------------------------------
_default: LongTermMemoryStore | None = None


def get_long_term() -> LongTermMemoryStore:
    """进程内默认实例。"""
    global _default
    if _default is None:
        _default = LongTermMemoryStore()
    return _default


async def remember(
    content: str,
    *,
    session_id: str | None = None,
    memory_type: str = TYPE_DIRECTION,
    importance: float = 0.5,
    run_id: str | None = None,
) -> int:
    """模块级快捷写入口。"""
    return await get_long_term().remember(
        content,
        session_id=session_id,
        memory_type=memory_type,
        importance=importance,
        run_id=run_id,
    )


async def recall(query: str, top_k: int = 5, *, session_id: str | None = None) -> list[str]:
    """模块级快捷读入口。"""
    return await get_long_term().recall(query, top_k=top_k, session_id=session_id)


async def remember_run(
    *,
    session_id: str | None = None,
    run_id: str | None = None,
    question: str | None = None,
    paper_ids: list[str] | None = None,
    summary: str | None = None,
) -> list[int]:
    """模块级快捷入口：`finalize` 节点会按这个名字探测并调用（见 finalize.py）。"""
    return await get_long_term().remember_run(
        session_id=session_id,
        run_id=run_id,
        question=question,
        paper_ids=paper_ids,
        summary=summary,
    )


__all__ = [
    "DEFAULT_MAX_ITEMS",
    "TYPE_DIRECTION",
    "TYPE_PREFERENCE",
    "TYPE_READ_PAPER",
    "LongTermMemoryStore",
    "get_long_term",
    "recall",
    "remember",
    "remember_run",
]

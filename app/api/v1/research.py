"""研究接口：SSE 流式 · HITL 恢复 · 状态与引用查询（架构 §7）。

## 这一层刻意做得很薄
图怎么跑、事件怎么发，都已经封装在 `graph.py` 与 `SSEEmitter.stream()` 里。
API 层只负责三件事，多一件都不做：
1. **协议转换**：把 SSE 事件流包成 `text/event-stream` 响应；
2. **生命周期**：请求期间持有 checkpointer，断开时正确收尾；
3. **落库**：把 run 的元信息写进 `research_runs`（用于列表页与状态查询）。

## 两个必须做对的地方
- **`X-Accel-Buffering: no`**：少了它，本地测试正常，一到 Nginx 后面就退化成
  "等 3 分钟再一次性吐全部内容"，流式效果完全失效。
- **`asyncio.CancelledError` 必须原样抛出**：客户端断开时不能吞掉它，
  否则会留下一个永远跑不完的生成器（并且 token 还在烧）。
"""

from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.agents.graph import open_graph, thread_config
from app.db.models import Citation, ResearchRun, ResearchSession
from app.db.session import session_scope
from app.observability.events import SSEEmitter
from app.schemas.research import ResearchRequest

router = APIRouter(prefix="/research", tags=["research"])
logger = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- 工具
def sse_headers() -> dict[str, str]:
    """SSE 必备响应头（架构 §7）。"""
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # 关掉网关缓冲，否则流会被攒着一次性发
        "Connection": "keep-alive",
    }


def _initial_state(req: ResearchRequest, run_id: str, session_id: str) -> dict:
    """喂给图的初始 state。

    只给"入参类"字段，其余（sub_questions / retrieved / citations…）由图自己长出来 ——
    这正是 `ResearchState` 声明成 `total=False` 的用意。
    """
    return {
        "run_id": run_id,
        "session_id": session_id,
        "question": req.question,
        "filters": req.filters.model_dump() if req.filters else {},
        "sub_questions": [],
        "retrieved": [],
        "readings": [],
        "failed_papers": [],
        "events": [],
        "citations": [],
        "revision_count": 0,
        "token_usage": {},
    }


async def _register_run(run_id: str, session_id: str, question: str) -> None:
    """登记会话与 run（幂等）—— 供列表页与 `GET /research/{run_id}` 使用。"""
    async with session_scope() as db:
        if await db.get(ResearchSession, session_id) is None:
            db.add(ResearchSession(id=session_id, title=question[:200]))
        if await db.get(ResearchRun, run_id) is None:
            db.add(
                ResearchRun(
                    id=run_id,
                    session_id=session_id,
                    thread_id=run_id,
                    question=question,
                    status="running",
                )
            )


async def _update_run_status(run_id: str, status: str) -> None:
    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
        if run is not None:
            run.status = status


# --------------------------------------------------------------------------- 主接口
@router.post("/stream", summary="SSE 流式研究（核心接口）")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    run_id = uuid.uuid4().hex[:16]
    session_id = run_id  # 一次提问即一个会话；后续「会话内追问」会复用同一个 session_id
    emitter = SSEEmitter(run_id)

    async def event_gen():
        try:
            await _register_run(run_id, session_id, req.question)
            async with open_graph() as graph:
                config = thread_config(run_id, emitter=emitter)
                async for frame in emitter.stream(
                    graph, _initial_state(req, run_id, session_id), config
                ):
                    yield frame

            await _update_run_status(run_id, "completed")
        except asyncio.CancelledError:
            # 客户端断开：记录后原样抛出，不要吞掉
            logger.info("research.client_disconnected", run_id=run_id)
            await _update_run_status(run_id, "cancelled")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("research.failed", run_id=run_id)
            await _update_run_status(run_id, "failed")
            # 流已经开始就发不了 HTTP 错误码了，只能走 SSE error 事件（架构 §8）
            yield emitter.error("api", "internal_error", str(exc)[:300], recoverable=False).to_sse()
            yield emitter.done(status="error").to_sse()
        finally:
            await emitter.flush_token_usage()

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=sse_headers()
    )


# --------------------------------------------------------------------------- HITL
class ResumeRequest(BaseModel):
    """恢复请求。

    `sub_questions` 非空表示用户**编辑过**子问题（中断点 A 的核心交互）——
    会先写回 state 再续跑，否则用户改的东西等于没改。
    """

    sub_questions: list[dict] | None = Field(
        default=None, description="用户编辑后的子问题；留空表示沿用原拆解直接继续"
    )


@router.post("/{run_id}/resume", summary="HITL 恢复（用户确认后继续）")
async def research_resume(run_id: str, body: ResumeRequest | None = None) -> StreamingResponse:
    emitter = SSEEmitter(run_id)
    body = body or ResumeRequest()

    async def event_gen():
        try:
            async with open_graph() as graph:
                config = thread_config(run_id, emitter=emitter)
                if body.sub_questions:
                    # 先落回 state，再续跑 —— 顺序反了用户编辑就白改了
                    await graph.aupdate_state(config, {"sub_questions": body.sub_questions})
                    logger.info(
                        "research.sub_questions_edited",
                        run_id=run_id,
                        count=len(body.sub_questions),
                    )
                # inputs 传 None 表示"从中断点恢复"；传 {} 会从 START 重跑、白烧一遍 token
                async for frame in emitter.stream(graph, None, config, resume=True):
                    yield frame
            await _update_run_status(run_id, "completed")
        except asyncio.CancelledError:
            logger.info("research.resume_disconnected", run_id=run_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("research.resume_failed", run_id=run_id)
            yield emitter.error("api", "resume_failed", str(exc)[:300], recoverable=False).to_sse()
            yield emitter.done(status="error").to_sse()
        finally:
            await emitter.flush_token_usage()

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=sse_headers()
    )


# --------------------------------------------------------------------------- 查询
@router.get("/{run_id}", summary="查询运行状态与最终报告")
async def get_run(run_id: str) -> dict:
    """状态取自 checkpointer（它是 state 的权威来源），元信息取自数据库。

    之所以查 checkpoint 而不是把报告也落库：报告体积大且可能被后续 revision 覆盖，
    重复存一份只会带来不一致。数据库只负责"这个 run 存在过、什么状态"。
    """
    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} 不存在")
        meta = {
            "run_id": run.id,
            "session_id": run.session_id,
            "question": run.question,
            "status": run.status,
        }

    values: dict = {}
    try:
        async with open_graph() as graph:
            snapshot = await graph.aget_state(thread_config(run_id))
            values = dict(getattr(snapshot, "values", None) or {})
    except Exception as exc:  # noqa: BLE001 - 取不到 state 不该让查询接口 500
        logger.warning("research.state_unavailable", run_id=run_id, error=str(exc)[:200])

    return {
        **meta,
        "stage": values.get("stage"),
        "sub_questions": values.get("sub_questions") or [],
        "verification": values.get("verification") or {},
        "revision_count": values.get("revision_count", 0),
        "token_usage": values.get("token_usage") or {},
        "draft": values.get("draft") or "",
        "citations": values.get("citations") or [],
        # 结构化对比表（产品定义的核心产出物 A）。
        # 这里返回的是"缺字段留空"的原始结果，前端应把空值渲染成「—」而不是隐藏整行 ——
        # 让用户看见"这里确实没有数据"，本身就是产品承诺的一部分。
        "comparison": values.get("comparison") or [],
        # 本次召回到的长期记忆（用于向用户解释"为什么推荐里没有你读过的那些"）
        "memory_context": values.get("memory_context") or [],
        # 按节点细分的 token 用量（成本瓶颈定位）
        "token_by_node": values.get("token_by_node") or {},
    }


@router.get("/{run_id}/citations", summary="引用证据链（前端画溯源卡片）")
async def get_citations(run_id: str) -> dict:
    async with session_scope() as db:
        rows = (
            (await db.execute(select(Citation).where(Citation.run_id == run_id)))
            .scalars()
            .all()
        )

    return {
        "run_id": run_id,
        "count": len(rows),
        # 引用准确率的自动口径：verified=True 的占比（架构 §12）
        "verified_rate": (
            round(sum(1 for r in rows if r.verified) / len(rows), 4) if rows else None
        ),
        "citations": [
            {
                "claim": r.claim,
                "arxiv_id": r.arxiv_id,
                "chunk_id": r.chunk_id,
                "quote": r.quote,
                "locator": r.locator,
                "source": r.source,
                "verified": r.verified,
                "note": r.note,
            }
            for r in rows
        ],
    }

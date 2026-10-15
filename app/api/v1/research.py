"""研究接口：SSE 流式 · HITL 恢复 · 状态与引用查询（架构 §7）。

## 这一层刻意做得很薄
图怎么跑、事件怎么发，都已经封装在 `graph.py` 与 `SSEEmitter.stream()` 里。
API 层只负责四件事，多一件都不做：
1. **协议转换**：把 SSE 事件流包成 `text/event-stream` 响应；
2. **生命周期**：请求期间持有 checkpointer，断开时正确收尾；
3. **落库**：把 run 的元信息写进 `research_runs`（用于列表页与状态查询）；
4. **指标埋点**：中断点 A 上"用户到底改没改拆解"写进 `research_runs.langgraph_state`
   （产品定义 §8 的「子问题拆解可用率」只有这一处能观测到 —— 图本身不知道用户改了什么）。

## 两个必须做对的地方
- **`X-Accel-Buffering: no`**：少了它，本地测试正常，一到 Nginx 后面就退化成
  "等 3 分钟再一次性吐全部内容"，流式效果完全失效。
- **`asyncio.CancelledError` 必须原样抛出**：客户端断开时不能吞掉它，
  否则会留下一个永远跑不完的生成器（并且 token 还在烧）。

## 这一层同时也是"输入侧安全闸门"（架构 §10）
`app/api/guards.py` 在这一层被调用，因为这里是**用户可控文本进入图之前的最后一站**：
- `question`（研究问题）与 resume 时用户编辑过的 `sub_questions[*].text/keywords`
  全部过 `validate_input`（长度 / 控制字符 / Prompt 注入模式）；
- 结论是"三态"而不是布尔：`allow` 放行、`degrade` 用清洗后的文本继续 + 打告警、
  `reject` 直接 400/422 —— 命中注入给 **400**（形状改对了也没用），
  空/超长/控制字符给 **422**（与 FastAPI 自己的请求体校验同口径）；
- 校验**通过后才继续**，且后续流程一律使用清洗过的文本。

不改请求形状：`ResearchRequest` 的既有字段与类型完全不动（F9 只**加了一个可选字段**
`session_id`，不传即维持原行为），只是在 handler 入口加了一道判定，
所以老客户端的合法请求照旧能跑（`tests/test_api_security.py` 把这条钉住）。

## 「会话」与「checkpoint 的 thread」不是一回事（F9 最容易搞混的地方）
- `session_id`：**产品概念**，把多次提问串成一段对话（短期记忆的读取键）。
  客户端可传（追问），不传则服务端用 `run_id` 兜底（新会话）。
- `thread_id`：**LangGraph 的断点续跑键**，永远等于 `run_id`。
  共用 thread_id 的后果不是"报错"，而是新提问接着上一轮已经跑完的图状态走 ——
  拆解被上一轮的证据污染，而且全程无异常。测试把这条钉在
  `tests/test_session_followup.py`。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import case, func, select

from app.agents.graph import open_graph, thread_config
from app.analysis.timeline import build_timeline
from app.api.guards import guard_run_id, guard_sub_questions, guard_text
from app.core.resilience import CircuitOpenError
from app.db.models import Citation, ResearchRun, ResearchSession
from app.db.session import session_scope
from app.export import RunNotFoundError, render_run_markdown
from app.export.related_work import render_related_work_markdown
from app.followup import answer_from_read_papers
from app.observability.events import SSEEmitter
from app.schemas.research import ResearchRequest

router = APIRouter(prefix="/research", tags=["research"])
logger = structlog.get_logger(__name__)

# 熔断打开时的错误码：与 `app/agents/nodes/_common.CIRCUIT_OPEN_CODE` 同口径，
# 前端只要按 code 分支，就能把"下游被熔断"和"内部错误"分开显示，而不是都当成 500。
CIRCUIT_OPEN_CODE = "llm_circuit_open"


def failure_event_code(exc: BaseException) -> tuple[str, bool]:
    """把异常映射成 (SSE error.code, recoverable)。

    `recoverable` 的判断依据很具体：熔断打开意味着"下游可能在冷却后自己恢复"，
    用户重试是有意义的；而其它未预期异常是代码问题，重试只会再失败一次。
    """
    if isinstance(exc, CircuitOpenError):
        return CIRCUIT_OPEN_CODE, True
    return "internal_error", False


# 对外 error.message 只允许是"一句人话"，不能是异常原文：DB / 网络层的异常文本
# 可能带 SQL 片段、连接串、本机绝对路径（Windows 下还含中文用户名）——这是开源项目，
# 每个 clone 者的部署环境都不该从错误响应里被推断出来。完整异常已由 logger.exception
# 落在服务端日志，排查路径是 code + run_id → 日志。
# ⚠️ code / recoverable 的取值是前端契约（`web/src/hooks/useResearchStream.ts` 按
# recoverable=false 作废流式预览），脱敏只动 message，一个 code 都不能吞。
FAILURE_MESSAGES: dict[str, str] = {
    CIRCUIT_OPEN_CODE: "下游服务暂时不可用（熔断保护中），请稍后重试。",
    "internal_error": "研究流程执行失败，服务端日志已记录异常详情；重试仍失败请携带 run_id 反馈。",
    "resume_failed": "续跑执行失败，服务端日志已记录异常详情；重试仍失败请携带 run_id 反馈。",
}


def failure_event_message(code: str) -> str:
    """按错误码给一句可展示的安全文案；未知 code 落到通用文案（契约只认 code 本身）。"""
    return FAILURE_MESSAGES.get(code, FAILURE_MESSAGES["internal_error"])


# --------------------------------------------------------------------------- 工具
def sse_headers() -> dict[str, str]:
    """SSE 必备响应头（架构 §7）。"""
    return {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # 关掉网关缓冲，否则流会被攒着一次性发
        "Connection": "keep-alive",
    }


def _initial_state(req: ResearchRequest, run_id: str, session_id: str, question: str) -> dict:
    """喂给图的初始 state。

    只给"入参类"字段，其余（sub_questions / retrieved / citations…）由图自己长出来 ——
    这正是 `ResearchState` 声明成 `total=False` 的用意。

    ⚠️ `question` 是**过了输入闸门、清洗过**的那一份（见 `app.api.guards.guard_text`）：
    控制字符已去掉、注入判定已完成。这里刻意不接受 `req.question` 原值 ——
    否则"判的是清洗后的文本、用的却是原始文本"这个坑就又回来了。

    `top_k` / `rerank` / `language` 是**请求级**开关，必须在这里进 state：
    节点若各自去读 `get_settings()`（进程级 lru_cache 单例），这两个开关就只是
    请求体上的装饰 —— 用户拨了重排开关、行为一点不变，而且从 SSE 与日志上都看不出来。
    """
    return {
        "run_id": run_id,
        "session_id": session_id,
        "question": question,
        "filters": req.filters.model_dump() if req.filters else {},
        "top_k": req.top_k,
        "rerank": req.rerank,
        "language": req.language,
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
                    # ★ thread_id 必须是 run 级，**不能**跟着 session_id 走：
                    # LangGraph 的 checkpoint 以 thread_id 为键，一个会话里共用同一个 thread
                    # 会让第二次提问"从中断点续跑"上一轮已经跑完的图状态（而不是从 START 起跑），
                    # 表现为新问题的拆解被上一轮的 readings/citations 污染。
                    # 会话是靠 `session_id` 关联多个 run 的，与 checkpoint 的键无关。
                    thread_id=run_id,
                    question=question,
                    status="running",
                )
            )


#: 走到这些状态就算"这一次跑完了"，需要落 `ended_at`。
#: `awaiting_input` 不在其中 —— 那是停在 HITL 中断点等人，run 还没结束。
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})


async def _update_run_status(run_id: str, status: str) -> None:
    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
        if run is not None:
            run.status = status
            # `ended_at` 此前全仓没有任何写入点：`app/export/markdown.py` 拿它算耗时，
            # 于是每一次研究的导出报告**无条件**写着"run 未正常收尾，起止时间不完整"，
            # 而 run 其实是 completed —— 导出的报告里出现自相矛盾的陈述。
            # 存 UTC：`started_at` 是 `func.now()`（SQLite 下即 UTC），两边同一时钟才能相减。
            if status in TERMINAL_STATUSES and run.ended_at is None:
                run.ended_at = datetime.now(UTC)


# --------------------------------------------------------------------------- 指标埋点（产品定义 §8）
# 写进 `research_runs.langgraph_state`（既有的 JSON 列），**不动任何表结构**：
#
#   langgraph_state["plan_edited"]      : true | false | null
#       false = 用户在中断点 A **没有改拆解**就直接继续（"可用率"的分子）
#       true  = 用户改了
#       null  = 想判但判不了（读不到图产出的基线，例如 checkpoint 已丢）
#   langgraph_state["plan_edit_detail"] : {"original_count", "final_count", "changes": [...]}
#       changes 取值："count" | "text" | "keywords" | "category" | "no_baseline"
#       只记**改了哪一类**，不存用户原文（那是用户可控文本，没必要在库里再存一份副本）
#
# 消费方的口径应当是：
#   可用率 = #(plan_edited is false) / #(plan_edited is not null)，分母只算"真的 resume 过"的 run。
PLAN_EDIT_FLAG = "plan_edited"
PLAN_EDIT_DETAIL = "plan_edit_detail"


def plan_edit_diff(original: list[dict], submitted: list[dict]) -> dict:
    """比较「图产出的拆解」与「用户 resume 时提交的拆解」，产出 §8 可用率要的那一位。

    三个判定细节是刻意的：
    - `submitted` 为空 == 接口文档写明的"沿用原拆解直接继续" → 判 **未修改**（不是"无法判断"）；
    - 不比较 `id`：面板回填时 id 可能被重新生成或换序，拿它判"改过"会把没改的算成改的；
    - `keywords` 按**集合**比较：顺序对检索没有语义（BM25 只看成关键词组）。
    """
    original_count = len(original)
    final_count = len(submitted)

    if not submitted:
        return {"edited": False, "changes": [], "original_count": original_count, "final_count": 0}
    if not original:
        # 基线读不到（checkpoint 丢失 / plan 没产出）：不能把"无法判断"记成"用户改了"，
        # 那会直接压低可用率，而指标失真的方向恰好是最难发现的那种。
        return {"edited": None, "changes": ["no_baseline"], "original_count": 0, "final_count": final_count}

    changes: list[str] = []
    if final_count != original_count:
        changes.append("count")
    for idx, new in enumerate(submitted):
        old = original[idx] if idx < original_count else {}
        pairs = (
            ("text", (new.get("text") or "").strip(), (old.get("text") or "").strip()),
            (
                "keywords",
                sorted(str(k) for k in (new.get("keywords") or [])),
                sorted(str(k) for k in (old.get("keywords") or [])),
            ),
            ("category", new.get("category") or None, old.get("category") or None),
        )
        for kind, left, right in pairs:
            if left != right and kind not in changes:
                changes.append(kind)

    return {"edited": bool(changes), "changes": changes, "original_count": original_count, "final_count": final_count}


async def _graph_sub_questions(graph: object, config: dict) -> list[dict]:
    """取图中断点 A 留下的拆解（判定"用户改没改"的基线）；读不到就返回空列表。"""
    try:
        snapshot = await graph.aget_state(config)  # type: ignore[attr-defined]
        values = dict(getattr(snapshot, "values", None) or {})
        return list(values.get("sub_questions") or [])
    except Exception as exc:  # noqa: BLE001 - 取不到基线只让指标变成 null，不影响这次研究
        logger.warning("research.plan_baseline_unavailable", error=str(exc)[:200])
        return []


async def _record_plan_edit(run_id: str, diff: dict) -> None:
    """把 `plan_edited` 合并进 `research_runs.langgraph_state`。

    ⚠️ 这个函数**会被调用两次**（resume 入口一次、流结束后一次）：
    `finalize` 节点落库时用一份新的摘要整体覆盖了 `langgraph_state`
    （见 `app/agents/nodes/finalize._persist`，那个文件不属于本层）。
    只在入口写一次的后果是"信号跑完就没了"，而指标列恰恰只在跑完之后才被消费。
    入口那一次的价值是：进程中途死掉 / 客户端断开时，改没改拆解这个事实已经留下来了。

    本身绝不抛异常：埋点失败不能毁掉一次已经成功的 resume。
    """
    try:
        async with session_scope() as db:
            run = await db.get(ResearchRun, run_id)
            if run is None:
                return
            # JSON 列必须**整体赋新值**：原地改 `run.langgraph_state[...]` SQLAlchemy 看不到差异
            state = dict(run.langgraph_state or {})
            state[PLAN_EDIT_FLAG] = diff["edited"]
            state[PLAN_EDIT_DETAIL] = {
                "original_count": diff["original_count"],
                "final_count": diff["final_count"],
                "changes": list(diff["changes"]),
            }
            run.langgraph_state = state
    except Exception as exc:  # noqa: BLE001 - 指标是辅助信息，读不到就继续
        logger.warning("research.plan_edit_persist_failed", run_id=run_id, error=str(exc)[:200])


# --------------------------------------------------------------------------- HITL 等待时长
# `research_runs.started_at ~ ended_at` 是**墙钟**，而 HITL 把一次研究劈成了两段：
# 中间那段是"人在看拆解、在想怎么改"，可以是一分钟也可以是通宵。直接相减会得到一个
# 自称"运行耗时"、实际混合了人类思考时间的数字 —— 导出报告里那一行就是它。
# 所以中断时打一个时间点、续跑时把它结算成累计秒数，两份都写进 `langgraph_state`
# （沿用同一份"不动表结构"的埋点载体，见上面 PLAN_EDIT_FLAG 的注释）。
AWAITING_SINCE = "awaiting_since"
HITL_WAIT_SECONDS = "hitl_wait_seconds"


async def _mutate_run_state(run_id: str, apply) -> None:
    """对 `langgraph_state` 做一次读-改-写。

    JSON 列必须**整体赋新值**，原地改字典 SQLAlchemy 看不到差异（同 `_persist_plan_edit_metric`）。
    埋点失败不该影响这次研究，所以这里吞异常只记日志。
    """
    try:
        async with session_scope() as db:
            run = await db.get(ResearchRun, run_id)
            if run is None:
                return
            state = dict(run.langgraph_state or {})
            apply(state)
            run.langgraph_state = state
    except Exception as exc:  # noqa: BLE001
        logger.warning("research.hitl_wait_tracking_failed", run_id=run_id, error=str(exc)[:200])


async def _open_awaiting_window(run_id: str) -> None:
    await _mutate_run_state(run_id, lambda s: s.__setitem__(AWAITING_SINCE, datetime.now(UTC).isoformat()))


async def _close_awaiting_window(run_id: str) -> None:
    def apply(state: dict) -> None:
        since = state.pop(AWAITING_SINCE, None)
        if not since:
            return  # 没在中断点停过（或已被结算过）
        try:
            waited = (datetime.now(UTC) - datetime.fromisoformat(since)).total_seconds()
        except ValueError:
            return
        if waited > 0:
            state[HITL_WAIT_SECONDS] = round(float(state.get(HITL_WAIT_SECONDS) or 0) + waited, 1)

    await _mutate_run_state(run_id, apply)


# --------------------------------------------------------------------------- 主接口
@router.post("/stream", summary="SSE 流式研究（核心接口）")
async def research_stream(req: ResearchRequest) -> StreamingResponse:
    # ★ 输入闸门在进入 handler 的第一行就执行：
    # 判定失败要发生在"还没开始流式"之前 —— 流一旦开始就只能靠 SSE error 事件报错，
    # 拿不到 HTTP 状态码，客户端也就分不清"我的请求不该发"与"服务端跑挂了"。
    question = guard_text(req.question, field="question")

    run_id = uuid.uuid4().hex[:16]
    # ★ F9「会话内追问」的接入口：客户端传了 session_id 就**加入那个会话**，
    # 本次提问因此能读到同一会话此前几轮的问答（短期记忆 `messages` 表，见
    # `app/memory/short_term.py` 与 `planner` 的注入点）。
    # 不传就保持原行为：`session_id = run_id`，一次提问即一个会话。
    # ⚠️ 只有 session_id 会跟着会话走，`thread_id` 始终是 run 级（见 `_register_run` 的注释），
    # 长期记忆也始终是**用户全局**、不按 session_id 过滤（见 `app/memory/long_term.py` 的★小节）。
    session_id = req.session_id or run_id
    emitter = SSEEmitter(run_id)

    async def event_gen():
        interrupted_emitted = False
        try:
            await _register_run(run_id, session_id, question)
            async with open_graph() as graph:
                config = thread_config(run_id, emitter=emitter)
                async for frame in emitter.stream(
                    graph, _initial_state(req, run_id, session_id, question), config
                ):
                    yield frame
                    # ★ 中断帧一发出就把状态落库，不等这条流跑完。
                    # 前端看到 interrupted 就会关掉 SSE 连接去渲染可编辑拆解 ——
                    # 状态若只在流末尾写，它要么停在 running、要么被下面的断开分支
                    # 改写成 cancelled，随后用户点"继续"会被 resume 门禁拒掉。
                    # （前端契约测试 stream.e2e.spec.ts 就是这条路径撞红的。）
                    if not interrupted_emitted and frame.startswith("event: interrupted"):
                        interrupted_emitted = True
                        await _update_run_status(run_id, AWAITING_INPUT_STATUS)
                        await _open_awaiting_window(run_id)

            # 状态取自发射器：停在 HITL 中断的 run 写 `awaiting_input` 而不是 `completed`。
            # 以前这里无条件写 completed，于是"正在等用户确认子问题"的 run
            # 在列表页显示成"已完成"—— 用户不会再点它，HITL 等于被静默丢弃。
            await _update_run_status(run_id, _final_status(emitter.last_status))
        except asyncio.CancelledError:
            # 客户端断开：记录后原样抛出，不要吞掉
            logger.info("research.client_disconnected", run_id=run_id)
            # 已经停在中断点的 run 不是"被取消"，是"在等用户"—— 断开是这里的正常行为，
            # 写成 cancelled 会把这条 run 从 resume 门禁外面推出去，用户再也继续不了。
            if not interrupted_emitted:
                await _update_run_status(run_id, "cancelled")
            raise
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("research.failed", run_id=run_id)
            await _update_run_status(run_id, "failed")
            emitter.last_status = "error"
            # 流已经开始就发不了 HTTP 错误码了，只能走 SSE error 事件（架构 §8）
            code, recoverable = failure_event_code(exc)
            yield emitter.error("api", code, failure_event_message(code), recoverable=recoverable).to_sse()
            yield emitter.done(status="error").to_sse()
        finally:
            await emitter.flush_token_usage()

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=sse_headers()
    )


def _final_status(last_status: str) -> str:
    """把"流怎么结束的"翻成 `research_runs.status`。"""
    if last_status == "interrupted":
        return AWAITING_INPUT_STATUS
    if last_status == "error":
        return "failed"
    return "completed"


# `research_runs.status` 的"停在 HITL 中断点 A"取值：`_final_status()` 与流里的
# 中断帧写入，列表页、`GET /{run_id}`、前端 `awaiting_input` 渲染消费的是同一个值
# （`test_stream_interrupts_after_plan_and_reports_awaiting_input` 把它钉成回归断言）。
AWAITING_INPUT_STATUS = "awaiting_input"

#: 已经跑完的状态：门禁只拦这两个。
#:
#: 为什么不用 `status == awaiting_input` 当唯一放行条件：这条 run 到底有没有停在
#: 中断点 A，权威事实在 LangGraph 的 checkpoint 里，而 `research_runs.status` 是
#: 流跑完才写的**副本**。客户端在中断点关掉 SSE（前端就是这么做的：拿到 interrupted
#: 就去渲染可编辑拆解）时，副本会留在 running、甚至被断开分支写成 cancelled ——
#: 拿副本当唯一放行条件，等于让"用户点继续"依赖"前端有没有把流读完"。
#: （这条真实踩过：前端契约测试 `stream.e2e.spec.ts` 因 resume 409 而红。）
#:
#: 反过来，被挡掉的 completed/failed 已经覆盖了这道门要防的事：
#: 对跑完的 run 反复续跑（白烧 LLM 配额、把报告重生成一遍）。
NON_RESUMABLE_STATUSES = frozenset({"completed", "failed"})


async def _ensure_resumable(run_id: str) -> None:
    """resume 门禁：跑完的 run 不能再续跑，其余放行（未知 run_id 404）。

    刻意**只读不占位**：续跑期间不改状态，进程中途死掉/客户端断开后 run 仍可从断点
    重试，不会卡在 "running" 死档。代价是"同一瞬间并发两次 resume"都能过门禁 ——
    串行重放已被拒绝（跑完即 completed），这个窄窗口换成"崩溃后一定能重试"是值的。
    """
    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
        status = run.status if run is not None else None

    if status is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "run_not_found",
                "run_id": run_id,
                "reason": "该 run 不存在",
                "hint": "run_id 来自 SSE run_started 事件的 data.run_id。",
            },
        )
    if status in NON_RESUMABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "not_resumable",
                "run_id": run_id,
                "status": status,
                "reason": "该 run 已经跑完，不能再次续跑",
                "hint": "报告与引用请直接用 `GET /api/v1/research/{run_id}`；想重做请发起新的研究。",
            },
        )


# --------------------------------------------------------------------------- HITL
class ResumeRequest(BaseModel):
    """恢复请求。

    `sub_questions` 非空表示用户**编辑过**子问题（中断点 A 的核心交互）——
    会先写回 state 再续跑，否则用户改的东西等于没改。
    留空则是"沿用原拆解直接继续"，这条也会被 §8 的可用率埋点记成 `plan_edited=false`。
    """

    sub_questions: list[dict] | None = Field(
        default=None, description="用户编辑后的子问题；留空表示沿用原拆解直接继续"
    )


@router.post("/{run_id}/resume", summary="HITL 恢复（用户确认后继续）")
async def research_resume(run_id: str, body: ResumeRequest | None = None) -> StreamingResponse:
    guard_run_id(run_id)
    body = body or ResumeRequest()
    # ★ 用户在面板上编辑过的子问题是**完全用户可控的输入**，且 resume 之后紧跟着就是
    # retrieve/read/write —— 跳过这道闸门等于在"中断后继续"这条路上开了个注入后门。
    sub_questions = guard_sub_questions(body.sub_questions)
    # ★ 状态门禁：放在返回 StreamingResponse **之前** —— 流一旦开始就只能靠 SSE error
    # 事件报"不许续跑"，客户端拿不到状态码。被拒的 resume 一个 token 也不该烧。
    await _ensure_resumable(run_id)
    # 用户点下"继续"的那一刻，等待窗口就结算掉：这段人类时间不该进导出的"耗时"。
    await _close_awaiting_window(run_id)
    emitter = SSEEmitter(run_id)

    async def event_gen():
        plan_diff: dict | None = None
        try:
            async with open_graph() as graph:
                config = thread_config(run_id, emitter=emitter)
                # ★ 指标埋点：必须在 `aupdate_state` **之前**取基线 ——
                # 一旦把用户提交的拆解写回 state，"图原本产出的是什么"就再也取不到了，
                # 于是每条 resume 看起来都"没改"，可用率恒等于 100%。
                plan_diff = plan_edit_diff(await _graph_sub_questions(graph, config), sub_questions)
                await _record_plan_edit(run_id, plan_diff)
                logger.info(
                    "research.plan_edit",
                    run_id=run_id,
                    edited=plan_diff["edited"],
                    changes=plan_diff["changes"],
                )
                if sub_questions:
                    # 先落回 state，再续跑 —— 顺序反了用户编辑就白改了
                    await graph.aupdate_state(config, {"sub_questions": sub_questions})
                    logger.info(
                        "research.sub_questions_edited",
                        run_id=run_id,
                        count=len(sub_questions),
                    )
                # inputs 传 None 表示"从中断点恢复"；传 {} 会从 START 重跑、白烧一遍 token
                async for frame in emitter.stream(graph, None, config, resume=True):
                    yield frame
            await _update_run_status(run_id, _final_status(emitter.last_status))
        except asyncio.CancelledError:
            logger.info("research.resume_disconnected", run_id=run_id)
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("research.resume_failed", run_id=run_id)
            emitter.last_status = "error"
            code, recoverable = failure_event_code(exc)
            # 这里刻意**不**把状态写成 failed：run 仍停在检查点可续跑的位置，
            # 保持 `awaiting_input` 才能让用户稍后从断点重试（与门禁取值自洽）。
            code = code if code != "internal_error" else "resume_failed"
            yield emitter.error("api", code, failure_event_message(code), recoverable=recoverable).to_sse()
            yield emitter.done(status="error").to_sse()
        finally:
            await emitter.flush_token_usage()
            # 补写第二次：`finalize` 在流里跑，它落库时会用摘要**整体覆盖** langgraph_state，
            # 入口那一次写下的 plan_edited 到这里已经没了（原因见 `_record_plan_edit`）。
            if plan_diff is not None:
                await _record_plan_edit(run_id, plan_diff)

    return StreamingResponse(
        event_gen(), media_type="text/event-stream", headers=sse_headers()
    )


# --------------------------------------------------------------------------- 查询
# --------------------------------------------------------------------------- 运行列表
# ⚠️ 必须声明在 `/{run_id}` **之前**：FastAPI 按声明顺序匹配，
# 空路径本身不冲突，但这条一旦挪到后面，读起来就会让人以为它被动态路由吃掉了。
@router.get("", summary="最近的运行列表")
async def list_runs(limit: int = 20, offset: int = 0, status: str | None = None) -> dict:
    """前端靠这条决定"报告页/会话页该展示哪一次运行"。

    没有它的话 `run_id` 只能靠用户从日志里抄 —— 那等于把 F5/F8/F9 全部锁死。
    顺带把每次运行的引用数与支撑率一起给出，列表页才能一眼挑出"哪次跑得值"。
    """
    limit = max(1, min(limit, 100))
    offset = max(0, offset)

    async with session_scope() as db:
        cond = [ResearchRun.status == status] if status else []

        total = (
            await db.execute(select(func.count()).select_from(ResearchRun).where(*cond))
        ).scalar_one()

        rows = (
            (
                await db.execute(
                    select(ResearchRun)
                    .where(*cond)
                    .order_by(ResearchRun.started_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            )
            .scalars()
            .all()
        )

        # 一次分组查询拿全部引用的统计，避免 N 个 run 打 N 次库。
        run_ids = [r.id for r in rows]
        stats: dict[str, tuple[int, int]] = {}
        if run_ids:
            agg = await db.execute(
                select(
                    Citation.run_id,
                    func.count().label("total"),
                    func.coalesce(func.sum(case((Citation.verified.is_(True), 1), else_=0)), 0).label("ok"),
                )
                .where(Citation.run_id.in_(run_ids))
                .group_by(Citation.run_id)
            )
            stats = {row[0]: (int(row[1]), int(row[2])) for row in agg}

    items = []
    for r in rows:
        total_c, ok_c = stats.get(r.id, (0, 0))
        items.append(
            {
                "run_id": r.id,
                "session_id": r.session_id,
                "question": r.question,
                "status": r.status,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "ended_at": r.ended_at.isoformat() if r.ended_at else None,
                "citation_count": total_c,
                # 支撑率：分母是**已校验**的引用数。`verified` 可为 NULL（还没校验），
                # 把 NULL 算进分母会让"刚跑完还没校验"和"校验后一半没支撑"看起来一样。
                "verified_rate": (round(ok_c / total_c, 4) if total_c else None),
            }
        )

    return {"total": int(total), "limit": limit, "offset": offset, "items": items}


@router.get("/{run_id}", summary="查询运行状态与最终报告")
async def get_run(run_id: str) -> dict:
    """状态取自 checkpointer（它是 state 的权威来源），元信息取自数据库。

    之所以查 checkpoint 而不是把报告也落库：报告体积大且可能被后续 revision 覆盖，
    重复存一份只会带来不一致。数据库只负责"这个 run 存在过、什么状态"。
    """
    guard_run_id(run_id)
    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} 不存在")
        meta = {
            "run_id": run.id,
            "session_id": run.session_id,
            "question": run.question,
            "status": run.status,
            # §8「子问题拆解可用率」的埋点（true/false/null，形状见 `PLAN_EDIT_FLAG` 注释）。
            # 挂在 meta 上而不是 checkpoint 里：它记录的是"用户对这个 run 做了什么"，
            # 属于 run 的元信息，权威来源就是 `research_runs.langgraph_state`。
            "plan_edited": (run.langgraph_state or {}).get(PLAN_EDIT_FLAG),
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
    guard_run_id(run_id)
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
                # 定位路径的可观测面：exact | quote | swapped_chunk（read 侧），
                # verify_reanchored=verify 硬校验时是否又重锚过一次。
                # 迁移前的历史行是 None —— 与"记录了且为 exact/False"刻意区分。
                "resolution_how": r.resolution_how,
                "verify_reanchored": r.verify_reanchored,
            }
            for r in rows
        ],
    }


@router.get("/{run_id}/export.md", summary="导出 Markdown 报告（产品定义 F8）")
async def export_markdown(run_id: str) -> PlainTextResponse:
    """把一次研究导出成可直接进开题报告的 .md。

    渲染逻辑全在 `app/export/markdown.py`（只认数据库，不碰 checkpointer），
    这一层只做协议映射 —— 与上面几个查询接口保持同一个口径。
    """
    guard_run_id(run_id)
    try:
        text = await render_run_markdown(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在") from exc

    # `attachment` 而不是 `inline`：这个接口的用途是"下载一份能归档的文件"，
    # 直接在浏览器里渲染成网页会把 Markdown 的相对链接与表格弄没，
    # 而文件名带上 run_id，用户存下来之后还能对回那次研究。
    return PlainTextResponse(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="paperpilot-{run_id}.md"'},
    )


# --------------------------------------------------------------------------- F18 / F12 / F9
@router.get("/{run_id}/related-work.md", summary="导出「相关工作」章节草稿（产品定义 F18）")
async def export_related_work(run_id: str) -> PlainTextResponse:
    """把一次研究组织成能直接进开题报告"相关工作"一节的草稿。

    与 `export.md` 的区别不是排版，而是**组织方式**：整份报告按产出物 A~E 排，
    而这里按方法/主题分组（同一类方法放在一起对比），因为"相关工作"这一节的
    读者要看的正是"这个方向上有几派、各自的问题在哪"。
    """
    guard_run_id(run_id)
    try:
        text = await render_related_work_markdown(run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在") from exc

    return PlainTextResponse(
        text,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="paperpilot-{run_id}-related-work.md"'
        },
    )


@router.get("/{run_id}/timeline", summary="方法演进时间线（产品定义 F12）")
async def run_timeline(run_id: str) -> dict:
    """按年份排开本次研究涉及的论文，并如实说明"能不能谈演进关系"。

    ⚠️ `addresses_gap`（谁补了谁的坑）只有在 `citation_edges` 里有引用边证据时才非空。
    没跑过 `POST /api/v1/papers/references/build` 时它恒为空数组 ——
    这是刻意的：用年份先后推断"后者补了前者的坑"是编故事，不是发现。
    """
    guard_run_id(run_id)
    try:
        return await build_timeline(run_id=run_id)
    except RunNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在") from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail={"code": "invalid_input", "reason": str(exc)}
        ) from exc


class FollowupBody(BaseModel):
    """F9 追问的请求体。"""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000, description="基于这次研究继续问的问题")


@router.post("/{run_id}/followup", summary="基于已读论文追问，不重新检索（产品定义 F9）")
async def followup_on_run(run_id: str, body: FollowupBody) -> dict:
    """在**上一轮已经读过并校验过的证据**上回答，一次 LLM 调用，不碰索引也不联网。

    这是 F9 里"省"的那一半；`POST /research/stream` 带 `session_id` 是"上下文"那一半
    （拆解时能看到本会话此前的问答，指代能被补全，但仍会走完整检索链路）。
    两条路径的分工写在这里而不是混成一个接口，是因为它们的成本与失败模式完全不同：
    追问复用不了证据时必须**如实返回 no_evidence**，而不是悄悄退化成一次全量研究。
    """
    guard_run_id(run_id)
    question = guard_text(body.question, field="question")
    try:
        return await answer_from_read_papers(run_id, question)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=f"run {run_id} 不存在") from exc

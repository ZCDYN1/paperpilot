"""评测侧接口：触发评测（架构 §7 的 `POST /api/v1/eval/run`）与查状态。

这一层薄得像纸：判定、取数、算分全在 `app/eval/service.py`（它再转调 `app/eval/` 里
已有的三套评测实现）。路由只做四件事 —— 输入闸门、把服务层异常映射成状态码、
把长任务交给后台、把结果原样吐出去。

## 状态码口径（与 `papers.py` / `research.py` 一致）
- `422` 形状不合法（模式不在枚举里、golden_path 跑出仓库、configs 写错、run_ids 为空/超长、
  字段多余）—— 客户端的正确反应是**改请求**；
- `409` 已有评测在跑 —— 正确反应是**等那条跑完**（响应体直接给了它的 id 与轮询地址）；
- `501` 缺可选依赖 ragas —— 正确反应是**照抄安装命令**，不是重试；
- `404` `eval_run_id` 不存在（评测记录被删过，或 id 记错了）；
- `202` 已受理并开始后台执行。**注意 202 不等于评测成功**：
  评测的真实结果只有 `GET /api/v1/eval/runs/{id}` 知道。

## 为什么触发接口返回 202 而不是同步 200
一次消融是 golden 集全量 × 六种检索配置，还要加载 embedding/重排模型；
RAGAS 更是每条样本都要 LLM 拆句逐句判定，分钟级起步。同步等待会同时踩中
"客户端超时"与"评测跑完前谁都别想用这台机器的模型"两个坑，所以：立刻回 id、
状态与结果走查询接口（服务层模块头注里有完整的取舍记录）。

## 后台任务与进程内互斥的诚实边界
互斥状态活在进程内存里（`app/eval/service._ACTIVE`）。**服务重启后，DB 里停在
`running` 的那条记录不会再被更新**，而闸门也放开了 —— 查询接口因此同时返回
`status` 与 `is_running` 两个字段：前者是记录的状态，后者是"本进程还在不在跑它"。
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from app.api.guards import guard_text
from app.eval import service as eval_service

router = APIRouter(prefix="/eval", tags=["eval"])
logger = structlog.get_logger(__name__)

# 与 `eval_runs.name` 的 String(128) 同宽：超长的名字在这里就该 422，
# 而不是在后台任务的 INSERT 上撞出一个 failed 记录。
MAX_NAME_CHARS = 128


class EvalRunRequest(BaseModel):
    """`POST /api/v1/eval/run` 的请求体。三种模式共用一个接口，字段按模式取用。

    `extra="forbid"`：拼错的字段名必须当场报出来。评测接口的字段（`top_k`、`configs`）
    静默失效的后果是"跑了一份和想象中配置不同的评测"，那比 404 难查得多。
    """

    model_config = {"extra": "forbid"}

    mode: Literal["ablation", "metrics", "ragas"] = Field(
        default="ablation",
        description="ablation=检索消融（只需索引，不需 LLM）；metrics=§8 成功指标（只读 DB）；"
        "ragas=RAGAS 端到端（需可选依赖 + 真实 LLM 裁判）",
    )
    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_CHARS)
    golden_path: str | None = Field(
        default=None,
        max_length=500,
        description="仓库内的 golden JSONL；省略即用 app/eval/datasets/golden_40.jsonl",
    )
    top_k: int | None = Field(default=None, ge=1, le=500, description="每路召回条数，默认取 RETRIEVE_TOP_K")
    configs: str | None = Field(
        default=None,
        max_length=64,
        description="只跑指定消融配置：序号 4,5,6 / key rrf,rerank / 圈号 ④⑤，默认全部六种",
    )
    run_ids: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=eval_service.MAX_RUN_IDS_PER_REQUEST,
        description="RAGAS 模式要评的 research run；省略即库里全部 run",
    )
    # 上限 5 == `ragas_runner.METRIC_NAMES` 的个数；不在这里 import ragas_runner 取长度，
    # 是为了让这个模块的导入面保持最小（名字是否合法由服务层闸门统一判定并给出 422 hint）。
    metrics: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=5,
        description="RAGAS 指标名子集，默认全部五个",
    )


def _detail(code: str, reason: str, *, field: str | None = None, hint: str | None = None, **extra: Any) -> dict[str, Any]:
    """统一的错误形状 `{code, field, reason, hint}`（与 papers.py 同口径，客户端只需认一种）。"""
    out: dict[str, Any] = {"code": code, "reason": reason}
    if field:
        out["field"] = field
    if hint:
        out["hint"] = hint
    out.update(extra)
    return out


# --------------------------------------------------------------------------- 触发评测
# ⚠️ 路由顺序纪律（同 papers.py 的 `/venue-tier`）：**静态路径必须声明在动态路由之前**。
# `/run`（POST）与 `/runs/{eval_run_id}`（GET）方法不同、不会互相吞，但 FastAPI 按声明顺序匹配，
# 一旦以后加 `GET /runs/latest` 这类静态段，顺序错了不会报错、只会返回 404 式的意外。
@router.post(
    "/run",
    status_code=202,
    summary="触发评测（架构 §7）：后台执行，立刻返回 eval_run_id",
)
async def trigger_eval_run(body: EvalRunRequest, request: Request) -> dict[str, Any]:
    """起一次评测：`ablation` / `metrics` 不需要 ragas，`ragas` 需要可选依赖 + 真实 LLM。

    受理即返回 `202`，评测在后台跑（结果落 `eval_runs.metrics` 与 `eval_samples`）。
    同一时刻只允许一条评测：第二条收到 `409`，响应体里带着正在跑的那条的 id 与轮询地址。
    """
    name = guard_text(body.name, field="name", max_chars=MAX_NAME_CHARS) if body.name else None
    try:
        started = await eval_service.start_eval_run(
            body.mode,
            name=name,
            golden_path=body.golden_path,
            top_k=body.top_k,
            configs=body.configs,
            run_ids=body.run_ids,
            metric_names=body.metrics,
        )
    except eval_service.EvalDependencyMissingError as exc:
        # 501 而不是 500：这是"少装了一个可选 extra"，修复动作是照抄命令
        raise HTTPException(
            status_code=501,
            detail=_detail(
                "eval_extra_missing",
                str(exc),
                field="mode",
                hint=f"安装评测依赖后重启服务：{eval_service.EVAL_INSTALL_CMD}。"
                "检索消融（mode=ablation）与 §8 指标（mode=metrics）都不需要 ragas。",
                install_command=eval_service.EVAL_INSTALL_CMD,
            ),
        ) from exc
    except eval_service.EvalAlreadyRunningError as exc:
        raise HTTPException(
            status_code=409,
            detail=_detail(
                "eval_already_running",
                str(exc),
                hint="不做排队：排队需要一个能活过进程重启的任务队列（新增依赖或新建迁移设施），"
                "而并发评测还会抢同一份索引与模型、并把消融表的耗时列变成噪声。"
                "请轮询 running_eval_run_id，跑完后再触发。",
                running_eval_run_id=exc.eval_run_id,
                running_mode=exc.mode,
                poll=_poll_path(request, exc.eval_run_id),
            ),
        ) from exc
    except eval_service.EvalInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=_detail("invalid_input", str(exc), field=exc.field, hint=exc.hint),
        ) from exc

    logger.info("eval.triggered", eval_run_id=started["eval_run_id"], mode=started["mode"])
    return {
        **started,
        "poll": _poll_path(request, started["eval_run_id"]),
        "message": "评测已在后台启动；完成后 status 变为 completed / unavailable / failed。",
    }


# --------------------------------------------------------------------------- 查状态与结果
@router.get("/runs/{eval_run_id}", summary="查询一次评测的状态与结果")
async def get_eval_run(
    eval_run_id: int,
    request: Request,
    sample_limit: int = Query(default=200, ge=1, le=1000, description="最多返回多少条明细"),
) -> dict[str, Any]:
    """`status`：`running` / `completed` / `unavailable` / `failed`。

    `unavailable` 是"环境不具备条件"（索引为空、mock 当裁判、没有可评的 run），
    不是代码坏了 —— 它和 `failed` 分开，评测记录里才不会全是假故障。
    两种情况下都**没有明细行、没有分数**：宁缺毋假（见 `app/eval/ragas_runner` 的不变量）。
    """
    try:
        record = await eval_service.get_eval_run(eval_run_id, sample_limit=sample_limit)
    except eval_service.EvalRunNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=_detail("eval_run_not_found", str(exc), field="eval_run_id"),
        ) from exc
    record["poll"] = _poll_path(request, eval_run_id)
    return record


def _poll_path(request: Request, eval_run_id: int) -> str:
    """轮询地址。带前缀（`root_path`）：反代挂在子路径下时相对路径才是对的。"""
    prefix = (request.scope.get("root_path") or "").rstrip("/")
    return f"{prefix}/api/v1/eval/runs/{eval_run_id}"


__all__ = ["router"]

"""全局聚合统计 —— 首页四张指标卡与系统状态页的数据源。

## 为什么要单独一个聚合接口，而不是让前端拼
这些数字每个都很便宜，但让前端拼就要发 5 个请求再自己 join，
其中"平均引用支撑率"还得把每次运行的引用数全拉回来在前端算 ——
那是把一个 GROUP BY 能做完的事拆成了 N 次往返。

## 口径全部在这里定死，前端只负责显示
- `avg_verified_rate` 的分母是**引用总条数**，不是运行次数（一次运行可能一条引用都没有）；
- 没有数据时返回 `null` 而不是 `0`。`0%` 会被读成"支撑率很差"，
  `null` 才是"还没跑过、无从谈起" —— 这个区分是产品承诺的一部分。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from sqlalchemy import case, func, select

from app.db.models import Citation, Paper, PaperChunk, ResearchRun, TokenUsage
from app.db.session import session_scope

router = APIRouter(prefix="/stats", tags=["stats"])

DEFAULT_DAYS = 7


@router.get("/overview", summary="首页聚合指标")
async def overview(days: int = Query(default=DEFAULT_DAYS, ge=1, le=90)) -> dict[str, Any]:
    days = max(1, min(days, 90))

    async with session_scope() as db:
        papers = (await db.execute(select(func.count()).select_from(Paper))).scalar_one()
        chunks = (await db.execute(select(func.count()).select_from(PaperChunk))).scalar_one()

        runs_total, runs_completed = (
            await db.execute(
                select(
                    func.count(),
                    func.coalesce(
                        func.sum(case((ResearchRun.status == "completed", 1), else_=0)), 0
                    ),
                ).select_from(ResearchRun)
            )
        ).one()

        # `verified` 可为 NULL（还没校验）。NULL 既不计入分子也不计入分母：
        # 把"没校验"当成"没支撑"会凭空压低支撑率。
        cited_total, cited_ok = (
            await db.execute(
                select(
                    func.count(),
                    func.coalesce(
                        func.sum(case((Citation.verified.is_(True), 1), else_=0)), 0
                    ),
                )
                .select_from(Citation)
                .where(Citation.verified.is_not(None))
            )
        ).one()

        token_expr = func.coalesce(func.sum(TokenUsage.prompt_tokens), 0) + func.coalesce(
            func.sum(TokenUsage.completion_tokens), 0
        )
        tokens = (await db.execute(select(token_expr).select_from(TokenUsage))).scalar_one()

        run_day = func.date(ResearchRun.started_at)
        runs_by_day = (
            await db.execute(
                select(run_day.label("d"), func.count())
                .where(run_day >= func.date(func.datetime("now", f"-{days} days")))
                .group_by(run_day)
            )
        ).all()

        usage_day = func.date(TokenUsage.created_at)
        tokens_by_day = (
            await db.execute(
                select(usage_day.label("d"), token_expr)
                .where(usage_day >= func.date(func.datetime("now", f"-{days} days")))
                .group_by(usage_day)
            )
        ).all()

    day_map: dict[str, dict[str, Any]] = {}
    for d, n in runs_by_day:
        key = str(d)
        day_map.setdefault(key, {"date": key, "runs": 0, "tokens": 0})["runs"] = int(n)
    for d, n in tokens_by_day:
        key = str(d)
        day_map.setdefault(key, {"date": key, "runs": 0, "tokens": 0})["tokens"] = int(n)

    return {
        "papers": int(papers),
        "chunks": int(chunks),
        "runs": int(runs_total),
        "runs_completed": int(runs_completed or 0),
        "avg_verified_rate": (round(int(cited_ok) / int(cited_total), 4) if cited_total else None),
        "total_tokens": int(tokens or 0),
        "by_day": sorted(day_map.values(), key=lambda row: row["date"]),
    }

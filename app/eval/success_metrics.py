"""产品定义 §8 两条「此前根本无法计算」的成功指标 —— 从真实落库数据算，算不出就拒绝。

## 这两条指标原来为什么不可测（本模块存在的全部理由）

1. **对比表字段完整率 ≥ 80%**：对比表此前**从不落库**（12 张表里没有一张存对比行），
   行只活在 graph state 与 SSE 帧里 —— 没有持久化数据，"完整率"连分母都构造不出来。
   现在 `finalize._persist` 把每行落进 `comparison_rows`，本模块据此统计。
2. **子问题拆解可用率 ≥ 85%（用户不修改直接执行的比例）**：此前没有任何地方记录
   用户在中断点 A 是否改了拆解。现在 API 层把 `plan_edited` 写进
   `research_runs.langgraph_state`（生产方在 `app/api/v1/research.py`，本模块只是消费方）。

## 样本量诚实：这是本模块的第一设计约束

仓库已经有过前科：18 条引用（2 条问题）算出的自动口径 0.61 被当成能力数字引用，
后来一次真实运行给出 0.35（26 条引用）—— README 只能把两者都标注成"量级"。
所以这里的规则写死在代码里，不靠使用者自觉：

- **任何比率都必须带着分子/分母出现**（分母的单位也写出来：格 / run）；
- **分母为 0 → 拒绝计算**，如实报"无数据"。历史 run 对两条指标都没有可判数据，
  这个脚本第一次跑出来的大概率就是两个"拒绝计算"，这是**预期结果**，不是失败；
- **分母低于门槛 → 同样拒绝**（`MIN_CELLS` / `MIN_RUNS`，理由见常量注释），
  并打印实际计数，让"差多少数据"成为可行动的信息；
- **算得出也要标注样本规模**，比率永远写成 `0.42（13/31 格）`这种形态，
  裸百分比在本模块里不存在。

## 口径：空字段是承诺，不是缺失

`ComparisonRow`（`app/agents/schemas.py`）的三条不可协商规则里，空字段 =
「原文里没找到依据」的刻意留空。完整率因此只能定义为**"有多少格真的找到了依据"**，
并且永远不许对空格做插补 —— 一旦有人为了把数字推过 80% 去填占位值，
这条指标就从"如实"变成了"编造的完成率"，产品承诺整体作废。
模块把这句话写进报告抬头，就是为了让指标本身无法被这样误用。

plan_edited 的三态语义照抄生产方定义（`app/api/v1/research.py` 的注释）：
`false` = 没改直接继续（分子）、`true` = 改了、`null`/缺键 = **判不了或没记录**，
一律**不进分母**、**更不记成 0** —— 把"没记录"算成"用户改了"会系统性压低可用率，
而历史 run 全部没有这个位，若当成 0 分母，指标会报出一个纯粹的假数。
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import session as db_session
from app.db.models import ComparisonRecord, ResearchRun

# 与 `app/agents/nodes/comparator.py` 的 `_build_notes.columns` 同口径的 9 个"字段"。
# arxiv_id / title 是**身份列**（系统回填、不是证据字段），cited_at 是被引数的
# 配套日期（产品规则：两者成对出现，单独数会让分母虚增）—— 都不参与完整率。
MEASURED_FIELDS: tuple[str, ...] = (
    "method_name",
    "core_idea",
    "datasets",
    "improvement",
    "cost",
    "limitation",
    "code_url",
    "venue",
    "cited_by_count",
)
_FIELD_LABELS: dict[str, str] = {
    "method_name": "方法名",
    "core_idea": "核心思路",
    "datasets": "数据集",
    "improvement": "相对提升",
    "cost": "代价",
    "limitation": "局限",
    "code_url": "代码链接",
    "venue": "发表会议",
    "cited_by_count": "被引数",
}

# 分母门槛。取数的理由要能当面说清：
# - 45 格 = 5 行 × 9 字段。一行表只有 9 个格子，单 run 的表（约 5~8 行）刚刚够到
#   "能看一眼"的边；低于 5 行时一次留空就是 >10% 的摆动，比率毫无稳度可言。
# - 5 个 run：可用率的分母是"真的 resume 过且判得了"的 run。低于 5 时一个用户
#   改没改就是 20% 的跳动 —— 这正是 18 条引用被当成能力数字的那类错误。
MIN_CELLS = 45
MIN_RUNS = 5

# plan_edited 的键名与生产方保持一致。当前生产方（`app/api/v1/research.py` 的
# `PLAN_EDIT_FLAG`，2026-10 核对）把它写在 **langgraph_state 顶层**：
#     research_runs.langgraph_state["plan_edited"] : true | false | null
# 生产方若迁移到 `interrupt_decisions` 嵌套形状，这里两个位置都认 —— 顶层优先。
# 消费方独立猜键名是最糟糕的失败方式（永远读出"未记录"却不报错），所以键名集中在这一个常量对上。
PLAN_EDIT_FLAG = "plan_edited"
_INTERRUPTIONS_BUCKET = "interrupt_decisions"

SessionScope = Callable[[], AbstractAsyncContextManager[AsyncSession]]


# --------------------------------------------------------------------------- 结果结构
@dataclass(slots=True)
class MetricOutcome:
    """一条指标的产出。`value` 为 None 时表示**拒绝计算**，原因写在 `reason`。"""

    key: str
    label: str
    target: str
    value: float | None
    numerator: int | None
    denominator: int | None
    unit: str
    status: str  # ok | no_data | insufficient_sample
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def display(self) -> str:
        """比率永远带着分子/分母出现 —— 不允许存在裸百分比这种形态。"""
        if self.value is None:
            return f"拒绝计算（{self.reason}）"
        pct = f"{self.value:.4f}"
        return f"{pct} = {self.numerator}/{self.denominator} {self.unit}"


def _cell_filled(row: ComparisonRecord, name: str) -> bool:
    """一格是否"找到了依据"。空串 / None = 刻意留空，判 False；其余（含数字 0）判 True。"""
    v = getattr(row, name, None)
    return not (v is None or (isinstance(v, str) and v.strip() == ""))


def _refuse_no_data(key: str, label: str, target: str, unit: str, reason: str) -> MetricOutcome:
    return MetricOutcome(
        key=key, label=label, target=target,
        value=None, numerator=None, denominator=None, unit=unit,
        status="no_data", reason=reason,
    )


# --------------------------------------------------------------------------- 指标一：字段完整率
async def measure_comparison_completeness(
    scope: SessionScope | None = None, *, min_cells: int = MIN_CELLS
) -> MetricOutcome:
    """对比表字段完整率 = 有依据的格 / 参与统计的格（9 字段 × 落库行数）。"""
    scope = scope or db_session.session_scope
    async with scope() as db:
        rows = (await db.execute(select(ComparisonRecord))).scalars().all()

    key, label, target, unit = (
        "comparison_field_completeness",
        "对比表字段完整率（产品定义 §8，验收线 >= 0.80）",
        ">= 0.80",
        "格",
    )
    if not rows:
        return _refuse_no_data(
            key, label, target, unit,
            "库里没有任何已落库的对比表行 —— comparison_rows 是新增表，历史 run 没有数据，"
            "不做合成兜底。指标自新 run 起可测。",
        )

    denominator = len(rows) * len(MEASURED_FIELDS)
    numerator = sum(1 for r in rows for f in MEASURED_FIELDS if _cell_filled(r, f))
    per_field = {
        f: (sum(1 for r in rows if _cell_filled(r, f)), len(rows)) for f in MEASURED_FIELDS
    }
    extras = {
        "n_rows": len(rows),
        "n_runs": len({r.run_id for r in rows}),
        "per_field": {
            _FIELD_LABELS[f]: f"{filled}/{total}" for f, (filled, total) in per_field.items()
        },
    }

    if denominator < min_cells:
        return MetricOutcome(
            key=key, label=label, target=target, value=None,
            numerator=numerator, denominator=denominator, unit=unit,
            status="insufficient_sample",
            reason=(
                f"分母 {denominator} 格（{extras['n_runs']} 个 run / {extras['n_rows']} 行 × "
                f"{len(MEASURED_FIELDS)} 字段）低于门槛 {min_cells} 格，拒绝给比率；"
                "计数如实列出，供判断还差多少数据"
            ),
            extras=extras,
        )
    return MetricOutcome(
        key=key, label=label, target=target,
        value=numerator / denominator, numerator=numerator, denominator=denominator,
        unit=unit, status="ok",
        extras=extras,
    )


# --------------------------------------------------------------------------- 指标二：拆解可用率
def _plan_edit_flag(state: Any) -> Any:
    """取 plan_edited 位。缺键 / 非 dict → None（调用方用"键是否存在"区分 未记录/null）。"""
    if not isinstance(state, dict):
        return None
    if PLAN_EDIT_FLAG in state:
        return state[PLAN_EDIT_FLAG]
    # 生产方迁移形状后的回退：langgraph_state["interrupt_decisions"]["plan_edit"]
    decisions = state.get(_INTERRUPTIONS_BUCKET)
    if isinstance(decisions, dict):
        for alt in ("plan_edit", PLAN_EDIT_FLAG):
            if alt in decisions:
                return decisions[alt]
    return None


def classify_plan_flags(runs: list[tuple[str, Any]]) -> dict[str, list[str]]:
    """把每个 run 的 `plan_edited` 归成四类（纯函数，可直接单测）。

    - `usable`   ：flag is False（没改直接继续 —— 分子）
    - `edited`   ：flag is True
    - `null`     ：flag 显式为 None（生产方"想判但判不了"，如 checkpoint 已丢）
    - `unrecorded`：langgraph_state 里根本没有可判的位（**全部历史 run** / 未走到 HITL）
    后两类都**不进分母**：判不了与没记录都不是"用户改过拆解"的证据，
    把它们记成 True/False 任意一边都是在编指标。
    """
    buckets: dict[str, list[str]] = {"usable": [], "edited": [], "null": [], "unrecorded": []}
    for run_id, state in runs:
        if isinstance(state, dict) and _plan_edit_flag(state) is None and not (
            PLAN_EDIT_FLAG in state or isinstance(state.get(_INTERRUPTIONS_BUCKET), dict)
        ):
            # 没有任何一处写过这个位：与"写过但值为 null"分开计 —— 前者是历史缺数据，
            # 后者是生产方判过但判不了，两类都不进分母，但报告里必须各归各位。
            buckets["unrecorded"].append(run_id)
            continue
        flag = _plan_edit_flag(state)
        # bool 判定必须**先于**任何 int 判定：Python 里 `isinstance(True, int)` 为真，
        # 反过来先判 int 会把 JSON 载入的 1/0 脏值误收成"可判"，污染分母。
        if isinstance(flag, bool):
            buckets["usable" if flag is False else "edited"].append(run_id)
        else:
            buckets["null"].append(run_id)  # 键在但值不是可判布尔（含显式 None）：按"判不了"处理
    return buckets


async def measure_plan_availability(
    scope: SessionScope | None = None, *, min_runs: int = MIN_RUNS
) -> MetricOutcome:
    """子问题拆解可用率 = #(plan_edited is False) / #(plan_edited 是布尔)。"""
    scope = scope or db_session.session_scope
    async with scope() as db:
        runs = [
            (r[0], r[1])
            for r in (await db.execute(select(ResearchRun.id, ResearchRun.langgraph_state))).all()
        ]

    key, label, target, unit = (
        "plan_availability",
        "子问题拆解可用率（产品定义 §8，验收线 >= 0.85）",
        ">= 0.85",
        "run",
    )
    buckets = classify_plan_flags(runs)
    denominator = len(buckets["usable"]) + len(buckets["edited"])
    numerator = len(buckets["usable"])
    extras = {
        "n_runs_total": len(runs),
        "usable": len(buckets["usable"]),
        "edited": len(buckets["edited"]),
        "flag_null": len(buckets["null"]),
        "unrecorded": len(buckets["unrecorded"]),
    }

    if denominator == 0:
        return MetricOutcome(
            key=key, label=label, target=target, value=None,
            numerator=None, denominator=0, unit=unit, status="no_data",
            reason=(
                f"库里 {len(runs)} 个 run 中没有任何一个带可判的 plan_edited 位"
                f"（未记录 {len(buckets['unrecorded'])}、显式 null {len(buckets['null'])}）—— "
                "该信号随 resume 埋点上线才开始产生，历史 run 如实按未知处理，不进分母"
            ),
            extras=extras,
        )
    if denominator < min_runs:
        return MetricOutcome(
            key=key, label=label, target=target, value=None,
            numerator=numerator, denominator=denominator, unit=unit,
            status="insufficient_sample",
            reason=(
                f"判得了的 run 只有 {denominator} 个（门槛 {min_runs}），单个用户行为就能让比率"
                f"跳动 {1 / denominator:.0%} 以上，拒绝给比率；计数如实列出"
            ),
            extras=extras,
        )
    return MetricOutcome(
        key=key, label=label, target=target,
        value=numerator / denominator, numerator=numerator, denominator=denominator,
        unit=unit, status="ok", extras=extras,
    )


# --------------------------------------------------------------------------- 渲染
def render_markdown(
    outcomes: list[MetricOutcome],
    *,
    env_lines: list[str],
    generated_at: datetime | None = None,
) -> str:
    """渲染 Markdown 报告。所有比率以 `值 = 分子/分母 单位` 的形态出现。"""
    generated_at = generated_at or datetime.now()
    lines: list[str] = ["# 产品定义 §8 成功指标实测（对比表字段完整率 / 子问题拆解可用率）", ""]
    lines.append(
        "> 口径提醒：字段完整率衡量的是**如实**（有依据才填），空格 = 「原文未提供」的产品承诺，"
        "**不允许以任何方式插补空格来提高这个比率** —— 插补后的数字测的就不再是这款产品。"
    )
    lines.append(
        "> 样本量规则：分母为 0 或低于门槛时**拒绝计算**，只打印真实计数；"
        "算得出时也随比率打印样本规模（仓库曾用 18 条样本的比率当能力数字，前车之鉴）。"
    )
    lines.append("")
    lines.append("## 运行环境")
    lines.append("")
    lines.extend(f"- {line}" for line in env_lines)
    lines.append(f"- 生成时间：{generated_at:%Y-%m-%d %H:%M:%S}")
    lines.append("")

    lines.append("## 指标结果")
    lines.append("")
    lines.append("| 指标 | 验收线 | 实测（含样本量） | 状态 |")
    lines.append("|---|---|---|---|")
    for m in outcomes:
        status_cn = {
            "ok": "已计算",
            "no_data": "无数据，拒绝计算",
            "insufficient_sample": "样本不足，拒绝计算",
        }.get(m.status, m.status)
        lines.append(f"| {m.label} | {m.target} | {m.display()} | {status_cn} |")
    lines.append("")

    for m in outcomes:
        lines.append(f"### {m.label}")
        lines.append("")
        lines.append(f"- 实测：**{m.display()}**")
        if m.reason:
            lines.append(f"- 说明：{m.reason}")
        e = m.extras
        if m.key == "comparison_field_completeness":
            lines.append(
                f"- 样本：{e.get('n_runs', 0)} 个 run 的对比表，共 {e.get('n_rows', 0)} 行 × "
                f"{len(MEASURED_FIELDS)} 字段 = {m.denominator if m.denominator is not None else e.get('n_rows', 0) * len(MEASURED_FIELDS)} 格"
            )
            pf = e.get("per_field") or {}
            if pf:
                lines.append("- 分字段填上率（有依据的格/行）：" + "，".join(
                    f"{name} {frac}" for name, frac in pf.items()
                ))
        if m.key == "plan_availability":
            lines.append(
                f"- 样本：库内 run 总数 {e.get('n_runs_total', 0)}；"
                f"未改直接继续 {e.get('usable', 0)}，改过 {e.get('edited', 0)}，"
                f"flag=null（判不了）{e.get('flag_null', 0)}，"
                f"未记录该位（历史 run / 未走到 HITL）{e.get('unrecorded', 0)} —— "
                "**后两类不进分母，也不记 0**"
            )
        lines.append("")

    lines.append("## 脚注")
    lines.append("")
    lines.append(
        "1. 两条指标的数据都是本轮才开始的：对比表自 `finalize` 落 `comparison_rows` 起可测，"
        "`plan_edited` 自 API 层 resume 埋点上线起可测；在此之前它们**没有历史数据**，"
        "报告输出「拒绝计算」是诚实结果，不是脚本故障。"
    )
    lines.append(
        "2. 分母门槛：字段完整率 " + str(MIN_CELLS) + " 格、可用率 " + str(MIN_RUNS) +
        " 个 run（理由见 `app/eval/success_metrics.py` 常量注释）。低于门槛只报计数。"
    )
    lines.append(
        "3. 参与统计的 9 个字段：" + "、".join(_FIELD_LABELS[f] for f in MEASURED_FIELDS) +
        "。arxiv_id/title 是身份列、cited_at 是被引数的配套日期，都不计入分母。"
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- 入口
async def run_and_render(
    scope: SessionScope | None = None,
    *,
    min_cells: int = MIN_CELLS,
    min_runs: int = MIN_RUNS,
) -> tuple[str, list[MetricOutcome]]:
    """算两条指标 + 渲染报告（CLI 与测试共用的入口）。"""
    scope = scope or db_session.session_scope
    completeness = await measure_comparison_completeness(scope, min_cells=min_cells)
    availability = await measure_plan_availability(scope, min_runs=min_runs)
    outcomes = [completeness, availability]

    from app.core.config import get_settings

    settings = get_settings()
    # 环境里先数一遍 run（不额外开事务：上面两个函数已经各查过自己的表）
    async with scope() as db:
        n_runs = len((await db.execute(select(ResearchRun.id))).scalars().all())
    env_lines = [
        f"数据库：`{settings.sqlite_url}`",
        f"research_runs 总数：{n_runs}",
        f"LLM provider：`{settings.llm_provider}`（mock run 的产出同样计入行数，"
        "但 mock 的对比表来自占位抽取，完整率数字不代表真实能力）",
        f"embedding provider：`{settings.embedding_provider}`",
    ]
    markdown = render_markdown(outcomes, env_lines=env_lines)
    return markdown, outcomes


# 项目根：以 __file__ 定位，从任意 CWD 执行 `python -m app.eval.success_metrics` 都落在同一处
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "eval_results" / "success_metrics_latest.md"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.success_metrics",
        description="产品定义 §8：对比表字段完整率 + 子问题拆解可用率（只读 SQLite，零 key）",
    )
    parser.add_argument("--output", default=None, help=f"Markdown 输出路径，默认 {DEFAULT_OUTPUT}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    markdown, outcomes = asyncio.run(run_and_render())
    out_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")

    # 控制台只打安全行（本机 console 是 GBK，装饰性符号会抛 UnicodeEncodeError）
    for m in outcomes:
        print(f"- {m.label}：{m.display()}")
        if m.reason:
            print(f"  说明：{m.reason}")
    print(f"[OK] 报告已写入 {out_path}")
    # 拒绝计算不算脚本失败（这是诚实产出）；退出码 0 让 run_eval/CI 能区分"脚本坏了"与"没数据"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

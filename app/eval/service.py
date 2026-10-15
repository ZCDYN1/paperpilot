"""评测服务层 —— 把 `scripts/run_eval.py` 的评测能力变成**可编程调用、可查状态**的入口。

## 为什么这一层只补三件事，而不是再写一份评测

三套评测的实际逻辑早就在 `app/eval/` 里：消融是 `ablation.run_and_render`、
生成侧是 `ragas_runner.run_ragas_eval`、§8 成功指标是 `success_metrics.run_and_render`。
`scripts/run_eval.py` 从来只做「编排 + 打印」。所以本模块**转调它们**，只补接口需要的三件事：

1. **落库** —— 结果写进已经在用的两张表：`eval_runs.config` / `eval_runs.metrics`（两个 JSON 列）
   与 `eval_samples`。CLI 与界面因此读的是同一份数据，而不是各算各的。
   ⚠️ 状态、耗时、错误、逐配置汇总全部塞进已有的 `metrics` JSON 列 ——
   **不给已有表加列**（SQLite 的 `create_all` 不 ALTER 老表，加列在生产库上是静默故障）。
2. **后台化** —— 评测是分钟级长任务（要载 embedding 模型、要拿 LLM 当裁判逐条判定）。
   放在请求生命周期里跑等于把 HTTP 连接当成任务队列：客户端超时会取消整个 ASGI 调用，
   一次 40 条查询 × 6 配置的消融就白跑。所以用 `asyncio.create_task` 起**脱离请求**的任务，
   接口只回 `eval_run_id`，状态与结果走 `GET /api/v1/eval/runs/{id}`。
3. **单实例闸门** —— 同一时刻只允许一个评测在跑。

## 为什么是 409 而不是排队

排队需要一个**活过进程重启**的任务队列。这里两个前提都不成立：

- 引入队列（celery / rq / arq / 任何任务库）等于**新增依赖**，明令禁止；
- 用内存队列 + DB 状态表自己搭，就得处理「worker 崩在两条任务之间」——
  那是一次迁移工程（架构 §9 的表纪律只允许新建表，不允许改老表）。

而并发评测的代价是**具体的、可观察的**，不是"怕慢"这么简单：

- 消融表有一列是**每条查询检索耗时的中位数**，它的用途是量化 Reranker 的延迟代价。
  两个评测并发跑会互相抢 CPU/GPU，那一列当场失去意义 —— 一张看起来精确到毫秒的表
  就这样变成了噪声，而这正是 §5.3 整张表的立表之本；
- `HybridRetriever` 会读同一份 FAISS/BM25 索引目录，`build_index` 是**原子重建**写同一批文件；
- 本地 BGE embedding + 交叉编码器 + RAGAS 的裁判模型同吃一份显存/内存，
  两个一起加载通常直接 OOM，得到的是"评测失败"而不是"慢一点"。

所以第二个请求**明确拒绝**（409 + 正在跑的那条 run 的 id + 查询地址），
让调用方自己决定等待与重试 —— 这比把请求挂进一个无人可见的队列里诚实得多。

## 降级在本层依然是「不可用」而不是「失败」

`run_ragas_eval` 在 mock LLM / 没配 key / 没有 run 时返回 `available=False` 的报告，
且**结构上不允许携带任何分数**（`EvalReport.__post_init__` 的不变量）。本层原样继承这个口径：
那类 run 落成 `status="unavailable"` + `reason`，而不是 `status="failed"` ——
"环境不具备条件"与"代码坏了"必须是两种状态，否则评测记录里全是假故障。

同理，索引为空时消融评测返回 `unavailable`（并指路 `scripts/build_index.py`），
不返回一张全 0 的表 —— 全 0 会被读成"检索一塌糊涂"。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select

from app.core.config import REPO_ROOT, Settings, get_settings
from app.db import session as db_session
from app.db.models import EvalRun, EvalSample
from app.eval.ablation import (
    ABLATION_CONFIGS,
    AblationConfig,
    # `_provider_notice` 是 ablation 的私有函数，这里**刻意复用而不是重写一份降级判定**：
    # 报告里的"本次结果是 mock"与落库的 `summary.degraded` 必须同一个来源，
    # 否则会出现报告警告、库里却写着"结果可信"这种自相矛盾的记录。
    _provider_notice,
    run_and_render,
    select_configs,
)
from app.eval.dataset import GoldenItem, load_golden
from app.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)

# 三种模式，对应仓库里已有的三套评测。刻意不做 "all"：
# 一次 API 调用只该对应一次可解释的落库记录，混跑三种评测会把 status/reason 搅成一团粥。
MODE_ABLATION = "ablation"
MODE_METRICS = "metrics"
MODE_RAGAS = "ragas"
MODES: tuple[str, ...] = (MODE_ABLATION, MODE_METRICS, MODE_RAGAS)

# 状态机。`unavailable` = 请求合法、评测逻辑没坏，但环境不具备条件（索引空 / mock 裁判 / 无 run）。
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"
TERMINAL_STATUSES = frozenset({STATUS_COMPLETED, STATUS_UNAVAILABLE, STATUS_FAILED})

# 501 里给的可照抄命令（与 `app/eval/ragas_runner._RAGAS_MISSING_REASON` 同一份口径）。
EVAL_INSTALL_CMD = 'pip install -e ".[eval]"'

# 一次请求最多评测多少个 run：RAGAS 每条样本都要 LLM 拆句 + 逐句判定，
# 没有上限时"全部 run"会随库增长无界膨胀成一个永远跑不完的后台任务。
MAX_RUN_IDS_PER_REQUEST = 50

# golden 文件只允许读仓库内的 .jsonl：这个参数是用户可控的，
# 不设牢笼就等于开了一个"任意路径读文件"的口（`../../` 一路走到 ~/.env）。
GOLDEN_SUFFIXES = frozenset({".jsonl"})


# --------------------------------------------------------------------------- 异常
class EvalError(Exception):
    """评测服务层的异常基类（路由据此映射状态码，不裸抛内部异常）。"""


class EvalInputError(EvalError):
    """入参不合法 → 422。带 `field` 是为了和 `papers.py` 的 `{code, field, reason, hint}` 同形状。"""

    def __init__(self, reason: str, *, field: str = "body", hint: str | None = None) -> None:
        super().__init__(reason)
        self.field = field
        self.hint = hint


class EvalAlreadyRunningError(EvalError):
    """已有评测在跑 → 409。带上正在跑的那条 `eval_run_id`，调用方可以直接去轮询它。"""

    def __init__(self, reason: str, *, eval_run_id: int, mode: str) -> None:
        super().__init__(reason)
        self.eval_run_id = eval_run_id
        self.mode = mode


class EvalDependencyMissingError(EvalError):
    """缺可选依赖 → 501。修复动作是照抄安装命令，不是重试，也不是报 bug。"""

    def __init__(self, reason: str, *, install_cmd: str = EVAL_INSTALL_CMD) -> None:
        super().__init__(reason)
        self.install_cmd = install_cmd


class EvalRunNotFoundError(EvalError):
    """`eval_runs` 里没有这个 id → 404（与"有 id 但还在跑"区分开）。"""


# --------------------------------------------------------------------------- 产出结构
@dataclass(slots=True)
class EvalSampleRow:
    """一行 `eval_samples`。字段名与表列一一对应，落库时不做任何二次加工。"""

    question: str
    answer: str | None = None
    contexts: list[str] = field(default_factory=list)
    scores: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalOutcome:
    """一次评测的产出：CLI 打印与落库**共用同一份**，避免两处口径分叉。

    `ok=False` 时 `detail`/`samples` 一律为空 —— 这条纪律继承自 `EvalReport` 的不变量：
    没有真实算出来的东西不进报告，哪怕报告因此变空。
    """

    ok: bool
    summary: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)
    samples: list[EvalSampleRow] = field(default_factory=list)
    markdown: str | None = None
    reason: str | None = None


def _json_safe(value: Any) -> Any:
    """把 ragas/numpy 可能塞进来的 NaN、numpy 标量收敛成 JSON 友好值。

    NaN 必须变成 None：`json.dumps` 默认会把 NaN 写成裸 `NaN` 记号（非法 JSON），
    于是这一列以后任何严格 JSON 读取方（前端、导出、别的语言）都会解析失败。
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- 输入闸门
def resolve_golden_path(raw: str | Path | None, *, base: Path | None = None) -> Path | None:
    """把用户给的 golden 路径解析成**不会跑出仓库**的绝对路径；`None` 表示用默认数据集。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        raise EvalInputError("golden_path 不能是空串；要么省略它用默认数据集，要么给一个真实文件。", field="golden_path")
    root = base or REPO_ROOT
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()  # 先解析再比目录：符号链接绕不过去

    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvalInputError(
            f"golden_path 只能指向仓库内的数据集文件（解析后为 {resolved}，仓库根为 {root}）",
            field="golden_path",
            hint="省略该字段即使用 app/eval/datasets/golden_40.jsonl；或给仓库内的相对路径。",
        ) from exc

    if resolved.suffix not in GOLDEN_SUFFIXES:
        raise EvalInputError(
            f"golden_path 必须是 .jsonl（每行一条金标准），收到的是 {resolved.suffix or '无扩展名'}",
            field="golden_path",
        )
    if not resolved.is_file():
        raise EvalInputError(f"找不到 golden 数据集文件：{resolved}", field="golden_path")
    return resolved


def parse_configs(spec: str | None) -> list[AblationConfig] | None:
    """`configs` 字符串 → 配置列表（原样复用 `ablation.select_configs`，不重写解析规则）。

    解析失败转成 422：`select_configs` 抛的是 `ValueError`，而它的消息里已经带好了
    「可用写法：1:vector / 2:bm25 …」，正是给调用方看的那句话。
    """
    if spec is None or not spec.strip():
        return None
    try:
        return select_configs(spec)
    except ValueError as exc:
        raise EvalInputError(str(exc), field="configs", hint="序号 4,5,6 / key rrf,rerank / 圈号 ④⑤。") from exc


def parse_metric_names(names: Sequence[str] | None) -> list[str] | None:
    """校验 RAGAS 指标名，未知名字**当场 422** 而不是让它静默进 skipped。

    `run_ragas_eval` 本身对未知指标是"跳过"处理的（它要能在批里容错），
    但那是运行期的宽容；请求期就该告诉调用方这个词拼错了。
    """
    if names is None:
        return None
    from app.eval.ragas_runner import METRIC_NAMES

    cleaned = [str(n).strip() for n in names if str(n).strip()]
    unknown = [n for n in cleaned if n not in METRIC_NAMES]
    if unknown:
        raise EvalInputError(
            f"未知的 RAGAS 指标：{', '.join(unknown)}",
            field="metrics",
            hint="可用指标：" + " / ".join(METRIC_NAMES),
        )
    return cleaned or None


def guard_run_ids(run_ids: Sequence[str] | None) -> list[str] | None:
    """`run_id` 逐条过与 papers/research 同一套字符集闸门（校验分叉后弱的那一处就是入口）。"""
    if run_ids is None:
        return None
    from app.api.guards import guard_run_id

    if not run_ids:
        raise EvalInputError("run_ids 不能是空列表；要评全部 run 请省略该字段。", field="run_ids")
    if len(run_ids) > MAX_RUN_IDS_PER_REQUEST:
        raise EvalInputError(
            f"run_ids 一次最多 {MAX_RUN_IDS_PER_REQUEST} 个（每条都要 LLM 裁判逐句判定）",
            field="run_ids",
        )
    cleaned: list[str] = []
    for raw in run_ids:
        value = (raw or "").strip()
        if not value:
            raise EvalInputError("run_ids 里有空串", field="run_ids")
        try:
            cleaned.append(guard_run_id(value))
        except Exception as exc:  # noqa: BLE001 - HTTPException → 本层统一异常类型
            raise EvalInputError(
                "run_id 只允许字母、数字与 . _ -，长度 1~64", field="run_ids"
            ) from exc
    return cleaned


# --------------------------------------------------------------------------- 依赖闸门
def ensure_ragas_available(settings: Settings | None = None) -> None:
    """RAGAS 模式的前置探测：**在起后台任务之前**把"没装可选依赖"暴露成 501。

    为什么要同步探：让后台任务去发现，调用方只能先拿到一个 202、
    再在很久之后读到一条 `failed` —— 那件缺的依赖本来在提交时就已知的。

    探测方式是调 `ragas_runner._load_ragas()`（它就是"真正 import ragas 的那一行"），
    而不是 `import ragas` 自己写一遍：两处判据必须同一个来源，
    否则会出现「闸门放过了、runner 说没装」这种自相矛盾的记录。
    """
    del settings  # 探测与具体配置无关；保留参数是为了调用点整洁
    try:
        from app.eval import ragas_runner
    except ImportError as exc:  # pragma: no cover - 模块在本仓库里，正常不会走到
        raise EvalDependencyMissingError(f"未找到 RAGAS 运行器：{exc}") from exc
    try:
        ragas_runner._load_ragas()
    except ImportError as exc:
        raise EvalDependencyMissingError(
            f"ragas 未安装（{exc}）：RAGAS 是可选依赖，主流程与检索消融都不需要它"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - 装了但坏掉同样是"依赖不可用"
        raise EvalDependencyMissingError(f"ragas 导入失败：{type(exc).__name__}: {exc}") from exc


# --------------------------------------------------------------------------- 三套评测（可编程入口）
def build_retriever(settings: Settings | None = None) -> HybridRetriever:
    """按需构造检索器（索引目录取 settings.faiss_index_dir；模型加载发生在构造里）。"""
    return HybridRetriever(settings=settings or get_settings())


async def run_ablation_eval(
    *,
    retriever: HybridRetriever | None = None,
    items: list[GoldenItem] | None = None,
    golden_path: Path | str | None = None,
    top_k: int | None = None,
    configs: list[AblationConfig] | None = None,
    settings: Settings | None = None,
) -> EvalOutcome:
    """检索层消融 / 指标评测：不需要 ragas、不需要 LLM，只要有索引就能跑。

    Args:
        retriever: 已建好索引的检索器。`None` = 本函数自己按 settings 构造（API 路径）。
                   CLI 传它进来，是为了保留"先打印索引信息、再跑评测"的输出顺序。
        items: golden 查询集；`None` = 从 `golden_path`（或默认数据集）加载。
    """
    settings = settings or get_settings()
    if retriever is None:
        # 构造检索器会读 FAISS + 反序列化 BM25（几十 MB 的同步 I/O）。丢线程池：
        # 评测跑在后台任务里，但后台任务也在**同一个事件循环**上 —— 在这里同步等，
        # 就等于让所有正在流式输出的 SSE 连接一起卡住。（真正的算力大头在
        # embedder/reranker 里，它们自己已经 `to_thread` 了，见各自模块注释。）
        retriever = await asyncio.to_thread(build_retriever, settings)
    if items is None:
        items = load_golden(golden_path)

    if retriever.is_empty:
        # 不返回一张全 0 的表：全 0 会被读成"检索一塌糊涂"，而事实是"没东西可检索"。
        return EvalOutcome(
            ok=False,
            reason="索引为空 —— 先跑 `python scripts/build_index.py` 建索引，再触发消融评测",
            # n_configs 报的是"本该跑几种"（None = 全部六种），而不是 len(None)=0
            summary={
                "index_size": 0,
                "n_queries": len(items),
                "n_configs": len(configs) if configs is not None else len(ABLATION_CONFIGS),
            },
        )

    markdown, results = await run_and_render(
        retriever, items, settings=settings, golden_path=golden_path, top_k=top_k, configs=configs
    )
    notice, degraded = _provider_notice(settings, retriever.embedder, retriever.reranker.name)
    effective_top_k = top_k or settings.retrieve_top_k

    return EvalOutcome(
        ok=True,
        markdown=markdown,
        summary={
            "n_queries": len(items),
            "n_configs": len(results),
            "index_size": retriever.size,
            "n_papers": len(retriever._titles),
            "top_k": effective_top_k,
            "embedding_provider": settings.embedding_provider,
            "reranker": retriever.reranker.name,
            "degraded": degraded,
            "provider_notice": notice,
            "golden": str(golden_path) if golden_path else "默认（app/eval/datasets/golden_40.jsonl）",
            # 明细行的口径随模式不同（同一列 `contexts` 两种模式装的东西不一样），
            # 所以把口径写进 summary：读评测记录的人不必回头翻代码猜。
            "contexts_kind": "该配置检回的 paper 级 arxiv_id 有序列表（不是 chunk 正文）",
        },
        detail={"configs": [_ablation_config_row(r) for r in results]},
        samples=_ablation_sample_rows(results, items),
    )


def _ablation_config_row(result: Any) -> dict[str, Any]:
    """一种配置的汇总行（列名沿用 `RetrievalMetrics` 的字段名，不另起一套叫法）。"""
    m = result.metrics
    return {
        "key": result.key,
        "label": result.label,
        "recall_at_5": round(m.recall_at_5, 4),
        "precision_at_5": round(m.precision_at_5, 4),
        "ndcg_at_10": round(m.ndcg_at_10, 4),
        "mrr": round(m.mrr, 4),
        "median_latency_ms": round(float(result.median_latency_ms), 3),
        "n_queries": m.n_queries,
        "n_skipped": m.n_skipped,
    }


def _ablation_sample_rows(results: list[Any], items: list[GoldenItem]) -> list[EvalSampleRow]:
    """逐 (配置 × 查询) 落一行明细，回答"是哪条查询拖后腿、这一条检索到了什么"。

    `contexts` 这一列在本模式下装的是**该配置检回的 paper 级 arxiv_id 有序列表**
    （检索明细本身就是 id 序列），分数进 `scores`；RAGAS 模式下同一列装 chunk 正文。
    两种模式的口径写进 `summary`，读的人不必猜。
    """
    rows: list[EvalSampleRow] = []
    for result in results:
        by_qid = {row.get("query_id"): row for row in result.metrics.per_query}
        for idx, item in enumerate(items):
            ranked = list(result.ranked[idx]) if idx < len(result.ranked) else []
            latencies = result.per_query_latency_ms
            scores: dict[str, Any] = {
                "config_key": result.key,
                "config_label": result.label,
                "latency_ms": round(float(latencies[idx]), 3) if idx < len(latencies) else None,
            }
            per_query = by_qid.get(item.id)
            if per_query is None:
                # 金标准为空的查询被跳过而不是计 0 分（`evaluate_retrieval` 的口径），
                # 这里如实打标，免得明细里"缺一行"被读成漏跑。
                scores["skipped_no_gold"] = True
            else:
                scores.update(
                    {
                        k: _json_safe(v)
                        for k, v in per_query.items()
                        if k != "query_id"
                    }
                )
            rows.append(
                EvalSampleRow(
                    question=item.question,
                    answer=None,
                    contexts=ranked,
                    scores=scores,
                )
            )
    return rows


async def run_success_metrics_eval(
    *, scope: Any = None, min_cells: int | None = None, min_runs: int | None = None
) -> EvalOutcome:
    """产品定义 §8 的两条成功指标（对比表字段完整率 / 子问题拆解可用率）。只读 DB、零依赖。

    "拒绝计算"（无数据 / 样本不足）在这里是**正常产出**：`ok=True`，
    每条指标的 `status` 与分子分母如实进 `detail`，不粉饰成 0 分。
    """
    from app.eval import success_metrics

    kwargs: dict[str, Any] = {}
    if min_cells is not None:
        kwargs["min_cells"] = min_cells
    if min_runs is not None:
        kwargs["min_runs"] = min_runs
    markdown, outcomes = await success_metrics.run_and_render(scope, **kwargs)

    rows = [
        EvalSampleRow(
            question=o.label,
            answer=o.display(),
            contexts=[],
            scores=_json_safe(
                {
                    "key": o.key,
                    "target": o.target,
                    "value": o.value,
                    "numerator": o.numerator,
                    "denominator": o.denominator,
                    "unit": o.unit,
                    "status": o.status,
                    "reason": o.reason,
                    **({"extras": o.extras} if o.extras else {}),
                }
            ),
        )
        for o in outcomes
    ]
    return EvalOutcome(
        ok=True,
        markdown=markdown,
        summary={
            "n_metrics": len(outcomes),
            "statuses": {o.key: o.status for o in outcomes},
            "note": "拒绝计算（无数据/样本不足）是诚实产出，不是失败",
        },
        detail={
            "metrics": [
                {
                    "key": o.key,
                    "label": o.label,
                    "target": o.target,
                    "value": o.value,
                    "numerator": o.numerator,
                    "denominator": o.denominator,
                    "unit": o.unit,
                    "status": o.status,
                    "reason": o.reason,
                }
                for o in outcomes
            ]
        },
        samples=rows,
    )


async def run_ragas_eval(
    *,
    run_ids: Sequence[str] | str | None = None,
    metric_names: Sequence[str] | str | None = None,
    golden_path: Path | str | None = None,
) -> EvalOutcome:
    """RAGAS 端到端评测：需要 ragas 包 + 真实 LLM 裁判。

    降级（没装 ragas / `LLM_PROVIDER=mock` / 没配 key / 没有可评的 run）由
    `run_ragas_eval` 自己返回 `available=False` + 原因，本函数原样转成 `ok=False`
    且 **detail/samples 全空** —— 分数不是本层算的，也就轮不到本层往里填。

    `run_ids` / `metric_names` 容忍单个字符串（CLI 老写法），内部一律归一成列表：
    `run_ragas_eval(run_ids="abc")` 会被 `list(str)` 拆成 ["a","b","c"]，
    那种 id 必然查不到，最终只会得到一句"没有可评测的 run"。
    """
    from app.eval import ragas_runner

    ids = _as_list(run_ids)
    metrics_arg = _as_list(metric_names)

    report = await ragas_runner.run_ragas_eval(
        run_ids=ids, metrics=metrics_arg, golden_path=golden_path
    )
    available = bool(getattr(report, "available", True))
    if not available:
        return EvalOutcome(
            ok=False,
            reason=getattr(report, "reason", None) or "RAGAS 不可用（未给出原因）",
            markdown=str(report),
            summary={
                "skipped_metrics": list(getattr(report, "skipped_metrics", []) or []),
                "generation_config": _json_safe(getattr(report, "generation_config", {}) or {}),
                "run_ids": list(getattr(report, "run_ids", []) or []),
            },
        )

    per_sample = list(getattr(report, "per_sample", []) or [])
    samples, paired = await _ragas_sample_rows(
        ragas_runner, list(report.run_ids or []), per_sample, golden_path
    )
    return EvalOutcome(
        ok=True,
        markdown=str(report),
        summary={
            "n_samples": int(getattr(report, "n_samples", len(per_sample))),
            "n_with_ground_truth": int(getattr(report, "n_with_ground_truth", 0)),
            "run_ids": list(getattr(report, "run_ids", []) or []),
            "citation_accuracy": _json_safe(getattr(report, "citation_accuracy", None)),
            "manual_accuracy": _json_safe(getattr(report, "manual_accuracy", None)),
            "contexts_kind": "chunk 正文（被引用到的那些），见 ragas_runner._contexts_from_db 的口径说明",
        },
        detail={
            "metrics": _json_safe(dict(getattr(report, "metrics", {}) or {})),
            "skipped_metrics": list(getattr(report, "skipped_metrics", []) or []),
            "generation_config": _json_safe(getattr(report, "generation_config", {}) or {}),
            # 明细行数与分数行数配不上对时如实标出来：宁缺毋假（见下面的配对规则）
            "per_sample_persisted": paired,
        },
        samples=samples,
    )


def _as_list(value: Sequence[str] | str | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


async def _ragas_sample_rows(
    ragas_runner: Any,
    run_ids: list[str],
    per_sample: list[dict[str, Any]],
    golden_path: Path | str | None,
) -> tuple[list[EvalSampleRow], bool]:
    """按 run 取回样本明细，并把逐样本分数**按同一顺序**配对。

    返回 `(明细行, 是否配对成功)`。为什么要再取一次数：`EvalReport.per_sample` 只有
    ragas 的分数行（键是指标列名），不含 question/answer/contexts。重新走
    `build_ragas_dataset` 而不是在 runner 里顺手把样本带出来，是为了不改评测运行器的
    对外契约（它已经被 CLI 与测试依赖着）。

    配对不上（行数不一致，或某条 run 已被删除）时**只落明细不落分数**并把标志交出去：
    把某一行的分数错配到另一条样本上，比少几个分数严重得多。
    """
    datasets = []
    for rid in run_ids:
        try:
            datasets.append(await ragas_runner.build_ragas_dataset(rid, golden_path=golden_path))
        except ragas_runner.RunNotFoundError:
            continue

    paired = len(datasets) == len(per_sample)
    rows: list[EvalSampleRow] = []
    for i, sample in enumerate(datasets):
        scores: dict[str, Any] = {"run_id": sample.run_id, "golden_id": sample.golden_id}
        if paired:
            scores.update(_json_safe(dict(per_sample[i])))
        rows.append(
            EvalSampleRow(
                question=sample.question,
                answer=sample.answer,
                contexts=list(sample.contexts),
                scores=scores,
            )
        )
    return rows, paired


# --------------------------------------------------------------------------- 落库
async def _insert_running(*, name: str, dataset: str | None, mode: str, params: dict[str, Any]) -> int:
    """建 `eval_runs` 行并立刻标 running，返回自增 id。

    ⚠️ JSON 列必须**整体重新赋值**（`row.metrics = {...}`）才会被 SQLAlchemy 判定为脏：
    这里的列类型是裸 `JSON`，不是 `MutableDict`，原地 `update` 不会产生 UPDATE 语句。
    """
    async with db_session.session_scope() as session:
        row = EvalRun(
            name=name,
            dataset=dataset,
            config={"mode": mode, "params": params},
            metrics={"status": STATUS_RUNNING, "mode": mode, "started_at": _now_iso()},
        )
        session.add(row)
        await session.flush()
        return int(row.id)


async def _persist_finished(
    eval_run_id: int,
    *,
    status: str,
    outcome: EvalOutcome | None,
    error: str | None,
    duration_s: float,
) -> None:
    """写终态 metrics + 明细行。

    两条纪律：
    - **明细只在 `completed` 时落库**。`unavailable` / `failed` 都不写行 ——
      "有明细行但没有分数"会被下游读成"评过了"；
    - 但 `unavailable` 的 `reason` 与 `summary`（跳过了哪些指标、索引有多大）**要落**：
      那正是"为什么这份报告是空的"的唯一答案，丢了它调用方只能看到一个空壳状态。
    """
    async with db_session.session_scope() as session:
        row = await session.get(EvalRun, eval_run_id)
        if row is None:  # 行被删了（级联删除评测记录）：静默返回，别在后台任务里抛栈
            return
        metrics: dict[str, Any] = {
            "status": status,
            "finished_at": _now_iso(),
            "duration_s": round(duration_s, 3),
        }
        if row.metrics:
            metrics["started_at"] = row.metrics.get("started_at")
            metrics["mode"] = row.metrics.get("mode")
        if outcome is not None:
            metrics["reason"] = outcome.reason
            metrics["summary"] = _json_safe(outcome.summary)
            metrics["result"] = _json_safe(outcome.detail)
            metrics["n_samples"] = len(outcome.samples) if status == STATUS_COMPLETED else 0
        if error:
            metrics["error"] = error
        row.metrics = metrics  # ← 整体赋值，见上面的 MutableDict 说明

        if outcome is not None and status == STATUS_COMPLETED:
            for s in outcome.samples:
                session.add(
                    EvalSample(
                        eval_run_id=eval_run_id,
                        question=s.question,
                        answer=s.answer,
                        contexts=_json_safe(s.contexts),
                        scores=_json_safe(s.scores),
                    )
                )


# --------------------------------------------------------------------------- 单实例闸门 + 后台执行
@dataclass(slots=True)
class ActiveRun:
    """进程内"正在跑的评测"。刻意只有这一处状态，不假装它能跨重启。"""

    eval_run_id: int
    mode: str
    task: asyncio.Task[None] | None = None
    started_at: float = field(default_factory=time.monotonic)


_ACTIVE: ActiveRun | None = None


def current_run() -> dict[str, Any] | None:
    """当前在跑的评测（`{eval_run_id, mode, elapsed_s}`），没有则 None。"""
    if _ACTIVE is None:
        return None
    return {
        "eval_run_id": _ACTIVE.eval_run_id,
        "mode": _ACTIVE.mode,
        "elapsed_s": round(time.monotonic() - _ACTIVE.started_at, 3),
    }


def clear_active_run() -> None:
    """放开闸门。正常路径由 `_execute` 的 finally 调用；测试与运维可用它复位。"""
    global _ACTIVE  # noqa: PLW0603 - 进程内单例，函数即为唯一写入口
    _ACTIVE = None


def _dispatch(mode: str, params: dict[str, Any], runtime: dict[str, Any]) -> Any:
    """按模式转调到对应的评测入口（新增模式只改这里，路由与本层其余代码不动）。

    `params` 是给 `eval_runs.config` 看的 JSON 视图，`runtime` 是执行要用的对象
    （如已解析的 `AblationConfig` 列表）。分开存是因为 JSON 列里只能放 key 字符串，
    把两者混成一个就会发生"把 `'rrf'` 当配置对象去 `retriever.search(...)`"这种事故。
    """
    if mode == MODE_ABLATION:
        return run_ablation_eval(
            golden_path=params.get("golden_path"),
            top_k=params.get("top_k"),
            configs=runtime.get("configs"),
        )
    if mode == MODE_METRICS:
        return run_success_metrics_eval()
    return run_ragas_eval(
        run_ids=params.get("run_ids"),
        metric_names=params.get("metric_names"),
        golden_path=params.get("golden_path"),
    )


async def _execute(
    eval_run_id: int, mode: str, params: dict[str, Any], runtime: dict[str, Any] | None = None
) -> None:
    """后台任务体：跑评测 → 落库 → 释放闸门。任何异常都吃掉并记成 failed。

    为什么必须自己兜住：任务已经脱离请求生命周期，没人能看见它的栈；
    漏一个异常就是一条永远停在 `running` 的记录 + 一道永久占住的闸门。
    """
    started = time.perf_counter()
    outcome: EvalOutcome | None = None
    status = STATUS_FAILED
    error: str | None = None
    try:
        outcome = await _dispatch(mode, params, runtime or {})
        status = STATUS_COMPLETED if outcome.ok else STATUS_UNAVAILABLE
    except Exception as exc:  # noqa: BLE001 - 后台任务的最后一道兜底
        error = f"{type(exc).__name__}: {exc}"
        logger.error("eval.run_failed", eval_run_id=eval_run_id, mode=mode, error=error[:300])
    finally:
        duration = time.perf_counter() - started
        # 明细只在真正算出结果时落库；unavailable/failed 都不写行，
        # 免得"有明细行但没有分数"被下游误读成"评过了"。
        try:
            await _persist_finished(
                eval_run_id,
                status=status,
                # failed 时 outcome 为 None（什么都没算出来）；
                # completed / unavailable 都带下去 —— 后者要留下"为什么没结果"。
                outcome=outcome,
                error=error,
                duration_s=duration,
            )
        except Exception as exc:  # noqa: BLE001 - 落库失败也不能把闸门卡死
            logger.error("eval.persist_failed", eval_run_id=eval_run_id, error=f"{type(exc).__name__}: {exc}"[:300])
        if _ACTIVE is not None and _ACTIVE.eval_run_id == eval_run_id:
            clear_active_run()
        logger.info("eval.run_finished", eval_run_id=eval_run_id, mode=mode, status=status)


def _dataset_label(mode: str, params: dict[str, Any]) -> str:
    """`eval_runs.dataset`：一眼能看出这批结果是拿什么算的。"""
    if mode == MODE_ABLATION:
        golden = params.get("golden_path")
        return Path(golden).stem if golden else "golden_40"
    if mode == MODE_RAGAS:
        ids = params.get("run_ids")
        return f"research_runs:{len(ids)}" if ids else "research_runs:all"
    return "success_metrics"


async def start_eval_run(
    mode: str,
    *,
    name: str | None = None,
    golden_path: str | Path | None = None,
    top_k: int | None = None,
    configs: str | None = None,
    run_ids: Sequence[str] | None = None,
    metric_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """校验入参 → 抢闸门 → 建 running 记录 → 起后台任务，立刻返回。

    校验一律发生在**抢闸门之前**：坏请求不该让后面所有评测收到 409。
    """
    global _ACTIVE  # noqa: PLW0603 - 见下面的抢占顺序说明

    if mode not in MODES:
        raise EvalInputError(
            f"未知的评测模式 {mode!r}",
            field="mode",
            hint="可用：" + " / ".join(MODES),
        )
    if mode == MODE_RAGAS:
        # 探测要真的 import ragas（它会拉 datasets 等重依赖，秒级）。放在请求协程里同步做
        # 就会卡住事件循环 —— 而此时其它 SSE 流式会话正在同一个循环上，一起被冻住。
        await asyncio.to_thread(ensure_ragas_available)

    resolved_golden = resolve_golden_path(golden_path)
    parsed_configs = parse_configs(configs) if mode == MODE_ABLATION else None
    guarded_run_ids = guard_run_ids(run_ids) if mode == MODE_RAGAS else None
    parsed_metrics = parse_metric_names(metric_names) if mode == MODE_RAGAS else None

    if _ACTIVE is not None:
        raise EvalAlreadyRunningError(
            "已有评测在跑：评测会加载 embedding/重排模型并读写同一份检索索引，"
            "并发跑既会污染消融表的耗时列，也容易 OOM。",
            eval_run_id=_ACTIVE.eval_run_id,
            mode=_ACTIVE.mode,
        )

    eval_name = (name or "").strip() or f"{mode}-{datetime.now(UTC):%Y%m%d-%H%M%S}"
    # `params` 进 `eval_runs.config`（必须 JSON 友好：路径是字符串、配置是 key 列表），
    # `runtime` 才是执行要用的对象（Path / list[AblationConfig]）。
    params: dict[str, Any] = {
        "golden_path": str(resolved_golden) if resolved_golden else None,
        "top_k": top_k,
        "configs": [c.key for c in parsed_configs] if parsed_configs else None,
        "run_ids": guarded_run_ids,
        "metric_names": parsed_metrics,
    }
    runtime: dict[str, Any] = {
        # 只有消融配置需要"对象形态"的执行参数；路径与 id 都是字符串，两处同源。
        "configs": parsed_configs,
    }
    dataset = _dataset_label(mode, params)

    # ★ 抢占发生在**任何 await 之前**：`_ACTIVE` 先占位、行 id 随后补。
    # 若先 await 建库行再置位，两个并发请求会同时通过上面的 `is not None` 检查，
    # 409 闸门就在高并发下形同虚设（而它正是这个接口唯一的互斥手段）。
    holder = ActiveRun(eval_run_id=0, mode=mode)
    _ACTIVE = holder
    try:
        eval_run_id = await _insert_running(name=eval_name, dataset=dataset, mode=mode, params=params)
        holder.eval_run_id = eval_run_id
        task = asyncio.create_task(_execute(eval_run_id, mode, params, runtime))
        holder.task = task
    except Exception:
        _ACTIVE = None
        raise

    logger.info("eval.run_started", eval_run_id=eval_run_id, mode=mode)
    return {
        "eval_run_id": eval_run_id,
        "status": STATUS_RUNNING,
        "mode": mode,
        "name": eval_name,
        "dataset": dataset,
    }


async def get_eval_run(eval_run_id: int, *, sample_limit: int = 200) -> dict[str, Any]:
    """查一条评测的状态与结果。

    返回的 `status` 来自 `eval_runs.metrics`（唯一事实来源）；
    `is_running` 则同时看进程内的闸门状态 —— 进程重启后留在 `running` 的行
    在这里会显示 `is_running=false`，调用方据此判断"这条已经不会被更新了"。
    """
    async with db_session.session_scope() as session:
        row = await session.get(EvalRun, int(eval_run_id))
        if row is None:
            raise EvalRunNotFoundError(f"找不到 eval_run_id={eval_run_id}（eval_runs 中不存在）")
        metrics = dict(row.metrics or {})
        sample_rows = list(
            (
                await session.execute(
                    select(EvalSample)
                    .where(EvalSample.eval_run_id == row.id)
                    .order_by(EvalSample.id)
                    .limit(sample_limit)
                )
            ).scalars().all()
        )
        n_samples = int(
            (
                await session.execute(
                    select(func.count()).select_from(EvalSample).where(EvalSample.eval_run_id == row.id)
                )
            ).scalar_one()
        )

    status = str(metrics.get("status") or STATUS_RUNNING)
    return {
        "eval_run_id": row.id,
        "name": row.name,
        "dataset": row.dataset,
        "mode": metrics.get("mode") or (row.config or {}).get("mode"),
        "status": status,
        "finished": status in TERMINAL_STATUSES,
        "is_running": _ACTIVE is not None and _ACTIVE.eval_run_id == row.id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": metrics.get("started_at"),
        "finished_at": metrics.get("finished_at"),
        "duration_s": metrics.get("duration_s"),
        "reason": metrics.get("reason"),
        "error": metrics.get("error"),
        "config": row.config or {},
        "summary": metrics.get("summary") or {},
        "result": metrics.get("result") or {},
        "n_samples": n_samples,
        "samples_truncated": n_samples > len(sample_rows),
        "samples": [
            {
                "id": s.id,
                "question": s.question,
                "answer": s.answer,
                "contexts": s.contexts or [],
                "scores": s.scores or {},
            }
            for s in sample_rows
        ],
    }


__all__ = [
    "EVAL_INSTALL_CMD",
    "MAX_RUN_IDS_PER_REQUEST",
    "MODES",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_RUNNING",
    "STATUS_UNAVAILABLE",
    "EvalAlreadyRunningError",
    "EvalDependencyMissingError",
    "EvalError",
    "EvalInputError",
    "EvalOutcome",
    "EvalRunNotFoundError",
    "EvalSampleRow",
    "build_retriever",
    "clear_active_run",
    "current_run",
    "ensure_ragas_available",
    "get_eval_run",
    "guard_run_ids",
    "parse_configs",
    "parse_metric_names",
    "resolve_golden_path",
    "run_ablation_eval",
    "run_ragas_eval",
    "run_success_metrics_eval",
    "start_eval_run",
]

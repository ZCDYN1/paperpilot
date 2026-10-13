"""RAGAS 评测运行器 —— 把「一次 research run 的结果」变成可复算的生成侧指标。

## 这个模块只做三件事，且每件事都对应一类必须守住的原则

1. **`citation_accuracy(run_id)`** —— 不依赖 LLM 的**自动口径**引用准确率。
   架构 §12 原文："`citations` 表中 `verified = true` 的占比 …… 这条指标是**免费**的"。
   因为 Verifier 节点已把校验结果写回 `verified` 字段，所以它只是对数据库做一次计数，
   在**没有任何 API key、没有装 ragas** 的环境下也能给出真实数字。
   ⚠️ 但自动口径的天花板是 Verifier 本身：`LLM_PROVIDER=mock` 下软校验按设计永远放行，
   `verified` 会恒为 True —— 那是结构保证，不是证据。所以配套
   **`manual_accuracy(csv_path)`（人工口径）**：读 `scripts/spotcheck_citations.py`
   导出、人工填写的抽检 CSV，报告里两个口径**并列**（架构 §12 双口径要求）。

2. **`build_ragas_dataset(run_id)`** —— 从 DB（或活的最终 state）组装
   `question / answer(draft) / contexts / ground_truth`。
   它**刻意不 import ragas**：RAGAS 的数据结构是它的实现细节，不该泄漏到
   "取数"这一步。否则没装 ragas 时连"我们手里有哪些数据"都问不出来。

3. **`run_ragas_eval(...)`** —— 真正调用 RAGAS 计算五个指标，并返回结构化报告。

## 为什么"降级"是本模块的重点而不是边角料

RAGAS 的指标本质是**"用 LLM 当裁判"**（faithfulness 甚至要 LLM 先拆句再逐句判真伪）。
这意味着它天然有两种失效方式，而这两种都会**悄悄**发生：

- **没装 ragas** —— 顶层 import 会直接 ImportError，把"想看看数据"的人挡在门外；
- **有 ragas 但没有可用的裁判模型**（未配 key / `LLM_PROVIDER=mock`）——
  若此时仍然返回一串数字，那是**在拿 mock 的确定性占位输出当分数**。
  这种"看起来跑通了"的假分数比不跑更糟：它会进入对比表、被人引用、误导后续所有决策。

所以本模块对后者的处理只有一条：**返回 `available=False` + 原因 + 被跳过的指标，绝不产出任何分数**。
这条不变量由 `EvalReport.__post_init__` 结构性保证：`available=False` 时携带任何分数会直接抛错，
而不是靠"写代码的人记得别填"。

## 生成配置为什么必须进报告

架构 §12 写得很直白：`faithfulness` 是**生成端**指标，受生成模型与 prompt 的影响远大于受检索影响；
"检索变好 1 分、换模型波动 10 分"，不记配置的分数**没有可比性**。
因此报告里必须带上 `provider / model / temperature / prompt_version`，
让"这一版分数"能被复现、能被放到同一基线上比较。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean
from typing import Any

from langchain_core.embeddings import Embeddings
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.db import session as db_session
from app.db.models import Citation, Message, PaperChunk, ResearchRun
from app.eval.dataset import GoldenItem, load_golden

# 架构 §12 的五个指标。顺序即报告里的展示顺序（faithfulness 排第一，因为它最重要）。
METRIC_NAMES: tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

# 需要"人工参考答案"（golden 的 reference_answer）才能算的指标。
# 没有 ground_truth 时**必须跳过**而不是算 0 分 —— 把"没标注"错算成"答错了"，
# 会系统性低估效果，是最容易让评测表失真的那种错误。
REFERENCE_METRICS: frozenset[str] = frozenset({"context_recall", "answer_correctness"})

# 需要 embedding 当裁判的指标（answer_relevancy 要算"逆向生成的问句"与原问句的相似度；
# answer_correctness 要算语义一致度）。mock embedding 是无语义的哈希随机向量，
# 用它当裁判只会产出一个**看起来像分数、实际是噪声**的数，所以这些指标在 mock embedding 下同样跳过。
EMBEDDING_METRICS: frozenset[str] = frozenset({"answer_relevancy", "answer_correctness"})

# 报告中"未安装 ragas"的统一话术。调用方可能直接把它展示给用户，所以写成可执行指引。
_RAGAS_MISSING_REASON = 'ragas 未安装，请 pip install -e ".[eval]"'


class RunNotFoundError(ValueError):
    """请求的 run_id 在库里不存在。

    单独定义类型（而不是让 SQLAlchemy 的 `NoResultFound` 或 KeyError 冒出去）：
    上层据此区分"这条 run 没有数据"与"评测逻辑坏了"，并可对单条 run 跳过而不中断整批。
    """


# --------------------------------------------------------------------------- 数据结构
@dataclass(slots=True)
class RagasSample:
    """一条 RAGAS 单轮样本（对应架构里"一次 run"）。

    刻意**不用 ragas 的 `SingleTurnSample`**：取数层不该依赖评测框架。
    真正跑评测时（`_to_ragas_dataset`）再做一次等价转换即可。
    """

    run_id: str
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str | None = None
    golden_id: str | None = None
    # 原始 citation 行（保持顺序），保留它是为了能回答"这个分数是哪几条引用支撑的"
    citations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_ground_truth(self) -> bool:
        """是否有参考答案。RAGAS 的 reference 类指标只有在它为真时才算得出来。"""
        return bool(self.ground_truth and self.ground_truth.strip())


@dataclass
class EvalReport:
    """一次 RAGAS 评测的结构化结果。

    调用方只看 `available` + `reason` 就能决定 UI 怎么展示：
    - `available=True`：`metrics` 是**真实**分数，`generation_config` 说明它是在什么配置下得到的；
    - `available=False`：`reason` 说明为什么没算，`skipped_metrics` 说明连带被跳过了哪些。
    """

    available: bool
    reason: str | None = None
    skipped_metrics: list[str] = field(default_factory=list)
    # ★ 只有真正跑过 RAGAS 才会有内容。降级路径一律为空 —— 这是"绝不返回假分数"的落点。
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] = field(default_factory=list)
    # 生成配置（provider / model / temperature / prompt_version）：架构 §12 的可比性前提
    generation_config: dict[str, Any] = field(default_factory=dict)
    run_ids: list[str] = field(default_factory=list)
    n_samples: int = 0
    n_with_ground_truth: int = 0
    citation_accuracy: dict[str, Any] | None = None
    # ★ 人工口径（抽检）：架构 §12 要求与自动口径**并列汇报**。
    # None = 没有已填写的抽检文件（`scripts/spotcheck_citations.py` 生成）。
    manual_accuracy: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """把"不产出假分数"从口头约定升级为**结构性不变量**。

        为什么要用抛错而不是静默清空：静默清空会让"某处误填了分数"这种 bug
        以"数字莫名其妙不见了"的形式出现，排查成本极高；直接抛错则当场暴露。
        """
        if not self.available and (self.metrics or self.per_sample):
            raise ValueError(
                "EvalReport 不变量被破坏：available=False 时不得携带任何分数（metrics/per_sample 必须为空）"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON 友好视图，供 UI / CLI 直接序列化。"""
        return {
            "available": self.available,
            "reason": self.reason,
            "skipped_metrics": list(self.skipped_metrics),
            "metrics": dict(self.metrics),
            "per_sample": list(self.per_sample),
            "generation_config": dict(self.generation_config),
            "run_ids": list(self.run_ids),
            "n_samples": self.n_samples,
            "n_with_ground_truth": self.n_with_ground_truth,
            "citation_accuracy": self.citation_accuracy,
            "manual_accuracy": self.manual_accuracy,
        }


def _unavailable(
    reason: str,
    *,
    skipped_metrics: list[str] | None = None,
    generation_config: dict[str, Any] | None = None,
    run_ids: list[str] | None = None,
    n_samples: int = 0,
    n_with_ground_truth: int = 0,
) -> EvalReport:
    """构造一个"不可用"报告的唯一入口。

    把降级路径全部收敛到这个函数，是为了让**每一条**降级分支都自动满足
    "metrics 为空"这条不变量 —— 新增降级条件时也不可能漏填分数。
    """
    return EvalReport(
        available=False,
        reason=reason,
        skipped_metrics=list(skipped_metrics if skipped_metrics is not None else METRIC_NAMES),
        generation_config=generation_config or {},
        run_ids=list(run_ids or []),
        n_samples=n_samples,
        n_with_ground_truth=n_with_ground_truth,
    )


# --------------------------------------------------------------------------- 生成配置
def _prompt_version() -> str:
    """用 prompt 模板内容的哈希当作"prompt 版本"。

    为什么不用手写版本号：手写的版本号会忘记改 —— 改了 planner.md 却忘了 bump，
    于是"同一版本号"下其实换了 prompt，分数就不再可比。内容哈希无法被遗忘，
    只要模板变了它就变；模板没变它就稳定，正好满足"可比"的要求。
    取不到模板目录时返回 "unknown"（不抛错：报告缺一个字段，不该让整次评测失败）。
    """
    try:
        import app.agents.prompts as prompts_mod

        prompt_dir = Path(prompts_mod.__file__).parent
        digest = hashlib.sha256()
        for path in sorted(prompt_dir.glob("*.md")):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest()[:12]
    except Exception:  # noqa: BLE001 - 版本指纹失败不应中断评测
        return "unknown"


def _ragas_version() -> str:
    """当前 ragas 版本（未安装则返回说明）。用于报告中标注"分数由哪个版本产出"。"""
    try:
        import ragas

        return str(getattr(ragas, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "未安装"


def generation_config(settings: Settings | None = None) -> dict[str, Any]:
    """本次评测的**生成配置** —— 架构 §12 要求的"分数可比性前提"。

    `faithfulness` 等指标由 LLM 生成/判定，换模型或改 prompt 的波动常常大于检索侧的改进。
    不把这些配置记下来，表格里的数字就无法被复现、也无法与另一版对比。
    """
    s = settings or get_settings()
    model = s.ollama_model if s.llm_provider == "ollama" else s.llm_model
    return {
        "llm_provider": s.llm_provider,
        "llm_model": model,
        "temperature": s.llm_temperature,
        "prompt_version": _prompt_version(),
        "embedding_provider": s.embedding_provider,
        "ragas_version": _ragas_version(),
    }


# --------------------------------------------------------------------------- 自动口径引用准确率
async def citation_accuracy(run_id: str) -> dict[str, Any]:
    """**自动口径**引用准确率：`verified = true` 的引用占比（架构 §12）。

    为什么这条最重要：它**不依赖任何模型**。Verifier 节点在逐条校验时已经
    把结论写回 `citations.verified`，所以这里只是对数据库做一次分组计数 ——
    零成本、可复现、无 key 也能得到真实数字。它同时也是"引用准确率 ≥ 90%"
    双口径里的自动那一半（人工抽检口径另有其流程）。

    `verified` 为 `None`（尚未校验）或 `False` 都**不计入**分子：
    指标问的是"已确认为真的引用有多少"，把"没校验"算进去等于自欺欺人。

    Returns:
        `{"total": n, "verified": k, "accuracy": k/n, "method": "auto"}`；
        没有引用时 `accuracy` 取 `0.0`（占比无定义，但不能让调用方拿到除零异常）。
    """
    async with db_session.session_scope() as session:
        rows = (
            await session.execute(select(Citation.verified).where(Citation.run_id == run_id))
        ).scalars().all()

    total = len(rows)
    verified = sum(1 for value in rows if value is True)
    return {
        "total": total,
        "verified": verified,
        "accuracy": (verified / total) if total else 0.0,
        "method": "auto",
    }


# --------------------------------------------------------------------------- 人工口径引用准确率
# 抽检 CSV 的默认位置：与 scripts/spotcheck_citations.py 的 --out 默认值一致。
# 报告端据此**自动**并列人工口径（文件不存在则跳过，不编造数字）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPOTCHECK_CSV = PROJECT_ROOT / "eval_results" / "citation_spotcheck.csv"

# 人工判定列允许的取值（大小写/全半角宽松匹配在 _normalize_verdict 里做）。
# ambiguous 计入分母但不计入分子：拿不准的引用不该被算成"名副其实"。
_VERDICT_SUPPORTED = "supported"
_VERDICT_UNSUPPORTED = "unsupported"
_VERDICT_AMBIGUOUS = "ambiguous"


def _normalize_verdict(raw: str | None) -> str | None:
    """把人工填写的判定归一成 supported/unsupported/ambiguous；无法识别返回 None。

    兼容常见的手填变体（yes/no、true/false、中文"是/否"），因为抽检大多在 Excel 里做，
    不能指望填写者严格照抄枚举值 —— 但也不能把乱填当成"没填"默默丢弃，
    调用方通过 `invalid` 计数能看到有多少条被判为不可识别。
    """
    v = (raw or "").strip().lower()
    if not v:
        return None
    if v in {_VERDICT_SUPPORTED, "yes", "true", "1", "是", "支持", "名副其实"}:
        return _VERDICT_SUPPORTED
    if v in {_VERDICT_UNSUPPORTED, "no", "false", "0", "否", "不支持", "名不副实"}:
        return _VERDICT_UNSUPPORTED
    if v in {_VERDICT_AMBIGUOUS, "ambiguous", "unclear", "存疑", "待定"}:
        return _VERDICT_AMBIGUOUS
    return None


def manual_accuracy(csv_path: Path | str | None = None) -> dict[str, Any]:
    """**人工口径**引用准确率：读抽检 CSV 的 `human_verdict` 列做统计（架构 §12 双口径的人工那一半）。

    为什么自动口径不够：`citations.verified` 是 Verifier（含 LLM 软校验）自己写回的，
    mock/宽松模型下它**结构上恒为 True**（本机库里的 30 条全 True 就是这么来的）。
    只报自动口径等于自己批改自己的作业。人工口径让人核对过的样本说话，
    `agreement`（人工判定与自动 `verified` 的一致率）就是"Verifier 是否虚高放行"的校准值。

    口径：
    - `accuracy = supported / (supported + unsupported + ambiguous)`，
      未判定行**不进分母**（把"没标"当"不合格"会系统性低估；当"合格"则是自欺）；
    - `agreement` 只在已判定且 `auto_verified` 非空的行上计算，无可比行时为 `None`；
    - `auto_accuracy` 用同一批已判定行的 `auto_verified` 列顺带复算，便于两口径**同分母对比**。

    Args:
        csv_path: 抽检 CSV（`scripts/spotcheck_citations.py` 导出、填好后回填的路径）。
                  默认 `eval_results/citation_spotcheck.csv`。

    Raises:
        FileNotFoundError: 文件不存在（调用方据此区分"没抽过检"与"抽了但没判"）。
    """
    import csv

    path = Path(csv_path) if csv_path is not None else DEFAULT_SPOTCHECK_CSV
    if not path.exists():
        raise FileNotFoundError(f"抽检文件不存在：{path}")

    supported = unsupported = ambiguous = invalid = pending = 0
    agree_rows = 0
    agree = 0
    auto_true = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            verdict = _normalize_verdict(row.get("human_verdict"))
            if verdict is None:
                if not (row.get("human_verdict") or "").strip():
                    pending += 1
                else:
                    invalid += 1
                continue
            if verdict == _VERDICT_SUPPORTED:
                supported += 1
            elif verdict == _VERDICT_UNSUPPORTED:
                unsupported += 1
            else:
                ambiguous += 1
            # 一致率：人工判定 vs 自动 verified（空=未校验，不进分母）
            auto_raw = (row.get("auto_verified") or "").strip().lower()
            if auto_raw in {"true", "1", "yes"}:
                auto_true += 1
                agree_rows += 1
                agree += 1 if verdict == _VERDICT_SUPPORTED else 0
            elif auto_raw in {"false", "0", "no"}:
                agree_rows += 1
                agree += 1 if verdict == _VERDICT_UNSUPPORTED else 0

    total_judged = supported + unsupported + ambiguous
    return {
        "total_judged": total_judged,
        "supported": supported,
        "unsupported": unsupported,
        "ambiguous": ambiguous,
        "pending": pending,
        "invalid": invalid,
        "accuracy": (supported / total_judged) if total_judged else 0.0,
        "agreement": (agree / agree_rows) if agree_rows else None,
        "auto_accuracy": (auto_true / total_judged) if total_judged else None,
        "csv": str(path),
        "method": "manual",
    }


# --------------------------------------------------------------------------- 数据集组装
def _normalize_question(text: str) -> str:
    """问题归一化：折叠空白，用于把 run 的 question 对上 golden 条目。

    只做空白折叠、不做更激进的归一（如去标点/大小写），因为 golden 的中文问题
    一旦过度归一就可能误配到另一个问题 —— 误配会让 `ground_truth` 张冠李戴，
    比"匹配不上、留空"危险得多。
    """
    return " ".join((text or "").split())


def _golden_index(items: list[GoldenItem], golden_item_id: str | None) -> dict[str, GoldenItem]:
    """把 golden 集索引成 `{归一化问题: 条目}`；指定 `golden_item_id` 时只留该条。"""
    index: dict[str, GoldenItem] = {}
    for item in items:
        if golden_item_id is not None and item.id != golden_item_id:
            continue
        index.setdefault(_normalize_question(item.question), item)
    return index


def _contexts_from_state(state: dict[str, Any]) -> list[str]:
    """从活的最终 state 的 `retrieved` 里抽 chunk 文本（论文级命中 → 摊平成 chunk）。

    state 里的 `retrieved` 形如 `{arxiv_id, title, chunks:[{chunk_id, text, ...}]}`，
    是**真正被检索到**的上下文；而 DB 只持久化了被引用的 chunk，
    所以活 state 能得到更完整的 contexts —— 这也是本函数存在的唯一理由。
    """
    contexts: list[str] = []
    seen: set[str] = set()
    for hit in state.get("retrieved") or []:
        for chunk in hit.get("chunks") or []:
            text = (chunk.get("text") or "").strip()
            if text and text not in seen:
                seen.add(text)
                contexts.append(text)
    return contexts


async def _contexts_from_db(session, run_id: str, citation_rows: list[Citation]) -> list[str]:
    """从 DB 还原 contexts：引用到的 chunk 正文，取不到时退回引用片段本身。

    ⚠️ 口径说明（必须写清楚，否则会被误读）：DB 只持久化了**被引用**的 chunk，
    没有存全量检索结果。所以这条路径得到的 contexts 是"报告实际用作证据的那些片段"，
    是全量 retrieved 的一个子集。对 `faithfulness` 而言这恰恰是**最该被核查**的集合
    （断言是否有支撑，只会看它引用的那些），因此作为 DB 缺省口径是合理的近似。
    """
    chunk_ids = {c.chunk_id for c in citation_rows if c.chunk_id is not None}
    texts: dict[int, str] = {}
    if chunk_ids:
        rows = (
            await session.execute(
                select(PaperChunk.id, PaperChunk.text).where(PaperChunk.id.in_(chunk_ids))
            )
        ).all()
        texts = {int(cid): text for cid, text in rows}

    contexts: list[str] = []
    seen: set[str] = set()
    for cit in citation_rows:
        candidate = ""
        if cit.chunk_id is not None and cit.chunk_id in texts:
            candidate = (texts[cit.chunk_id] or "").strip()
        if not candidate:
            # chunk 找不到（被删/外键置空）时用引用片段兜底 —— 有证据总比空上下文好
            candidate = (cit.quote or "").strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            contexts.append(candidate)
    return contexts


async def build_ragas_dataset(
    run_id: str,
    *,
    state: dict[str, Any] | None = None,
    golden_path: Path | str | None = None,
    golden_item_id: str | None = None,
) -> RagasSample:
    """组装一次 run 的 RAGAS 单轮样本。

    Args:
        run_id: `research_runs.id`。
        state: 可选的**活的最终 state**。传了就用它（能拿到全量 `retrieved` 与 `draft`）；
               不传则从 DB 还原（question ← research_runs，answer ← messages 里 writer 写的
               assistant 消息，contexts ← 引用到的 chunk 正文）。
        golden_path: golden 集路径，默认用 `app/eval/datasets/golden_20.jsonl`。
        golden_item_id: 直接指定 golden 条目 id（当 run 的问题与 golden 措辞对不上时用）。

    Raises:
        RunNotFoundError: DB 里没有这条 run（且未提供 `state`）。
    """
    citations_raw: list[dict[str, Any]] = []
    ground_truth: str | None = None
    golden_id: str | None = None

    if state is not None:
        # ---- 活 state 路径：contexts 用全量 retrieved，answer 用 draft ----
        question = str(state.get("question") or "")
        answer = str(state.get("draft") or "")
        contexts = _contexts_from_state(state)
        citations_raw = [dict(c) for c in (state.get("citations") or [])]
    else:
        # ---- DB 路径 ----
        async with db_session.session_scope() as session:
            run = await session.get(ResearchRun, run_id)
            if run is None:
                raise RunNotFoundError(f"找不到 run_id={run_id!r}（research_runs 中不存在）")
            question = run.question or ""

            # draft 由 finalize 以"assistant 消息"形式落进 messages 表（架构 §6 的轻量摘要不含正文），
            # 取最新一条：会话内追问会产生多条，最新的一条才是本次 run 的成稿。
            answer = (
                await session.execute(
                    select(Message.content)
                    .where(Message.run_id == run_id, Message.role == "assistant")
                    .order_by(Message.id.desc())
                    .limit(1)
                )
            ).scalars().first() or ""

            citation_rows = list(
                (
                    await session.execute(
                        select(Citation).where(Citation.run_id == run_id).order_by(Citation.id)
                    )
                ).scalars().all()
            )
            citations_raw = [
                {
                    "claim": c.claim,
                    "chunk_id": c.chunk_id,
                    "paper_id": c.arxiv_id,
                    "quote": c.quote,
                    "locator": c.locator,
                    "source": c.source,
                    "verified": c.verified,
                }
                for c in citation_rows
            ]
            contexts = await _contexts_from_db(session, run_id, citation_rows)

    # ---- ground_truth：来自 golden 的 reference_answer；没有就留空（跳过相关指标）----
    # 用受保护的方式读 golden：文件缺失/格式坏了都不该让"取数"这一步失败。
    try:
        golden_items = load_golden(golden_path)
    except Exception:  # noqa: BLE001 - golden 不可用 → 视作"无参考答案"
        golden_items = []
    golden = _golden_index(golden_items, golden_item_id).get(_normalize_question(question))
    if golden is not None:
        golden_id = golden.id
        ground_truth = golden.reference_answer or None

    return RagasSample(
        run_id=run_id,
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        golden_id=golden_id,
        citations=citations_raw,
    )


# --------------------------------------------------------------------------- ragas 惰性接入
class _RagasAPI:
    """惰性导入的 ragas 符号集合。

    把"从哪 import 什么"集中在这一个小类里，接口一旦随版本变动只需改一处；
    更重要的是：**只有真正要跑评测时**才会触发这些 import（见 `_load_ragas`），
    于是"没装 ragas"的环境仍能使用本模块的取数与引用准确率功能。
    """

    def __init__(self) -> None:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: PLC0415
        from ragas import metrics as ragas_metrics  # noqa: PLC0415
        from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: PLC0415
        from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415

        self.EvaluationDataset = EvaluationDataset
        self.SingleTurnSample = SingleTurnSample
        self.evaluate = evaluate
        self.metrics = ragas_metrics
        self.LangchainLLMWrapper = LangchainLLMWrapper
        self.LangchainEmbeddingsWrapper = LangchainEmbeddingsWrapper


def _load_ragas() -> _RagasAPI:
    """惰性构造 ragas 接入点。

    ⚠️ **绝不在模块顶层 import ragas**：
    1. ragas 依赖较重（拉 datasets / openai 等），会给所有只想用取数功能的人增加启动成本；
    2. 它是可选依赖（`pip install -e ".[eval]"`），没装时顶层 import 会让**整个模块不可用**，
       连"跑一下 citation_accuracy"都做不到 —— 而那恰恰是无 key 环境唯一能兑现的指标。

    测试通过 monkeypatch 本函数来模拟"环境里没有 ragas"。
    """
    return _RagasAPI()


class _EmbedderAdapter(Embeddings):
    """把本项目的 `BaseEmbedder` 适配成 LangChain `Embeddings`，供 RAGAS 当裁判用。

    为什么需要它：RAGAS 的 `LangchainEmbeddingsWrapper` 认的是 LangChain 接口，
    而我们的 embedder 是高内聚的自有接口（`encode_sync` → 已归一化矩阵）。
    这里只做最小胶水，不复制任何逻辑，避免两套实现漂移。
    """

    def __init__(self, embedder: Any) -> None:  # noqa: ANN401 - 避免为类型引入 retrieval 层依赖
        self._embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode_sync(list(texts)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode_sync([text])[0].tolist()


# --------------------------------------------------------------------------- 指标选择
def _select_metric_pairs(
    api: _RagasAPI,
    requested: list[str],
    *,
    all_have_ground_truth: bool,
    embeddings_usable: bool,
) -> tuple[list[tuple[str, Any]], list[str]]:
    """把指标名解析成 (报告列名, 指标对象)，并算出被跳过的指标与原因。

    两条跳过规则，各自对应一类"会产出假分数"的陷阱：

    - **缺 ground_truth** → 跳过 `context_recall` / `answer_correctness`。
      这两者以人工参考答案为分母，缺了它再算出来的不是"答得不好"，而是**没有意义**。
    - **embedding 不可信（provider=mock）** → 跳过 `answer_relevancy` / `answer_correctness`。
      mock embedding 是无语义的哈希向量，拿它当语义裁判得到的数只是噪声。

    `context_precision` 有两条实现：有 reference 时用带参考版（更准），
    没有时用 `LLMContextPrecisionWithoutReference`（仍能衡量"排前面的片段是否真有用"，
    正好对应架构 §12 里它的用途 —— 验证 Reranker 的价值，本就不该依赖参考答案）。
    """
    pairs: list[tuple[str, Any]] = []
    skipped: list[str] = []

    for name in requested:
        if name == "faithfulness":
            pairs.append((name, api.metrics.Faithfulness()))
        elif name == "answer_relevancy":
            if not embeddings_usable:
                skipped.append(name)
                continue
            pairs.append((name, api.metrics.AnswerRelevancy()))
        elif name == "context_precision":
            metric_cls = (
                api.metrics.LLMContextPrecisionWithReference
                if all_have_ground_truth
                else api.metrics.LLMContextPrecisionWithoutReference
            )
            pairs.append((name, metric_cls()))
        elif name in REFERENCE_METRICS:
            if not all_have_ground_truth:
                skipped.append(name)
                continue
            metric_cls = api.metrics.ContextRecall if name == "context_recall" else api.metrics.AnswerCorrectness
            pairs.append((name, metric_cls()))
        else:
            # 未识别的指标名：跳过并让调用方看见，而不是静默忽略（静默会让人以为算了）
            skipped.append(name)

    return pairs, skipped


def _embeddings_usable(settings: Settings) -> bool:
    """embedding 当裁判是否可信。

    只有 `mock` 明确不可信（它不携带任何语义）。`local` / `api` 都是真实向量空间，
    即便本地权重尚未下载，也会在真正调用时报出可操作的错误，而不是产出假分数。
    """
    return settings.embedding_provider != "mock"


# --------------------------------------------------------------------------- 主入口
async def _resolve_run_ids(run_ids: list[str] | None) -> list[str]:
    """把 `run_ids=None` 解析成"库里现有的全部 run"（按开始时间升序）。"""
    if run_ids is not None:
        return list(run_ids)
    async with db_session.session_scope() as session:
        return list(
            (
                await session.execute(select(ResearchRun.id).order_by(ResearchRun.started_at))
            ).scalars().all()
        )


async def run_ragas_eval(
    run_ids: list[str] | None = None,
    metrics: list[str] | None = None,
    *,
    golden_path: Path | str | None = None,
    with_citation_accuracy: bool = True,
    spotcheck_csv: Path | str | None = None,
) -> EvalReport:
    """对一批 run 跑 RAGAS 评测，返回结构化报告。

    Args:
        run_ids: 要评测的 run；`None` 表示"库里现有的全部 run"。
                 不存在的 run 会被**跳过**（不抛 KeyError），从而不中断整批。
        metrics: 要算的指标名（见 `METRIC_NAMES`），默认全部五个。
        golden_path: golden 集路径，用于取 `ground_truth`。
        with_citation_accuracy: 是否附带自动口径引用准确率（它不依赖 LLM，几乎免费）。
        spotcheck_csv: 人工抽检 CSV（`scripts/spotcheck_citations.py` 导出并填写）。
                 存在且已填写时并列进 `report.manual_accuracy`；缺失则留 None（不编数字）。

    Returns:
        `EvalReport`。**任何一步不可用都返回 `available=False` 的报告，绝不抛异常、绝不填假分数。**
    """
    requested = list(metrics) if metrics is not None else list(METRIC_NAMES)
    settings = get_settings()
    config = generation_config(settings)
    manual = _manual_accuracy_if_present(spotcheck_csv)

    # ---- 降级①：没装 ragas ----
    # 放在最前面，因为它是"环境缺依赖"，与具体有哪些 run、配没配 key 无关。
    try:
        api = _load_ragas()
    except ImportError:
        return _with_manual(_unavailable(_RAGAS_MISSING_REASON, generation_config=config), manual)
    except Exception as exc:  # noqa: BLE001 - ragas 装了但坏掉时同样降级而非崩溃
        return _with_manual(_unavailable(f"ragas 导入失败：{exc}", generation_config=config), manual)

    # ---- 降级②：没有可用的裁判模型 ----
    # RAGAS 的指标全靠 LLM 生成/判定。mock 或未配 key 时算出的不是"分数"而是占位输出，
    # 因此**不伪造分数**，直接说明原因并把它连带跳过的指标列清楚。
    if settings.llm_provider == "mock":
        reason = (
            "LLM_PROVIDER=mock：RAGAS 指标依赖 LLM 当裁判，mock 下无法产出可信分数，已跳过评测。"
            '如需真实评测，请配置 OPENAI_API_KEY（或改用真实 provider）后重跑。'
        )
        return _with_manual(
            _unavailable(reason, generation_config=config, skipped_metrics=requested), manual
        )
    if not settings.llm_configured:
        reason = (
            "未配置可用的 LLM（缺少 OPENAI_API_KEY / OPENAI_BASE_URL）："
            "RAGAS 无法在无裁判模型的情况下计算，已跳过评测。"
        )
        return _with_manual(
            _unavailable(reason, generation_config=config, skipped_metrics=requested), manual
        )

    # ---- 组装样本（不存在的 run 跳过）----
    all_run_ids = await _resolve_run_ids(run_ids)
    samples: list[RagasSample] = []
    for rid in all_run_ids:
        try:
            samples.append(await build_ragas_dataset(rid, golden_path=golden_path))
        except RunNotFoundError:
            continue

    if not samples:
        requested_ids = ", ".join(all_run_ids) if all_run_ids else "（未指定）"
        return _with_manual(
            _unavailable(
                f"没有可评测的 run（run_id={requested_ids} 不存在或无可评测内容）",
                generation_config=config,
                run_ids=all_run_ids,
            ),
            manual,
        )

    n_with_gt = sum(1 for s in samples if s.has_ground_truth)

    # ---- 选择指标 ----
    all_have_gt = n_with_gt == len(samples)
    pairs, skipped = _select_metric_pairs(
        api,
        requested,
        all_have_ground_truth=all_have_gt,
        embeddings_usable=_embeddings_usable(settings),
    )

    # ---- 真正调用 RAGAS ----
    try:
        scores = _evaluate_samples(api, samples, pairs, settings)
    except Exception as exc:  # noqa: BLE001 - 运行期任何失败都降级，绝不半真半假地返回
        report = _unavailable(
            f"RAGAS 评测执行失败：{type(exc).__name__}: {exc}",
            generation_config=config,
            run_ids=[s.run_id for s in samples],
            n_samples=len(samples),
            n_with_ground_truth=n_with_gt,
        )
        report.skipped_metrics = list(requested)
        if with_citation_accuracy:
            report.citation_accuracy = await _aggregate_citation_accuracy(samples)
        report.manual_accuracy = manual
        return report

    # 只保留**真实算出来**的指标；某个指标若整批都没算出值，就归入 skipped 而不是填 0。
    metric_values: dict[str, float] = {}
    for name, metric_obj in pairs:
        column = getattr(metric_obj, "name", name)
        values = [
            float(row[column])
            for row in scores
            if isinstance(row.get(column), (int, float)) and row.get(column) is not None
        ]
        if values:
            metric_values[name] = fmean(values)
        else:
            skipped.append(name)

    report = EvalReport(
        available=True,
        reason=None,
        skipped_metrics=skipped,
        metrics=metric_values,
        per_sample=scores,
        generation_config=config,
        run_ids=[s.run_id for s in samples],
        n_samples=len(samples),
        n_with_ground_truth=n_with_gt,
    )
    if with_citation_accuracy:
        report.citation_accuracy = await _aggregate_citation_accuracy(samples)
    report.manual_accuracy = manual
    return report


def _manual_accuracy_if_present(csv_path: Path | str | None) -> dict[str, Any] | None:
    """抽检文件存在且可解析时返回人工口径统计，否则返回 None。

    为什么"没有文件"不是错误而是 None：人工抽检是周期性动作，
    评测报告随时可能被重跑，没抽过检也得有自动口径 —— 但**绝不能因此造一个人工数**。
    """
    try:
        stats = manual_accuracy(csv_path)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - CSV 坏了不该拖垮整份报告，跳过人工口径即可
        return None
    return stats if (stats["total_judged"] or stats["pending"]) else None


def _with_manual(report: EvalReport, manual: dict[str, Any] | None) -> EvalReport:
    report.manual_accuracy = manual
    return report


def _evaluate_samples(
    api: _RagasAPI,
    samples: list[RagasSample],
    pairs: list[tuple[str, Any]],
    settings: Settings,
) -> list[dict[str, Any]]:
    """把样本交给 RAGAS 求值，返回逐样本的分数行。

    单独抽出这一层是为了让"真正产生分数"的代码**只有这一处** ——
    `run_ragas_eval` 里除此之外没有别的地方会写 `metrics`，
    这样"绝不返回假分数"这条不变量在代码结构上就是显然的。
    """
    from app.core.llm import get_llm
    from app.retrieval.embedder import get_embedder

    # 用我们自己的 provider 当裁判：LangChain ChatModel → ragas 包装器
    llm = api.LangchainLLMWrapper(get_llm(temperature=0.0))
    embeddings = api.LangchainEmbeddingsWrapper(_EmbedderAdapter(get_embedder(settings)))

    ragas_samples = [
        api.SingleTurnSample(
            user_input=s.question,
            response=s.answer,
            retrieved_contexts=list(s.contexts),
            # ground_truth 为空时**不要**传 None 给 reference：给空串让 RAGAS 见到"这一列存在但为空"，
            # 从而在我们已按需跳过相关指标的前提下，不会因为缺列而整批失败。
            reference=s.ground_truth or "",
        )
        for s in samples
    ]
    dataset = api.EvaluationDataset(samples=ragas_samples)

    result = api.evaluate(
        dataset,
        metrics=[obj for _name, obj in pairs],
        llm=llm,
        embeddings=embeddings,
        # raise_exceptions=False：单条样本失败不应让整批判定崩掉；
        # 失败的指标会以 NaN 出现，随后被上面的"只保留真实值"逻辑剔除。
        raise_exceptions=False,
        show_progress=False,
    )
    return list(result.scores)


async def _aggregate_citation_accuracy(samples: list[RagasSample]) -> dict[str, Any]:
    """把所有样本的引用合并统计，得到整批的自动口径引用准确率。"""
    total = 0
    verified = 0
    for sample in samples:
        total += len(sample.citations)
        verified += sum(1 for c in sample.citations if c.get("verified") is True)
    return {
        "total": total,
        "verified": verified,
        "accuracy": (verified / total) if total else 0.0,
        "method": "auto",
    }


__all__ = [
    "DEFAULT_SPOTCHECK_CSV",
    "EMBEDDING_METRICS",
    "METRIC_NAMES",
    "REFERENCE_METRICS",
    "EvalReport",
    "RagasSample",
    "RunNotFoundError",
    "build_ragas_dataset",
    "citation_accuracy",
    "generation_config",
    "manual_accuracy",
    "run_ragas_eval",
]

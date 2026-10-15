"""RAGAS 评测运行器的测试（临时库隔离，无需模型权重 / API key）。

**测试目标不是"跑出好看的分数"**（没有 key 根本跑不出分数），而是守住三条承诺：

1. **自动口径引用准确率算术正确** —— 它是无 key 环境下唯一能兑现的指标，
   算错比没有更糟（会被当成"引用准确率达标"的证据）。
2. **取数正确** —— `question / answer / contexts / ground_truth` 真的从 DB 组装对了。
3. **★ 降级绝不返回假分数** —— 这是本任务的重点。三种降级（未装 ragas / mock 或无 key /
   run 不存在）都必须：返回结构化结果、说明原因、且 `metrics` 与 `per_sample` **全空**。
   其中"mock 下不得有任何分数"单列一条断言，作为**防假分数回归**的守门测试。

隔离方式沿用 `test_memory.py` 的做法：临时 SQLite 库 + 替换 `app.db.session.session_scope`，
不碰真实的 `data/sqlite/paperpilot.db`。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 导入以把 12 张表注册进 metadata
from app.core.config import get_settings
from app.db import session as db_session
from app.db.base import Base
from app.db.models import Citation, Message, Paper, PaperChunk, ResearchRun, ResearchSession
from app.eval.ragas_runner import (
    METRIC_NAMES,
    EvalReport,
    RunNotFoundError,
    build_ragas_dataset,
    citation_accuracy,
    generation_config,
    run_ragas_eval,
)


# --------------------------------------------------------------------------- 夹具
@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """每个用例前后清掉 `Settings` 的 lru_cache。

    测试里会用 `monkeypatch.setenv` 改 LLM/embedding provider；若不清缓存，
    单例会把上一个用例的配置带进来，导致"我明明设了 mock 却走了真实分支"这类
    极难排查的假绿。
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
async def scope(monkeypatch, tmp_path) -> AsyncIterator:
    """临时库的事务作用域，并把 `app.db.session.session_scope` 替换成它。

    直接打在模块属性上，是因为 `ragas_runner` 以 `db_session.session_scope()` 的形式调用，
    属性查找发生在调用时，替换后即生效。
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'ragas.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    monkeypatch.setattr(db_session, "session_scope", _scope)
    yield _scope
    await engine.dispose()


async def _seed_run(
    scope,
    *,
    run_id: str = "r1",
    question: str = "测试问题？",
    answer: str = "这是报告正文。",
    chunks: list[tuple[str, str]] | None = None,
    citations: list[dict] | None = None,
) -> dict:
    """造一条完整的 run：session + paper + chunks + run + assistant 消息 + citations。

    返回落库后拿到的 chunk_id，便于断言 contexts 取的是**chunk 正文**而非引用片段。
    """
    chunks = chunks if chunks is not None else [("2401.00001", "这是被检索到的 chunk 正文。")]
    citations = citations or []

    chunk_ids: dict[str, int] = {}
    async with scope() as session:
        session.add(ResearchSession(id=f"s-{run_id}", title="t"))
        # PaperChunk.arxiv_id 有外键，先建 Paper
        for arxiv_id, _text in chunks:
            if await session.get(Paper, arxiv_id) is None:
                session.add(Paper(arxiv_id=arxiv_id, title=f"论文 {arxiv_id}"))
        await session.flush()
        for arxiv_id, text in chunks:
            chunk = PaperChunk(arxiv_id=arxiv_id, chunk_index=0, section="abstract", text=text)
            session.add(chunk)
            await session.flush()
            chunk_ids[text] = chunk.id

        session.add(
            ResearchRun(
                id=run_id,
                session_id=f"s-{run_id}",
                thread_id=run_id,
                question=question,
                status="done",
                langgraph_state={},
            )
        )
        # draft 由 finalize 以 assistant 消息落进 messages 表
        session.add(
            Message(session_id=f"s-{run_id}", run_id=run_id, role="assistant", content=answer, node_name="writer")
        )
        for cit in citations:
            session.add(
                Citation(
                    run_id=run_id,
                    claim=cit.get("claim", "论断"),
                    chunk_id=cit.get("chunk_id"),
                    arxiv_id=cit.get("arxiv_id", "2401.00001"),
                    quote=cit.get("quote", ""),
                    verified=cit.get("verified"),
                )
            )
    return chunk_ids


def _write_golden(tmp_path, items: list[dict]):
    """写一份临时 golden 集（JSONL）。"""
    path = tmp_path / "golden.jsonl"
    path.write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items), encoding="utf-8"
    )
    return path


def _fake_ragas_api():
    """一个只提供"指标类 + name"的最小 ragas 替身，用于验证**聚合与列名映射**逻辑。

    真实的 `evaluate()` 需要联网调用裁判模型，单测里跑不了；但"把指标对象映射成报告列名、
    再对逐样本分数求均值"这段纯逻辑是本模块自己的代码，必须被覆盖。
    """

    def make(name: str):
        class _Metric:
            def __init__(self) -> None:
                self.name = name

        return _Metric

    return SimpleNamespace(
        metrics=SimpleNamespace(
            Faithfulness=make("faithfulness"),
            AnswerRelevancy=make("answer_relevancy"),
            ContextRecall=make("context_recall"),
            AnswerCorrectness=make("answer_correctness"),
            LLMContextPrecisionWithReference=make("llm_context_precision_with_reference"),
            LLMContextPrecisionWithoutReference=make("llm_context_precision_without_reference"),
        )
    )


# =========================================================================== 1. 引用准确率
async def test_citation_accuracy_arithmetic(scope):
    """`verified=true` 的占比要算对：4 真 / 1 假 / 1 未校验 → 4/6，且未校验不计入分子。"""
    await _seed_run(
        scope,
        citations=[
            {"claim": "c1", "verified": True},
            {"claim": "c2", "verified": True},
            {"claim": "c3", "verified": True},
            {"claim": "c4", "verified": True},
            {"claim": "c5", "verified": False},
            {"claim": "c6", "verified": None},  # 未校验：既不算真也不算假
        ],
    )

    result = await citation_accuracy("r1")

    assert result["total"] == 6
    assert result["verified"] == 4
    assert result["accuracy"] == pytest.approx(4 / 6)
    assert result["method"] == "auto"


async def test_citation_accuracy_no_citations_is_zero_not_crash(scope):
    """没有引用时返回 0.0 而不是抛 ZeroDivisionError —— 调用方不该为"空"付出异常代价。"""
    await _seed_run(scope)
    result = await citation_accuracy("r1")
    assert result == {"total": 0, "verified": 0, "accuracy": 0.0, "method": "auto"}


# =========================================================================== 2. 数据集组装
async def test_build_ragas_dataset_assembles_from_db(scope, tmp_path):
    """从 DB 组装 question / answer(draft) / contexts（contexts 必须取 **chunk 正文**）。"""
    chunk_text = "重排带来额外的推理延迟与算力开销。"
    chunk_ids = await _seed_run(
        scope,
        question="RAG 重排的代价是什么？",
        answer="重排的代价主要是额外延迟。",
        chunks=[("2401.00001", chunk_text)],
        citations=[
            {"claim": "重排有延迟代价", "chunk_id": None, "quote": "无关的引用片段", "verified": True},
        ],
    )
    # 让引用的 chunk_id 真正指向 chunk，验证 contexts 取的是 **chunk 正文**而非引用片段。
    # scope() 退出时会 commit，所以这次修改是持久的。
    async with scope() as session:
        from sqlalchemy import select

        cit = (
            await session.execute(select(Citation).where(Citation.run_id == "r1"))
        ).scalars().one()
        cit.chunk_id = chunk_ids[chunk_text]

    sample = await build_ragas_dataset("r1", golden_path=tmp_path / "missing.jsonl")

    assert sample.question == "RAG 重排的代价是什么？"
    assert sample.answer == "重排的代价主要是额外延迟。"
    assert sample.contexts == ["重排带来额外的推理延迟与算力开销。"]  # 正文，而不是「无关的引用片段」
    assert sample.run_id == "r1"


async def test_build_ragas_dataset_ground_truth_matches_golden(scope, tmp_path):
    """问题能对上 golden 条目时，ground_truth 取该条目的 reference_answer。"""
    await _seed_run(scope, question="SciTLDR 用于什么任务？")
    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q003",
                "question": "SciTLDR 用于什么任务？",
                "gold_paper_ids": ["2411.18583"],
                "reference_answer": "用于评测自动文献综述生成。",
            }
        ],
    )

    sample = await build_ragas_dataset("r1", golden_path=golden)

    assert sample.golden_id == "q003"
    assert sample.ground_truth == "用于评测自动文献综述生成。"
    assert sample.has_ground_truth is True


async def test_build_ragas_dataset_missing_golden_leaves_ground_truth_empty(scope, tmp_path):
    """没有对应 golden 条目时 ground_truth 留空且**不报错**（此时应跳过 reference 类指标）。"""
    await _seed_run(scope, question="一个 golden 里没有的问题？")
    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q999",
                "question": "完全不同的另一个问题？",
                "gold_paper_ids": ["2411.18583"],
                "reference_answer": "参考答案。",
            }
        ],
    )

    sample = await build_ragas_dataset("r1", golden_path=golden)

    assert sample.ground_truth is None
    assert sample.has_ground_truth is False
    assert sample.golden_id is None


async def test_build_ragas_dataset_run_not_found_raises(scope):
    """run 不存在时抛**专用异常**（不是 KeyError），便于上层精确跳过。"""
    with pytest.raises(RunNotFoundError):
        await build_ragas_dataset("nope")


async def test_build_ragas_dataset_prefers_live_state(scope, tmp_path):
    """传入活的最终 state 时，用它的 draft 与全量 retrieved（比 DB 只能拿到被引 chunk 更完整）。"""
    state = {
        "question": "问题？",
        "draft": "来自 state 的报告。",
        "retrieved": [
            {"arxiv_id": "a1", "chunks": [{"chunk_id": 1, "text": "检索片段一"}, {"chunk_id": 2, "text": "检索片段二"}]}
        ],
        "citations": [{"claim": "x", "verified": True}],
    }
    sample = await build_ragas_dataset("any-run", state=state, golden_path=tmp_path / "missing.jsonl")

    assert sample.question == "问题？"
    assert sample.answer == "来自 state 的报告。"
    assert sample.contexts == ["检索片段一", "检索片段二"]


# =========================================================================== 3. 降级路径
def _pretend_ragas_installed(monkeypatch) -> None:
    """放行"没装 ragas"这道闸门，好测它**之后**的降级分支。

    下面三条测的是"裁判模型不可用 / 库里没有 run"，与 ragas 本身无关。
    若不加这层打桩，只有装了 `[eval]` 可选依赖的机器才能测到它们 ——
    而按 README 只装 `pip install -e ".[dev]"` 的人（包括 CI）会得到
    `reason='ragas 未安装'`，断言"提到 mock/LLM"当场红。
    "闸门自己放行"这一分支由 `test_..._ragas_missing` 单独守，判据不重叠。
    """
    from app.eval import ragas_runner

    # 返回值在这三条路径里根本不会被用到（都提前 return 了），给个空壳即可。
    monkeypatch.setattr(ragas_runner, "_load_ragas", lambda: SimpleNamespace())


async def test_run_ragas_eval_missing_run_returns_empty_report(scope, monkeypatch):
    """DB 里没有对应 run → 返回空报告（available=False + 说明），而不是 KeyError。"""
    _pretend_ragas_installed(monkeypatch)
    # 先把裁判模型配成"可用"，确保走到的是"没有 run"这一分支而不是"没配 key"分支
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    get_settings.cache_clear()

    report = await run_ragas_eval(run_ids=["does-not-exist"])

    assert report.available is False
    assert report.reason is not None and "没有可评测的 run" in report.reason
    assert report.metrics == {}
    assert report.per_sample == []


async def test_run_ragas_eval_mock_returns_unavailable(monkeypatch):
    """LLM_PROVIDER=mock → available=False、有 reason、列出被跳过的指标。"""
    _pretend_ragas_installed(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    report = await run_ragas_eval()

    assert report.available is False
    assert report.reason is not None and "mock" in report.reason
    assert report.skipped_metrics == list(METRIC_NAMES)


async def test_run_ragas_eval_unconfigured_key_returns_unavailable(monkeypatch):
    """provider=openai 但没配 key → 同样降级（不能拿"连不上"当"算出分"）。"""
    _pretend_ragas_installed(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    get_settings.cache_clear()

    report = await run_ragas_eval()

    assert report.available is False
    assert report.reason is not None and "LLM" in report.reason
    assert report.metrics == {}


async def test_run_ragas_eval_degrades_when_ragas_missing(monkeypatch):
    """monkeypatch 模拟"环境里没有 ragas"：必须降级而不是 ImportError 冒出去。"""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    from app.eval import ragas_runner

    def _boom():
        raise ImportError("No module named 'ragas'")

    monkeypatch.setattr(ragas_runner, "_load_ragas", _boom)

    report = await run_ragas_eval()

    assert report.available is False
    assert report.reason is not None and "ragas 未安装" in report.reason
    assert report.metrics == {}


# =========================================================================== 4. ★ 防假分数回归
async def test_mock_mode_never_produces_any_score(monkeypatch):
    """★ 守门测试：mock 降级路径下**不得出现任何分数**（含 metrics / per_sample）。

    这条断言存在的唯一目的，就是防止有人为了"让流程看起来跑通"而在降级分支里填一个
    0.0 或占位分数。假分数会进入对比表、被当成结论引用 —— 比"没有分数"危害大得多。
    """
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()

    report = await run_ragas_eval()

    assert report.available is False
    assert report.metrics == {}
    assert report.per_sample == []
    # 序列化视图里也必须是空的分数表 —— 调用方无论从哪条路径取数都拿不到假分数。
    assert report.as_dict()["metrics"] == {}
    assert report.as_dict()["per_sample"] == []


def test_eval_report_invariant_rejects_scores_when_unavailable():
    """结构性不变量：available=False 却带分数 → 直接抛错。

    把"绝不返回假分数"从口头约定变成构造期校验，新增降级分支时也无法绕过。
    """
    with pytest.raises(ValueError):
        EvalReport(available=False, reason="x", metrics={"faithfulness": 1.0})
    with pytest.raises(ValueError):
        EvalReport(available=False, reason="x", per_sample=[{"faithfulness": 1.0}])
    # 正常情况不应抛错
    EvalReport(available=False, reason="x")
    EvalReport(available=True, metrics={"faithfulness": 1.0})


# =========================================================================== 5. 真实聚合逻辑
async def test_run_ragas_eval_aggregates_real_scores_when_available(scope, monkeypatch, tmp_path):
    """裁判可用时，逐样本分数要按列名正确聚合成均值，并带上生成配置。

    用替身替掉 ragas 接入与 `_evaluate_samples`（真实评估需联网），
    专门验证"指标对象 → 报告列名 → 均值"这段本模块自己的逻辑。
    """
    await _seed_run(
        scope,
        question="SciTLDR 用于什么任务？",
        citations=[{"claim": "c", "verified": True}, {"claim": "c2", "verified": False}],
    )
    golden = _write_golden(
        tmp_path,
        [
            {
                "id": "q003",
                "question": "SciTLDR 用于什么任务？",
                "gold_paper_ids": ["2411.18583"],
                "reference_answer": "用于评测自动文献综述生成。",
            }
        ],
    )

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")  # 让 embedding 类指标不被跳过
    get_settings.cache_clear()

    from app.eval import ragas_runner

    monkeypatch.setattr(ragas_runner, "_load_ragas", _fake_ragas_api)
    monkeypatch.setattr(
        ragas_runner,
        "_evaluate_samples",
        lambda api, samples, pairs, settings: [
            {
                "faithfulness": 1.0,
                "answer_relevancy": 0.8,
                "llm_context_precision_with_reference": 0.5,
                "context_recall": 0.6,
                "answer_correctness": 0.7,
            }
        ],
    )

    report = await run_ragas_eval(run_ids=["r1"], golden_path=golden)

    assert report.available is True
    assert report.metrics == {
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
        "context_precision": 0.5,  # ★ 列名 llm_context_precision_with_reference 被映射回架构口径名
        "context_recall": 0.6,
        "answer_correctness": 0.7,
    }
    assert report.skipped_metrics == []
    assert report.n_samples == 1
    assert report.n_with_ground_truth == 1
    # 生成配置必须进报告（架构 §12 的可比性前提）
    assert report.generation_config["llm_model"]
    assert "temperature" in report.generation_config
    assert report.generation_config["prompt_version"]
    # 引用准确率：1 真 / 1 假 → 0.5
    assert report.citation_accuracy == {"total": 2, "verified": 1, "accuracy": 0.5, "method": "auto"}


async def test_reference_metrics_skipped_without_ground_truth(scope, monkeypatch, tmp_path):
    """没有 ground_truth 时跳过 reference 类指标，其余照算（且绝不把缺失算成 0 分）。"""
    await _seed_run(scope, question="golden 里没有的问题？")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "api")
    get_settings.cache_clear()

    from app.eval import ragas_runner

    monkeypatch.setattr(ragas_runner, "_load_ragas", _fake_ragas_api)
    monkeypatch.setattr(
        ragas_runner,
        "_evaluate_samples",
        lambda api, samples, pairs, settings: [
            {"faithfulness": 0.9, "answer_relevancy": 0.4, "llm_context_precision_without_reference": 0.3}
        ],
    )

    report = await run_ragas_eval(run_ids=["r1"], golden_path=tmp_path / "missing.jsonl")

    assert report.available is True
    assert "context_recall" in report.skipped_metrics
    assert "answer_correctness" in report.skipped_metrics
    assert "context_recall" not in report.metrics
    assert "answer_correctness" not in report.metrics
    # 无参考版 context_precision 的列名被映射回 context_precision
    assert report.metrics["context_precision"] == 0.3


def test_generation_config_records_comparability_fields():
    """生成配置必须含 model / temperature / prompt_version —— 否则分数不可比（架构 §12）。"""
    cfg = generation_config()
    assert set(cfg) >= {"llm_provider", "llm_model", "temperature", "prompt_version", "ragas_version"}
    # prompt_version 是内容哈希（12 位十六进制）；取不到模板时为 unknown
    assert cfg["prompt_version"] == "unknown" or len(cfg["prompt_version"]) == 12


# =========================================================================== 6. ★ 批量自动口径的分母/分子
async def test_batched_citation_accuracy_excludes_unverified_from_numerator(scope, monkeypatch, tmp_path):
    """整批评测面板上的引用准确率：`verified=None`（还没校验）**不计入分子**。

    单条 run 的 `citation_accuracy()` 已经有用例，但 `run_ragas_eval` 走的是另一段实现
    （`_aggregate_citation_accuracy`，吃的是样本里带出的 citations）。两处口径必须一致，
    否则同一个数字在 CLI 与面板上会不一样。

    写成 1 真 / 1 假 / 1 未校验：正确答案是 1/3。若哪天有人改成
    `if c.get("verified") is not False`（把"没校验"当成"通过"），分子会变成 2、
    面板直接虚高到 2/3 —— 而"引用准确率 ≥ 90%"这条验收就再也没人守得住了。
    """
    await _seed_run(
        scope,
        question="SciTLDR 用于什么任务？",
        citations=[
            {"claim": "c-true", "verified": True},
            {"claim": "c-false", "verified": False},
            {"claim": "c-pending", "verified": None},  # 尚未校验：不进分子
        ],
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()

    from app.eval import ragas_runner

    monkeypatch.setattr(ragas_runner, "_load_ragas", _fake_ragas_api)
    monkeypatch.setattr(
        ragas_runner,
        "_evaluate_samples",
        lambda api, samples, pairs, settings: [{"faithfulness": 1.0}],
    )

    report = await run_ragas_eval(run_ids=["r1"], golden_path=tmp_path / "missing.jsonl")

    assert report.citation_accuracy == {
        "total": 3,
        "verified": 1,
        "accuracy": pytest.approx(1 / 3),
        "method": "auto",
    }

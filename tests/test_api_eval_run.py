"""`POST /api/v1/eval/run` + `GET /api/v1/eval/runs/{id}` 的对外契约（架构 §7 最后一条待实现接口）。

全部离线：不联网、不下模型、不调任何 LLM。打桩的纪律是**桩的返回结构照抄真实 schema**：

- 检索数据用 `tests/conftest.make_corpus()` 造的**真实 `ChunkRecord`**，索引真建在
  `api_client` 的临时 FAISS 目录里（走真实 `build_index`，mock embedder 不下载权重）；
- golden 用**真实 `GoldenItem`**（字段一个不落），只是把 40 条裁成 2 条，
  免得断言里塞 240 行明细；
- RAGAS 沿用 `app/eval/ragas_runner` 里那两个既有 monkeypatch 点（`_load_ragas` /
  `_evaluate_samples`），返回**真实的 `EvalReport` 字段名与 ragas 的指标列名**
  （`llm_context_precision_with_reference` → `context_precision` 的映射是 runner 自己的
  逻辑，必须被真实覆盖到，而不是绕开）。

覆盖：三种模式各自的落库形状、后台执行与轮询、409 并发拒绝与释放、501 缺依赖、
404/422 闸门、崩溃后闸门不卡死，以及"结果真的进了 `eval_runs` / `eval_samples`"。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select

import app.db.models  # noqa: F401 - 注册表到 metadata
import app.eval.service as eval_service
import app.eval.success_metrics as success_metrics
from app.core.config import get_settings
from app.db.models import (
    Citation,
    EvalRun,
    EvalSample,
    Message,
    Paper,
    PaperChunk,
    ResearchRun,
    ResearchSession,
)
from app.db.session import session_scope
from app.eval import ragas_runner
from app.eval.dataset import GoldenItem, load_golden
from app.retrieval.hybrid_retriever import HybridRetriever
from tests.api_support import api_client, assert_no_secret_values
from tests.conftest import FOCUSED_KEYWORDS, FOCUSED_QUERY, PAPER_A, PAPER_C, make_corpus

# 与 test_ragas_runner 同一口径的"裁判可用"环境：provider=openai + key + base_url
# （`Settings.llm_configured` 要求两者都非空）。真正的 LLM 调用被桩替掉，一个包都不发。
JUDGE_ENV = {
    "LLM_PROVIDER": "openai",
    "OPENAI_API_KEY": "test-key",
    "OPENAI_BASE_URL": "https://example.invalid/v1",
    # mock embedding 是无语义哈希向量，runner 会据此跳过 answer_relevancy / answer_correctness。
    # 设成 api 只是**声明**"这一项该算" —— 取向量的那一步在被桩掉的 `_evaluate_samples` 里，
    # 所以既不会下载权重，也不会打网络。
    "EMBEDDING_PROVIDER": "api",
}

RUN_URL = "/api/v1/eval/run"
POLL_TMPL = "/api/v1/eval/runs/{}"


# --------------------------------------------------------------------------- 夹具
@pytest.fixture
async def client(monkeypatch, tmp_path) -> AsyncIterator:
    eval_service.clear_active_run()
    async with api_client(monkeypatch, tmp_path) as c:
        yield c
    eval_service.clear_active_run()


@pytest.fixture
async def indexed_client(monkeypatch, tmp_path) -> AsyncIterator:
    """离线建好索引的客户端：消融模式"得有东西可检索"这个前提在这里兑现。"""
    eval_service.clear_active_run()
    async with api_client(monkeypatch, tmp_path) as c:
        records, titles = make_corpus()
        retriever = eval_service.build_retriever()  # settings 已指向 tmp_path 下的 FAISS 目录
        await retriever.build_index(records, titles)
        assert retriever.size == len(records)
        yield c
    eval_service.clear_active_run()


@pytest.fixture
async def judge_client(monkeypatch, tmp_path) -> AsyncIterator:
    """RAGAS 模式：裁判配好（但求值全桩掉，真实求值要联网调模型）。"""
    eval_service.clear_active_run()
    async with api_client(monkeypatch, tmp_path, **JUDGE_ENV) as c:
        yield c
    eval_service.clear_active_run()


# --------------------------------------------------------------------------- 小工具
async def _wait_terminal(client, eval_run_id: int, *, timeout: float = 20.0) -> dict:
    """轮询到终态。后台任务是 `asyncio.create_task` 起的，必须等而不能假设同步完成。"""
    deadline = time.monotonic() + timeout
    last: dict = {}
    while time.monotonic() < deadline:
        resp = await client.get(POLL_TMPL.format(eval_run_id))
        assert resp.status_code == 200, resp.text
        last = resp.json()
        if last["finished"]:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"评测 {eval_run_id} 没在 {timeout}s 内进入终态；最后状态：{last}")


def _golden_items() -> list[GoldenItem]:
    """两条真实形状的 golden（`GoldenItem` 的字段一个不落），够验证明细行数与算术。"""
    return [
        GoldenItem(
            id="t1",
            question=FOCUSED_QUERY,
            keywords=tuple(FOCUSED_KEYWORDS),
            gold_paper_ids=(PAPER_A,),
            categories=("cs.CL",),
            reference_answer="hybrid retrieval with reranker fusion",
        ),
        # 第二条跨到另一个领域：分类约束不同，用来验证配置之间确实有差异
        GoldenItem(
            id="t2",
            question="vision transformer image classification",
            keywords=("vision", "transformer", "classification"),
            gold_paper_ids=(PAPER_C,),
            categories=("cs.CV",),
            reference_answer="vision transformer for image classification",
        ),
    ]


def _fake_ragas_api():
    """ragas 符号集合的替身：只提供指标类与 `.name`（列名沿用 ragas 的真实写法）。"""

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


def _ragas_missing() -> None:
    """替身：模拟"环境里没有 ragas 包"（`_load_ragas` 就是真正 import 它的那一行）。"""
    raise ImportError("No module named 'ragas'")


async def _seed_run(question: str, *, chunk_text: str) -> str:
    """造一条可被 RAGAS 取数的 run：session + paper + chunk + run + assistant 消息 + citation。"""
    run_id = "eval-ragas-run"
    async with session_scope() as db:
        db.add(ResearchSession(id="s-eval-ragas", title="eval"))
        db.add(Paper(arxiv_id=PAPER_A, title="Paper A"))
        chunk = PaperChunk(
            arxiv_id=PAPER_A, chunk_index=0, section="abstract", source="abstract", text=chunk_text
        )
        db.add(chunk)
        await db.flush()
        db.add(
            ResearchRun(
                id=run_id,
                session_id="s-eval-ragas",
                thread_id=run_id,
                question=question,
                status="done",
                langgraph_state={},
            )
        )
        db.add(
            Message(
                session_id="s-eval-ragas",
                run_id=run_id,
                role="assistant",
                content="这是成稿正文，带一条引用。",
                node_name="writer",
            )
        )
        db.add(
            Citation(
                run_id=run_id,
                claim="引用支撑的论断",
                chunk_id=chunk.id,
                arxiv_id=PAPER_A,
                quote="hybrid retrieval",
                verified=True,
            )
        )
    return run_id


# =========================================================================== 模式一：检索消融
async def test_ablation_mode_runs_in_background_and_persists(indexed_client, monkeypatch):
    """202 立刻返回 id → 后台跑完 → 状态与逐配置指标进 `eval_runs.metrics`、明细进 `eval_samples`。"""
    monkeypatch.setattr(eval_service, "load_golden", lambda path=None: _golden_items())

    resp = await indexed_client.post(RUN_URL, json={"mode": "ablation", "name": "api-smoke"})
    assert resp.status_code == 202, resp.text
    started = resp.json()
    assert started["status"] == "running"
    assert started["mode"] == "ablation"
    assert started["poll"] == POLL_TMPL.format(started["eval_run_id"])
    assert isinstance(started["eval_run_id"], int)

    done = await _wait_terminal(indexed_client, started["eval_run_id"])
    assert done["status"] == "completed", done
    assert done["is_running"] is False
    assert done["reason"] is None and done["error"] is None
    assert done["duration_s"] >= 0

    # summary：真实跑出来的规模
    summary = done["summary"]
    assert summary["n_queries"] == 2 and summary["n_configs"] == 6
    assert summary["index_size"] == 10 and summary["n_papers"] == 3
    assert summary["embedding_provider"] == "mock"
    # mock embedding 必须被如实标成降级 —— 否则读的人会把随机数当成检索质量
    assert summary["degraded"] is True
    assert any("mock" in line for line in summary["provider_notice"])

    # 六种配置的汇总行（列名沿用 `RetrievalMetrics` 的字段名）
    rows = {c["key"]: c for c in done["result"]["configs"]}
    assert set(rows) == {"vector", "bm25", "weighted", "rrf", "rerank", "filter"}
    assert rows["bm25"]["n_queries"] == 2 and rows["bm25"]["n_skipped"] == 0
    assert rows["bm25"]["recall_at_5"] >= 0.5
    for row in rows.values():
        assert {"median_latency_ms", "mrr", "ndcg_at_10", "precision_at_5"} <= set(row)

    # 明细：2 条查询 × 6 种配置 = 12 行
    assert done["n_samples"] == 12
    assert not done["samples_truncated"]
    bm25_first = [s for s in done["samples"] if s["scores"]["config_key"] == "bm25" and s["question"] == FOCUSED_QUERY]
    assert len(bm25_first) == 1
    # 受控语料里 ② 纯 BM25 对 FOCUSED_QUERY 完全捞回 PAPER_A（与 test_ablation 同一被测事实）
    assert bm25_first[0]["scores"]["recall@5"] == 1.0
    assert bm25_first[0]["contexts"][0] == PAPER_A
    assert bm25_first[0]["scores"]["latency_ms"] >= 0

    # 不只看接口回显：直接查库，确认落的是同一份数据
    async with session_scope() as db:
        run = await db.get(EvalRun, started["eval_run_id"])
        sample_rows = (
            await db.execute(select(EvalSample).where(EvalSample.eval_run_id == run.id))
        ).scalars().all()
    assert len(sample_rows) == 12
    assert run.name == "api-smoke" and run.dataset == "golden_40"
    assert run.metrics["status"] == "completed"
    assert run.config["mode"] == "ablation"
    assert run.config["params"]["configs"] is None  # 没挑配置 → 六种全跑


async def test_ablation_mode_configs_filter_is_applied(indexed_client, monkeypatch):
    """`configs` 不是摆设：选两种就只跑两种，`top_k` 也真的进 summary。"""
    monkeypatch.setattr(eval_service, "load_golden", lambda path=None: _golden_items())

    resp = await indexed_client.post(
        RUN_URL, json={"mode": "ablation", "configs": "②④", "top_k": 3}
    )
    assert resp.status_code == 202, resp.text
    done = await _wait_terminal(indexed_client, resp.json()["eval_run_id"])

    assert [c["key"] for c in done["result"]["configs"]] == ["bm25", "rrf"]
    assert done["summary"]["n_configs"] == 2
    assert done["summary"]["top_k"] == 3
    assert done["n_samples"] == 4  # 2 查询 × 2 配置
    async with session_scope() as db:
        run = await db.get(EvalRun, resp.json()["eval_run_id"])
        assert run.config["params"]["configs"] == ["bm25", "rrf"]


async def test_ablation_mode_uses_real_default_golden(indexed_client):
    """不打桩的一遍：真实 `load_golden(None)` = 仓库里的 golden_40（40 条），明细可分页截断。"""
    resp = await indexed_client.post(RUN_URL, json={"mode": "ablation", "configs": "rrf"})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["eval_run_id"]
    done = await _wait_terminal(indexed_client, run_id)

    assert done["status"] == "completed"
    assert done["summary"]["n_queries"] == 40
    assert done["summary"]["n_configs"] == 1
    assert "默认" in done["summary"]["golden"]
    assert done["n_samples"] == 40  # 40 查询 × 1 配置

    paged = await indexed_client.get(POLL_TMPL.format(run_id), params={"sample_limit": 5})
    body = paged.json()
    assert body["n_samples"] == 40
    assert len(body["samples"]) == 5
    assert body["samples_truncated"] is True


async def test_ablation_mode_empty_index_is_unavailable_not_zero(client, monkeypatch):
    """索引为空 → `unavailable` + 指路建索引，**不落一行明细、不给一张全 0 的表**。"""
    monkeypatch.setattr(eval_service, "load_golden", lambda path=None: _golden_items())
    empty_dir = Path(get_settings().faiss_index_dir).parent / "faiss-empty"
    monkeypatch.setattr(
        eval_service, "build_retriever", lambda settings=None: HybridRetriever(index_dir=empty_dir)
    )

    resp = await client.post(RUN_URL, json={"mode": "ablation"})
    assert resp.status_code == 202, resp.text
    done = await _wait_terminal(client, resp.json()["eval_run_id"])

    assert done["status"] == "unavailable"
    assert "build_index" in done["reason"]
    assert done["result"] == {}
    assert done["n_samples"] == 0 and done["samples"] == []
    # 闸门已释放：unavailable 也算跑完，不该把后面所有请求挡成 409
    assert eval_service.current_run() is None


# =========================================================================== 模式二：§8 成功指标
async def test_metrics_mode_persists_two_outcome_rows(client, monkeypatch):
    """`mode=metrics` 不需要索引也不需要 ragas：两条 §8 指标各落一行 `eval_samples`。"""
    seen: dict = {}
    original = success_metrics.run_and_render

    async def spy(scope=None, *, min_cells: int = 45, min_runs: int = 5):
        seen["min_cells"] = min_cells
        seen["min_runs"] = min_runs
        # 必须调**保存下来的原函数**：直接调 `success_metrics.run_and_render` 就是调自己
        return await original(scope, min_cells=min_cells, min_runs=min_runs)

    monkeypatch.setattr(success_metrics, "run_and_render", spy)

    resp = await client.post(RUN_URL, json={"mode": "metrics"})
    assert resp.status_code == 202, resp.text
    done = await _wait_terminal(client, resp.json()["eval_run_id"])

    assert done["status"] == "completed", done
    assert done["dataset"] == "success_metrics"
    assert (seen["min_cells"], seen["min_runs"]) == (45, 5)

    metrics_rows = done["result"]["metrics"]
    assert [m["key"] for m in metrics_rows] == ["comparison_field_completeness", "plan_availability"]
    for m in metrics_rows:
        # 空库上这两条大概率是"拒绝计算"—— 那是诚实产出，不是失败
        assert m["status"] in {"ok", "no_data", "insufficient_sample"}
        assert m["target"].startswith(">=")
        assert m["value"] is None or 0.0 <= m["value"] <= 1.0

    assert done["n_samples"] == 2
    assert [row["scores"]["key"] for row in done["samples"]] == [
        "comparison_field_completeness",
        "plan_availability",
    ]
    assert all(row["scores"]["status"] for row in done["samples"])
    # display()：比率永远带分子/分母出现，本模式里不存在裸百分比
    assert done["samples"][0]["answer"] and "拒绝计算" in done["samples"][0]["answer"]


# =========================================================================== 模式三：RAGAS
async def test_ragas_mode_persists_scores_and_sample_rows(judge_client, monkeypatch):
    """RAGAS 端到端：分数进 `eval_runs.metrics.result.metrics`，逐样本进 `eval_samples`。"""
    # 用 golden 里真实存在的一条问题当 run 的 question，才能验证 ground_truth 对上了
    golden_question = load_golden(None)[0].question
    chunk_text = "hybrid retrieval with bm25 and a reranker fusion"
    await _seed_run(golden_question, chunk_text=chunk_text)

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

    resp = await judge_client.post(RUN_URL, json={"mode": "ragas", "run_ids": ["eval-ragas-run"]})
    assert resp.status_code == 202, resp.text
    done = await _wait_terminal(judge_client, resp.json()["eval_run_id"])

    assert done["status"] == "completed", done
    assert done["result"]["metrics"] == {
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
        # ragas 的列名被映射回架构口径名（runner 的逻辑，这里必须覆盖到）
        "context_precision": 0.5,
        "context_recall": 0.6,
        "answer_correctness": 0.7,
    }
    assert done["result"]["per_sample_persisted"] is True
    assert done["result"]["skipped_metrics"] == []
    # 生成配置必须随结果落库，否则分数不可比（架构 §12）
    assert done["result"]["generation_config"]["llm_provider"] == "openai"
    assert done["summary"]["n_samples"] == 1 and done["summary"]["n_with_ground_truth"] == 1
    assert done["summary"]["citation_accuracy"] == {
        "total": 1,
        "verified": 1,
        "accuracy": 1.0,
        "method": "auto",
    }
    assert done["dataset"] == "research_runs:1"

    assert done["n_samples"] == 1
    row = done["samples"][0]
    assert row["question"] == golden_question
    assert row["answer"] == "这是成稿正文，带一条引用。"
    assert row["contexts"] == [chunk_text]  # chunk 正文，不是引用片段
    assert row["scores"]["run_id"] == "eval-ragas-run"
    assert row["scores"]["golden_id"] == "q001"
    assert row["scores"]["faithfulness"] == 1.0


async def test_ragas_mode_mock_judge_never_stores_scores(client, monkeypatch):
    """`LLM_PROVIDER=mock` 下 RAGAS 返回 available=False：库里只能有原因，**不能有任何分数**。"""
    monkeypatch.setattr(ragas_runner, "_load_ragas", _fake_ragas_api)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("mock 裁判下不该走到求值 —— 那会产出假分数")

    monkeypatch.setattr(ragas_runner, "_evaluate_samples", _must_not_be_called)

    resp = await client.post(RUN_URL, json={"mode": "ragas"})
    assert resp.status_code == 202, resp.text
    done = await _wait_terminal(client, resp.json()["eval_run_id"])

    assert done["status"] == "unavailable"
    assert "mock" in done["reason"]
    assert done["result"] == {}
    assert done["n_samples"] == 0 and done["samples"] == []
    assert done["summary"]["skipped_metrics"]  # 连带跳过了哪些指标要说清楚
    assert done["summary"]["generation_config"]["llm_provider"] == "mock"


# =========================================================================== 409 单实例闸门
async def test_concurrent_eval_is_rejected_with_409_then_release(client, monkeypatch):
    """第二条评测并发进来 → 409（带正在跑的 id 与轮询地址）；跑完后闸门释放。"""
    hold = asyncio.Event()
    started: list[str] = []

    async def hanging_ablation(**kwargs):
        started.append("called")
        await hold.wait()
        return eval_service.EvalOutcome(ok=True, summary={"n_queries": 0}, detail={"configs": []})

    monkeypatch.setattr(eval_service, "run_ablation_eval", hanging_ablation)

    first = await client.post(RUN_URL, json={"mode": "ablation"})
    assert first.status_code == 202, first.text
    run_id = first.json()["eval_run_id"]
    for _ in range(200):  # 等后台任务真正开跑（不是等它跑完）
        if started:
            break
        await asyncio.sleep(0.01)
    assert started == ["called"]

    second = await client.post(RUN_URL, json={"mode": "metrics"})
    assert second.status_code == 409, second.text
    detail = second.json()["detail"]
    assert detail["code"] == "eval_already_running"
    assert detail["running_eval_run_id"] == run_id
    assert detail["running_mode"] == "ablation"
    assert detail["poll"] == POLL_TMPL.format(run_id)
    assert "排队" in detail["hint"]  # 为什么不做排队，写在响应里而不只是代码注释里

    # 正在跑的那条仍可查询，状态是 running
    poll = await client.get(POLL_TMPL.format(run_id))
    body = poll.json()
    assert body["status"] == "running"
    assert body["is_running"] is True and body["finished"] is False
    assert body["started_at"] and body["finished_at"] is None

    # 拒绝并发**不会**留下多余记录
    async with session_scope() as db:
        assert len((await db.execute(select(EvalRun))).scalars().all()) == 1

    hold.set()
    done = await _wait_terminal(client, run_id)
    assert done["status"] == "completed"
    assert eval_service.current_run() is None

    # 释放之后同样的请求就能受理
    third = await client.post(RUN_URL, json={"mode": "metrics"})
    assert third.status_code == 202, third.text
    await _wait_terminal(client, third.json()["eval_run_id"])


async def test_background_task_crash_releases_gate(client, monkeypatch):
    """后台任务抛异常 → 记录标 failed + 原因，闸门照样释放（否则一次崩溃永久锁死这个接口）。"""

    async def boom(**kwargs):
        raise RuntimeError("索引读坏了")

    monkeypatch.setattr(eval_service, "run_ablation_eval", boom)

    resp = await client.post(RUN_URL, json={"mode": "ablation"})
    assert resp.status_code == 202
    done = await _wait_terminal(client, resp.json()["eval_run_id"])

    assert done["status"] == "failed"
    assert "RuntimeError" in done["error"] and "索引读坏了" in done["error"]
    assert done["result"] == {} and done["samples"] == []
    assert eval_service.current_run() is None


# =========================================================================== 501 缺可选依赖
async def test_ragas_mode_without_ragas_returns_501(client, monkeypatch):
    """没装 ragas 时**在受理之前**就 501（而不是先给 202、很久后回一条 failed）。"""
    monkeypatch.setattr(ragas_runner, "_load_ragas", _ragas_missing)

    resp = await client.post(RUN_URL, json={"mode": "ragas"})
    assert resp.status_code == 501, resp.text
    detail = resp.json()["detail"]
    assert detail["code"] == "eval_extra_missing"
    assert detail["install_command"] == 'pip install -e ".[eval]"'
    assert "ragas" in detail["reason"]
    assert "ablation" in detail["hint"]  # 指一条现在就能走的路

    # 既没建记录、也没占闸门：另外两种模式不受影响
    async with session_scope() as db:
        assert len((await db.execute(select(EvalRun))).scalars().all()) == 0
    assert eval_service.current_run() is None
    ok = await client.post(RUN_URL, json={"mode": "metrics"})
    assert ok.status_code == 202, ok.text
    await _wait_terminal(client, ok.json()["eval_run_id"])


# =========================================================================== 422 / 404 闸门
async def test_request_shape_gates(client, monkeypatch):
    """坏形状一律 422 且**一条评测都不该起**（闸门在最外层，抢锁之前）。"""
    calls: list = []

    async def spy(**kwargs):
        calls.append(kwargs)
        return eval_service.EvalOutcome(ok=True)

    monkeypatch.setattr(eval_service, "run_success_metrics_eval", spy)
    # 让 mode=ragas 走到参数校验（本用例只验形状，不验依赖；依赖缺失另有 501 用例）
    monkeypatch.setattr(eval_service, "ensure_ragas_available", lambda settings=None: None)

    bad_bodies = [
        {"mode": "nope"},  # 不在枚举里
        {"name": "x" * 129},  # 超过 eval_runs.name 的 String(128)
        {"golden_path": str(Path.home() / ".env")},  # 仓库外的绝对路径
        {"golden_path": "README.md"},  # 不是 .jsonl
        {"golden_path": "app/eval/datasets/does_not_exist.jsonl"},  # 文件不存在
        {"top_k": 0},  # ge=1
        {"top_k": 501},  # le=500
        {"configs": "⑦"},  # 消融配置没有第七种 → 服务层闸门
        {"mode": "ragas", "run_ids": []},  # 空列表
        {"mode": "ragas", "run_ids": ["bad id"]},  # 含空格 → run_id 字符集闸门
        {"mode": "ragas", "metrics": ["not_a_ragas_metric"]},  # 未知指标名（运行期会被静默跳过）
        {"mode": "ablation", "unknown_field": 1},  # extra=forbid
    ]
    for body in bad_bodies:
        resp = await client.post(RUN_URL, json=body)
        assert resp.status_code == 422, body
        assert "detail" in resp.json()
    assert calls == []
    assert eval_service.current_run() is None

    # 422 一条都不该落库
    async with session_scope() as db:
        assert len((await db.execute(select(EvalRun))).scalars().all()) == 0

    # 反向确认：合法形状走进服务层
    ok = await client.post(RUN_URL, json={"mode": "metrics", "name": "nightly"})
    assert ok.status_code == 202, ok.text
    await _wait_terminal(client, ok.json()["eval_run_id"])
    assert calls == [{}]


async def test_golden_path_stays_inside_the_repo(client):
    """golden_path 是用户可控路径：跑出仓库 / 非 .jsonl / 文件不存在，三种都在受理前 422。"""
    for bad in ("../../Windows/win.ini", "app/eval/datasets/golden_40.jsonl.bak", "nonexistent.jsonl"):
        resp = await client.post(RUN_URL, json={"mode": "ablation", "golden_path": bad})
        assert resp.status_code == 422, bad
        assert resp.json()["detail"]["field"] == "golden_path"
    assert eval_service.current_run() is None


async def test_get_eval_run_unknown_or_malformed_id(client):
    """404 = 库里没有这条；422 = id 根本不是整数。两者不能混成一种。"""
    missing = await client.get(POLL_TMPL.format(9999))
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "eval_run_not_found"

    malformed = await client.get(POLL_TMPL.format("not-a-number"))
    assert malformed.status_code == 422


# =========================================================================== 路由注册与响应安全
async def test_eval_routes_registered_and_responses_carry_no_secrets(client):
    """静态 `/run` 声明在动态 `/runs/{eval_run_id}` 之前；SPA 兜底没把 API 吞成 HTML；响应无密钥。"""
    spec = (await client.get("/openapi.json")).json()["paths"]
    assert "/api/v1/eval/run" in spec and "post" in spec["/api/v1/eval/run"]
    assert "/api/v1/eval/runs/{eval_run_id}" in spec
    assert spec["/api/v1/eval/run"]["post"]["responses"].get("202"), (
        "触发接口必须是 202（已受理、未跑完），不能让人以为是同步结果"
    )

    resp = await client.post(RUN_URL, json={"mode": "metrics"})
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert_no_secret_values(body)
    done = await _wait_terminal(client, body["eval_run_id"])
    assert_no_secret_values(done)

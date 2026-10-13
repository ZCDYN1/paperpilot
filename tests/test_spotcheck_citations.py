"""人工抽检口径（`scripts/spotcheck_citations.py` + `ragas_runner.manual_accuracy`）的测试。

守住四条承诺（对应架构 §12「两个口径都要报」）：
1. 抽样**可复现**（同库同 seed 同一批），导出 CSV 含人工判定所需的全部材料；
2. 人工准确率算术正确：**未判定行不进分母**（否则样本缺失会拉偏结论）；
3. `ambiguous` 计入分母不计入分子（拿不准不算"名副其实"）；
4. 报告能并列两口径，且**没有抽检文件时绝不凭空造人工数字**。

全程临时 SQLite + 临时 CSV，不碰真实 `data/`，也不需要任何 API key。
"""

from __future__ import annotations

import csv as csvlib
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 注册全部表到 metadata
from app.db import session as db_session
from app.db.base import Base
from app.db.models import Citation
from app.eval.ragas_runner import EvalReport, manual_accuracy, run_ragas_eval

# 以"模块"方式加载 scripts/spotcheck_citations.py（scripts 不是包的一部分）
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "spotcheck_citations.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("spotcheck_citations", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spot = _load_script()


# --------------------------------------------------------------------------- 纯函数
def test_pick_sample_is_reproducible_and_clamped():
    pool = [{"id": i} for i in range(50)]
    a = spot.pick_sample(pool, 20, seed=42)
    b = spot.pick_sample(pool, 20, seed=42)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len(a) == 20
    assert len(spot.pick_sample(pool, 20, seed=1)) == 20
    small = [{"id": i} for i in range(5)]
    assert len(spot.pick_sample(small, 20, seed=42)) == 5  # 不足时全量，不报错
    with pytest.raises(ValueError):
        spot.pick_sample(pool, 0, seed=42)


def test_write_csv_columns_and_blank_verdict(tmp_path):
    out = tmp_path / "spot.csv"
    rows = [{"citation_id": 1, "claim": "x", "chunk_text": "y", "auto_verified": "True"}]
    spot.write_csv(rows, out)
    with out.open(encoding="utf-8-sig") as fh:
        reader = csvlib.DictReader(fh)
        assert reader.fieldnames == spot.CSV_COLUMNS
        got = list(reader)
    assert got[0]["human_verdict"] == ""  # 必须留空给人工填
    assert got[0]["claim"] == "x"


# --------------------------------------------------------------------------- manual_accuracy 算术
def _make_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csvlib.DictWriter(fh, fieldnames=["citation_id", "auto_verified", "human_verdict"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_manual_accuracy_denominator_and_agreement(tmp_path):
    p = tmp_path / "s.csv"
    _make_csv(
        p,
        [
            {"auto_verified": "True", "human_verdict": "supported"},
            {"auto_verified": "True", "human_verdict": "unsupported"},   # 自动放行但人工打脸 → 拉低一致率
            {"auto_verified": "True", "human_verdict": "ambiguous"},
            {"auto_verified": "False", "human_verdict": "unsupported"},
            {"auto_verified": "True", "human_verdict": ""},               # 未判定：不进任何分母
            {"auto_verified": "", "human_verdict": "supported"},          # 未校验：不进一致率分母
            {"auto_verified": "True", "human_verdict": "瞎写的"},         # invalid
        ],
    )
    stats = manual_accuracy(p)
    assert stats["total_judged"] == 5
    assert stats["supported"] == 2
    assert stats["unsupported"] == 2
    assert stats["ambiguous"] == 1
    # accuracy 分子只数 supported：2/5
    assert stats["accuracy"] == pytest.approx(2 / 5)
    # 一致率只在 auto_verified 非空的已判定行（前 4 行）上算：第 1 行 agree、第 2 行 disagree、
    # 第 3 行 ambiguous vs True → disagree、第 4 行 agree → 2/4
    assert stats["agreement"] == pytest.approx(0.5)
    assert stats["pending"] == 1
    assert stats["invalid"] == 1
    assert stats["auto_accuracy"] == pytest.approx(3 / 5)  # 同分母下自动口径（5 条已判定里 3 条 True）
    assert stats["method"] == "manual"


def test_manual_accuracy_verdict_variants(tmp_path):
    p = tmp_path / "s.csv"
    _make_csv(p, [{"auto_verified": "True", "human_verdict": v} for v in ["yes", "是", "SUPPORTED", "No", "存疑"]])
    stats = manual_accuracy(p)
    assert (stats["supported"], stats["unsupported"], stats["ambiguous"]) == (3, 1, 1)


def test_manual_accuracy_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        manual_accuracy(tmp_path / "nope.csv")


def test_manual_accuracy_no_auto_column(tmp_path):
    p = tmp_path / "s.csv"
    _make_csv(p, [{"auto_verified": "", "human_verdict": "supported"}])
    stats = manual_accuracy(p)
    assert stats["agreement"] is None  # 没有可比行时报 None，而不是 0/1 误导


# --------------------------------------------------------------------------- 报告并列两口径
@pytest.fixture(autouse=True)
def _no_real_spotcheck(monkeypatch, tmp_path):
    """把默认抽检文件挪到临时目录，避免测试读到本机 eval_results 里的真实文件。"""
    monkeypatch.setattr(
        "app.eval.ragas_runner.DEFAULT_SPOTCHECK_CSV", tmp_path / "default_absent.csv"
    )


async def test_run_ragas_eval_degraded_still_carries_both_calibers(monkeypatch, tmp_path):
    """ragas 缺失 → available=False，但人工口径（若有填好的 CSV）必须照常并列。"""
    p = tmp_path / "s.csv"
    _make_csv(p, [{"auto_verified": "True", "human_verdict": "unsupported"}])

    def boom():
        raise ImportError("no ragas")

    monkeypatch.setattr("app.eval.ragas_runner._load_ragas", boom)
    report = await run_ragas_eval(run_ids=[], spotcheck_csv=p)
    assert report.available is False
    assert report.metrics == {} and report.per_sample == []  # 防假分数不变量仍然成立
    assert report.manual_accuracy and report.manual_accuracy["total_judged"] == 1
    assert report.as_dict()["manual_accuracy"]["accuracy"] == 0.0


async def test_run_ragas_eval_without_csv_has_no_manual_number(monkeypatch):
    """没有抽检文件时 manual_accuracy 必须是 None —— 不许凭空造一个'人工准确率 0/1'。"""

    def boom():
        raise ImportError("no ragas")

    monkeypatch.setattr("app.eval.ragas_runner._load_ragas", boom)
    report = await run_ragas_eval(run_ids=[])
    assert report.available is False
    assert report.manual_accuracy is None


def test_evalreport_accepts_manual_with_metrics():
    r = EvalReport(available=True, metrics={"faithfulness": 0.5},
                   manual_accuracy={"total_judged": 3, "accuracy": 0.6667, "method": "manual"})
    assert r.as_dict()["manual_accuracy"]["accuracy"] == pytest.approx(0.6667)


# --------------------------------------------------------------------------- 导料：只读真库结构
@pytest.fixture
async def temp_db(monkeypatch, tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'spot.db'}")
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


async def test_fetch_pool_maps_rows(temp_db):
    async with temp_db() as session:
        session.add_all(
            [
                Citation(run_id="r1", claim="c1", quote="q1", verified=True),
                Citation(run_id="r1", claim="c2", quote="q2", verified=False, note="引文未逐字匹配"),
                Citation(run_id="r2", claim="c3", quote="q3", verified=None),
            ]
        )
    pool = await spot.fetch_pool()
    assert len(pool) == 3
    by_claim = {r["claim"]: r for r in pool}
    assert by_claim["c1"]["auto_verified"] == "True"
    assert by_claim["c2"]["auto_verified"] == "False"
    assert by_claim["c2"]["chunk_text"] == ""      # chunk_id 为空 → 原文缺失，人工据此判 unsupported
    assert by_claim["c3"]["auto_verified"] == ""   # None = 未校验，导出为空串


async def test_cmd_export_end_to_end(temp_db, tmp_path, capsys):
    async with temp_db() as session:
        for i in range(25):
            session.add(Citation(run_id=f"r{i % 3}", claim=f"c{i}", quote=f"q{i}", verified=True))
    out = tmp_path / "out.csv"
    code = await spot.cmd_export(n=20, seed=7, out_path=out)
    assert code == 0
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csvlib.DictReader(fh))
    assert len(rows) == 20
    assert all(r["human_verdict"] == "" for r in rows)
    assert spot.DEFAULT_N == 20  # 架构 §12 的"抽 20 条"


# --------------------------------------------------------------------------- ★ 回填闭环
async def test_export_fill_backfill_round_trip(temp_db, tmp_path, capsys):
    """导出 → 人工填 `human_verdict` → `--from-csv` 回填 的**闭环**。

    回填（`cmd_stats` → `manual_accuracy`）只认导出文件里的 `human_verdict` 与
    `auto_verified` 两个列名。一旦导出侧改了列名（比如把 `auto_verified` 写成 `verified`），
    回填**不会报错**，只会统计出"0 条已判定"，人工口径就此静默失效 ——
    而填表的人以为自己已经填完了。这条用例是那个错位唯一的探测器。

    三种判定都填：supported / unsupported / ambiguous（外加一条未校验的 None），
    顺带把"未判定不进分母""未校验不进一致率分母"这两条口径在真实导出文件上复核一遍。
    """
    from app.db.models import Paper, PaperChunk

    async with temp_db() as session:
        session.add(Paper(arxiv_id="2401.00001", title="t"))
        chunk = PaperChunk(arxiv_id="2401.00001", chunk_index=0, section="abstract", text="RRF 融合原文。")
        session.add(chunk)
        await session.flush()
        session.add_all(
            [
                Citation(run_id="r1", claim="c-ok", quote="q1", chunk_id=chunk.id, verified=True),
                Citation(run_id="r1", claim="c-bad", quote="q2", chunk_id=chunk.id, verified=False),
                Citation(run_id="r1", claim="c-none", quote="q3", chunk_id=chunk.id, verified=None),
                Citation(run_id="r2", claim="c-blank", quote="q4", chunk_id=chunk.id, verified=True),
            ]
        )

    out = tmp_path / "spot.csv"
    assert await spot.cmd_export(n=20, seed=3, out_path=out) == 0

    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csvlib.DictReader(fh))
    assert len(rows) == 4
    verdicts = {"c-ok": "supported", "c-bad": "unsupported", "c-none": "ambiguous"}
    for row in rows:
        row["human_verdict"] = verdicts.get(row["claim"], "")  # c-blank 留空 = 还没判
    with out.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csvlib.DictWriter(fh, fieldnames=spot.CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    stats = manual_accuracy(out)
    assert stats["total_judged"] == 3, "未判定那条不该进分母"
    assert stats["pending"] == 1
    assert stats["accuracy"] == pytest.approx(1 / 3)  # ambiguous 在分母里、不在分子里
    # 一致率只在 auto_verified 非空的已判定行上算：c-ok(agree) / c-bad(agree)，c-none 未校验不计
    assert stats["agreement"] == pytest.approx(1.0)
    assert stats["auto_accuracy"] == pytest.approx(1 / 3)

    code = spot.cmd_stats(out)
    assert code == 0
    printed = capsys.readouterr().out
    assert "人工准确率" in printed
    assert "0.3333" in printed, "回填必须把算出来的人工口径打到 stdout"


def test_cmd_stats_handles_missing_file_and_unfilled_csv(tmp_path, capsys):
    """两种"还没做完"的边界：文件不存在要给非零码（脚本可被 CI 区分），没填判定要说清楚。"""
    assert spot.cmd_stats(tmp_path / "nope.csv") == 1

    p = tmp_path / "blank.csv"
    _make_csv(p, [{"auto_verified": "True", "human_verdict": ""}])
    capsys.readouterr()
    assert spot.cmd_stats(p) == 0
    assert "还没有任何 human_verdict 判定" in capsys.readouterr().out

"""抽检抽样池的 provider 过滤测试（`scripts/spotcheck_citations.py`）。

守住的口径（对应产品定义 §8 双口径汇报）：
1. mock run 的 `verified` 结构恒真（verifier 分支永远放行），**默认不进人工抽样池** ——
   让人核对它们既浪费人力，又会把人工口径系统性拉高；
2. 没跑到 finalize 的 run 没有 token_usage 行，provider 确实不可考 → 标 `unknown`，
   默认同样剔除（来源不可考的行不值得占用人力），但可用 `--provider unknown/all` 显式取回，
   绝不悄悄猜测、也不悄悄丢弃；
3. 导出 CSV 带 `run_provider` 列，填表的人能看到每行的来源；
4. `cmd_stats` 对新增列无感，对**旧版导出的 CSV**（没有 run_provider 列）也不许崩。

纯函数部分（classify_provider / filter_pool_by_provider）不依赖 DB；
集成部分用临时 SQLite，不碰真实 `data/`。
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
from app.db.models import Citation, TokenUsage

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "spotcheck_citations.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("spotcheck_citations_sampling", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


spot = _load_script()


# --------------------------------------------------------------------------- 纯函数：provider 判定
def test_classify_provider():
    assert spot.classify_provider([]) == "unknown"          # 没跑到 finalize：无 token_usage 行
    assert spot.classify_provider([None]) == "unknown"      # 有行但 model 为空：同样不可考
    assert spot.classify_provider(["mock"]) == "mock"       # mock 模式 llm_identity 固定 model="mock"
    assert spot.classify_provider(["mock", "mock"]) == "mock"
    assert spot.classify_provider(["qwen2.5:7b"]) == "real"
    assert spot.classify_provider(["mock", None]) == "mock"  # 空 model 不算"非 mock 证据"
    assert spot.classify_provider(["mock", "gpt-4o"]) == "real"  # 出现过非 mock → real（保守偏向可抽）


def test_filter_pool_by_provider_default_excludes_mock_and_unknown():
    pool = [
        {"citation_id": 1, "run_provider": "real"},
        {"citation_id": 2, "run_provider": "mock"},
        {"citation_id": 3, "run_provider": "unknown"},
        {"citation_id": 4, "run_provider": "real"},
    ]
    kept, dropped = spot.filter_pool_by_provider(pool)  # 默认 real
    assert [r["citation_id"] for r in kept] == [1, 4]
    assert dropped == {"real": 0, "mock": 1, "unknown": 1}


def test_filter_pool_by_provider_all_keeps_everything():
    pool = [
        {"citation_id": 1, "run_provider": "real"},
        {"citation_id": 2, "run_provider": "mock"},
        {"citation_id": 3, "run_provider": "unknown"},
    ]
    kept, dropped = spot.filter_pool_by_provider(pool, "all")
    assert len(kept) == 3
    assert dropped == {"real": 0, "mock": 0, "unknown": 0}


def test_filter_pool_by_provider_explicit_mock_and_unknown():
    pool = [
        {"citation_id": 1, "run_provider": "real"},
        {"citation_id": 2, "run_provider": "mock"},
        {"citation_id": 3, "run_provider": "unknown"},
    ]
    kept, _ = spot.filter_pool_by_provider(pool, "mock")
    assert [r["citation_id"] for r in kept] == [2]
    kept, _ = spot.filter_pool_by_provider(pool, "unknown")
    assert [r["citation_id"] for r in kept] == [3]


def test_filter_pool_missing_key_treated_as_unknown():
    # 缺 run_provider 键（比如别的代码手工拼的行）→ 按 unknown 处理，绝不当 real 混进样本
    kept, dropped = spot.filter_pool_by_provider([{"citation_id": 1}])
    assert kept == []
    assert dropped["unknown"] == 1


def test_filter_pool_rejects_bad_provider():
    with pytest.raises(ValueError):
        spot.filter_pool_by_provider([], "whatever")


def test_csv_columns_contains_run_provider():
    assert "run_provider" in spot.CSV_COLUMNS


# --------------------------------------------------------------------------- 集成：临时库导出
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


async def _seed_mixed_runs(temp_db):
    """三种来源各一个 run：real（qwen2.5:7b）/ mock / unknown（没有 token_usage 行）。"""
    async with temp_db() as session:
        session.add_all(
            [
                Citation(run_id="run-real", claim="c-real", quote="q", verified=True),
                Citation(run_id="run-mock", claim="c-mock", quote="q", verified=True),
                Citation(run_id="run-orphan", claim="c-orphan", quote="q", verified=True),
                TokenUsage(run_id="run-real", node_name="all", model="qwen2.5:7b"),
                TokenUsage(run_id="run-real", node_name="synthesize", model="qwen2.5:7b"),
                TokenUsage(run_id="run-mock", node_name="all", model="mock"),
            ]
        )


async def test_fetch_pool_tags_run_provider(temp_db):
    await _seed_mixed_runs(temp_db)
    pool = await spot.fetch_pool()
    by_claim = {r["claim"]: r for r in pool}
    assert by_claim["c-real"]["run_provider"] == "real"
    assert by_claim["c-mock"]["run_provider"] == "mock"
    assert by_claim["c-orphan"]["run_provider"] == "unknown"


async def test_cmd_export_default_only_real(temp_db, tmp_path, capsys):
    await _seed_mixed_runs(temp_db)
    out = tmp_path / "spot.csv"
    assert await spot.cmd_export(n=20, seed=7, out_path=out) == 0
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csvlib.DictReader(fh))
    assert [r["claim"] for r in rows] == ["c-real"]  # mock 与 unknown 都被默认剔除
    assert rows[0]["run_provider"] == "real"         # 来源列写进了 CSV
    printed = capsys.readouterr().out
    # 诚实汇报：池子多大、剔了多少、为什么剔 mock
    assert "库里共 3 条" in printed
    assert "mock 1 条" in printed and "unknown 1 条" in printed


async def test_cmd_export_provider_all_and_unknown(temp_db, tmp_path):
    await _seed_mixed_runs(temp_db)
    out = tmp_path / "all.csv"
    assert await spot.cmd_export(n=20, seed=7, out_path=out, provider="all") == 0
    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csvlib.DictReader(fh))
    assert {r["run_provider"] for r in rows} == {"real", "mock", "unknown"}

    out2 = tmp_path / "unknown.csv"
    assert await spot.cmd_export(n=20, seed=7, out_path=out2, provider="unknown") == 0
    with out2.open(encoding="utf-8-sig") as fh:
        rows2 = list(csvlib.DictReader(fh))
    assert [r["claim"] for r in rows2] == ["c-orphan"]


async def test_cmd_export_says_plainly_when_pool_short(temp_db, tmp_path, capsys):
    await _seed_mixed_runs(temp_db)
    out = tmp_path / "short.csv"
    assert await spot.cmd_export(n=20, seed=7, out_path=out) == 0  # real 只有 1 条 < 20
    printed = capsys.readouterr().out
    assert "不足请求的 20 条" in printed  # 明说，而不是悄悄少给


async def test_cmd_export_all_filtered_out_returns_nonzero(temp_db, tmp_path, capsys):
    async with temp_db() as session:
        session.add(Citation(run_id="run-mock", claim="c", quote="q", verified=True))
        session.add(TokenUsage(run_id="run-mock", node_name="all", model="mock"))
    out = tmp_path / "none.csv"
    assert await spot.cmd_export(n=20, seed=7, out_path=out) == 1
    assert not out.exists()  # 池子空时不写出一个只有表头的 CSV 假装成功
    assert "--provider all" in capsys.readouterr().out


# --------------------------------------------------------------------------- cmd_stats 兼容性
def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csvlib.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def test_cmd_stats_on_old_csv_without_run_provider(tmp_path, capsys):
    """旧版导出的 CSV 没有 run_provider 列 —— 回填统计必须照常工作（列名按 key 取，不校验表头）。"""
    old_columns = [c for c in spot.CSV_COLUMNS if c != "run_provider"]
    p = tmp_path / "old.csv"
    _write_csv(
        p,
        old_columns,
        [{"auto_verified": "True", "human_verdict": "supported", "claim": "x"}],
    )
    assert spot.cmd_stats(p) == 0
    assert "人工准确率" in capsys.readouterr().out


def test_cmd_stats_on_new_csv_with_run_provider(tmp_path, capsys):
    p = tmp_path / "new.csv"
    _write_csv(
        p,
        spot.CSV_COLUMNS,
        [
            {"auto_verified": "True", "human_verdict": "supported", "run_provider": "real"},
            {"auto_verified": "True", "human_verdict": "unsupported", "run_provider": "real"},
        ],
    )
    assert spot.cmd_stats(p) == 0
    printed = capsys.readouterr().out
    assert "0.5000" in printed  # 新增列不影响算术

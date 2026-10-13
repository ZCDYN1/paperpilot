"""Reranker 工厂与真实 BGE 路径的单测（**不下载任何权重**）。

覆盖目标（交付标准：陌生 clone 者可复现）：
1. 工厂三分支：disabled→Null / lexical→Lexical / local→BGE；
2. local 但依赖或权重缺失：**自动降级 lexical**，且 warning 里带**可照抄的获取命令**
   （必须含 `hf-mirror`，因为实测国内直连 huggingface.co 不通）；
3. `strict=True` 时同样缺失必须**当场抛错**而不是留到第一个检索请求；
4. BGEReranker 调用形态：`FlagReranker(model_name_or_path, use_fp16=False)` ——
   首参透传，`RERANK_MODEL` 写本地目录也走得通；`compute_score(pairs, normalize=True)`
   的返回值（单条 float / 多条 list）都能正确排序与截断。

BGE 分支全部用 monkeypatch 的假类验证，绝不触发真实 import/下载。
"""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.retrieval import reranker as rr
from app.retrieval.reranker import (
    BGEReranker,
    LexicalReranker,
    NullReranker,
    download_instructions,
    get_reranker,
    model_locally_available,
)


def make_settings(**overrides) -> Settings:
    base = dict(rerank_enabled=True, rerank_provider="local", rerank_model="BAAI/bge-reranker-v2-m3")
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- 工厂三分支
def test_factory_disabled_returns_null():
    r = get_reranker(make_settings(rerank_enabled=False))
    assert isinstance(r, NullReranker)
    # Null 仅保序：分数必须严格随原序递减
    assert r.rerank_sync("q", ["a", "b", "c"]) == [(0, 1.0), (1, 0.5), (2, 1 / 3)]


def test_factory_lexical_provider():
    assert isinstance(get_reranker(make_settings(rerank_provider="lexical")), LexicalReranker)


def test_factory_local_with_weights(monkeypatch):
    monkeypatch.setattr(rr, "_flagembedding_installed", lambda: True)
    monkeypatch.setattr(rr, "model_locally_available", lambda m: True)
    r = get_reranker(make_settings())
    assert isinstance(r, BGEReranker)
    assert r.name == "bge-reranker"  # ablation 的可信度警告靠这个名字判定
    assert r.model_name == "BAAI/bge-reranker-v2-m3"


# --------------------------------------------------------------------------- 缺失 → 降级 / strict 抛错
class RecordingLogger:
    """structlog 直写 stdout，caplog 抓不到；这里直接拦模块 logger 记录 warning 参数。"""

    def __init__(self):
        self.warnings: list[dict] = []

    def warning(self, event, **kw):
        self.warnings.append({"event": event, **kw})

    def info(self, event, **kw):  # pragma: no cover - 工厂也会打 info，保持接口完整
        pass


@pytest.fixture
def rec_logger(monkeypatch):
    logger = RecordingLogger()
    monkeypatch.setattr(rr, "logger", logger)
    return logger


def test_local_missing_dependency_falls_back_with_copyable_hint(monkeypatch, rec_logger):
    monkeypatch.setattr(rr, "_flagembedding_installed", lambda: False)
    r = get_reranker(make_settings())
    assert isinstance(r, LexicalReranker)
    assert rec_logger.warnings, "降级必须打 warning，不许无声"
    text = str(rec_logger.warnings)
    assert "[local]" in text               # pip 命令
    assert "hf-mirror" in text             # 国内镜像（直连不通是实测事实）


def test_local_missing_weights_falls_back_with_download_guide(monkeypatch, rec_logger):
    monkeypatch.setattr(rr, "_flagembedding_installed", lambda: True)
    monkeypatch.setattr(rr, "model_locally_available", lambda m: False)
    r = get_reranker(make_settings())
    assert isinstance(r, LexicalReranker)
    text = str(rec_logger.warnings)
    assert "HF_ENDPOINT=https://hf-mirror.com" in text
    assert "snapshot_download" in text
    assert "[local]" in text


def test_strict_raises_when_dependency_missing(monkeypatch):
    monkeypatch.setattr(rr, "_flagembedding_installed", lambda: False)
    with pytest.raises(RuntimeError, match=r"\[local\]"):
        get_reranker(make_settings(), strict=True)


def test_strict_raises_when_weights_missing(monkeypatch):
    monkeypatch.setattr(rr, "_flagembedding_installed", lambda: True)
    monkeypatch.setattr(rr, "model_locally_available", lambda m: False)
    with pytest.raises(RuntimeError, match="hf-mirror"):
        get_reranker(make_settings(), strict=True)


# --------------------------------------------------------------------------- 本地目录 / 仓库名两种写法
def test_local_dir_counts_as_available(tmp_path):
    d = tmp_path / "bge-reranker-v2-m3"
    d.mkdir()
    assert not model_locally_available(str(d))  # 目录在但没 config.json → 仍不可用
    (d / "config.json").write_text("{}", encoding="utf-8")
    assert model_locally_available(str(d))


def test_repo_name_absent_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rr, "_cached_on_hf", lambda m: False)
    assert not model_locally_available("BAAI/bge-reranker-v2-m3")


# --------------------------------------------------------------------------- BGEReranker 调用形态（假 FlagReranker）
class FakeFlagReranker:
    """记录构造参数与 compute_score 入参；返回可控分数。"""

    last_args: tuple = ()
    scores: list = [0.1, 0.9, 0.5]

    def __init__(self, model_name_or_path, use_fp16=False, **kwargs):
        FakeFlagReranker.last_args = (model_name_or_path, use_fp16)

    def compute_score(self, pairs, normalize=False):
        FakeFlagReranker.last_pairs = pairs
        FakeFlagReranker.last_normalize = normalize
        if len(pairs) == 1:
            return float(self.scores[0])
        return list(self.scores)[: len(pairs)]


@pytest.fixture
def fake_flag(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("FlagEmbedding")
    mod.FlagReranker = FakeFlagReranker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    FakeFlagReranker.last_args = ()
    FakeFlagReranker.scores = [0.1, 0.9, 0.5]
    return FakeFlagReranker


def test_bge_passes_model_name_through(fake_flag, tmp_path):
    """本地目录写法必须原样透传给 FlagReranker 的首参（model_name_or_path）。"""
    d = tmp_path / "reranker-dir"
    d.mkdir()
    (d / "config.json").write_text("{}", encoding="utf-8")
    r = BGEReranker(str(d))
    r._ensure_model()
    assert fake_flag.last_args == (str(d), False)  # use_fp16=False（CPU 一致性）


def test_bge_score_ordering_and_top_n(fake_flag):
    r = BGEReranker("BAAI/bge-reranker-v2-m3")
    ranked = r.rerank_sync("q", ["d0", "d1", "d2"])
    assert fake_flag.last_pairs == [["q", "d0"], ["q", "d1"], ["q", "d2"]]
    assert fake_flag.last_normalize is True
    # scores=[0.1,0.9,0.5] → 降序应为 (1,0.9),(2,0.5),(0,0.1)
    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.1)]
    assert r.rerank_sync("q", ["d0", "d1", "d2"], top_n=2) == ranked[:2]


def test_bge_single_doc_scalar_return_handled(fake_flag):
    fake_flag.scores = [0.7, 0.9, 0.5]  # 只取第一个：单条时 compute_score 返回 float
    r = BGEReranker("BAAI/bge-reranker-v2-m3")
    assert r.rerank_sync("q", ["only"]) == [(0, 0.7)]


def test_bge_lazy_loading(fake_flag):
    """构造不得加载模型（否则每次建索引都要拖起 torch）。"""
    r = BGEReranker("BAAI/bge-reranker-v2-m3")
    assert r._model is None
    r._ensure_model()
    assert r._model is not None


def test_bge_load_failure_message_is_actionable(fake_flag, monkeypatch):
    """from_pretrained 失败（网络断/权重坏）→ RuntimeError 必须带镜像下载命令。"""

    class Boom(FakeFlagReranker):
        def __init__(self, *a, **k):
            raise OSError("We couldn't connect to huggingface.co")

    import sys
    import types

    mod = types.ModuleType("FlagEmbedding")
    mod.FlagReranker = Boom
    monkeypatch.setitem(sys.modules, "FlagEmbedding", mod)
    with pytest.raises(RuntimeError, match="hf-mirror") as ei:
        BGEReranker("BAAI/bge-reranker-v2-m3")._ensure_model()
    assert "couldn't connect" in str(ei.value).lower() or "huggingface" in str(ei.value)


# --------------------------------------------------------------------------- 指引文案本身
def test_download_instructions_cover_both_platforms_and_local_dir():
    txt = download_instructions("BAAI/bge-reranker-v2-m3")
    assert "$env:HF_ENDPOINT" in txt              # Windows PowerShell 写法
    assert "export HF_ENDPOINT=https://hf-mirror.com" in txt  # Git Bash/Linux
    assert "models/bge-reranker-v2-m3" in txt     # 本地目录写法 + RERANK_MODEL 指回该目录
    assert "RERANK_MODEL=models/bge-reranker-v2-m3" in txt

"""路径类配置的"锚在仓库根"性质（开源可复现的底线之一）。

为什么单独钉这个性质：`data_dir` / `sqlite_url` 这类默认值一旦写成相对路径
（`./data`），行为就变成"**在哪个目录启动，就在哪个目录新建一套空库**"。
它不报错 —— 用户看到的是"索引明明建了，却什么都检索不到"，
而排查方向会被带到 embedding、FAISS、过滤条件上去，实际是启动目录的问题。
别人 clone 后第一个踩到的往往就是它，所以这里把"与 CWD 无关"钉成断言。
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import REPO_ROOT, Settings


def _bare_settings() -> Settings:
    """不读任何 `.env`，只看代码里的默认值（默认值才是 clone 后第一次运行的形态）。"""
    return Settings(_env_file=None)


def test_repo_root_is_the_project_root():
    assert (REPO_ROOT / "pyproject.toml").exists(), f"REPO_ROOT 指错了：{REPO_ROOT}"
    assert (REPO_ROOT / "app" / "core" / "config.py").exists()


def test_storage_defaults_are_absolute_and_under_repo_root(tmp_path, monkeypatch):
    """★ 换目录启动必须指向同一套库 —— 这条挂了，就等于把上面 docstring 的坑放回去。"""
    monkeypatch.chdir(tmp_path)
    s = _bare_settings()

    for label, value in (
        ("data_dir", s.data_dir),
        ("faiss_index_dir", s.faiss_index_dir),
        ("sqlite_dir", s.sqlite_dir),
        ("checkpoint_db", s.checkpoint_db),
    ):
        p = Path(value)
        assert p.is_absolute(), f"{label} 不是绝对路径：{p}"
        assert p == REPO_ROOT / "data" or p.is_relative_to(REPO_ROOT / "data"), (
            f"{label} 跑到了 data/ 外面：{p}"
        )
        assert Path(tmp_path) not in p.parents, f"{label} 跟着 CWD 走了：{p}"


def test_sqlite_url_is_an_absolute_posix_path_under_data():
    """URL 里的反斜杠（Windows 默认 `str(Path)` 形状）会让 aiosqlite 认不出路径；
    看着像数据库问题，其实是 URL 解析问题。所以钉死"斜杠 + 绝对 + 在 data 下"。"""
    url = _bare_settings().sqlite_url
    assert url.startswith("sqlite+aiosqlite:///"), url
    path_part = url[len("sqlite+aiosqlite:///") :]
    assert "\\" not in path_part, f"URL 里出现了反斜杠：{path_part}"
    assert Path(path_part).is_absolute(), f"URL 里的库路径不是绝对路径：{path_part}"
    assert Path(path_part).is_relative_to(REPO_ROOT / "data"), path_part


def test_env_file_lookup_does_not_depend_on_cwd():
    """`.env` 的查找位置也必须是仓库根，否则"在项目根之外启动"会静默地不读配置 ——
    表现是所有值退回默认，而用户以为自己在用自己那份 .env。"""
    env_file = Settings.model_config.get("env_file")
    assert Path(env_file).is_absolute(), f"env_file 是相对路径：{env_file}"
    assert Path(env_file) == REPO_ROOT / ".env"

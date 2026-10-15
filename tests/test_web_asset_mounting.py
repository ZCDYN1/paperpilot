"""`app/main.py::_mount_web_ui` 的对外契约：兜底路由不许把缺失的构建产物伪装成正常页面。

★ 为什么要单独立一个文件：
`web/dist/assets` 在后端启动时不存在，旧实现的兜底路由会对 `/assets/*.js`
返回 200 + text/html（index.html 本体），浏览器拿到非 JS 的 MIME 会**静默拒绝**
把它当 module 执行 —— 白屏、控制台一行报错都没有、网络面板全是绿，极难定位（实测踩过）。
现在钉死两种情形：assets 存在 → 正常给文件；assets 缺失 → 404 红叉，一眼可辨。

测试直接调 `_mount_web_ui(app, dist_dir=...)` 而不是 `create_app()`：
这里要验的只有静态挂载与兜底路由，不该把数据库 / lifespan / API 路由拖进来。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.main import _mount_web_ui

# 与真实产物形态一致：module 脚本从 /assets 加载
INDEX_HTML = (
    '<!doctype html><html><head>'
    '<script type="module" src="/assets/index-hash1234.js"></script>'
    "</head><body><div id=\"root\"></div></body></html>"
)
ASSET_JS = "console.log('paperpilot');"


def _make_dist(tmp_path: Path, *, with_assets: bool) -> Path:
    dist = tmp_path / "web" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    if with_assets:
        assets = dist / "assets"
        assets.mkdir()
        (assets / "index-hash1234.js").write_text(ASSET_JS, encoding="utf-8")
    return dist


def _client(tmp_path: Path, *, with_assets: bool) -> TestClient:
    app = FastAPI()
    assert _mount_web_ui(app, dist_dir=_make_dist(tmp_path, with_assets=with_assets)) is True
    return TestClient(app)


def test_assets_are_served_when_dir_present(tmp_path):
    client = _client(tmp_path, with_assets=True)
    resp = client.get("/assets/index-hash1234.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]
    assert resp.text == ASSET_JS
    # SPA 外壳照常：任意非 API 路径都该拿到 index.html
    assert client.get("/console").status_code == 200


def test_single_missing_asset_file_is_404(tmp_path):
    # 挂载在、但 index.html 引用的那个 hash 文件被删（换了新构建没重启）：
    # StaticFiles 自己就该 404，不许退化成 HTML
    dist = _make_dist(tmp_path, with_assets=True)
    (dist / "assets" / "index-hash1234.js").unlink()
    client = TestClient(_mounted_app(dist))
    resp = client.get("/assets/index-hash1234.js")
    assert resp.status_code == 404
    assert ASSET_JS not in resp.text


def test_assets_404_not_html_when_assets_dir_missing(tmp_path):
    # 核心回归用例：assets/ 整个不存在（首次构建没跑 / 被删）。
    # 曾经的事故形态是 200 + HTML + index.html 内容，这里必须是 404 且响应体不是 index.html。
    client = _client(tmp_path, with_assets=False)
    resp = client.get("/assets/index-hash1234.js")
    assert resp.status_code == 404
    assert INDEX_HTML not in resp.text


def test_spa_shell_still_works_when_assets_missing(tmp_path):
    # assets 缺失只该影响 /assets/*，不该顺手把 SPA 外壳也拒掉：
    # 页面至少能加载、由前端自己提示（或让人在网络面板里看到 /assets 的红）。
    client = _client(tmp_path, with_assets=False)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "root" in resp.text


def test_fallback_never_swallows_api_and_healthz(tmp_path):
    # 原有约束不许被这次改动破坏：拼错的 API / healthz / 兜底之外的保留前缀都该 404
    for assets_ok in (True, False):
        client = _client(tmp_path / f"ok{assets_ok}", with_assets=assets_ok)
        for path in ("/api/v1/nope", "/healthz", "/docs-typo", "/redoc-typo"):
            resp = client.get(path)
            assert resp.status_code == 404, path
            assert INDEX_HTML not in resp.text, path


def test_no_index_means_no_mount(tmp_path):
    # 完全没构建过：不挂 SPA，也不该抛异常（退回 Streamlit 的路径由 launch.py 负责）
    app = FastAPI()
    assert _mount_web_ui(app, dist_dir=tmp_path / "web" / "dist") is False


def _mounted_app(dist: Path) -> FastAPI:
    app = FastAPI()
    assert _mount_web_ui(app, dist_dir=dist) is True
    return app

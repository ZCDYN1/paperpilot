"""PaperPilot 一键启动器（`启动PaperPilot.bat` 调用，也可以自己跑）。

顺序：起后端 → 等 /healthz 真的返回 200 → 起面板 → 等面板就绪 → 开浏览器。
必须等就绪再开浏览器，否则用户看到的是 502 而不是界面。

两个子进程**共用当前控制台窗口**，所以关掉这个窗口 = 两个服务一起停；
额外注册清理钩子，保证 Ctrl+C 时不留下占着服务端口的僵尸进程。
"""

from __future__ import annotations

import atexit
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _resolve_api_port(default: int = 8000) -> int:
    """从环境变量 / `.env` 里取 API_PORT，取不到就用 8000。

    ★ 为什么要在这里手写一个小解析而不是 `from app.core.config import get_settings`：
    启动器必须在"依赖还没装"的状态下也能跑 —— import app.* 会立刻因为
    FastAPI/Pydantic 缺失而崩掉，连"请先装依赖"的提示都发不出来。
    而 Settings 里的 api_port 此前没有任何消费者：用户改了 .env 的 API_PORT，
    启动器照样在 8000 上拉 uvicorn，配置静默无效。
    优先级照搬 pydantic-settings 的语义：真实环境变量 > .env > 代码默认。
    """
    raw = os.environ.get("API_PORT", "").strip()
    if not raw:
        env_file = ROOT / ".env"
        try:
            for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                if key.strip() == "API_PORT":
                    raw = value.strip().strip("\"'")
                    break
        except OSError:  # .env 读不了（被占用/编码怪异）不该挡住启动
            raw = ""
    if not raw:
        return default
    # 回显用户填的值前先压成 ASCII：控制台此刻还是 GBK（main() 里才 reconfigure），
    # 把 .env 里的怪字符原样 print 可能直接抛 UnicodeEncodeError，启动器反而先崩。
    shown = raw.encode("ascii", "replace").decode("ascii")
    try:
        port = int(raw)
    except ValueError:
        print(f"[!] .env/环境里的 API_PORT={shown!r} 不是合法端口，改用默认 {default}。")
        return default
    if not 1 <= port <= 65535:
        print(f"[!] API_PORT={port} 超出合法范围，改用默认 {default}。")
        return default
    return port


API_PORT = _resolve_api_port()
UI_PORT = 8501
API_URL = f"http://127.0.0.1:{API_PORT}"
UI_URL = f"http://127.0.0.1:{UI_PORT}"

# React 控制台构建过一次就由后端同源提供（`app/main.py` 的 `_mount_web_ui`），
# 这时 :8000 就是完整产品，不必再拉一个 Streamlit。
# 没构建过就退回 Streamlit 面板 —— 装不了 Node 的人不该因此连界面都看不到。
WEB_INDEX = ROOT / "web" / "dist" / "index.html"

_children: list[subprocess.Popen] = []


# ------------------------------------------------------------------ 工具函数
def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - 未就绪时的 connection refused 是预期路径
        return False


def _newest_mtime(paths) -> float:
    newest = 0.0
    for p in paths:
        try:
            newest = max(newest, p.stat().st_mtime)
        except OSError:  # 文件不存在/被占用都不该让启动器崩
            continue
    return newest


def _web_source_files() -> list[Path]:
    """影响构建产物的源文件：`web/src/**` 加上构建配置本身。"""
    src = ROOT / "web" / "src"
    files: list[Path] = []
    if src.is_dir():
        for path in src.rglob("*"):
            if path.is_file():
                files.append(path)
    for name in ("index.html", "vite.config.ts", "tailwind.config.js", "package.json", "tsconfig.json"):
        files.append(ROOT / "web" / name)
    return files


def _missing_bundled_assets() -> list[str]:
    """index.html 里引用了、但磁盘上并不存在的 `/assets/...` 文件列表。

    ★ 为什么光比时间戳不够：mtime 只能回答"产物旧了"，答不出"产物缺了"。
    index.html 还在、assets/ 却被删了或第一次 build 没跑出来时，
    时间戳检查一切正常，用户得到的却是白屏。
    """
    try:
        html = WEB_INDEX.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    missing: list[str] = []
    seen: set[str] = set()
    # 引用形态：<script type="module" src="/assets/index-xxxx.js"></script>、link href 等
    for ref in re.findall(r"""["'](/assets/[^"'\s]+)["']""", html):
        ref = ref.split("?", 1)[0].split("#", 1)[0]
        if ref in seen:
            continue
        seen.add(ref)
        if not (WEB_INDEX.parent / ref.lstrip("/")).is_file():
            missing.append(ref)
    return missing


def _warn_if_web_bundle_is_stale() -> bool:
    """构建产物比源码旧、或 index.html 引用的文件缺了就明确说出来 ——
    "改了界面但看到的还是旧的/干脆白屏"几乎总是这两个原因。

    这里只警告不自动构建：`npm run build` 要几十秒、还要 Node，双击启动的人
    未必装得起，替他默默跑掉反而更难解释。返回 True 表示产物是新的且完整。
    """
    if not WEB_INDEX.is_file():
        return False  # 没构建过：走的是 Streamlit 退回路径，那条路自己会说明

    missing = _missing_bundled_assets()
    if missing:
        print()
        print("[X] web/dist/index.html 引用了下面这些构建产物，但文件并不存在：")
        for ref in missing[:8]:
            print(f"      {ref}")
        if len(missing) > 8:
            print(f"      …… 共 {len(missing)} 个")
        print("    多半是 assets/ 被删了、或上一次 npm run build 中断了。")
        print("    不修的话浏览器会对 /assets/* 收到 404，界面一片空白。")
        print("    修复（需要本机有 Node 18+）：")
        print("      cd web && npm install && npm run build")
        print("    构建完**必须重启后端**（关掉启动窗口再双击一次）：")
        print("    /assets 的挂载是后端启动那一刻决定的，第一次构建出的 assets/ 不重启不认。")
        return False

    try:
        built = WEB_INDEX.stat().st_mtime
    except OSError:
        return False
    source = _newest_mtime(_web_source_files())
    if source <= built:
        return True
    print()
    print("[!] web/dist 里的前端构建产物比 web/src 旧 —— 界面上看不到刚加的功能是正常的。")
    print("    重新构建（需要本机有 Node 18+）：")
    print("      cd web && npm install && npm run build")
    print("    构建完要不要重启后端，取决于后端启动那一刻 assets 挂载好了没有：")
    print("    - web/dist/assets/ 当时就存在：刷新浏览器即可，静态文件是每次请求现读盘的；")
    print("    - 当时 assets/ 不存在（第一次构建）：必须关掉启动窗口再双击一次 ——")
    print("      /assets 挂载在启动时决定，不重启的话浏览器只拿到 404，界面白屏。")
    print("    改了 Python 侧（app/）的，一律要重启。")
    return False


def _wait_http(url: str, deadline: float) -> bool:
    end = time.time() + deadline
    while time.time() < end:
        for proc in _children:
            if proc.poll() is not None:  # 子进程已退出就别再等了，直接报失败
                return False
        if _http_ok(url):
            return True
        time.sleep(1.0)
    return False


def _spawn(cmd: list[str]) -> subprocess.Popen:
    proc = subprocess.Popen(cmd, cwd=str(ROOT))  # noqa: S603 - 命令由本文件拼装
    _children.append(proc)
    return proc


def _stop_all() -> None:
    for proc in reversed(_children):
        if proc.poll() is not None:
            continue
        if os.name == "nt":
            # uvicorn --reload / streamlit 都会再生子进程，terminate 只杀得掉父进程，
            # 端口就被孤儿占着 —— 必须整棵树强杀。
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
        else:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    _children.clear()


def _print_health() -> None:
    """把 /healthz 的结果翻译成人话：哪一项没就绪、该跑哪条命令修。"""
    try:
        with urllib.request.urlopen(f"{API_URL}/healthz", timeout=8) as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001
        print(f"  · 健康检查读不到：{exc}")
        return

    emb = data.get("embedding") or {}
    idx = data.get("index") or {}

    llm = f"{data.get('llm_provider', '?')} / {data.get('llm_model', '?')}"
    if not data.get("llm_configured"):
        llm += "   ⚠ 未就绪：填 key，或把 LLM_PROVIDER 改成 mock"
    print(f"  · 模型      {llm}")

    print(
        f"  · Embedding {data.get('embedding_provider', '?')}   "
        f"就绪={'是' if emb.get('ready') else '否'}"
    )
    if not emb.get("ready") and emb.get("hint"):
        print(f"    ↳ {emb['hint']}")
    print(f"  · 重排      {'开' if data.get('rerank_enabled') else '关'}")

    chunks = idx.get("chunks")
    papers = idx.get("papers")
    if idx.get("ready"):
        flag = "匹配" if idx.get("compatible") else "⚠ 与当前 embedding 不匹配"
        print(f"  · 本地索引  {papers} 篇 / {chunks} chunk（{flag}）")
        if not idx.get("compatible") and idx.get("hint"):
            print(f"    ↳ {idx['hint']}")
    else:
        print("  · 本地索引  尚未构建")
        if idx.get("hint"):
            print(f"    ↳ {idx['hint']}")
    print("  （提问时若语料不足，系统会自己联网抓 arXiv 补索引，不必手动建）")


# ------------------------------------------------------------------ 主流程
def _seed_sample_corpus_if_new() -> None:
    """全新 clone（还没有本地库）时，导入仓库自带的离线样例语料。

    为什么放在启动器里：少了这一步，第一次提问就会去联网抓 arXiv ——
    那需要合规的 User-Agent、几分钟限速等待，还要求网络可达 arXiv，
    而"双击看看它能不能跑"的人这三样都没有。
    已有本地库时 `load_sample_data.py` 自己会原样跳过，不会覆盖真实抓取的数据。
    """
    if (ROOT / "data" / "sqlite" / "paperpilot.db").exists():
        return
    print("  首次运行：导入离线样例语料（24 篇真实论文摘要，零联网）...")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "load_sample_data.py")],
        cwd=str(ROOT),
        check=False,
    )
    print()


def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # 本进程自己的输出流：中文 Windows 控制台默认 GBK，打印 ✓ 会直接抛 UnicodeEncodeError，
    # 于是"其实已经成功"的启动反而以 traceback 收场。这里刻意不 import app.*，
    # 免得依赖没装全时启动器本身崩掉、连"请先装依赖"的提示都发不出来。
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

    print("=" * 62)
    print("  PaperPilot 一键启动")
    print("=" * 62)
    print(f"  Python   {sys.executable}")
    print(f"  项目目录 {ROOT}")
    print(f"  .env     {'已加载' if (ROOT / '.env').exists() else '⚠ 不存在，将使用代码默认值'}")
    print()

    if not (ROOT / "app" / "main.py").exists():
        print("[X] 当前目录下找不到 app/main.py，启动器位置不对。")
        return 1

    _seed_sample_corpus_if_new()

    api_proc = ui_proc = None
    use_web_ui = WEB_INDEX.is_file()
    target_url = API_URL if use_web_ui else UI_URL
    # 产物过期要在**任何分支**之前说，包括"已经在跑、直接打开浏览器"那条：
    # 他双击后看到旧界面，多半就是这里，而不是"功能没做"。
    bundle_fresh = _warn_if_web_bundle_is_stale()

    # 已有实例在跑就不重复起：双击两次不该变成端口冲突
    api_up = _http_ok(f"{API_URL}/healthz")
    panel_up = use_web_ui or _http_ok(f"{UI_URL}/_stcore/health")
    if api_up and panel_up:
        print(f"[i] 检测到 PaperPilot 已经在运行，直接为你打开 {target_url} 。")
        print("    想重启请先关掉那个启动窗口。")
        if use_web_ui and not bundle_fresh:
            print("    而且上面说了：构建产物要么比源码旧、要么 index.html 引用的文件缺失，")
            print("    所以这次打开的界面**确实是上一版或干脆白屏**。")
            print("    先按上面的命令构建；提示里说了要重启后端的，别只刷新浏览器。")
        webbrowser.open(target_url)
        time.sleep(1.0)
        return 0

    atexit.register(_stop_all)

    print("[1/4] 启动后端 API ...")
    api_proc = _spawn(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)]
    )
    if not _wait_http(f"{API_URL}/healthz", deadline=180):
        code = api_proc.poll()
        print(f"[X] 后端没能在 180 秒内就绪（退出码 {code}）。")
        print(f"    常见原因：依赖没装全，或 {API_PORT} 端口被别的程序占着。")
        print(f"    排查：  先只起后端看报错 →  python -m uvicorn app.main:app --port {API_PORT}")
        _stop_all()
        return 1
    print(f"      ✓ http://127.0.0.1:{API_PORT}  文档 /docs")
    _print_health()

    print()
    if use_web_ui:
        print("[2/4] 使用 React 控制台（已由后端同源提供，无需单独起面板）...")
        print(f"      ✓ http://127.0.0.1:{API_PORT}")
    else:
        print("[2/4] 未找到 web/dist，启动 Streamlit 面板 ...")
        print("      （想要 React 控制台： cd web && npm install && npm run build）")
        ui_proc = _spawn(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "ui/streamlit_app.py",
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(UI_PORT),
                "--server.headless",
                "true",
            ]
        )
        if not _wait_http(f"{UI_URL}/_stcore/health", deadline=120):
            print(f"[X] 面板没能在 120 秒内就绪（退出码 {ui_proc.poll()}）。")
            _stop_all()
            return 1
        print(f"      ✓ http://127.0.0.1:{UI_PORT}")

    print()
    print("[3/4] 打开浏览器 ...")
    webbrowser.open(target_url)
    # 浏览器进程起来需要一点时间，太早退出会让首页空白
    time.sleep(1.5)

    print()
    print("[4/4] 已就绪。")
    print("-" * 62)
    print("  在浏览器页面里输入研究问题即可，例如：")
    print("      近两年 RAG 的重排（reranking）技术有哪些进展，各自的代价是什么？")
    print()
    print("  ★ 这个窗口别关 —— 关掉它后端和面板会一起停。")
    print("  ★ 想看实时日志就切回来看；想停止就按 Ctrl+C 或直接关窗口。")
    print("-" * 62)

    try:
        while True:
            for proc in _children:
                if proc.poll() is not None:
                    print(f"\n[X] 有个服务退出了（退出码 {proc.returncode}），正在停止其余服务。")
                    _stop_all()
                    return 1
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[i] 收到 Ctrl+C，正在停止 ...")
    finally:
        _stop_all()
    print("[i] 已全部停止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

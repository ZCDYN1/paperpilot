"""PaperPilot 一键启动器（`启动PaperPilot.bat` 调用，也可以自己跑）。

顺序：起后端 → 等 /healthz 真的返回 200 → 起面板 → 等面板就绪 → 开浏览器。
必须等就绪再开浏览器，否则用户看到的是 502 而不是界面。

两个子进程**共用当前控制台窗口**，所以关掉这个窗口 = 两个服务一起停；
额外注册清理钩子，保证 Ctrl+C 时不留下占着 8000 端口的僵尸进程。
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_PORT = 8000
UI_PORT = 8501
API_URL = f"http://127.0.0.1:{API_PORT}"
UI_URL = f"http://127.0.0.1:{UI_PORT}"

_children: list[subprocess.Popen] = []


# ------------------------------------------------------------------ 工具函数
def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 - 未就绪时的 connection refused 是预期路径
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
def main() -> int:
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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

    api_proc = ui_proc = None

    # 已有实例在跑就不重复起：双击两次不该变成端口冲突
    if _http_ok(f"{API_URL}/healthz") and _http_ok(f"{UI_URL}/_stcore/health"):
        print("[i] 检测到 PaperPilot 已经在运行，直接为你打开面板。")
        print("    想重启请先关掉那个启动窗口。")
        webbrowser.open(UI_URL)
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
        print("    常见原因：依赖没装全，或 8000 端口被别的程序占着。")
        print("    排查：  先只起后端看报错 →  python -m uvicorn app.main:app --port 8000")
        _stop_all()
        return 1
    print(f"      ✓ http://127.0.0.1:{API_PORT}  文档 /docs")
    _print_health()

    print()
    print("[2/4] 启动交互面板 ...")
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
    webbrowser.open(UI_URL)
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

"""轻量自检：不启动服务器，直接验证应用能否装配、`/healthz` 是否正常。

用途：
    python scripts/check_health.py

适合在装完依赖后立刻确认「地基是活的」，
也适合放进 CI 做冒烟测试（无需占端口、无需起后台进程）。
"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from app.core.console import force_utf8_output
from app.main import app


def main() -> int:
    # 用 with 包裹，确保 lifespan（启动/关闭钩子）真的被执行到
    with TestClient(app) as client:
        resp = client.get("/healthz")

    print(f"HTTP {resp.status_code}")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))

    if resp.status_code != 200:
        print("✗ /healthz 未返回 200，地基没立住", file=sys.stderr)
        return 1
    if resp.json().get("status") != "ok":
        print("✗ /healthz 的 status 不是 ok", file=sys.stderr)
        return 1

    print("✓ 地基正常：应用可装配 · lifespan 可执行 · /healthz 可用")
    return 0


if __name__ == "__main__":
    force_utf8_output()
    sys.exit(main())

"""控制台输出兜底：把 stdout/stderr 强制成 UTF-8。

**为什么需要它**：项目里的 CLI 输出大量使用中文与符号（✓ ⚠️ ①）。
在 Windows 中文系统上，Python 默认按 GBK 编码控制台流，
遇到无法映射的字符会抛 `UnicodeEncodeError` —— 于是一个**已经执行成功**的脚本
（比如索引已建好）反而以 traceback 和非零退出码收场，
新用户第一次跑就会误判成"项目坏了"。

这不是样式问题，是可复现性问题，所以在每个 CLI 入口显式调用一次。
"""

from __future__ import annotations

import sys


def force_utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # 被重定向成非 TextIOWrapper（如测试捕获流）时跳过
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


__all__ = ["force_utf8_output"]

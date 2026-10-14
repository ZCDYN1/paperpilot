"""F13 单篇深挖（产品定义 §5）。

对外只暴露三个名字，路由层 import 面越窄越好：
- `build_paper_detail`：详情（纯读库）
- `ask_about_paper`：深度问答（只在本篇 chunks 上，硬校验兜底）
- `PaperNotFoundError`：404 映射用

F15 的上传入口在 `app/ingest/pdf.py`，不在这里 —— 一个是"读"一个是"写"，
混进同一个包会让路由层的依赖平白长出一整条 ingest 链路。
"""

from app.papers.detail import (
    INSUFFICIENT_ANSWER,
    PaperNotFoundError,
    ask_about_paper,
    build_paper_detail,
)

__all__ = [
    "INSUFFICIENT_ANSWER",
    "PaperNotFoundError",
    "ask_about_paper",
    "build_paper_detail",
]

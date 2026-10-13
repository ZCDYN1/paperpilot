"""LangGraph 多 Agent 编排层（架构 §4）。

对外只需要两个入口：
- `build_graph(checkpointer)`：装配图。HITL 中断依赖 checkpointer 才能续跑。
- `open_graph()`：异步上下文，自动管理 `AsyncSqliteSaver` 生命周期。

状态与节点分别见 `state.py` / `nodes/`。
"""

from app.agents.graph import (
    build_graph,
    decide_after_verify,
    open_checkpointer,
    open_graph,
    route_after_corpus,
    route_revision_workers,
    thread_config,
)
from app.agents.state import Evidence, ResearchState, SubQuestion

__all__ = [
    "Evidence",
    "ResearchState",
    "SubQuestion",
    "build_graph",
    "decide_after_verify",
    "open_checkpointer",
    "open_graph",
    "route_after_corpus",
    "route_revision_workers",
    "thread_config",
]

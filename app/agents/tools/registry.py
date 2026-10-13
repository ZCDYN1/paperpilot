"""工具注册表。

**为什么需要一个注册表，而不是各处 `from ... import hybrid_search`？**
两个理由：
1. schema 是**对外契约**：把工具集合集中声明，才能一眼看出"Agent 到底能做什么"，
   也才能在 API 层做能力开关（例如禁用 `arxiv_search` 的纯离线部署）。
2. `create_react_agent` / `bind_tools` 这类框架入口都要一个 list。
   散落导入会让"新工具忘了挂上去"变成一个静默失效的功能（写了但没接进图）。
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from app.agents.tools.arxiv_search import arxiv_search
from app.agents.tools.hybrid_search import hybrid_search
from app.agents.tools.paper_meta import get_paper_detail

# 名称 → 工具。名称必须与 `@tool("name")` 里的一致，否则 `get_tool(name)` 会漏
TOOL_REGISTRY: dict[str, BaseTool] = {
    hybrid_search.name: hybrid_search,
    get_paper_detail.name: get_paper_detail,
    arxiv_search.name: arxiv_search,
}

# 默认可见的工具集：**不含 arxiv_search**。
# 理由：arxiv_search 会打外网且受 3s 限速，交给 LLM 自由调用极易把一次研究拖到分钟级；
# 抓取属于 `ensure_corpus` 的职责，应由编排层决定"要不要抓"，而不是让模型随口决定。
DEFAULT_TOOL_NAMES: tuple[str, ...] = ("hybrid_search", "get_paper_detail")


def get_tool(name: str) -> BaseTool:
    try:
        return TOOL_REGISTRY[name]
    except KeyError as exc:  # 明确报错，而不是让调用方拿到 None 后莫名 AttributeError
        raise KeyError(f"未注册的工具：{name!r}；已注册：{sorted(TOOL_REGISTRY)}") from exc


def get_tools(names: tuple[str, ...] | list[str] | None = None) -> list[BaseTool]:
    """按名称取工具列表；不传则返回默认集合。"""
    if names is None:
        names = DEFAULT_TOOL_NAMES
    return [get_tool(n) for n in names]


__all__ = ["DEFAULT_TOOL_NAMES", "TOOL_REGISTRY", "get_tool", "get_tools"]

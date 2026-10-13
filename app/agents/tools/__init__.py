"""Agent 工具集（架构 §3 / §4.2）。

工具是 Agent 与"外部世界"（检索层 / 数据库 / arXiv）之间唯一的接口，
因此每个工具的**入参都必须用 Pydantic 强校验**（架构 §10「工具侧」）。
"""

from app.agents.tools.registry import (
    DEFAULT_TOOL_NAMES,
    TOOL_REGISTRY,
    get_tool,
    get_tools,
)

__all__ = ["DEFAULT_TOOL_NAMES", "TOOL_REGISTRY", "get_tool", "get_tools"]

"""LangGraph 节点集合（架构 §4.2）。

每个节点都是 `async def xxx_node(state) -> dict`，只返回**增量**（要写回的 channel），
不返回整个 state —— 返回全量会让 reducer 把同一份数据重复累加
（例如 `retrieved` 被加两遍，检索结果凭空翻倍且没有任何报错）。
"""

from app.agents.nodes.comparator import compare_node
from app.agents.nodes.ensure_corpus import ensure_corpus_node
from app.agents.nodes.finalize import finalize_node
from app.agents.nodes.planner import plan_node
from app.agents.nodes.reader import read_invocations, read_node, reset_read_counter
from app.agents.nodes.retriever_worker import retrieve_node
from app.agents.nodes.revise import revise_node
from app.agents.nodes.verifier import verify_node
from app.agents.nodes.writer import write_node

__all__ = [
    "compare_node",
    "ensure_corpus_node",
    "finalize_node",
    "plan_node",
    "read_invocations",
    "read_node",
    "reset_read_counter",
    "retrieve_node",
    "revise_node",
    "verify_node",
    "write_node",
]

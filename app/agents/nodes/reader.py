"""`read` 节点 —— 汇总去重 + 摘要级结构化抽取（架构 §4.2）。

## 它为什么是"并行汇合陷阱"的落点（架构 §4.3-④）
`retrieve` 被 Send 扇出 N 份并行跑，它们都连一条普通边到 `read`。
担心的是"`read` 被触发 N 次，Reader 的 LLM 调用凭空翻 N 倍"。

**实测结论（langgraph 1.2.14，子问题数=3）**：
```
retrieve calls = 3
read calls     = 1      ← 普通边 + reducer 下，汇合节点只执行一次
retrieved len  = 3      ← N 份产出被 operator.add 正确聚合
```
LangGraph 的 Pregel 超级步语义保证了"同一超级步内所有并行分支完成后，
下游节点**只被调度一次**"。这与 langgraph 官方 map-reduce 示例
（`generate_topics` → Send → `generate_joke` → `best_joke`）的行为一致。
真正会翻倍的是**在汇合节点上用普通 list channel**：那会直接抛
`InvalidUpdateError`，而不是静默多跑。

## 但仍然是"必须处理"的：幂等哨兵（§4.3-④ 方案 2）
`read` 里加了一道 `stage` 哨兵（见 `_already_read`）。它不是本次的补救措施，
而是**回归护栏**：一旦将来有人把图改成"两个来源都能到达 read"（例如给 read
再接一条普通边），`read` 就会在一个超级步内被多次调度；此时哨兵能让重复执行
变成廉价 no-op，而不是把 token 悄悄翻倍。同时它会打 warning，
让"图结构被改坏了"这件事在日志里显形，而不是等账单出来才发现。

（另两种候选方案没选的理由：① 显式 barrier 需要额外 counter 通道 + 条件边回边，
在 Pregel 超级步语义下反而是多余的复杂度；③ 取消下游汇合会让每个分支各自 inline
一次抽取，丢掉跨子问题全局去重 —— 而"去重后共 24 篇进入抽取"正是产品要展示的数字。）
"""

from __future__ import annotations

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._common import (
    chunk_index,
    dedup_papers,
    node_event,
    structured_invoke,
)
from app.agents.prompts import load_prompt, render_papers_block
from app.agents.schemas import ReadOutput
from app.agents.state import STAGE_READ, Evidence, ResearchState
from app.core.config import get_settings

logger = structlog.get_logger(__name__)

# 供测试与观测使用：记录 read 节点被实际执行的次数。
# 这是"并行汇合只执行一次"这条结论的**实测口径**（架构 §4.3-④ 要求给出实测数字）。
#
# ⚠️ 用可变容器（dict）而不是裸 int：`from ... import READ_INVOCATIONS` 对 int 而言是
# **值拷贝**，模块内 `+= 1` 不会反映到导入方，于是计数器永远是 0 ——
# 一个"看起来接好了、其实没接上"的观测点，比没有观测点更误导。
_READ_STATS: dict[str, int] = {"invocations": 0}


def reset_read_counter() -> None:
    """复位计数器（每次验证/测试运行前调用，避免被上一次的数字污染）。"""
    _READ_STATS["invocations"] = 0


def read_invocations() -> int:
    """read 节点自上次复位以来被执行的次数。"""
    return _READ_STATS["invocations"]


def _already_read(state: ResearchState) -> bool:
    """幂等哨兵：本轮是否已经抽取过。

    判据同时看 `stage` 与 `readings`：
    - `stage == "read"` 说明本轮的 read 已经跑完；
    - `readings` 非空才说明它**产出了东西**（否则可能是一次失败的空跑，
      这种情况必须允许重试，不能被哨兵挡住）。
    `revise` 节点会把 `stage` 改成 `revising`，从而自动解除哨兵、放行第二轮抽取。
    """
    return state.get("stage") == STAGE_READ and bool(state.get("readings"))


async def read_node(state: ResearchState) -> dict:
    """汇合全部检索结果 → 去重 → 对 top-k 论文做结构化抽取。"""
    _READ_STATS["invocations"] += 1

    if _already_read(state):
        logger.warning("read.skipped_by_sentinel", invocations=_READ_STATS["invocations"])
        return {}

    settings = get_settings()
    retrieved = state.get("retrieved") or []
    papers = dedup_papers(retrieved)
    # 只对 top-k 论文做抽取：prompt 长度是真实成本，而"去重后 24 篇"
    # 里真正需要精读的是重排后的头部（产品定义 §4 的产出物 A 就是 8 篇对比表）。
    top = papers[: settings.rerank_top_n]

    if not top:
        logger.warning("read.no_papers")
        return {
            "readings": [],
            "stage": STAGE_READ,
            "events": [node_event("thought", node="read", delta="未检索到可用论文，跳过抽取。")],
        }

    block = render_papers_block(top, max_chunks_per_paper=settings.max_chunks_per_paper)
    messages = [
        SystemMessage(content=load_prompt("reader")),
        HumanMessage(
            content=(
                f"研究问题：{state.get('question', '')}\n\n"
                f"以下是本次检索命中的 {len(top)} 篇论文片段，请抽取「论断 → 原文证据」对：\n\n"
                f"{block}"
            )
        ),
    ]

    # 单篇失败不拖垮全流程（架构 §11）：这里是一批论文一次调用，
    # 所以失败时把整批计入 failed_papers 并**降级继续**（readings 为空），
    # 而不是让异常冒泡把整次研究打断 —— 用户拿到"部分结果"远好于拿到一个报错。
    known_chunks = chunk_index(retrieved)
    try:
        parsed, usage = await structured_invoke(ReadOutput, messages)
    except Exception as exc:  # noqa: BLE001
        failed = [p["arxiv_id"] for p in top]
        logger.warning("read.extract_failed", error=str(exc)[:200], papers=len(failed))
        return {
            "readings": [],
            "failed_papers": failed,
            "stage": STAGE_READ,
            "events": [
                node_event(
                    "error", node="read", code="extract_failed", message=str(exc)[:200],
                    recoverable=True,
                )
            ],
        }

    readings: list[dict] = []
    citations: list[Evidence] = []
    dropped = 0
    for item in parsed.claims:
        key = (item.paper_id, item.chunk_id) if item.chunk_id is not None else None
        chunk = known_chunks.get(key) if key else None
        if chunk is None:
            # 反幻觉（架构 §10 输出侧）：引用必须能在本次检索结果集合里找到对应 chunk，
            # 找不到就是幻觉，**剔除**而不是"存疑保留"——存疑的引用最终一定会被写进报告。
            dropped += 1
            logger.warning("read.hallucinated_citation", paper=item.paper_id, chunk=item.chunk_id)
            continue
        readings.append(
            {
                "paper_id": item.paper_id,
                "chunk_id": item.chunk_id,
                "claim": item.claim,
                "quote": item.quote,
                "locator": item.locator or chunk.get("section", "abstract"),
                # source 以**检索层记录的真实来源**为准，不以模型自述为准：
                # 模型很容易把摘要级证据美化成"正文"。证据强度是产品承诺，不能由模型决定。
                "source": chunk.get("source", "abstract"),
            }
        )
        citations.append(
            {
                "claim": item.claim,
                "chunk_id": int(item.chunk_id),
                "paper_id": item.paper_id,
                "quote": item.quote,
                "locator": item.locator or chunk.get("section", "abstract"),
                "source": chunk.get("source", "abstract"),
                "verified": False,  # 由 verify 节点写回
            }
        )

    logger.info("read.done", papers=len(top), claims=len(readings), dropped=dropped)

    return {
        "readings": readings,
        "citations": citations,
        "stage": STAGE_READ,
        "token_usage": usage,
        "events": [
            node_event(
                "thought",
                node="read",
                delta=f"去重后 {len(top)} 篇论文进入抽取，产出 {len(readings)} 条「论断 → 证据」对。",
            ),
        ],
    }


__all__ = ["read_invocations", "read_node", "reset_read_counter"]

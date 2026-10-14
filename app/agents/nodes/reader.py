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
    dedup_papers,
    node_event,
    prioritize_unread,
    structured_invoke,
)
from app.agents.nodes.evidence import (
    build_chunk_lookup,
    build_paper_chunks,
    resolve_citation,
)
from app.agents.prompts import load_prompt, render_papers_block
from app.agents.schemas import ReadOutput
from app.agents.state import STAGE_READ, Evidence, ResearchState
from app.core.config import get_settings
from app.memory.long_term import safe_read_paper_ids

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
    # 长期记忆（产品定义 F10「下次不重复推荐」）：已精读过的论文让出候选位。
    # ⚠️ 排除**只作用在"派哪几篇进精读"**，`state["retrieved"]` 一篇都不删。
    #    理由不是"已读论文还会被引用"—— 它没进 prompt，模型本来也看不见它。
    #    真正的理由是 `retrieved` 另有五个消费者：verify 用它建"合法 chunk 全集"
    #    （`chunk_index`）、compare / write 用它做对比与成稿、finalize 用它算
    #    "本次覆盖 N 篇"。在选篇这一步顺手删它，等于同时改掉了校验口径和报告里的数字。
    # 已读清单跨会话读取（不传 session_id）：`session_id = run_id` 且记忆在 finalize
    # 才写入，按会话过滤必然为空 —— 详见 app/memory/long_term.py 的★小节。
    read_ids = await safe_read_paper_ids()
    ordered = prioritize_unread(papers, read_ids)
    # 只对 top-k 论文做抽取：prompt 长度是真实成本，而"去重后 24 篇"
    # 里真正需要精读的是重排后的头部（产品定义 §4 的产出物 A 就是 8 篇对比表）。
    # ⚠️ 用 `read_top_k` 而不是 `rerank_top_n`：后者是 Settings 里早就声明、但**没有任何
    #    代码消费**的死配置 —— 用户照 README 调它压 token 成本会毫无效果。两者语义确实不同
    #    （精读几篇 vs 每个子问题留几篇候选 / 对比表几行），所以在这里接上它。
    top = ordered[: settings.read_top_k]
    if read_ids:
        # reused>0 说明未读论文不够、已读的补位回来了 —— 这是设计好的降级方向，
        # 但要可观测：否则"排除已读"到底是生效了还是没生效，只能靠猜。
        logger.info(
            "read.prioritized_unread",
            known_read=len(read_ids),
            selected=len(top),
            reused_read=sum(1 for p in top if str(p.get("arxiv_id")) in read_ids),
        )

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

    # 反幻觉闸门 + 引文重锚（见 nodes/evidence.py）：
    # `(paper_id, chunk_id)` 对不上时，先按逐字引文在同篇片段里重锚；
    # 重锚不成功才算幻觉剔除。这样"模型打错一个自增编号"不再等于"编造证据"。
    papers_by_id = build_paper_chunks(retrieved)
    chunk_lookup = build_chunk_lookup(retrieved)
    rendered_chunk_ids: set[int] = set()
    for paper in top:
        for chunk in (paper.get("chunks") or [])[: settings.max_chunks_per_paper]:
            if chunk.get("chunk_id") is not None:
                rendered_chunk_ids.add(int(chunk["chunk_id"]))

    # 单篇失败不拖垮全流程（架构 §11）：这里是一批论文一次调用，
    # 所以失败时把整批计入 failed_papers 并**降级继续**（readings 为空），
    # 而不是让异常冒泡把整次研究打断 —— 用户拿到"部分结果"远好于拿到一个报错。
    try:
        parsed, usage = await structured_invoke(ReadOutput, messages)
        if parsed is None:
            # `include_raw=True` 下解析失败**不抛异常**，只是 `parsed=None`。
            # 必须在这里当失败处理：否则下一行 `parsed.claims` 抛 AttributeError，
            # 把"模型输出不合规"这个可降级的事件变成整次研究 500。
            raise ValueError("结构化输出解析失败（模型未按要求产出 claims）")
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
    recovered = 0
    recovered_by: dict[str, int] = {}
    drop_reasons: dict[str, int] = {}
    for item in parsed.claims:
        hit = resolve_citation(
            papers_by_id,
            paper_id=item.paper_id,
            chunk_id=item.chunk_id,
            quote=item.quote,
            rendered_chunk_ids=rendered_chunk_ids,
            chunk_lookup=chunk_lookup,
            # read 侧只允许锚到"确实渲染给模型看过"的片段：它没法引用自己没读过的东西
            restrict_to_rendered=True,
        )
        if not hit.ok:
            # 反幻觉（架构 §10 输出侧）：引文逐字找不到对应 chunk，就是幻觉，
            # **剔除**而不是"存疑保留"——存疑的引用最终一定会被写进报告。
            dropped += 1
            drop_reasons[hit.reason or "unknown"] = drop_reasons.get(hit.reason or "unknown", 0) + 1
            # 带上引文开头：上一轮就是因为日志里只有 id、没有 quote，
            # 才分不清"编号写错"与"引文被改写"这两种修法完全不同的失败，绕了一圈。
            logger.warning(
                "read.hallucinated_citation",
                paper=item.paper_id,
                chunk=item.chunk_id,
                reason=hit.reason,
                quote_head=item.quote[:60],
            )
            continue
        chunk = hit.chunk
        paper_id = hit.paper_id or item.paper_id
        if hit.recovered:
            # id 写错了但引文是真的：救回来，并把归属改成引文真正所在的 chunk。
            # `how` 进日志是为了能在真实分布上统计"哪条路径在生效"，而不是靠猜有效性。
            recovered += 1
            recovered_by[hit.how] = recovered_by.get(hit.how, 0) + 1
            logger.info(
                "read.citation_reanchored",
                paper=paper_id,
                claimed_paper=item.paper_id,
                claimed_chunk=item.chunk_id,
                actual_chunk=chunk.get("chunk_id"),
                how=hit.how,
            )
        readings.append(
            {
                "paper_id": paper_id,
                "chunk_id": chunk.get("chunk_id"),
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
                "chunk_id": int(chunk["chunk_id"]),
                "paper_id": paper_id,
                "quote": item.quote,
                "locator": item.locator or chunk.get("section", "abstract"),
                "source": chunk.get("source", "abstract"),
                "verified": False,  # 由 verify 节点写回
                # 定位路径随证据一路流到 finalize 落库（citations.resolution_how）：
                # 此前它只进日志（read.done），事后想查"这条引用当初怎么救回来的"
                # 只能 grep 服务端日志。键是新增的，不参与 merge_evidence 的去重键。
                "resolution_how": hit.how,
            }
        )

    logger.info(
        "read.done",
        papers=len(top),
        claims=len(readings),
        dropped=dropped,
        reanchored=recovered,
        recovered_by=recovered_by or None,
        drop_reasons=drop_reasons or None,
    )

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
            # 救回统计进 SSE（此前只在上面的 read.done 日志里）：面板时间线要能直接
            # 回答"本次救回了几条、靠哪条路径"。**rescued=0 也照发** —— 根因已修掉时
            # 这条事件的常态就是 0，只在非零时出现等于把回归藏起来。
            node_event(
                "citation_rescue",
                node="read",
                checked=len(parsed.claims),
                rescued=recovered,
                dropped=dropped,
                recovered_by=recovered_by,
                drop_reasons=drop_reasons,
            ),
        ],
    }


__all__ = ["read_invocations", "read_node", "reset_read_counter"]

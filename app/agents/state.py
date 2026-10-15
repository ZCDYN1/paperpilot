"""LangGraph 状态定义（架构文档 §4.1）。

这个文件只有一件事要做对：**把"哪些字段是并行写入、因而必须有 reducer"划清楚**。
划错的后果不是"报个错"那么友好 —— 少了 reducer 会抛 `InvalidUpdateError`（还算好），
而 reducer 写错（比如该求和却用了覆盖）会**静默丢数据**，例如 token 统计少一半、
检索结果只剩最后一个分支的，这种 bug 在测试里常常看不出来。

三个 reducer 的存在理由：
| 字段 | 谁在并行写 | 为什么不能覆盖 |
|---|---|---|
| `retrieved` | N 个 retrieve 分支 | 覆盖的话只剩最后一个子问题的结果 |
| `readings` | read（revision 循环里会写多轮） | 覆盖会丢掉被回退轮次已抽取的论断 |
| `failed_papers` | 多个分支各自失败 | 覆盖会掩盖"还有哪些篇没读成" |
| `token_usage` | 所有分支 | 按 key 求和，否则并行分支的消耗互相覆盖 |
| `events` | 所有节点 | 累积成一条事件流，供 SSE 发射器消费 |
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class SubQuestion(TypedDict):
    """Planner 拆解出的子问题。

    `keywords` **必须是英文术语**（含同义/缩写扩展），因为主用法是
    「中文提问 → 检索英文论文」，而 BM25 的词面匹配跨不过中英语表 ——
    用中文关键词打英文索引，召回接近随机（架构 §4.1 注①）。
    校验落在 `app.schemas.research.SubQuestion`（Pydantic），这里只做类型声明。
    """

    id: str
    text: str
    keywords: list[str]
    category: str | None


class Evidence(TypedDict, total=False):
    """「论断 → 原文片段」的证据对。产品的核心承诺（引用可溯源）挂在它身上。"""

    claim: str
    chunk_id: int
    paper_id: str
    quote: str
    locator: str      # 章节 / 页码
    source: str       # html=正文原文 / abstract=摘要级降级证据，UI 必须如实展示
    verified: bool    # 由 verifier 写回；None 表示尚未校验
    verify_note: str  # verifier 的说明：失败原因，或"靠抽取残留容忍通过"的审计痕迹
                      # （精确通过仍为空串）；finalize 原样落进 citations.note
    # ★ verifier 重锚后**真正落在哪个片段上**（只有定位被纠正过时才存在）。
    # 为什么不直接改写 chunk_id：`merge_evidence` 的去重键是
    # `(paper_id, chunk_id, claim)`，改了键就等于"新增一条"而不是"覆盖这一条"，
    # 于是状态里会同时留下纠正前后两条 —— 那条纠正前的会在落库时被外键保护
    # 写成 NULL，等于凭空多出一条"看起来失败"的引用。
    # 所以纠正结果另记一笔，由 `finalize` 落库时优先采用。
    resolved_chunk_id: int
    resolved_paper_id: str
    resolved_source: str
    resolved_locator: str


def merge_token_usage(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """按 key 求和。并行分支各报各的，必须在 reducer 里累加而不是覆盖。"""
    keys = set(a) | set(b)
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in keys}


def merge_evidence(a: list[Evidence], b: list[Evidence]) -> list[Evidence]:
    """证据合并（**新值覆盖同键旧值**）。

    为什么 `citations` 也需要 reducer：`revise` 之后会再扇出一次 retrieve → read，
    新一轮的证据必须**追加**到已有证据上。若用普通 list，第二轮会覆盖第一轮。

    ⚠️ 同键冲突时必须是"新值覆盖"而不是"跳过"：`verify` 节点会把
    `verified` 字段写回同一批引用。若用"跳过"，校验结果会被**旧的无 verified 版本**顶掉，
    表现是"校验明明跑了，引用准确率却一直是 0"—— 这类静默失效最难查。
    去重键取 (paper_id, chunk_id, claim)：同一处原文支撑不同论断是合法的，不算重复。
    """
    merged: list[Evidence] = list(a)
    pos = {(e.get("paper_id"), e.get("chunk_id"), e.get("claim")): i for i, e in enumerate(merged)}
    for item in b:
        key = (item.get("paper_id"), item.get("chunk_id"), item.get("claim"))
        if key in pos:
            merged[pos[key]] = item
        else:
            pos[key] = len(merged)
            merged.append(item)
    return merged


class ResearchState(TypedDict, total=False):
    """全图共享状态。

    `total=False` 是**刻意的**：图入口只喂 `session_id / run_id / question` 三个字段，
    其余字段由图自身逐步填出来。若声明成 required，`ainvoke` 的入参就得把
    所有 key 补成 None，反而让"哪些字段是运行期产出的"变得模糊。
    """

    # ---- 入参 ----
    session_id: str
    run_id: str
    question: str
    filters: dict[str, Any]        # metadata 过滤（分类/年份），从 API 透传下来
    # ★ 下面三个是**请求级**开关。它们必须作为 state 通道存在，而不是让节点各自去读
    #   `settings`：`Settings` 是进程级的（`get_settings()` 带 lru_cache），
    #   一次提问的 top_k/rerank/语言属于这次请求，写进 state 才能被 checkpoint 回放，
    #   也才不会让并发的两个 run 互相串配置。
    top_k: int                    # 每个子问题重排后保留几篇；None 时节点退回 settings.RERANK_TOP_N
    rerank: bool                  # 是否走交叉编码器重排（延迟敏感可关；消融表就是靠它量化权衡）
    language: str                 # 成稿输出语言（"zh" / "en"，产品定义 F17）

    # ---- Planner ----
    sub_questions: list[SubQuestion]
    plan_rationale: str

    # ★ Send 扇出时注入给「单个分支」的私有字段。
    #   `route_to_workers` 通过 Send 把它塞进某个 retrieve 任务的入参；
    #   Send 只作用于单分支，不会污染全局 state。
    # ⚠️ 反过来也成立，而且更容易忘：**Send 的 payload 就是那个分支的全部 state**，
    #   全局通道不会自动带进来。所以分支要读的配置（filters / top_k / rerank / language）
    #   必须由 `graph._worker_payload` 显式放进去 —— 漏掉的表现不是报错，而是
    #   "用户设的过滤条件在检索那一步凭空消失"。
    sub_question: SubQuestion
    # revise 扇出用的补充子问题（与 sub_questions 分开，避免污染原计划）
    revision_sub_questions: list[SubQuestion]

    # ---- 语料保障 ----
    corpus_ready: bool
    corpus_stats: dict[str, Any]

    # ---- 并行分支产出（reducer 聚合）----
    retrieved: Annotated[list[dict[str, Any]], operator.add]
    readings: Annotated[list[dict[str, Any]], operator.add]
    failed_papers: Annotated[list[str], operator.add]
    events: Annotated[list[dict[str, Any]], operator.add]

    # ---- 成稿 ----
    citations: Annotated[list[Evidence], merge_evidence]
    verification: dict[str, Any]    # {"passed": bool, "unsupported": [...], "score": float}
    draft: str
    # 结构化对比表（产品定义的核心产出物 A）。由 comparator 节点产出。
    # 硬要求：**字段缺失一律留空，不允许编造** —— 这是本产品与
    # "普通 RAG 问答机器人"的分界线之一。
    comparison: list[dict[str, Any]]

    # ---- 记忆 ----
    # plan 阶段召回的相关长期记忆（已读论文 / 关注的子方向 / 输出偏好），
    # 由 planner 写入、finalize 更新，让"第二次问同一方向"不重复推荐。
    memory_context: list[str]

    # ---- 控制 ----
    revision_count: int
    # ★ 粗粒度阶段标记：兼作 read 节点的**幂等哨兵**（架构 §4.3-④ 方案 2）。
    stage: str

    # ---- 观测 ----
    token_usage: Annotated[dict[str, int], merge_token_usage]
    # 按节点细分的 token 用量：{node_name: {prompt_tokens, completion_tokens, ...}}
    # 刻意**不加 reducer**：它由 token_meter 汇总后一次性写入，
    # 不存在"并行分支各写各的"场景，用 reducer 反而会因为值是嵌套 dict 而报错。
    token_by_node: dict[str, Any]


# 阶段常量：用常量而不是裸字符串，避免"改了一处忘了另一处"导致哨兵永远不生效
STAGE_PLANNED = "planned"
STAGE_CORPUS = "corpus"
STAGE_RETRIEVED = "retrieved"
STAGE_READ = "read"
STAGE_VERIFIED = "verified"
STAGE_REVISING = "revising"
STAGE_WRITTEN = "written"
STAGE_DONE = "done"

__all__ = [
    "Evidence",
    "ResearchState",
    "STAGE_CORPUS",
    "STAGE_DONE",
    "STAGE_PLANNED",
    "STAGE_READ",
    "STAGE_RETRIEVED",
    "STAGE_REVISING",
    "STAGE_VERIFIED",
    "STAGE_WRITTEN",
    "SubQuestion",
    "merge_evidence",
    "merge_token_usage",
]

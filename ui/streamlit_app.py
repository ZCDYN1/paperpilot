"""PaperPilot 交互面板（架构 §13）。

三栏布局：控制栏 / Agent 时间线 + 报告 / 引用证据链。

## 两个实现要点
1. **消费 SSE 用 `requests(stream=True)` 逐行解析**，把事件 dispatch 到不同的
   `st.empty()` 占位符 —— 不要用 `st.write` 全量重绘，那样整页会闪。
2. **HITL 需要跨 rerun 保状态**：Streamlit 每次交互都重跑整个脚本，
   所以 `run_id` 与"待确认的子问题"必须放进 `st.session_state`，
   否则用户点一下按钮，上下文就没了。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import requests
import streamlit as st

# Streamlit 把 sys.path[0] 设成 ui/，补上项目根才能 import app.*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="PaperPilot", page_icon="📚", layout="wide")

# --------------------------------------------------------------- 状态初始化
for key, default in {
    "run_id": None,
    "awaiting_resume": False,
    "pending_sub_questions": [],
    "timeline": [],
    "report": "",
    "citations": [],
    "retrieval_hits": [],
    "verification": None,
    "last_error": None,
    # 下面三项只存在于 state，不在 SSE 事件流里 —— 流结束后补拉一次
    "comparison": [],
    "token_by_node": {},
    "memory_context": [],
}.items():
    st.session_state.setdefault(key, default)


# ================================================================ SSE 消费
def consume_stream(url: str, payload: dict | None, timeline_box, report_box, meta_box) -> None:
    """消费一次 SSE 流，实时渲染到给定占位符。"""
    st.session_state.timeline = []
    st.session_state.report = ""
    st.session_state.citations = []
    st.session_state.retrieval_hits = []
    st.session_state.verification = None
    st.session_state.awaiting_resume = False
    st.session_state.last_error = None

    event_type = ""
    try:
        with requests.post(url, json=payload, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("event:"):
                    event_type = raw.split(":", 1)[1].strip()
                    continue
                if not raw.startswith("data:"):
                    continue

                try:
                    data = json.loads(raw.split(":", 1)[1].strip())
                except json.JSONDecodeError:
                    continue

                # ---- 各类事件的渲染 ----
                if event_type == "run_started":
                    st.session_state.run_id = data.get("run_id")

                elif event_type == "node_started":
                    meta_box.caption(f"⏳ 正在执行节点：`{data.get('node')}`")
                    st.session_state.timeline.append(data.get("label") or data.get("node", ""))

                elif event_type == "plan_created":
                    st.session_state.pending_sub_questions = data.get("sub_questions") or []
                    timeline_box.markdown(_render_timeline(st.session_state.timeline, data))

                elif event_type == "corpus":
                    st.session_state.timeline.append(
                        f"语料：抓取 {data.get('fetched')} / 缓存 {data.get('cached')}"
                        f" / 共 {data.get('total')}（充足={data.get('corpus_ready')}）"
                    )
                    timeline_box.markdown(_render_timeline(st.session_state.timeline, None))

                elif event_type == "retrieval":
                    hits = data.get("hits") or []
                    st.session_state.retrieval_hits.extend(hits)
                    st.session_state.timeline.append(f"检索命中 {len(hits)} 篇")

                elif event_type == "verification":
                    st.session_state.verification = data
                    st.session_state.timeline.append(
                        f"引用校验：{'通过' if data.get('passed') else '不通过'}"
                        f"（score={data.get('score')}，第 {data.get('revision_count')} 次）"
                    )
                    timeline_box.markdown(_render_timeline(st.session_state.timeline, None))

                elif event_type in ("thought", "token"):
                    st.session_state.report += data.get("delta") or ""
                    report_box.markdown(st.session_state.report[-4000:])

                elif event_type == "final_report":
                    st.session_state.report = data.get("markdown") or st.session_state.report
                    # 引用链只在这一个事件里出现（`verify` 只发校验结论，不发逐条明细）。
                    # 存下来才能在报告下方展示"哪些条目最终没通过校验"，
                    # 而不是让用户只看到一个总分。
                    st.session_state.citations = data.get("citations") or []
                    report_box.markdown(st.session_state.report)

                elif event_type == "interrupted":
                    st.session_state.awaiting_resume = True
                    payload_sub = (data.get("payload") or {}).get("sub_questions") or []
                    if payload_sub:
                        st.session_state.pending_sub_questions = payload_sub
                    st.session_state.timeline.append(
                        f"⏸ 已暂停（{data.get('at')}）：{data.get('reason')}"
                    )
                    timeline_box.markdown(_render_timeline(st.session_state.timeline, None))

                elif event_type == "error":
                    st.session_state.last_error = data
                    st.session_state.timeline.append(f"✗ 错误：{data.get('message')}")
                    timeline_box.markdown(_render_timeline(st.session_state.timeline, None))

                elif event_type == "token_usage":
                    meta_box.caption(
                        f"💰 token={data.get('total')} · 估算成本={data.get('cost_estimate')}"
                    )

        # 流结束后补拉一次详情：对比表 / 分节点 token / 记忆召回
        # **不在事件流里** —— 它们体量偏大且属于"最终产物"，走查询接口更合适。
        run_id = st.session_state.run_id
        if run_id:
            try:
                detail = requests.get(f"{API_BASE}/api/v1/research/{run_id}", timeout=30).json()
                st.session_state.comparison = detail.get("comparison") or []
                st.session_state.token_by_node = detail.get("token_by_node") or {}
                st.session_state.memory_context = detail.get("memory_context") or []
            except Exception:  # noqa: BLE001 - 补拉失败不该影响已经渲染好的内容
                pass

    except Exception as exc:  # noqa: BLE001
        st.session_state.last_error = {"message": str(exc)}
        st.error(f"请求失败：{exc}")


def _render_timeline(lines: list[str], plan: dict | None) -> str:
    out = ["**Agent 时间线**"]
    out += [f"- {ln}" for ln in lines[-24:]]
    if plan:
        out.append("")
        out.append(f"**拆解出 {len(plan.get('sub_questions') or [])} 个子问题**")
        for sq in plan.get("sub_questions") or []:
            out.append(f"- {sq.get('text', '')}")
            out.append(f"  - keywords: `{sq.get('keywords')}`")
    return "\n".join(out)


# 对比表的列定义（顺序即展示顺序）。空值统一渲染成「—」——
# 隐藏空值会让用户误以为"这篇论文没有局限"，而事实是"我们没能从证据里抽出来"，
# 两者含义完全不同，后者必须让用户看见。
COMPARISON_COLUMNS: list[tuple[str, str]] = [
    ("title", "论文"),
    ("method_name", "方法"),
    ("core_idea", "核心思路"),
    ("datasets", "数据集"),
    ("improvement", "相对提升"),
    ("cost", "代价"),
    ("limitation", "局限（原文自述）"),
    ("venue", "发表会议"),
    ("cited_by_count", "被引"),
]


def _comparison_to_markdown(rows: list[dict]) -> str:
    """把对比表导出成 Markdown（用于直接粘进开题报告）。"""
    lines = [
        "| " + " | ".join(label for _, label in COMPARISON_COLUMNS) + " |",
        "|" + "---|" * len(COMPARISON_COLUMNS),
    ]
    for row in rows:
        cells = []
        for key, _ in COMPARISON_COLUMNS:
            val = row.get(key)
            cells.append("—" if val in (None, "", []) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ==================================================================== 评测页辅助
# 评测产物目录（消融报告落在这里）。用 ROOT 定位而不是相对 CWD ——
# Streamlit 从任意目录启动时 CWD 都可能不同，用相对路径会读不到文件。
EVAL_RESULTS_DIR = ROOT / "eval_results"

# 报告展示的默认优先级：**可信口径优先**。
# 消融报告的数字只有在**同轮、同索引、同 embedding** 下才可比，而 embedding/重排器口径就写在
# 报告头部，所以排序表达的是"哪一份的口径更接近当前配置"，不是"哪一份更新"：
# 真 BGE 交叉编码器那轮 > lexical 降级那轮 > 其它（未登记的文件按文件名排在后面）。
# 名字不在这里的报告依然会出现在下拉里 —— 这里只排序，不筛选，避免新产物被静默藏起来。
_PREFERRED_ABLATION_REPORTS: tuple[str, ...] = (
    "ablation_24papers_bge_rerank.md",
    "ablation_24papers.md",
    "ablation_latest.md",  # `make eval-ablation` 的默认产物，口径未登记，排在已保留的两份之后
)


def _list_ablation_reports() -> list[Path]:
    """列出 `eval_results/*.md`，按「真实数字优先」排序，供下拉选择。"""
    if not EVAL_RESULTS_DIR.exists():
        return []
    files = {p.name: p for p in EVAL_RESULTS_DIR.glob("*.md")}
    ordered = [files.pop(name) for name in _PREFERRED_ABLATION_REPORTS if name in files]
    ordered += [files[name] for name in sorted(files)]
    return ordered


def _fetch_citation_accuracy(api_base: str, run_id: str) -> dict:
    """从后端取一条 run 的**自动口径**引用准确率。

    口径 = `citations.verified == True` 的占比（架构 §12 的免费口径，Verifier 节点已写回字段）。
    这里**先确认 run 存在**（`GET /research/{run_id}` 不存在会返回 404），
    否则一个不存在的 run_id 会被后端当成"0 条引用"、显示成"准确率 —"，
    那等于拿"查无此 run"冒充"这次研究没有引用"，两种含义完全不同。

    说明：后端没有单独的 `citation_accuracy` 路由，数值由 `/{run_id}/citations`
    返回的 `verified_rate` / `citations[].verified` 现算，与 `ragas_runner.citation_accuracy` 口径一致。
    """
    detail = requests.get(f"{api_base}/api/v1/research/{run_id}", timeout=15)
    if detail.status_code == 404:
        raise ValueError(f"run_id={run_id} 不存在（后端返回 404）")
    detail.raise_for_status()

    payload = requests.get(
        f"{api_base}/api/v1/research/{run_id}/citations", timeout=15
    ).json()
    citations = payload.get("citations") or []
    total = payload.get("count", len(citations))
    verified = sum(1 for c in citations if c.get("verified") is True)
    accuracy = payload.get("verified_rate")
    if accuracy is None and total:
        accuracy = round(verified / total, 4)
    return {"total": total, "verified": verified, "accuracy": accuracy, "method": "auto"}


def _run_ragas_local(run_id: str | None):
    """在面板进程内直接调用 RAGAS 运行器（后端未暴露评测路由）。

    这样做的依据是运行器的硬约束：**不可用时不产出任何分数**（`EvalReport.available=False`
    时携带分数会直接抛错）。所以面板可以放心地按 `available` 分支展示 —— 它不可能显示假分数。
    `run_ids=None` 表示"库里现有的全部 run"。
    """
    from app.eval.ragas_runner import run_ragas_eval

    return asyncio.run(run_ragas_eval(run_ids=[run_id] if run_id else None))


# ==================================================================== 侧边栏
with st.sidebar:
    st.title("📚 PaperPilot")
    st.caption("arXiv 论文深度研究助手")
    st.divider()
    api_base = st.text_input("后端地址", value=API_BASE)
    st.divider()
    st.caption(
        "**使用前提**：后端要配置 LLM。\n\n"
        "在 `.env` 里填 `OPENAI_API_KEY`（兼容 DeepSeek / 通义等），"
        "或把 `LLM_PROVIDER` 设为 `mock` 做离线演示。"
    )

st.title("PaperPilot")
st.caption("基于 LangGraph 多 Agent 的 arXiv 论文深度研究工具")

tab_research, tab_search, tab_eval, tab_status = st.tabs(
    ["🧭 深度研究", "🔍 检索试玩", "📊 评测", "⚙️ 系统状态"]
)

# ================================================================== 深度研究
with tab_research:
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "研究问题",
            value="近两年 RAG 的重排（reranking）技术有哪些进展，各自的代价是什么？",
        )
        top_k = st.slider("每个子问题召回论文数", 1, 20, 8, key="research_top_k")
    with col2:
        st.write("")
        st.write("")
        start = st.button("🚀 开始研究", type="primary", use_container_width=True)

    timeline_box = st.empty()
    report_box = st.empty()
    meta_box = st.empty()

    # ---- 发起研究 ----
    if start and question.strip():
        with st.spinner("研究中…"):
            consume_stream(
                f"{api_base}/api/v1/research/stream",
                {"question": question.strip(), "top_k": top_k},
                timeline_box,
                report_box,
                meta_box,
            )

    # ---- HITL：中断后展示可编辑的子问题 ----
    if st.session_state.awaiting_resume:
        st.divider()
        st.info("⏸ 已暂停：系统拆解出子问题，**你可以直接修改后再继续**（这是刻意留的干预点）。")
        edited = []
        current = st.session_state.pending_sub_questions
        for i, sq in enumerate(current):
            text = st.text_input(f"子问题 {i + 1}", value=sq.get("text", ""), key=f"sq_{i}_{st.session_state.run_id}")
            edited.append({**sq, "text": text})

        if st.button("✅ 确认并继续", type="primary"):
            run_id = st.session_state.run_id
            with st.spinner("继续研究中…"):
                consume_stream(
                    f"{api_base}/api/v1/research/{run_id}/resume",
                    {"sub_questions": edited},
                    timeline_box,
                    report_box,
                    meta_box,
                )

    # ---- 结果 ----
    if st.session_state.verification:
        v = st.session_state.verification
        (st.success if v.get("passed") else st.warning)(
            f"引用校验 {'通过' if v.get('passed') else '未通过'}"
            f" · score={v.get('score')} · 无支撑论断 {len(v.get('unsupported') or [])} 条"
        )

    # ---- 引用明细 ----
    # `verified` 由 verify 节点逐条写回：硬校验（引文逐字出现在定位到的 chunk 里）
    # 与 LLM 软校验两层都过才置 True，这里只做计数、不做判定。
    # ⚠️ 刻意不区分"精确命中"与"重锚救回"：两者在 state 里同形，救回条数只在后端日志
    # `read.done`（reanchored / recovered_by）里 —— 面板没有这些数据，就不假装能显示。
    citations = st.session_state.citations
    if citations:
        passed = sum(1 for c in citations if c.get("verified") is True)
        st.caption(
            f"引用 {len(citations)} 条 · 通过校验 {passed} · 未通过 {len(citations) - passed}"
            "（口径：引文逐字命中 + LLM 软校验两层都过，含经重锚定位后通过的引用）"
        )

    if st.session_state.report:
        st.divider()
        st.markdown("### 研究报告")
        st.markdown(st.session_state.report)
        st.download_button(
            "⬇️ 导出 Markdown",
            data=st.session_state.report,
            file_name=f"paperpilot_{st.session_state.run_id or 'report'}.md",
        )

    # ---- 结构化对比表（产品定义的核心产出物 A）----
    # 空字段渲染成「—」而不是隐藏整行：让用户看见"这里确实没有数据"，
    # 本身就是产品承诺（不编造）的一部分。
    if st.session_state.comparison:
        st.divider()
        st.markdown("### 结构化对比表")
        st.caption("字段缺失一律留空（显示为「—」）—— **不编造**是硬规则。")
        st.dataframe(st.session_state.comparison, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ 导出对比表 (Markdown)",
            data=_comparison_to_markdown(st.session_state.comparison),
            file_name=f"comparison_{st.session_state.run_id or 'table'}.md",
        )

    if st.session_state.token_by_node:
        with st.expander("💰 Token 按节点分布（定位成本瓶颈）"):
            st.json(st.session_state.token_by_node)

    if st.session_state.memory_context:
        with st.expander(f"🧠 本次召回的长期记忆（{len(st.session_state.memory_context)} 条）"):
            for m in st.session_state.memory_context:
                st.markdown(f"- {m}")

    if st.session_state.retrieval_hits:
        with st.expander(f"🔎 检索命中的论文（{len(st.session_state.retrieval_hits)} 篇）"):
            for h in st.session_state.retrieval_hits:
                st.markdown(
                    f"- **{h.get('title', '')[:80]}** `{h.get('arxiv_id')}` "
                    f"· 分数 {h.get('score')} · 章节 `{h.get('section')}` · 来源 `{h.get('source')}`"
                )

# ================================================================== 检索试玩
with tab_search:
    st.caption("不跑完整研究流程，直接试混合检索（向量 + BM25 → RRF → 重排）。")


    @st.cache_resource(show_spinner=True)
    def get_retriever():
        """按 `.env` 配置构造检索栈（本机当前是 `EMBEDDING_PROVIDER=local` 的 bge-m3）。

        这里**不再强制 mock**：落盘索引就是 bge-m3 建的
        （`data/faiss/paperpilot.manifest.json` 的 `embedding_identity`），
        查询端必须用同一个 embedder —— 换一个模型就落进另一个语义空间，
        而维度常常撞车（都是 1024），不报错、只是结果变差（静默退化）。
        真实的一致性检查在调用处比对 `index_identity` 与 `embedder.identity`。
        """
        from app.retrieval.hybrid_retriever import HybridRetriever

        return HybridRetriever()


    c1, c2 = st.columns([3, 1])
    with c1:
        q2 = st.text_input("查询", value="reranking methods for retrieval augmented generation")
        kw2 = st.text_input("英文关键词（逗号分隔）", value="reranking,rerank,reranker,retrieval")
    with c2:
        n2 = st.slider("返回数", 1, 10, 5, key="demo_top_n")
        go2 = st.button("检索", key="demo_go")

    if go2:
        try:
            retriever = get_retriever()
            if retriever.index_identity and retriever.index_identity != retriever.embedder.identity:
                st.warning(
                    f"索引由 `{retriever.index_identity}` 建立，当前查询端是 "
                    f"`{retriever.embedder.identity}` —— 两者不在同一语义空间，"
                    "向量路的结果不可信。请重建索引：`python scripts/build_index.py`"
                )
            kws = [k.strip() for k in kw2.split(",") if k.strip()] or None
            hits = asyncio.run(retriever.search(q2, kws, rerank_top_n=n2))
            st.success(f"命中 {len(hits)} 篇（索引 {retriever.size} chunk）")
            for i, h in enumerate(hits, 1):
                with st.expander(f"{i}. [{h.score:.4f}] {h.title[:80]}", expanded=i == 1):
                    st.caption(
                        f"`{h.arxiv_id}` · 章节 **{h.rep_chunk.section}** · "
                        f"来源 **{h.rep_chunk.source}** · 路径 {'/'.join(h.matched_by)}"
                    )
                    st.text(h.rep_chunk.text[:800])
        except Exception as exc:  # noqa: BLE001
            st.error(f"检索失败：{exc}")

# ================================================================== 系统状态
with tab_status:
    try:
        health = requests.get(f"{api_base}/healthz", timeout=5).json()
        st.success(f"后端已连接 · 版本 {health.get('version')} · mock_mode={health.get('mock_mode')}")
        st.json(health)
        if not health.get("llm_configured"):
            st.warning("后端未配置 LLM：在 `.env` 填 `OPENAI_API_KEY`，或设 `LLM_PROVIDER=mock`。")
    except Exception as exc:  # noqa: BLE001
        st.error(f"后端连接失败：{exc}")
        st.code("uvicorn app.main:app --port 8000", language="bash")

    st.divider()
    st.markdown("#### 已实现")
    st.markdown(
        "- 抓取 arXiv → 章节级分块 → 落库（SQLite + WAL）\n"
        "- 混合检索：FAISS + BM25 → RRF 融合 → 交叉编码器重排 → chunk→paper 聚合\n"
        "- 语义检索与重排都用**本地权重**：索引由 `models/bge-m3` 建立"
        "（见 `data/faiss/paperpilot.manifest.json` 的 `embedding_identity`），"
        "重排是 `bge-reranker-v2-m3`\n"
        "- LangGraph 多 Agent：拆解 → 语料保障 → 并行检索 → 抽取 → 引用校验 → 对比表 → 撰写\n"
        "- SSE 流式（15 种事件）+ 两个 HITL 中断点\n"
        "- 引用可溯源：每条引用落在具体 chunk，带章节定位与证据强度\n"
        "- 引用定位三条路径：精确命中 → 字段错位纠正 → 同篇逐字引文重锚；"
        "每条都要求原文逐字对上才放行，全失败才判幻觉\n"
        "- 结构化对比表：逐字段回原文校验，缺字段留空「—」，可导出 Markdown\n"
        "- 长期记忆已接线：`plan` 召回并注入 prompt，`finalize` 写入方向/已读/偏好\n"
        "- 短期记忆写入（`finalize` 追加本轮问答）\n"
        "- Token **按节点**细分：面板与 `token_usage` 表都能看到每个节点的消耗\n"
        "- 注入防护 / 重试超时熔断"
    )
    st.markdown("#### 待办")
    st.markdown(
        "- 长期记忆只影响 `plan` 的 prompt：检索与精读阶段不消费已读清单"
        "（`memory_context` 只在 `planner` 里产出并写进 state），"
        "所以「避免重复精读同一批论文」还没落地\n"
        "- 短期记忆**只写不读**：`load_context` 目前只被单测调用，没有节点把它读回 prompt，"
        "因此「会话内追问」还不可用\n"
        "- 引用重锚的统计只落在后端日志（`read.done` 的 `reanchored` / `recovered_by`），"
        "既没进 SSE 事件也没进 `citations` 表，面板无法显示「本次救回了几条」"
    )

# ====================================================================== 评测
with tab_eval:
    st.caption("把仓库里的评测能力搬到面板上：检索消融报告、索引状态、引用准确率、RAGAS。")

    # ⭐ 这一句是整个页面的"诚实性说明"：用户必须先知道哪些数字不需要 key、哪些必须要真实 LLM，
    # 才不会被"看起来像分数"的假数字误导。
    st.info(
        "**不需要 API key 就能算的指标**：\n"
        "- ① 检索侧指标（Recall@5 / Precision@5 / nDCG@10 / MRR / 耗时）—— 只需要一个建好的索引；\n"
        "- ③ **引用准确率（自动口径）**—— 只统计 `citations.verified` 的比例，零成本、可复现。\n\n"
        "**需要真实 LLM 当裁判的指标**：\n"
        "- ④ RAGAS 的五个生成侧指标（faithfulness / answer_relevancy / context_precision / "
        "context_recall / answer_correctness）。\n"
        "  在 mock 或未配 key 时它们**不可计算** —— 面板只显示「不可用 + 原因」，**绝不显示成 0 分**。"
    )

    # ---------------------------------------------------------------- ① 检索消融
    st.divider()
    st.subheader("① 检索消融结果")
    st.caption(
        "数据来源：`eval_results/*.md`（`make eval-ablation` 的产物）。"
        "注意不同报告的 embedding 口径不同（local / api / mock），**跨 embedding 的数字不可横向比较**。"
    )
    reports = _list_ablation_reports()
    if not reports:
        st.warning(
            "未找到 `eval_results/*.md`。先跑一次消融生成报告：\n\n"
            "```bash\nmake eval-ablation\n```"
        )
    else:
        names = [p.name for p in reports]
        picked = st.selectbox(
            "选择报告文件",
            names,
            index=0,
            key="eval_report_pick",
            help="默认优先展示本地 BGE 跑出的真实数字。",
        )
        st.caption("可选报告：" + "、".join(f"`{n}`" for n in names))
        # 直接用 st.markdown 渲染 Markdown 表格 —— 报告本身就是 Markdown，不需要再解析成 DataFrame。
        st.markdown((EVAL_RESULTS_DIR / picked).read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- ② 索引状态
    st.divider()
    st.subheader("② 索引状态")
    st.caption("数据来源：后端 `GET /healthz` 的 `index` / `embedding` 字段。")
    try:
        health = requests.get(f"{api_base}/healthz", timeout=5).json()
        idx = health.get("index") or {}
        emb = health.get("embedding") or {}

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("索引就绪", "✅ 是" if idx.get("ready") else "❌ 否")
        c2.metric("与 embedding 匹配", "✅ 匹配" if idx.get("compatible") else "⚠️ 不匹配")
        c3.metric("chunk 数", idx.get("chunks") if idx.get("chunks") is not None else "—")
        c4.metric("论文数", idx.get("papers") if idx.get("papers") is not None else "—")
        st.caption(
            f"built_at：`{idx.get('built_at')}`　·　built_with：`{idx.get('built_with')}`　·　"
            f"当前 embedding：`{emb.get('identity')}`"
        )

        # ⚠️ `compatible=false` 是最容易"静默退化"的坑：换了 embedding 却不重建索引时，
        # 维度相同的话连报错都没有，检索质量悄悄变差。所以这里必须显著提示，
        # 并把后端已经写好的修复命令（hint 里就是它）原样透出，用户不用去翻文档。
        if not idx.get("compatible"):
            st.warning(
                "**索引与当前 embedding 配置不匹配** —— 检索会静默退化成噪声，这份索引的结果不可信。\n\n"
                + (idx.get("hint") or "请重建索引：`python scripts/build_index.py`")
            )
        elif not idx.get("ready"):
            st.warning(idx.get("hint") or "索引尚未构建：`python scripts/build_index.py`")
    except Exception as exc:  # noqa: BLE001
        st.error(f"读取 /healthz 失败：{exc}")
        st.code("uvicorn app.main:app --port 8000", language="bash")

    # ------------------------------------------------------------ ③ 引用准确率
    st.divider()
    st.subheader("③ 引用准确率（自动口径）")
    st.caption(
        "口径：`citations.verified == True` 的占比（架构 §12 的免费口径，由 Verifier 节点写回）。"
        "这是**自动口径**；它不等于「人工抽检口径」—— 只报自动口径等于自己批改自己的作业，"
        "两个口径都要报。"
    )
    # 分母是"落库的全部引用"，所以"经引用定位救回"的条目也一并计入分子 ——
    # 它们与精确命中的引用在库里同形（`/{run_id}/citations` 不返回定位方式，
    # 见 app/api/v1/research.py:291-321），面板因此**不宣称**能分辨两者，只把救回的条数指向日志。
    st.caption(
        "统计范围：包含经引用定位（字段错位纠正 / 同篇逐字引文重锚）后通过校验的引用；"
        "救回条数只在后端日志 `read.done` 的 `reanchored` / `recovered_by` 里，接口未透出。"
    )
    # 后端没有"列出所有 run"的接口，因此**不新增后端接口**，改为让用户手工填写 run_id。
    st.caption("提示：后端未提供 run 列表接口，请手工填写 run_id（可在「深度研究」页拿到）。")
    rid = st.text_input(
        "run_id", value=st.session_state.run_id or "", key="eval_run_id"
    )
    if st.button("查询引用准确率", key="eval_cite_go"):
        if not rid.strip():
            st.info("请先填写 run_id。")
        else:
            try:
                acc = _fetch_citation_accuracy(api_base, rid.strip())
                a1, a2, a3 = st.columns(3)
                a1.metric("引用总数", acc["total"])
                a2.metric("已校验为真", acc["verified"])
                a3.metric(
                    "准确率",
                    "—" if acc["accuracy"] is None else f"{acc['accuracy']:.1%}",
                )
                # 明确标注口径，避免"自动口径"被误当成权威结论。
                # 口径细节必须写清分母：后端的 `verified_rate` 是 `sum(verified) / 全部引用条数`
                # （app/api/v1/research.py 的 `/{run_id}/citations`），
                # 所以"未校验（verified=None）"与"校验失败"都留在分母里、都不进分子。
                # 把分母说成"已校验引用"会把两个完全不同的数混成一个。
                st.caption(
                    f"口径：**自动**（method={acc['method']}）—— 分母 = 本次落库的全部引用（"
                    f"{acc['total']} 条），分子 = `verified is True` 的 {acc['verified']} 条；"
                    "未校验（`verified=None`）与校验失败都计入分母、不计入分子。"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"查询失败：{exc}")

    # ------------------------------------------------------------------ ④ RAGAS
    st.divider()
    st.subheader("④ RAGAS 生成侧评测（可选）")
    st.caption(
        "RAGAS 的指标本质是「用 LLM 当裁判」，必须有真实生成模型；"
        "无 key / mock 时运行器返回 `available=False`，此处只显示**原因**，不显示任何分数。"
    )
    if st.button("运行 RAGAS 评测", key="eval_ragas_go"):
        with st.spinner("运行中…（缺少裁判模型时会立刻返回原因，不会产出假分数）"):
            try:
                report = _run_ragas_local(rid.strip() or None)
                if not report.available:
                    st.warning(f"RAGAS 不可用：{report.reason}")
                    if report.skipped_metrics:
                        st.caption(
                            "被跳过的指标：" + "、".join(f"`{m}`" for m in report.skipped_metrics)
                        )
                else:
                    st.success(
                        f"RAGAS 可用 · 样本 {report.n_samples} 条"
                        f"（其中含参考答案 {report.n_with_ground_truth} 条）"
                    )
                    st.dataframe(
                        [{"指标": k, "分数": round(v, 4)} for k, v in report.metrics.items()],
                        use_container_width=True,
                        hide_index=True,
                    )
                    if report.skipped_metrics:
                        st.caption(
                            "被跳过的指标：" + "、".join(f"`{m}`" for m in report.skipped_metrics)
                        )
                    with st.expander("生成配置（分数可比性的前提：不记配置的分数没有可比性）"):
                        st.json(report.generation_config)

                # 引用准确率不依赖 LLM，任何情况下都能附上（自动口径）
                if report.citation_accuracy:
                    ca = report.citation_accuracy
                    st.caption(
                        f"附带的自动口径引用准确率：{ca['verified']}/{ca['total']}"
                        f"（{ca['accuracy']:.1%}，method={ca.get('method')}）"
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"运行 RAGAS 失败：{exc}")


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

tab_research, tab_search, tab_status = st.tabs(["🧭 深度研究", "🔍 检索试玩", "⚙️ 系统状态"])

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


    @st.cache_resource(show_spinner=False)
    def get_retriever():
        """强制 mock embedding：当前落盘索引就是 mock 建的，换真模型查询会落进
        完全不同的语义空间、结果反而无意义。装了 `[local]` 并重建索引后可换成 `HybridRetriever()`。"""
        from app.retrieval.embedder import MockEmbedder
        from app.retrieval.hybrid_retriever import HybridRetriever
        from app.retrieval.reranker import LexicalReranker

        return HybridRetriever(embedder=MockEmbedder(dim=1024), reranker=LexicalReranker())


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
        "- LangGraph 多 Agent：拆解 → 语料保障 → 并行检索 → 抽取 → 引用校验 → 撰写\n"
        "- SSE 流式（15 种事件）+ 两个 HITL 中断点\n"
        "- 引用可溯源：每条引用落在具体 chunk，带章节定位与证据强度\n"
        "- 注入防护 / 重试超时熔断 / 短期+长期记忆"
    )
    st.markdown("#### 待办")
    st.markdown(
        "- 真实 BGE 语义检索（需 `pip install -e \".[local]\"` 后重建索引）\n"
        "- 长期记忆的写入/召回在图中尚未完全接线\n"
        "- Token 按节点细分统计（当前是整体一条）\n"
        "- 对比表结构化抽取（产品定义里的核心产出物 A）"
    )

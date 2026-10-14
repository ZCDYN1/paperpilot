"""端到端验证脚本：用 mock 三件套跑通全图（plan → finalize）。

用法（**必须在导入 `app.*` 之前**设好环境变量，否则 `Settings` 的 lru_cache 会把默认值固化）：

    LLM_PROVIDER=mock EMBEDDING_PROVIDER=mock RERANK_PROVIDER=lexical \
    MIN_CORPUS=3 python scripts/verify_agents.py

它会验证四件事：
1. 全链路不报错，产出中文综述（含引用列表、引用校验结果、AI 免责声明）；
2. **`read` 节点实际执行次数 == 1**（并行汇合陷阱的实测口径，架构 §4.3-④）；
3. `interrupt_after=["plan"]` 生效，且能用 `ainvoke(None, config)` 从中断点恢复
   （同时验证 `AsyncSqliteSaver` 落盘与续跑）；
4. `MAX_REVISION` 的 `give_up` 出口可用（直接调路由函数，不依赖 LLM 行为）。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

# ---- 环境必须在 import app.* 之前设好（Settings 是 lru_cache 单例）----
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "mock")
os.environ.setdefault("RERANK_PROVIDER", "lexical")
# 让 5 篇本地语料就算"达标"，从而**完全离线**（不触发 arXiv 抓取与 3s 限速）
os.environ.setdefault("MIN_CORPUS", "3")

# ---- ★ 数据库隔离：复制一份到临时目录，所有写入都落在副本上 ----
# 这个脚本会跑完整图（含 finalize 写长期记忆），而长期记忆现在是**跨会话生效**的：
# `read` 节点会拿"已读论文清单"排除精读候选（架构 §9.1）。所以往真实库里写合成 run 的
# 记忆不是无害的 —— 每跑一次就把一批语料标记成"已读"，反复运行最终会把整个语料标记完，
# 之后每次真实运行都只能走补位分支。而这种退化**不报错**，只会让功能悄悄失效。
# 用复制而不是空库：脚本依赖库里已入库的语料才能保持离线（见上面的 MIN_CORPUS）。
_SRC_DB_DIR = Path(__file__).resolve().parents[1] / "data" / "sqlite"
_TMP_DB_DIR = Path(tempfile.mkdtemp(prefix="paperpilot_verify_"))
for _src in _SRC_DB_DIR.glob("*.db*"):  # 含 -wal / -shm，漏掉会丢失最近未 checkpoint 的写入
    shutil.copy2(_src, _TMP_DB_DIR / _src.name)
os.environ["SQLITE_DIR"] = str(_TMP_DB_DIR)
os.environ["SQLITE_URL"] = f"sqlite+aiosqlite:///{_TMP_DB_DIR / 'paperpilot.db'}"
os.environ["CHECKPOINT_DB"] = str(_TMP_DB_DIR / "checkpoints.db")

from app.agents.graph import build_graph, decide_after_verify, open_checkpointer  # noqa: E402
from app.agents.nodes import read_invocations, reset_read_counter  # noqa: E402
from app.db.session import dispose_engine  # noqa: E402
from app.observability.events import SSEEmitter  # noqa: E402


def _banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


async def scenario_revise() -> dict:
    """实测反思回路：`verify` 恒不通过 → revise → **Send 扇出** → retrieve → read → give_up。

    为什么必须单独测这条路径：mock 的 verifier 默认判"通过"，happy path 永远走不到 revise。
    而 revise 是**修正点 3**（必须用 Send 扇出）的落点 —— 不用 Send 的话
    retrieve 会拿到空 `sub_question`，检索静默返回随机结果且不报错。

    做法是对 graph 模块里的 `verify_node` 引用打补丁（`build_graph` 在函数体内解析该名字，
    所以补在 `app.agents.graph` 上、且在 build 之前生效）。
    """
    import app.agents.graph as graph_mod

    counters = {"retrieve": 0, "revise": 0}
    real_verify, real_retrieve, real_revise = (
        graph_mod.verify_node,
        graph_mod.retrieve_node,
        graph_mod.revise_node,
    )

    async def failing_verify(state):
        out = await real_verify(state)
        v = dict(out.get("verification") or {})
        v["passed"] = False          # 强制不通过，逼出回退
        v["unsupported"] = v.get("unsupported") or ["（构造的未支撑论断）"]
        out["verification"] = v
        return out

    async def counting_retrieve(state):
        counters["retrieve"] += 1
        return await real_retrieve(state)

    async def counting_revise(state):
        counters["revise"] += 1
        return await real_revise(state)

    graph_mod.verify_node, graph_mod.retrieve_node, graph_mod.revise_node = (
        failing_verify,
        counting_retrieve,
        counting_revise,
    )
    try:
        run_id = f"revise-{uuid.uuid4().hex[:8]}"
        async with open_checkpointer() as saver:
            graph = build_graph(saver)
            config = {"configurable": {"thread_id": run_id}}
            await graph.ainvoke(
                {"session_id": run_id, "run_id": run_id, "question": "RAG 重排的代价是什么？"},
                config,
            )
            final = await graph.ainvoke(None, config)
    finally:
        graph_mod.verify_node, graph_mod.retrieve_node, graph_mod.revise_node = (
            real_verify,
            real_retrieve,
            real_revise,
        )

    return {
        "stage": final.get("stage"),
        "revision_count": final.get("revision_count"),
        "retrieve_calls": counters["retrieve"],
        "revise_calls": counters["revise"],
        "read_calls_total": read_invocations(),
        "has_unsupported_section": "未找到原文支撑的结论" in (final.get("draft") or ""),
    }


async def main() -> int:
    run_id = f"verify-{uuid.uuid4().hex[:8]}"
    question = "近两年 RAG 的重排（reranking）技术有哪些进展，各自的代价是什么？"

    reset_read_counter()

    async with open_checkpointer() as saver:
        graph = build_graph(saver)
        config = {"configurable": {"thread_id": run_id}}
        emitter = SSEEmitter(run_id=run_id)

        _banner("① 第一次调用：跑到中断点 A（plan 之后）")
        inputs = {"session_id": run_id, "run_id": run_id, "question": question}
        seen_nodes: list[str] = []
        async for frame in emitter.stream(graph, inputs, config):
            event = frame.split("\n", 1)[0].removeprefix("event: ")
            if event == "node_started":
                name = frame.split('"node": "', 1)[1].split('"', 1)[0]
                seen_nodes.append(name)
                print(f"  ▶ node_started: {name}")
            elif event in ("plan_created", "corpus", "interrupted", "done"):
                print(f"  ◆ {event}: {frame.splitlines()[1][:160]}")
        print(f"  中断前实际执行的节点：{seen_nodes}")

        snapshot = await graph.aget_state(config)
        subs = (snapshot.values or {}).get("sub_questions") or []
        print(f"  中断点可见的子问题数：{len(subs)}")
        for sq in subs:
            print(f"    - {sq['id']}: {sq['text'][:40]} | keywords={sq['keywords'][:4]}")

        _banner("② 恢复执行：await graph.ainvoke(None, config) —— 验证 AsyncSqliteSaver 续跑")
        # 传 None（而不是 {}）：None 表示"从中断点续跑"，{} 会被当成空输入从 START 重跑，
        # 白烧一遍 token（架构 §4.3-⑤）。
        final = await graph.ainvoke(None, config)

    stage = final.get("stage")
    main_read_calls = read_invocations()   # ★ 并行汇合实测口径：应为 1
    retrieved = final.get("retrieved") or []
    citations = final.get("citations") or []
    verification = final.get("verification") or {}
    draft = final.get("draft") or ""
    usage = final.get("token_usage") or {}

    _banner("③ 关键指标")
    print(f"  stage                : {stage}")
    print(f"  子问题数             : {len(subs)}")
    print(f"  ★ read 节点执行次数  : {main_read_calls}   （期望 1；若为 3 则并行汇合翻倍）")
    print(f"  retrieve 产出聚合长度: {len(retrieved)}   （= 各子问题命中数之和）")
    print(f"  去重后论文数         : {len({p['arxiv_id'] for p in retrieved})}")
    print(f"  引用条数             : {len(citations)}")
    print(f"  校验通过             : {verification.get('passed')} / score={verification.get('score')}")
    print(f"  token_usage          : {usage}")
    print(f"  报告字数             : {len(draft)}")

    _banner("④ 最终报告（前 40 行）")
    for line in draft.splitlines()[:40]:
        print("  " + line)

    _banner("⑤ give_up 出口（直接调路由函数，不受 LLM 行为影响）")
    print("  revision_count=0, 不通过 →", decide_after_verify({"verification": {"passed": False}, "revision_count": 0}))
    print("  revision_count=2, 不通过 →", decide_after_verify({"verification": {"passed": False}, "revision_count": 2}))
    print("  passed=True            →", decide_after_verify({"verification": {"passed": True}, "revision_count": 0}))

    _banner("⑥ 反思回路实测：verify 恒不通过 → revise(Send 扇出) → ... → give_up")
    reset_read_counter()  # 计数器复位，让本节数字自成一体
    rev = await scenario_revise()
    for k, v in rev.items():
        print(f"  {k:24}: {v}")
    print("  说明：retrieve 调用次数 = 首轮 3 + 每轮回退 3×2 轮 = 9；read = 3（每轮汇合 1 次）")

    ok = (
        stage == "done"
        and main_read_calls == 1             # ★ 核心断言：3 个并行分支，read 只执行 1 次
        and len(subs) == 3
        and len(citations) > 0
        and "本文由 AI 生成" in draft
        and rev["revise_calls"] == 2         # MAX_REVISION=2
        and rev["read_calls_total"] == 3     # 回退回路里每轮汇合仍**恰好一次**
        and rev["stage"] == "done"
        and rev["has_unsupported_section"]   # give_up 后必须显式标注未支撑结论
    )
    _banner("结论：" + ("✅ 全链路通过" if ok else "❌ 存在未通过项"))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    finally:
        # 先释放引擎再删目录：Windows 上 SQLite 句柄未关闭会让删除失败（ignore_errors 兜底）
        try:
            asyncio.run(dispose_engine())
        except Exception:  # noqa: BLE001 - 清理失败不影响脚本退出码
            pass
        shutil.rmtree(_TMP_DB_DIR, ignore_errors=True)

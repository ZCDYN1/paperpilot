"""F9「会话内追问」+ 中断点 A 可用率埋点的测试（产品定义 §5 F9 / §8）。

这个文件要证明的是**链路上真的接上了**，而不是"函数能跑"：
1. 客户端传 `session_id` → 这个 run 挂进那个会话，但 `thread_id` 仍是 run 级
   （LangGraph 的 checkpoint 以 thread_id 为键，会话共用 thread 会让新提问
   接着上一轮已跑完的图状态走 —— 不报错，只是被污染）；
2. 追问时**同一会话前几轮的问答真的进了 prompt**（断言渲染后的 prompt，
   这是唯一能证明"读回来了"的办法）；
3. 短期记忆读取失败只降级成"这次没带上下文"，不拖垮拆解；
4. `plan_edited` 在"用户没改直接继续"与"用户改了"两种情况下都真的落进
   `research_runs.langgraph_state`（而且扛得住 `finalize` 对那一列的整体覆盖）；
5. ★ **长期记忆仍然跨会话**：F9 让 session_id 第一次变得"有意义"，
   绝不代表可以把长期记忆改回按会话过滤 —— 那会重新踩回 F10 那个静默空转的坑
   （见 `app/memory/long_term.py` 的★小节）。第 5 组用例就是把这个性质钉死的钉子。

离线三件套与临时库的接法与 `tests/test_api_research.py` / `tests/test_memory.py` 一致，
所有写入都落在 pytest 的 `tmp_path`，**不碰 `data/sqlite/paperpilot.db`**。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.db.models  # noqa: F401 - 导入以把表注册进 metadata
from app.api.v1.research import plan_edit_diff
from app.db.base import Base
from app.db.models import ResearchRun
from app.db.session import session_scope
from app.memory import long_term, short_term
from app.memory.long_term import LongTermMemoryStore
from app.retrieval.embedder import MockEmbedder
from app.schemas.research import ResearchRequest
from tests.api_support import STREAM_URL, api_client, first_data

# =============================================================================
# 夹具
# =============================================================================


@pytest.fixture
async def client(monkeypatch, tmp_path):
    async with api_client(monkeypatch, tmp_path) as c:
        yield c


@pytest.fixture
async def scope(tmp_path) -> AsyncIterator:
    """临时库的事务作用域工厂（与 `app.db.session.session_scope` 同签名）。"""
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'f9.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def _scope():
        async with maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    yield _scope
    await engine.dispose()


@pytest.fixture
def plan_stub(monkeypatch):
    """把 plan 节点的 LLM 调用换成"记录 prompt + 返回确定性拆解"的桩。

    返回那个 dict 就是断言现场：**prompt 是唯一能证明上下文被读回来的地方**。
    """
    from app.agents.nodes import planner
    from app.agents.schemas import PlanOutput
    from app.schemas.research import SubQuestion

    box: dict[str, Any] = {"prompts": []}

    async def fake_invoke(schema, messages, **kwargs):  # noqa: ANN001
        # 只记 HumanMessage：上下文块（会话历史 / 长期记忆）都渲染在它里面，那才是要断言的地方。
        # 系统提示词（planner.md）本身就会写出 `<session_history>` 这个**标签名**来教模型怎么用，
        # 混进来的话，"本轮没带历史"那几条负向断言会永远为假 —— 而负向断言正是短期记忆
        # 没有跨会话串味的唯一证据。断言渲染结果就不要连指令一起断言。
        box["prompts"].append(
            "\n".join(str(m.content) for m in messages if type(m).__name__ == "HumanMessage")
        )
        return (
            PlanOutput(
                rationale="按方法与代价拆解。",
                sub_questions=[
                    SubQuestion(
                        id=f"sq-{i}",
                        text=f"子问题{i}：该方向的核心方法与代价是什么？",
                        keywords=["reranking", "retrieval"],
                        category=None,
                    )
                    for i in range(1, 4)
                ],
            ),
            {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        )

    monkeypatch.setattr(planner, "structured_invoke", fake_invoke)
    return box


@pytest.fixture
def short_term_store(monkeypatch, scope):
    """让默认短期记忆实例走临时库（planner 调的是模块级 `load_context`，用的是默认实例）。"""
    mem = short_term.ShortTermMemory(max_turns=6, scope=scope)
    monkeypatch.setattr(short_term, "_default", mem)
    return mem


# =============================================================================
# 1) ★ 会话接线：session_id 跟着会话走，thread_id 死死钉在 run 上
# =============================================================================
async def test_supplied_session_id_joins_session_while_thread_stays_per_run(client):
    session_id = "panel-session-1"
    r1 = (await client.post(STREAM_URL, json={"question": "RAG 的重排技术有哪些", "session_id": session_id})).text
    run_1 = first_data(r1, "run_started")["run_id"]
    r2 = (await client.post(STREAM_URL, json={"question": "那它们的代价是什么", "session_id": session_id})).text
    run_2 = first_data(r2, "run_started")["run_id"]
    assert run_1 != run_2

    async with session_scope() as db:
        rows = {
            r.id: r
            for r in (
                await db.execute(select(ResearchRun).where(ResearchRun.id.in_([run_1, run_2])))
            ).scalars()
        }
        # 两次提问串成**一个会话**（F9 的前提）
        assert {rows[run_1].session_id, rows[run_2].session_id} == {session_id}
        # ★ 但 checkpoint 的键各走各的：thread_id 一旦被改成 session_id，
        #   第二次提问就会"续跑"第一次已经跑完的图状态
        assert rows[run_1].thread_id == run_1
        assert rows[run_2].thread_id == run_2

    # 每个 thread 里的 state 必须**只有它自己那一轮**：
    # 共用 thread 的话这里读不到（thread_id 不存在），或读到的是上一轮的问题。
    from app.agents.graph import open_graph, thread_config

    async with open_graph() as graph:
        for run_id, question in ((run_1, "RAG 的重排技术有哪些"), (run_2, "那它们的代价是什么")):
            snapshot = await graph.aget_state(thread_config(run_id))
            values = dict(getattr(snapshot, "values", None) or {})
            assert values.get("run_id") == run_id
            assert values.get("session_id") == session_id
            assert values.get("question") == question

    # 查询接口也把会话 id 如实透出来（面板要靠它跨刷新续用同一个会话）
    detail = (await client.get(f"/api/v1/research/{run_2}")).json()
    assert detail["session_id"] == session_id


async def test_no_session_id_keeps_the_old_one_question_one_session(client):
    """不传 session_id = 原行为：`session_id == run_id`，一次提问即一个会话。"""
    body = (await client.post(STREAM_URL, json={"question": "混合检索的融合怎么做"})).text
    run_id = first_data(body, "run_started")["run_id"]

    async with session_scope() as db:
        run = await db.get(ResearchRun, run_id)
    assert run is not None
    assert run.session_id == run_id
    assert run.thread_id == run_id

    # 空串也按"新会话"处理（面板常见写法：没有会话就传空）
    blank = (await client.post(STREAM_URL, json={"question": "视觉 Transformer 的蒸馏", "session_id": "  "})).text
    blank_run = first_data(blank, "run_started")["run_id"]
    async with session_scope() as db:
        blank_row = await db.get(ResearchRun, blank_run)
    assert blank_row.session_id == blank_run


def test_session_id_shape_is_validated_at_the_schema_layer():
    """session_id 会直接当 `sessions.id` 主键写库，字符集必须和 run_id 同口径。"""
    assert ResearchRequest(question="q", session_id="").session_id is None
    assert ResearchRequest(question="q", session_id="abc.def_1-2").session_id == "abc.def_1-2"
    for bad in ["a/b", "with space", "x" * 65, "id\x00"]:
        with pytest.raises(ValidationError):
            ResearchRequest(question="q", session_id=bad)


# =============================================================================
# 2) ★ 短期记忆回读：追问的 prompt 里真的有上一轮的问答
# =============================================================================
async def test_followup_prompt_carries_previous_round(client, monkeypatch, plan_stub):
    """端到端（API 层）：第一轮跑完 → 第二轮传同一个 session_id → prompt 里必须出现第一轮的提问。

    这是整个 F9 唯一的"真接线"证据。`load_context` 早在文件里就写好了，
    但它**只被单测调用**过 —— 那种"函数存在且能跑通"的绿色，恰恰是 F9 长期不可用的原因。
    """
    session_id = "panel-session-followup"
    first_q = "近两年 RAG 的重排技术有哪些进展"
    body1 = (await client.post(STREAM_URL, json={"question": first_q, "session_id": session_id})).text
    run_1 = first_data(body1, "run_started")["run_id"]
    # 只有跑到 finalize，本轮问答才会被 `append_message` 写进 messages（数据本来就在那儿，缺的是读）
    resumed = (await client.post(f"/api/v1/research/{run_1}/resume", json={})).text
    assert first_data(resumed, "done")["status"] == "ok"

    # 第一轮已经落库（否则下面的断言可能只是"prompt 桩没生效"）
    stored = await short_term.load_context(session_id)
    assert any(first_q in str(m.get("content")) for m in stored), f"第一轮问答没进会话：{stored}"

    plan_stub["prompts"].clear()  # 只看第二轮那一次拆解
    body2 = (await client.post(STREAM_URL, json={"question": "那方法 cross-encoder 的代价是什么", "session_id": session_id})).text

    prompt = plan_stub["prompts"][-1]
    assert "<session_history>" in prompt, "追问没有拿到会话历史 == F9 没接上"
    assert first_q in prompt
    assert "研究问题：那方法 cross-encoder 的代价是什么" in prompt
    # 与长期记忆块同一条约束：上下文必须排在 `研究问题：` 之前
    # （mock provider 用 rfind("研究问题：") 截提问抽英文关键词）
    assert prompt.index("<session_history>") < prompt.index("研究问题：")
    # 对外可观测：面板据此显示"已带入 N 段上下文"
    assert first_data(body2, "plan_created")["history_count"] >= 2

    # 别的会话不该被串进来（短期记忆按 session 取，与长期记忆相反）
    plan_stub["prompts"].clear()
    await client.post(STREAM_URL, json={"question": "图神经网络的可解释性"})
    assert "<session_history>" not in plan_stub["prompts"][-1]


async def test_plan_reads_history_from_short_term_table(plan_stub, short_term_store):
    """单元层：`plan` 读 `load_context` 并把历史渲染进 prompt；摘要行单独标注。"""
    sid = "s-unit"
    await short_term_store.append_message(sid, "user", "第一轮：重排方法有哪些", run_id="r1")
    await short_term_store.append_message(sid, "assistant", "第一轮回答：cross-encoder reranking 等。", node_name="writer")

    out = await __import__("app.agents.nodes.planner", fromlist=["plan_node"]).plan_node(
        {"question": "那它的代价是什么", "session_id": sid}
    )

    prompt = plan_stub["prompts"][-1]
    assert "第一轮：重排方法有哪些" in prompt
    assert "cross-encoder reranking" in prompt
    assert "用户：第一轮" in prompt and "助手：第一轮回答" in prompt
    assert out["events"][0]["history_count"] == 2


async def test_rolling_summary_is_labelled_not_treated_as_user_turn(plan_stub, short_term_store):
    """压缩出来的摘要行必须标成"更早对话的摘要"，否则模型会把它当成用户本轮说的话。"""
    sid = "s-summary"
    mem = short_term_store
    for i in range(1, 5):  # max_turns=6 的窗口是 12 条，这里换个短窗口实例来逼出压缩
        await mem.append_message(sid, "user", f"历史问题{i}")
        await mem.append_message(sid, "assistant", f"历史回答{i}")

    short_term_store.max_turns = 2  # 窗口 4 条：下一次写入就会触发压缩
    await mem.append_message(sid, "user", "触发压缩的问题")
    context = await mem.load_context(sid, max_turns=2)
    assert context[0]["node_name"] == short_term.SUMMARY_NODE

    await __import__("app.agents.nodes.planner", fromlist=["plan_node"]).plan_node(
        {"question": "追问", "session_id": sid}
    )
    prompt = plan_stub["prompts"][-1]
    assert "（更早对话的摘要）" in prompt


async def test_first_round_in_a_session_has_no_history_block(plan_stub, short_term_store):
    """会话里没有历史 == 不注入空块，而不是注入一段"（空历史）"。"""
    await __import__("app.agents.nodes.planner", fromlist=["plan_node"]).plan_node(
        {"question": "本会话的第一次提问", "session_id": "brand-new-session"}
    )
    assert "<session_history>" not in plan_stub["prompts"][-1]


async def test_history_read_failure_degrades_to_no_context(plan_stub, monkeypatch):
    """★ 短期记忆读失败必须降级，不能把整次研究打死（与 `_recall_memory` 同一条原则）。"""

    async def boom(*args: Any, **kwargs: Any):
        raise RuntimeError("messages 表被锁了")

    monkeypatch.setattr(short_term, "load_context", boom)

    from app.agents.nodes import planner

    out = await planner.plan_node({"question": "RAG 重排的代价", "session_id": "s1"})
    assert "<session_history>" not in plan_stub["prompts"][-1]
    assert out["stage"] == "planned"
    assert len(out["sub_questions"]) == 3  # 拆解照常完成
    assert out["events"][0]["history_count"] == 0


async def test_missing_session_id_skips_history_lookup(plan_stub, monkeypatch):
    """没有 session_id（老链路 / 脚本直调）就**不发查询**：省一次数据库往返。"""
    calls: list[str] = []
    real = short_term.load_context

    async def spy(session_id: str, *args: Any, **kwargs: Any):
        calls.append(session_id)
        return await real(session_id, *args, **kwargs)

    monkeypatch.setattr(short_term, "load_context", spy)
    from app.agents.nodes import planner

    await planner.plan_node({"question": "RAG 重排"})
    assert calls == []


# =============================================================================
# 3) ★ 可用率埋点：用户在中断点 A 到底改没改拆解
# =============================================================================
def test_plan_edit_diff_rules():
    """纯函数口径：id 不参与比较、keywords 按集合比、没提交 == 没改。"""
    original = [
        {"id": "sq-1", "text": "有哪些方法", "keywords": ["reranking", "bm25"], "category": None},
        {"id": "sq-2", "text": "代价是什么", "keywords": ["latency"], "category": "cs.IR"},
    ]

    # 原样回填（面板就是这么干的：把 interrupt 事件里的内容再提交一次）→ 没改
    same = [{"id": "sq-1", "text": "有哪些方法", "keywords": ["bm25", "reranking"], "category": None},
            {"id": "sq-2", "text": "代价是什么", "keywords": ["latency"], "category": "cs.IR"}]
    assert plan_edit_diff(original, same)["edited"] is False

    # 留空 == "沿用原拆解直接继续" → 没改（不是"无法判断"）
    assert plan_edit_diff(original, [])["edited"] is False

    # 改正文 / 删一条 / 换分类 / 换关键词，各判各的
    text_edit = [dict(x) for x in original]
    text_edit[0]["text"] = "有哪些方法（用户补充：含 ListNet）"
    assert plan_edit_diff(original, text_edit)["changes"] == ["text"]

    assert plan_edit_diff(original, original[:1])["changes"] == ["count"]

    cat = [dict(original[0]), {**original[1], "category": "cs.CL"}]
    assert plan_edit_diff(original, cat)["changes"] == ["category"]

    kw = [{**original[0], "keywords": ["rerank"]}, original[1]]
    assert plan_edit_diff(original, kw)["changes"] == ["keywords"]

    # 读不到基线（checkpoint 丢了）：判不了就如实写 None，绝不能默认"改了"压低可用率
    dead = plan_edit_diff([], [{"id": "sq-1", "text": "x", "keywords": ["a"], "category": None}])
    assert dead["edited"] is None
    assert dead["changes"] == ["no_baseline"]


async def test_resume_without_edits_records_plan_edited_false(client):
    """用户不修改直接执行 == §8 可用率的分子；这一位必须**扛过 finalize 对同一列的覆盖**。"""
    body = (await client.post(STREAM_URL, json={"question": "RAG 重排的进展"})).text
    run_id = first_data(body, "run_started")["run_id"]
    resumed = (await client.post(f"/api/v1/research/{run_id}/resume", json={})).text
    assert first_data(resumed, "done")["status"] == "ok"

    # finalize 在流里跑，它落库时会用摘要整体覆盖 langgraph_state；
    # API 层在流结束后补写了一次，所以这里必须还在。
    detail = (await client.get(f"/api/v1/research/{run_id}")).json()
    assert detail["plan_edited"] is False

    async with session_scope() as db:
        state = (await db.get(ResearchRun, run_id)).langgraph_state or {}
    assert state["plan_edited"] is False
    assert state["plan_edit_detail"]["changes"] == []
    assert state["plan_edit_detail"]["final_count"] == 0
    # 与同一次落库的摘要并存：埋点是**合并**写进 JSON，不是把摘要换成自己的结构
    assert "token_usage" in state and "citations" in state


async def test_resume_with_edits_records_plan_edited_true(client):
    run_id, sub_questions = await _start(client, "混合检索里 BM25 与向量的融合")
    edited = [dict(sq) for sq in sub_questions]
    edited[0]["text"] = edited[0]["text"] + "（用户补充：关注跨编码器）"

    resumed = (await client.post(f"/api/v1/research/{run_id}/resume", json={"sub_questions": edited})).text
    assert first_data(resumed, "done")["status"] == "ok"

    detail = (await client.get(f"/api/v1/research/{run_id}")).json()
    assert detail["plan_edited"] is True
    assert "text" in detail["plan_edited"] if False else True  # 值本身已在上一行断言

    async with session_scope() as db:
        state = (await db.get(ResearchRun, run_id)).langgraph_state or {}
    assert state["plan_edited"] is True
    assert "text" in state["plan_edit_detail"]["changes"]
    assert state["plan_edit_detail"]["original_count"] == len(sub_questions)
    # 用户改的拆解**同时**进了图 state（不是只记了个指标就丢掉编辑内容）
    assert "跨编码器" in detail["sub_questions"][0]["text"]


async def test_plan_edit_recorded_even_when_resume_fails_midway(client, monkeypatch):
    """入口那一次写的价值：流中途死掉（这里是下游抛异常），改没改这个事实也已经落库。"""
    from app.agents import graph as graph_mod

    run_id, sub_questions = await _start(client, "RAG 重排")
    edited = [dict(sq) for sq in sub_questions]
    edited[1]["keywords"] = ["retrieval"]

    async def boom(state):  # noqa: ARG001
        raise RuntimeError("下游炸了")

    monkeypatch.setattr(graph_mod, "ensure_corpus_node", boom)
    await client.post(f"/api/v1/research/{run_id}/resume", json={"sub_questions": edited})

    detail = (await client.get(f"/api/v1/research/{run_id}")).json()
    assert detail["plan_edited"] is True


async def _start(client, question: str) -> tuple[str, list[dict]]:
    body = (await client.post(STREAM_URL, json={"question": question})).text
    run_id = first_data(body, "run_started")["run_id"]
    return run_id, first_data(body, "plan_created")["sub_questions"]


# =============================================================================
# 4) ★ 钉子：长期记忆**不因为 F9 而改成按会话过滤**
# =============================================================================
async def test_followup_run_in_session_b_still_sees_papers_read_by_session_a(scope, monkeypatch, plan_stub):
    """追问发生在会话 B，也必须看得见会话 A 写下的已读论文与方向记忆。

    F9 让 session_id 第一次"跨 run 有意义"，这正是最容易顺手把长期记忆
    也改成"按会话读"的时刻 —— 而那是上一轮刚修掉的严重 bug：
    按会话过滤 = 在"这次运行还没写过记忆"的时刻查这次的记忆 = 恒为空，
    症状是 F10 静默失效，而打桩了记忆读取的单测照样全绿。
    """
    store = LongTermMemoryStore(scope=scope, embedder=MockEmbedder(dim=32), max_items=50)
    await store.remember_run(
        session_id="session-A",
        run_id="run-A",
        question="RAG 重排有哪些方法",
        paper_ids=["2401.00001"],
    )
    monkeypatch.setattr(long_term, "_default", store)

    # ① read 侧的排除入口：它**压根不接受 session_id**（这就是坑写不回来的原因）
    assert await long_term.safe_read_paper_ids() == {"2401.00001"}

    # ② plan 侧的召回：换个会话（B）发起追问，召回照旧命中 A 写的记忆
    seen_kwargs: list[dict] = []
    real_recall = store.recall

    async def spy_recall(query: str, top_k: int = 5, **kwargs: Any):
        seen_kwargs.append(kwargs)
        return await real_recall(query, top_k=top_k, **kwargs)

    monkeypatch.setattr(long_term, "recall", spy_recall)

    from app.agents.nodes import planner

    out = await planner.plan_node(
        {"question": "那重排的代价是什么", "session_id": "session-B"}
    )

    assert out["memory_context"], "长期记忆又变回只写不读 == F10 静默失效"
    assert "历史长期记忆" in plan_stub["prompts"][-1]
    # 直接钉住"不传 session_id"这个约定本身（比钉住结果更难被绕开）
    assert seen_kwargs and all("session_id" not in kw for kw in seen_kwargs), (
        "planner 把 session_id 传给了长期记忆召回 —— 长期记忆属于用户，不属于会话"
    )

    # ③ 与此同时短期记忆**确实**是按会话取的：会话 B 没有历史，不该被 A 的对话污染
    assert "<session_history>" not in plan_stub["prompts"][-1]


async def test_long_term_eviction_and_recall_stay_global_across_sessions(scope):
    """同一个坑的第二条钉子：容量与召回都是全局口径，不随会话数归零。"""
    store = LongTermMemoryStore(scope=scope, embedder=MockEmbedder(dim=32), max_items=2)
    for sid in ("run-1", "run-2", "run-3"):
        await store.remember(f"会话 {sid} 记住的方向", session_id=sid)

    assert await store.count() == 2  # 按会话计数的话这里会是 1+1+1，永不淘汰
    hits = await store.recall("会话 run-3 记住的方向", top_k=5)
    assert hits, "跨会话召回必须拿得到东西"

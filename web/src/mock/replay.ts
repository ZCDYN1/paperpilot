import type { SSEEvent } from "@/api/events";
import {
  MOCK_RUN_ID,
  MOCK_SESSION_ID,
  fixtures,
  mockCitations,
  mockComparison,
  mockReport,
  mockTokenByNode,
  mockVerification,
} from "@/mock/fixtures";

/**
 * Mock SSE 流：把一次真实研究跑完整重放一遍，用来在没有后端时验证时间线组件。
 *
 * 事件名与 data 字段严格照 `app/observability/events.py` 的 helper 签名构造，
 * 所以后端改了字段，这里也会第一时间对不上（这是故意的：契约变更必须可见）。
 *
 * ★ 关键行为：`/stream` 会停在 `interrupted` 并正常收尾，剩下的事件由 `/resume` 那条流
 * 继续推 —— 与后端"两段流式"的真实形态一致，而不是把中断当动画糊过去。
 */

const now = () => Date.now();

function ev(name: string, data: Record<string, unknown>): SSEEvent {
  return { type: name as SSEEvent["type"], name, data: { schema_version: "1", ...data }, receivedAt: now() };
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
    const t = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    function onAbort() {
      clearTimeout(t);
      reject(new DOMException("aborted", "AbortError"));
    }
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/** 形状严格照 `app/schemas/research.py` 的 SubQuestion：只有 id / text / keywords / category */
const planSubQuestions = [
  { id: "q1", text: "Self-RAG / CRAG 这类方法如何决定检索时机？", keywords: ["self-rag", "adaptive retrieval", "on-demand retrieval"], category: "cs.CL" },
  { id: "q2", text: "各方法相对 baseline 的事实性提升幅度是多少？", keywords: ["factuality", "ASQA", "hallucination reduction"], category: "cs.CL" },
  { id: "q3", text: "额外训练或额外推理调用带来的工程代价？", keywords: ["inference latency", "fine-tuning cost", "retrieval overhead"], category: "cs.LG" },
  { id: "q4", text: "作者在原文里自述的局限有哪些？", keywords: ["limitations", "failure cases", "ablation"], category: "cs.CL" },
];

const reportChunks = mockReport.match(/[\s\S]{1,18}/g) ?? [];

/** 第一段：拆解 → HITL 中断 */
async function* phaseOne(signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  yield ev("run_started", { run_id: MOCK_RUN_ID, question: fixtures.question });
  await sleep(320, signal);

  yield ev("node_started", { node: "plan", label: "拆解研究问题", ts: now() });
  await sleep(260, signal);
  yield ev("thought", { node: "plan", delta: "识别到这是一个「方法横评」类问题，需要按机制 / 效果 / 代价 / 局限四个维度切分。" });
  await sleep(420, signal);

  yield ev("plan_created", { sub_questions: planSubQuestions, rationale: "四个维度各自可独立检索，并行召回后在对比表里汇合。", count: 4 });
  await sleep(200, signal);

  // ★ HITL 中断点 A：必须补发 interrupted，否则前端只看到流"卡住不动"
  yield ev("interrupted", {
    at: "plan_confirm",
    reason: "等待用户确认或修改子问题拆解",
    payload: { sub_questions: planSubQuestions },
    resume_token: MOCK_RUN_ID,
  });
  await sleep(160, signal);
  yield ev("done", { run_id: MOCK_RUN_ID, status: "interrupted" });
}

/** 第二段：恢复后一路跑到报告 */
async function* phaseTwo(edited: boolean, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  yield ev("run_started", { run_id: MOCK_RUN_ID, question: fixtures.question, resumed: true });
  yield ev("node_started", { node: "ensure_corpus", label: "准备语料", ts: now() });
  await sleep(300, signal);
  yield ev("corpus", { fetched: 18, cached: 6, total: 24, corpus_ready: true });
  await sleep(240, signal);

  yield ev("node_started", { node: "retrieve", label: "多路并行检索", ts: now() });
  for (const [i, q] of planSubQuestions.entries()) {
    const callId = `call_${i}`;
    yield ev("tool_call", {
      node: "retrieve",
      tool: "arxiv_search",
      args: { query: q.text.slice(0, 24), cat: "cs.CL", since: "2020", has_code: i === 2 },
      call_id: callId,
    });
    await sleep(380 + i * 120, signal);
    yield ev("tool_result", { call_id: callId, status: "ok", duration_ms: 380 + i * 120, count: 8 - i });
    yield ev("retrieval", {
      sub_question_id: q.id,
      count: 3,
      hits: [
        { arxiv_id: "2310.11511", title: "Self-RAG: Learning to Retrieve, Generate, and Critique", score: 0.874, section: "§3 Method", source: "arxiv" },
        { arxiv_id: "2401.15884", title: "Corrective Retrieval Augmented Generation", score: 0.812, section: "Abstract", source: "arxiv" },
        { arxiv_id: "2112.01488", title: "ColBERTv2", score: 0.764, section: "§4 Experiments", source: "local_index" },
      ],
    });
    await sleep(180, signal);
  }

  yield ev("node_started", { node: "read", label: "精读与证据抽取", ts: now() });
  await sleep(500, signal);
  yield ev("thought", { node: "read", delta: "去重后 4 篇论文进入抽取，产出 27 条「论断 → 证据」对。" });
  // ★ rescued=0 也照发：这是回归探测，不是装饰
  yield ev("citation_rescue", {
    node: "read",
    checked: 27,
    rescued: 0,
    dropped: 2,
    recovered_by: {},
    drop_reasons: { no_quote_match: 1, quote_not_in_chunk: 1 },
  });
  await sleep(360, signal);

  yield ev("node_started", { node: "compare", label: "结构化对比表抽取", ts: now() });
  await sleep(620, signal);
  yield ev("thought", { node: "compare", delta: `生成 ${mockComparison.length} 行 × 9 列对比表，${countEmptyCells(mockComparison)} 个单元格在原文中确实没有依据，按空处理。` });
  await sleep(300, signal);

  yield ev("node_started", { node: "verify", label: "引用校验", ts: now() });
  await sleep(560, signal);
  yield ev("verification", {
    passed: false,
    score: 0.926,
    unsupported: mockVerification.unsupported,
    revision_count: 0,
  });
  await sleep(320, signal);

  yield ev("node_started", { node: "revise", label: "剔除无支撑论断并重写", ts: now() });
  await sleep(480, signal);
  yield ev("verification", { passed: true, score: 0.963, unsupported: [], revision_count: 1 });
  await sleep(260, signal);

  yield ev("node_started", { node: "write", label: "生成报告", ts: now() });
  for (const delta of reportChunks) {
    yield ev("token", { node: "write", delta });
    await sleep(26, signal);
  }

  yield ev("final_report", { markdown: mockReport, citations: mockCitations.citations });
  await sleep(200, signal);
  yield ev("token_usage", {
    total: { prompt: 1_284_500, completion: 157_800, total: 1_442_300 },
    cost_estimate: edited ? 0.42 : 0.39,
    by_node: mockTokenByNode,
  });
  await sleep(160, signal);
  yield ev("done", { run_id: MOCK_RUN_ID, status: "ok" });
}

function countEmptyCells(rows: typeof mockComparison): number {
  return rows.reduce((sum, r) => sum + Object.values(r).filter((v) => v === null || v === "").length, 0);
}

export function mockStream(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
  if (path.includes("/resume")) {
    const edited = Array.isArray((body as { sub_questions?: unknown[] })?.sub_questions);
    return phaseTwo(edited, signal);
  }
  return phaseOne(signal);
}

export const mockSessionId = MOCK_SESSION_ID;

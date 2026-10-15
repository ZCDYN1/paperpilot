/**
 * SSE 事件契约 —— 与后端 `app/observability/events.py` 的 `EVENT_TYPES` 一一对应。
 *
 * ⚠️ 后端把这 16 种事件视为**对外契约**（枚举化，拼错的事件名直接抛错），
 * 且每帧 data 都带 `schema_version`。前端任何一处 switch 漏了某个事件类型，
 * 那条信息就静默消失了 —— 所以 Timeline 用「未知事件也要渲染」的兜底分支。
 *
 * 帧格式固定：`event: <type>\ndata: <json>\n\n`
 */

export const SSE_EVENT_TYPES = [
  "run_started",
  "node_started",
  "plan_created",
  "corpus",
  "tool_call",
  "tool_result",
  "retrieval",
  "thought",
  "token",
  "verification",
  "interrupted",
  "final_report",
  "token_usage",
  "error",
  "done",
  "citation_rescue",
] as const;

export type SSEEventType = (typeof SSE_EVENT_TYPES)[number];

export interface SSEData {
  schema_version?: string;
  [key: string]: unknown;
}

export interface SSEEvent {
  type: SSEEventType | "unknown";
  /** 原始事件名（后端新增了类型而前端枚举未更新时，靠它仍能显示） */
  name: string;
  data: SSEData;
  /** 本地接收时间，用于「耗时」可视化 */
  receivedAt: number;
}

/**
 * 后端声明"这段流式预览作废"的错误码。
 * 源：`app/agents/nodes/_common.py` 的 `STREAM_ABORT_CODE`（`TokenSink.abort` 只在
 * **这一次尝试真的吐过字**时才发；一个字都没吐过就不发，好让重试那一轮还能流）。
 *
 * 处置口径是「宁少流、不重流」：一次尝试已经吐过字又失败时，补发这条
 * `error{code, recoverable: true}` 声明该段作废并关闭本次 sink，重试那一轮
 * **不再往外吐**。所以前端收到它必须清掉已累积的预览，否则界面上会永远留着
 * 一段已作废的半截文本，而它和随后的 `final_report` 不一致。
 */
export const STREAM_ABORT_CODE = "token_stream_aborted";

/**
 * `error` 事件在真实后端里出现过的 `code` 全集（前端只按"要不要作废预览"分两族）：
 * - `token_stream_aborted`（`app/agents/nodes/_common.py` 的 `STREAM_ABORT_CODE`，
 *   `recoverable: true`）—— 唯一的**明示作废**信号，见上面那条注释；
 * - 降级/局部失败通知，**不作废预览**（那条流后面真的还会出报告）：
 *   `circuit_open` / `circuit_reopened`（`app/observability/resilience_events.py`，
 *   `node:"resilience"`）、`llm_circuit_open`（`_common._on_circuit_open`）、
 *   `extract_failed`（`app/agents/nodes/comparator.py`，`node:"compare"`）、
 *   `persist_failed`（`app/agents/nodes/finalize.py`，`node:"finalize"`），一律 `recoverable: true`；
 * - `internal_error` / `resume_failed`（`app/api/v1/research.py` 的失败分支，
 *   `node: "api"`，`recoverable: false`）—— 紧接着一定有一条
 *   `done{status:"error"}`，此时永远不会有 `final_report`，半截预览同样得清。
 *   处置逻辑在 `useResearchStream.ts` 的 `reduceRunEvent`。
 */

/** 时间线里按语义归组，决定图标与配色 */
export const EVENT_GROUP: Record<string, "lifecycle" | "plan" | "retrieval" | "llm" | "verify" | "report" | "alert"> = {
  run_started: "lifecycle",
  node_started: "lifecycle",
  done: "lifecycle",
  plan_created: "plan",
  interrupted: "plan",
  corpus: "retrieval",
  retrieval: "retrieval",
  tool_call: "retrieval",
  tool_result: "retrieval",
  thought: "llm",
  token: "llm",
  token_usage: "llm",
  verification: "verify",
  citation_rescue: "verify",
  final_report: "report",
  error: "alert",
};

export function groupOf(type: string): (typeof EVENT_GROUP)[string] {
  return EVENT_GROUP[type] ?? "lifecycle";
}

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

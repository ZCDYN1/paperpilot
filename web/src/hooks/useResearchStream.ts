import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";
import type { SSEEvent } from "@/api/events";
import type { ResearchFilters, ResearchRequest } from "@/api/domain";

/**
 * 驱动一次研究跑：消费 SSE 流、累积事件、处理 HITL 中断与恢复。
 *
 * ★ 为什么 `token` 要在 reducer 里合并而不是各自成一条事件：
 * 后端 writer 会推几百条 `token` 增量，逐条渲染等于几百次 React commit，
 * 时间线会被刷屏且明显掉帧。这里把它们并进同一条流式条目。
 */

export interface RunState {
  events: SSEEvent[];
  /** 合并后的流式正文（writer 的 token 增量累加） */
  streamed: string;
  status: "idle" | "running" | "awaiting_input" | "done" | "error";
  /** 中断点 A 等待确认的子问题 */
  pendingSubQuestions: Array<Record<string, unknown>>;
  runId: string | null;
  error: string | null;
}

const initial: RunState = {
  events: [],
  streamed: "",
  status: "idle",
  pendingSubQuestions: [],
  runId: null,
  error: null,
};

/**
 * `start` 的第二个参数：除 `question` / `session_id` 之外的完整请求体。
 *
 * ★ 这里原来是 `filters?: Record<string, unknown>` —— 松到把 UI 的
 * `{cat, since, hasCode}` 一路放行到 `POST /research/stream`，后端
 * `ResearchRequest` / `ResearchFilters` 都是 `extra="forbid"`，于是整请求 422，
 * 而 TS 一声不响。改成契约类型之后，字段名写错或多塞一个键在编译期就红。
 */
export type ResearchRunOptions = Omit<ResearchRequest, "question" | "session_id"> & {
  filters?: ResearchFilters;
};

export function useResearchStream() {
  const [state, setState] = useState<RunState>(initial);
  const abortRef = useRef<AbortController | null>(null);
  const sessionRef = useRef<string>("");

  useEffect(() => () => abortRef.current?.abort(), []);

  const apply = useCallback((event: SSEEvent) => {
    setState((prev) => {
      const next: RunState = { ...prev, events: [...prev.events, event] };
      const data = event.data as Record<string, unknown>;

      switch (event.name) {
        case "run_started":
          next.runId = (data.run_id as string) ?? prev.runId;
          next.status = "running";
          break;
        case "token":
          next.streamed = prev.streamed + String(data.delta ?? "");
          break;
        case "interrupted":
          next.status = "awaiting_input";
          next.pendingSubQuestions = Array.isArray((data.payload as { sub_questions?: unknown[] })?.sub_questions)
            ? ((data.payload as { sub_questions: Array<Record<string, unknown>> }).sub_questions)
            : [];
          break;
        case "error":
          next.error = `${String(data.code ?? "error")}: ${String(data.message ?? "")}`;
          break;
        case "done": {
          const s = String(data.status ?? "ok");
          next.status = s === "interrupted" ? "awaiting_input" : s === "error" ? "error" : "done";
          break;
        }
      }
      return next;
    });
  }, []);

  const consume = useCallback(
    async (stream: AsyncGenerator<SSEEvent>) => {
      setState((p) => ({ ...p, status: "running", error: null }));
      try {
        for await (const event of stream) apply(event);
      } catch (err) {
        if ((err as Error)?.name === "AbortError") {
          setState((p) => ({ ...p, status: "idle" }));
          return;
        }
        setState((p) => ({ ...p, status: "error", error: (err as Error).message }));
      }
    },
    [apply],
  );

  /**
   * 发起一次研究。`options` 由调用方按后端形状构造（见 `ResearchRunOptions`），
   * 这里只补 `question` 与内部生成的 `session_id`，不做任何字段翻译。
   */
  const start = useCallback(
    (question: string, options?: ResearchRunOptions) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      if (!sessionRef.current) sessionRef.current = `web-${Date.now().toString(36)}`;
      setState({ ...initial, status: "running" });
      void consume(
        api.research.stream({ ...options, question, session_id: sessionRef.current }, controller.signal),
      );
    },
    [consume],
  );

  // resume 要读最新的 runId，用 ref 拿，避免把它塞进依赖数组导致闭包反复重建
  const stateRef = useRef(state);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  /** 用户在子问题编辑器里改过就传数组，沿用原拆解就传 undefined */
  const resume = useCallback(
    (subQuestions?: Array<Record<string, unknown>>) => {
      const runId = stateRef.current.runId;
      if (!runId) {
        setState((p) => ({ ...p, status: "error", error: "没有可恢复的运行：先发起一次研究。" }));
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      setState((p) => ({ ...p, status: "running", pendingSubQuestions: [] }));
      void consume(api.research.resume(runId, subQuestions, controller.signal));
    },
    [consume],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(initial);
  }, []);

  /** 按节点聚合的耗时与事件数，给时间线顶部统计条用 */
  const nodeStats = useMemo(() => {
    const map = new Map<string, { count: number; first: number; last: number }>();
    for (const e of state.events) {
      const node = String((e.data as Record<string, unknown>).node ?? e.name);
      const cur = map.get(node) ?? { count: 0, first: e.receivedAt, last: e.receivedAt };
      map.set(node, { count: cur.count + 1, first: cur.first, last: e.receivedAt });
    }
    return [...map.entries()].map(([node, v]) => ({ node, ...v, ms: v.last - v.first }));
  }, [state.events]);

  return { ...state, nodeStats, start, resume, stop, reset, sessionId: sessionRef.current };
}

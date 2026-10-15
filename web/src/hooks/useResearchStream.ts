import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "@/api/client";
import { STREAM_ABORT_CODE, type SSEEvent } from "@/api/events";
import type { ResearchFilters, ResearchRequest } from "@/api/domain";

/**
 * 驱动一次研究跑：消费 SSE 流、累积事件、处理 HITL 中断与恢复。
 *
 * ★ token 事件走 `TokenBatcher` 而不是逐条 setState —— 原因见下面 `TOKEN_FLUSH_MS`。
 * ★ `streamed` 是**预览**，不是最终报告；权威内容在 `finalReport`（`final_report` 事件）。
 * ★ 纯逻辑（批处理 + reducer）都拆成模块级导出的 `TokenBatcher` / `reduceRunEvent` /
 *   `voidPreview`，这样**没有 DOM 也能被测**（本仓没装 jsdom，hook 本身跑不起来）。
 *   见 `src/hooks/__tests__/runStreamLogic.spec.ts`。
 */

export interface RunState {
  events: SSEEvent[];
  /**
   * 已提交的流式正文（writer 的 token 增量累加）。
   * ⚠️ 它**不等于**最终报告：⚠️ 未支撑一节、参考文献、页脚是代码追加的、不进流。
   * 权威内容永远以 `finalReport`（来自 `final_report` 事件）为准。
   */
  streamed: string;
  /** `final_report` 到达后的权威正文；没有它时界面只能显示预览 */
  finalReport: string | null;
  /**
   * true = 这段预览已作废，界面上必须标成"已作废"而不是留着。
   * 三个触发点（都发生在 `final_report` 之前，所以不可能盖掉权威内容）：
   * ① `error{code:"token_stream_aborted"}` —— 后端"宁少流、不重流"的明示；
   * ② `error{recoverable:false}` —— API 层的 `internal_error`，这次运行不会再出报告；
   * ③ `done{status:"error"}` —— 流终止而权威内容从未到达。
   */
  previewVoid: boolean;
  status: "idle" | "running" | "awaiting_input" | "done" | "error";
  /** 中断点 A 等待确认的子问题 */
  pendingSubQuestions: Array<Record<string, unknown>>;
  runId: string | null;
  error: string | null;
}

const initial: RunState = {
  events: [],
  streamed: "",
  finalReport: null,
  previewVoid: false,
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

/**
 * `token` 的批处理窗口。
 *
 * 真实一次生成实测 800+ 条 token 事件（架构 §8）。逐条 `setState` 的问题不只是
 * "800 次 commit"——每次都还要 `events: [...prev, e]` 整份复制一个正在增长的数组，
 * 并且让 `nodeStats` 对全量事件重算，合起来是 O(n²)。
 * 攒到 ~100ms 一次批量提交后，800 条变成约 10 次提交，观感上依然是连续打字。
 */
const TOKEN_FLUSH_MS = 100;

/** 后端在"已经吐过字又失败"时发的作废信号（`app/agents/nodes/_common.STREAM_ABORT_CODE`） */
const TOKEN_STREAM_ABORTED = STREAM_ABORT_CODE;

/** 定时器句柄的最小抽象：默认走 `globalThis`，测试里注入假时钟才能数清提交次数 */
export interface Scheduler {
  setTimeout(cb: () => void, ms: number): number;
  clearTimeout(id: number): void;
}

const defaultScheduler: Scheduler = {
  setTimeout: (cb, ms) => globalThis.setTimeout(cb, ms) as unknown as number,
  clearTimeout: (id) => globalThis.clearTimeout(id as unknown as Parameters<typeof globalThis.clearTimeout>[0]),
};

/**
 * `token` 的缓冲器：把几百条增量攒成一个批量提交（待办①）。
 *
 * 三条规矩：
 * - `push` 只写缓冲区，**绝不自带 setState**；窗口内第一条起计时；
 * - `flush` 按**到达顺序**整批交给 `onFlush`，然后清空（时间线不会错序）；
 * - `discard` 直接丢（开新一轮 / 重置 / 卸载用）。
 */
export class TokenBatcher {
  private batch: SSEEvent[] = [];
  private text = "";
  private timer: number | null = null;

  constructor(
    private readonly onFlush: (batch: SSEEvent[], text: string) => void,
    private readonly delayMs: number = TOKEN_FLUSH_MS,
    private readonly scheduler: Scheduler = defaultScheduler,
  ) {}

  /** 收到一条 token 增量 */
  push(event: SSEEvent): void {
    this.batch.push(event);
    this.text += String((event.data as Record<string, unknown>).delta ?? "");
    if (this.timer === null) {
      this.timer = this.scheduler.setTimeout(() => {
        this.timer = null;
        this.flush();
      }, this.delayMs);
    }
  }

  /** 到点或被非 token 事件打断：整批落地 */
  flush(): void {
    if (this.timer !== null) {
      this.scheduler.clearTimeout(this.timer);
      this.timer = null;
    }
    if (this.batch.length === 0) return;
    const batch = this.batch;
    const text = this.text;
    this.batch = [];
    this.text = "";
    this.onFlush(batch, text);
  }

  /** 丢掉还没提交的尾巴（不回调 onFlush） */
  discard(): void {
    if (this.timer !== null) {
      this.scheduler.clearTimeout(this.timer);
      this.timer = null;
    }
    this.batch = [];
    this.text = "";
  }

  /** 缓冲区里是否还压着没提交的增量 */
  get pending(): number {
    return this.batch.length;
  }
}

/**
 * 预览作废的**唯一**出口。
 *
 * ⚠️ 只清 `streamed` 是不够的：`EventTimeline` 的 `collapseTokens` 是从 `events`
 * 里把 token 增量拼成一行的，那半截文本会照样留在时间线上（用户看到的还是那段已作废的
 * 预览）。所以这里连 token 事件一起摘掉 —— error 行本身还在，用户知道为什么没了。
 */
export function voidPreview(prev: RunState): RunState {
  return {
    ...prev,
    events: prev.events.filter((e) => e.name !== "token"),
    streamed: "",
    previewVoid: true,
  };
}

/**
 * 一条非 token 事件如何改写整体状态（纯函数，不含 React、不含定时器）。
 * 拆出来的理由见文件头：本仓没装 jsdom，只有纯函数才能在 vitest 的 node 环境里被断言。
 */
export function reduceRunEvent(prev: RunState, event: SSEEvent): RunState {
  const data = event.data as Record<string, unknown>;
  const next: RunState = { ...prev, events: [...prev.events, event] };

  switch (event.name) {
    case "run_started":
      next.runId = (data.run_id as string) ?? prev.runId;
      next.status = "running";
      return next;

    case "interrupted":
      next.status = "awaiting_input";
      next.pendingSubQuestions = Array.isArray((data.payload as { sub_questions?: unknown[] })?.sub_questions)
        ? ((data.payload as { sub_questions: Array<Record<string, unknown>> }).sub_questions)
        : [];
      return next;

    case "error": {
      next.error = `${String(data.code ?? "error")}: ${String(data.message ?? "")}`;
      // ★ 待办②：两类作废都要清，只处理第一条是有意的漏（原实现就漏了后者）。
      // ① `token_stream_aborted`：后端明示"这段作废"（recoverable 仍是 true，
      //    因为重试那一轮会跑完 —— 但已流出的部分不作数，必须清）；
      // ② `recoverable:false` 的 error：API 层 `failure_event_code` 的 `internal_error`
      //    走的就是这条，后面不会再有 final_report，半截预览留着就是谎言。
      //    可恢复的 error（`circuit_open` / `llm_circuit_open` 这类降级通知）不清 ——
      //    那条流后面真的会接着出报告，清了反而是把有用的进度抹掉。
      const aborted = data.code === TOKEN_STREAM_ABORTED;
      const terminal = data.recoverable === false;
      return aborted || (terminal && prev.finalReport === null) ? voidPreview(next) : next;
    }

    case "final_report":
      // 权威内容到位后，预览区就该让位：streamed 只含模型生成的正文，
      // ⚠️ 节 / 参考文献 / 页脚是代码追加的，不进流，两者永远不相等。
      next.finalReport = String(data.markdown ?? "");
      next.previewVoid = false;
      return next;

    case "done": {
      const s = String(data.status ?? "ok");
      next.status = s === "interrupted" ? "awaiting_input" : s === "error" ? "error" : "done";
      // 待办②的另一半：`done{status:"error"}` 是"这次运行不会再出报告了"的终局信号
      //（`app/api/v1/research.py` 失败分支就是 error + done{status:"error"} 成对发的）。
      // 已经拿到 final_report 的话不动 —— 权威内容不该被一次收尾失败擦掉。
      return s === "error" && prev.finalReport === null ? voidPreview(next) : next;
    }
  }
  return next;
}

export function useResearchStream() {
  const [state, setState] = useState<RunState>(initial);
  const abortRef = useRef<AbortController | null>(null);
  const sessionRef = useRef<string>("");

  /** 待提交的 token 增量：到点、或遇到非 token 事件时一次性落地 */
  const [batcher] = useState(
    () =>
      new TokenBatcher((batch, text) => {
        setState((prev) => ({ ...prev, events: [...prev.events, ...batch], streamed: prev.streamed + text }));
      }),
  );

  // 卸载时丢掉未提交的增量（避免在已销毁的组件上 setState）并断开流
  useEffect(
    () => () => {
      batcher.discard();
      abortRef.current?.abort();
    },
    [batcher],
  );

  const apply = useCallback(
    (event: SSEEvent) => {
      // token 不进 setState，先进缓冲区
      if (event.name === "token") {
        batcher.push(event);
        return;
      }
      // 非 token 事件：先把缓冲区按原顺序落地，再处理它，时间线才不会错序
      batcher.flush();
      setState((prev) => reduceRunEvent(prev, event));
    },
    [batcher],
  );

  const consume = useCallback(
    async (stream: AsyncGenerator<SSEEvent>) => {
      setState((p) => ({ ...p, status: "running", error: null }));
      try {
        for await (const event of stream) apply(event);
      } catch (err) {
        batcher.flush();
        if ((err as Error)?.name === "AbortError") {
          setState((p) => ({ ...p, status: "idle" }));
          return;
        }
        // 连接断了 / 读流出错了：不会再有 final_report，这段预览同样不能留着当内容。
        // 已经拿到 final_report 就不动它（权威内容在手，预览早就不承担了展示职责）。
        setState((p) => {
          const failed: RunState = { ...p, status: "error", error: (err as Error).message };
          return p.finalReport === null ? voidPreview(failed) : failed;
        });
      } finally {
        // 流结束时缓冲区里可能还剩最后一段没到点的 token
        batcher.flush();
      }
    },
    [apply, batcher],
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
      batcher.discard();
      setState({ ...initial, status: "running" });
      void consume(
        api.research.stream({ ...options, question, session_id: sessionRef.current }, controller.signal),
      );
    },
    [consume, batcher],
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
      batcher.discard();
      // 新一轮流开始：上一段作废标记要跟着清掉，否则界面会挂着一条早已作废的横幅
      setState((p) => ({ ...p, status: "running", pendingSubQuestions: [], previewVoid: false }));
      void consume(api.research.resume(runId, subQuestions, controller.signal));
    },
    [consume, batcher],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const reset = useCallback(() => {
    abortRef.current?.abort();
    batcher.discard();
    setState(initial);
  }, [batcher]);

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

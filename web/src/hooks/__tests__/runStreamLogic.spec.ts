/**
 * 流式状态的纯逻辑测试（批处理 + reducer）。
 *
 * 为什么测这两个导出而不是测 hook 本身：本仓没装 jsdom/testing-library，
 * `useResearchStream()` 这种带 React 状态的 hook 在 node 环境里根本跑不起来。
 * 所以 `TokenBatcher` / `reduceRunEvent` 被拆成**不含 React 的模块级导出**，
 * 待办①（800+ 条要批处理）和待办②（作废要真清掉）才能在这里被证伪 ——
 * 否则这两条就只剩注释在自称"已处理"。
 */

import { describe, expect, it } from "vitest";
import type { SSEEvent } from "@/api/events";
import {
  TokenBatcher,
  reduceRunEvent,
  voidPreview,
  type RunState,
  type Scheduler,
} from "@/hooks/useResearchStream";

const base = (over: Partial<RunState> = {}): RunState => ({
  events: [],
  streamed: "",
  finalReport: null,
  previewVoid: false,
  status: "idle",
  pendingSubQuestions: [],
  runId: null,
  error: null,
  ...over,
});

const ev = (name: string, data: Record<string, unknown>): SSEEvent => ({
  type: name as SSEEvent["type"],
  name,
  data: { schema_version: "1", ...data },
  receivedAt: 0,
});

const token = (delta: string): SSEEvent => ev("token", { node: "write", delta });

/** 假时钟：只有能自己拨表，才数得清"提交了几次"这件真实掉帧的元凶 */
function fakeClock(): { scheduler: Scheduler; advance(ms: number): void; due: number } {
  const timers = new Map<number, { at: number; cb: () => void }>();
  let nowMs = 0;
  let nextId = 1;
  return {
    due: 0,
    scheduler: {
      setTimeout(cb, ms) {
        const id = nextId++;
        timers.set(id, { at: nowMs + ms, cb });
        return id;
      },
      clearTimeout(id) {
        timers.delete(id);
      },
    },
    advance(ms) {
      nowMs += ms;
      for (const [id, t] of [...timers.entries()]) {
        if (t.at <= nowMs) {
          timers.delete(id);
          t.cb();
        }
      }
    },
  };
}

/**
 * 把 hook 里那条「batcher 提交 → reduceRunEvent」的调用序列原样搬到测试里。
 * ⚠️ 提交回调必须写回**同一份** state —— 不然 `streamed` 一直是空的，
 * "作废有没有真清掉"就变成了一条永远为真的断言（这才是这个测试的全部价值）。
 */
function drive(
  state: RunState,
  events: SSEEvent[],
  opts: { commits?: Array<{ count: number; text: string }>; scheduler?: Scheduler } = {},
): RunState {
  let cur = state;
  const batcher = new TokenBatcher(
    (batch, text) => {
      opts.commits?.push({ count: batch.length, text });
      cur = { ...cur, events: [...cur.events, ...batch], streamed: cur.streamed + text };
    },
    100,
    opts.scheduler,
  );
  for (const e of events) {
    if (e.name === "token") {
      batcher.push(e);
      continue;
    }
    batcher.flush();
    cur = reduceRunEvent(cur, e);
  }
  batcher.flush();
  return cur;
}

describe("TokenBatcher：待办① 的 800+ 条 token 必须被合并提交", () => {
  it("1000 条 token 只产生个位数的提交次数，且增量按到达顺序拼全", () => {
    const clock = fakeClock();
    const commits: Array<{ count: number; text: string }> = [];
    const batcher = new TokenBatcher((batch, text) => commits.push({ count: batch.length, text }), 100, clock.scheduler);

    const total = 1000;
    for (let i = 0; i < total; i++) {
      batcher.push(token(`t${i}`));
      // 每 40 条让出一次事件循环（mock replay 的成组节奏），但不拨表 → 全落在同一个窗口里
      if (i % 40 === 39) clock.advance(8);
    }
    clock.advance(100);

    expect(commits.length, `${total} 条 token 竟然触发了 ${commits.length} 次提交 —— 没有批处理`).toBeLessThanOrEqual(3);
    const joined = commits.map((c) => c.text).join("");
    expect(joined).toBe(Array.from({ length: total }, (_, i) => `t${i}`).join(""));
    expect(commits.reduce((n, c) => n + c.count, 0)).toBe(total);
  });

  it("非 token 事件会先把缓冲区按序落地，时间线不会错序", () => {
    const clock = fakeClock();
    const landed: Array<{ name: string; events: SSEEvent[] }> = [];
    const batcher = new TokenBatcher((batch) => landed.push({ name: "tokens", events: batch }), 100, clock.scheduler);

    batcher.push(token("a"));
    batcher.push(token("b"));
    expect(landed.length, "push 不该自己提交").toBe(0);
    expect(batcher.pending).toBe(2);

    batcher.flush(); // 模拟收到 final_report 前的那一次 flush
    expect(landed[0]?.events.map((e) => String((e.data as { delta?: string }).delta))).toEqual(["a", "b"]);

    batcher.push(token("c"));
    batcher.discard(); // 开新一轮：未提交的必须被丢掉，而不是留着晚点冒出来
    expect(landed.length).toBe(1);
    expect(batcher.pending).toBe(0);
  });

  it("窗口是滚动的：持续到达的 token 会周期性提交，不是一次性憋到结尾", () => {
    const clock = fakeClock();
    const commits: number[] = [];
    const batcher = new TokenBatcher((batch) => commits.push(batch.length), 100, clock.scheduler);

    for (let window = 0; window < 5; window++) {
      for (let i = 0; i < 60; i++) batcher.push(token("x"));
      clock.advance(100);
    }
    expect(commits.length).toBe(5);
    expect(commits.every((c) => c === 60)).toBe(true);
  });
});

describe("reduceRunEvent / voidPreview：待办② 的「半截预览真被清掉」", () => {
  const streamed = base({
    events: [ev("node_started", { node: "write" }), token("前半句"), token("后半句")],
    streamed: "前半句后半句",
    status: "running",
    runId: "r1",
  });

  it("token_stream_aborted：正文清空 + token 事件一起摘掉（时间线那边也是这段预览的展示面）", () => {
    const next = reduceRunEvent(streamed, ev("error", { node: "write", code: "token_stream_aborted", message: "生成中断，正在重试", recoverable: true }));

    expect(next.streamed).toBe("");
    expect(next.previewVoid).toBe(true);
    expect(next.events.filter((e) => e.name === "token")).toEqual([]);
    expect(next.events.map((e) => e.name), "error 行要留着，用户才知道这段为什么没了").toContain("error");
    expect(next.error).toContain("token_stream_aborted");
  });

  it("普通可恢复 error（熔断降级通知）不作废：那条流后面还会出报告", () => {
    const next = reduceRunEvent(streamed, ev("error", { node: "resilience", code: "llm_circuit_open", message: "下游被打穿", recoverable: true }));
    expect(next.previewVoid).toBe(false);
    expect(next.streamed).toBe("前半句后半句");
    expect(next.error).toContain("llm_circuit_open");
  });

  it("不可恢复 error（API 层 internal_error）作废预览 —— 这次运行不会再有 final_report", () => {
    const next = reduceRunEvent(streamed, ev("error", { node: "api", code: "internal_error", message: "炸了", recoverable: false }));
    expect(next.previewVoid).toBe(true);
    expect(next.streamed).toBe("");
    expect(next.events.filter((e) => e.name === "token")).toEqual([]);
  });

  it("done{status:\"error\"}：没有权威内容时同样清空，不能留一段永不成稿的半截文本", () => {
    const next = reduceRunEvent(streamed, ev("done", { run_id: "r1", status: "error" }));
    expect(next.status).toBe("error");
    expect(next.previewVoid).toBe(true);
    expect(next.streamed).toBe("");
  });

  it("已经拿到 final_report 后运行才崩：权威内容优先，绝不被擦掉", () => {
    const withReport = reduceRunEvent(streamed, ev("final_report", { markdown: "# 权威正文", citations: [] }));
    expect(withReport.finalReport).toBe("# 权威正文");
    expect(withReport.previewVoid).toBe(false);

    const crashed = reduceRunEvent(withReport, ev("done", { run_id: "r1", status: "error" }));
    expect(crashed.finalReport).toBe("# 权威正文");
    expect(crashed.previewVoid).toBe(false);
    expect(crashed.events.filter((e) => e.name === "token").length, "报告已到位，没必要再擦时间线里的过程行").toBe(2);
  });

  it("voidPreview 是幂等的，且不动 finalReport", () => {
    const once = voidPreview(streamed);
    const twice = voidPreview(once);
    expect(twice).toEqual(once);
    expect(once.finalReport).toBeNull();
  });

  it("整条序列：作废之后重试轮不再吐字，界面停在作废态直到 final_report 接管", () => {
    const clock = fakeClock();
    const commits: Array<{ count: number; text: string }> = [];
    const opts = { commits, scheduler: clock.scheduler };

    const events = [
      ev("run_started", { run_id: "r1", question: "q" }),
      ...Array.from({ length: 300 }, (_, i) => token(`d${i}`)),
      ev("error", { node: "write", code: "token_stream_aborted", message: "已流出的字作废", recoverable: true }),
      // 后端「不重流」：作废之后一条 token 都没有
      ev("node_started", { node: "write", label: "生成报告" }),
      ev("final_report", { markdown: "# 权威正文", citations: [] }),
      ev("done", { run_id: "r1", status: "ok" }),
    ];
    const abortAt = events.findIndex((e) => e.name === "error");

    // 作废通知**刚落地**的那一刻：预览正文和时间线里的 token 行都必须已经没了
    const midCommits: Array<{ count: number; text: string }> = [];
    const atAbort = drive(base({ status: "running" }), events.slice(0, abortAt + 1), {
      commits: midCommits,
      scheduler: clock.scheduler,
    });
    expect(atAbort.previewVoid).toBe(true);
    expect(atAbort.streamed).toBe("");
    expect(atAbort.events.filter((e) => e.name === "token")).toEqual([]);
    expect(midCommits.length, "作废前也不该逐条提交").toBeLessThanOrEqual(2);

    // 走完剩余事件：权威内容接管，作废态跟着解除
    const state = drive(base({ status: "running" }), events, opts);
    expect(state.streamed, "final_report 之后不该再有残留预览").toBe("");
    expect(state.previewVoid).toBe(false);
    expect(state.finalReport).toBe("# 权威正文");
    expect(state.status).toBe("done");
    expect(state.events.filter((e) => e.name === "token")).toEqual([]);
    // 300 条增量只被合并成少量提交（不是 300 次 setState）
    expect(commits.length, `300 条 token 触发了 ${commits.length} 次提交`).toBeLessThanOrEqual(2);
    expect(commits.reduce((n, c) => n + c.count, 0)).toBe(300);
  });

  it("预览不靠 node_started(write)：只喂 token 也能出内容（待办③）", () => {
    // drive 末尾会 flush，所以这里等价于"只有增量、一个框架事件都没到"的那段时间
    const state = drive(base({ status: "running" }), [token("第一段"), token("第二段")]);
    expect(state.events.filter((e) => e.name === "node_started")).toEqual([]);
    expect(state.streamed).toBe("第一段第二段");
    expect(state.previewVoid).toBe(false);
  });
});

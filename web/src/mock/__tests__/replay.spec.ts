/**
 * mock 重放的契约保真测试。
 *
 * 为什么需要它：这一整轮 bug 的根因都是"mock 和真实后端形状不一致，
 * 于是 mock 模式一路绿灯、切真实后端全炸"。所以 mock 本身也要被断言 ——
 * 它必须复现后端 §8 写明的三条流式边界，否则它就是个会说谎的替身。
 *
 * ★ 全部通过驱动真实 generator 来验，不读 replay.ts 自己导出的计数常量 ——
 * 那些常量本身就是被测对象的一部分。
 */

import { afterEach, describe, expect, it } from "vitest";
import { mockStream, setMockStreamAbort } from "@/mock/replay";
import type { SSEEvent } from "@/api/events";

async function collect(path: string, body: unknown = {}): Promise<SSEEvent[]> {
  const out: SSEEvent[] = [];
  for await (const e of mockStream(path, body)) out.push(e);
  return out;
}

/** 走完两段（stream 停在 HITL，resume 续到 done），拿到第二段事件 */
async function runFull(body: unknown = {}): Promise<SSEEvent[]> {
  await collect("/api/v1/research/stream", body);
  return collect("/api/v1/research/mockrun00000001/resume", body);
}

const deltasOf = (events: SSEEvent[]) =>
  events.filter((e) => e.name === "token").map((e) => String((e.data as { delta?: string }).delta ?? ""));

const firstIndexOf = (events: SSEEvent[], pred: (e: SSEEvent) => boolean) =>
  events.findIndex((e) => pred(e));

afterEach(() => setMockStreamAbort(false));

describe("mock 流式重放必须复现后端 §8 的三条边界", () => {
  it(
    "边界③：token 事件量级接近真实（800+），且 node_started(write) 晚于第一个 token",
    { timeout: 60_000 },
    async () => {
      const events = await runFull();

      expect(events.length, "第二段一条事件都没有").toBeGreaterThan(0);
      expect(events.filter((e) => e.name === "token").length, "token 事件太少，压不出批处理问题").toBeGreaterThanOrEqual(800);

      const firstToken = firstIndexOf(events, (e) => e.name === "token");
      const writeStarted = firstIndexOf(
        events,
        (e) => e.name === "node_started" && (e.data as { node?: string }).node === "write",
      );
      expect(firstToken).toBeGreaterThan(-1);
      expect(writeStarted).toBeGreaterThan(-1);
      // updates 通道的固有顺序：前端不能把"节点开始"当流式起点
      expect(firstToken, "mock 里 node_started(write) 竟然早于第一个 token，与后端不符").toBeLessThan(writeStarted);
    },
  );

  it(
    "边界①：streamed 拼起来短于 final_report（⚠️ 节 / 参考文献 / 页脚不进流）",
    { timeout: 60_000 },
    async () => {
      const events = await runFull();
      const streamed = deltasOf(events).join("");
      const report = String((events.find((e) => e.name === "final_report")?.data as { markdown?: string })?.markdown ?? "");

      expect(streamed.length).toBeGreaterThan(0);
      expect(report.length).toBeGreaterThan(0);
      expect(report.length, "final_report 不比预览长，说明 mock 没复现「代码追加的节」").toBeGreaterThan(streamed.length);
      // streamed 是 report 的前缀：追加发生在尾部，而不是整篇重写
      expect(report.startsWith(streamed), "mock 里预览不是最终报告的前缀，边界建模错了").toBe(true);
    },
  );

  it(
    "边界②：作废之后不得再吐 token（宁少流、不重流）",
    { timeout: 60_000 },
    async () => {
      setMockStreamAbort(true);
      const events = await runFull();

      const abortAt = firstIndexOf(
        events,
        (e) => e.name === "error" && (e.data as { code?: string }).code === "token_stream_aborted",
      );
      expect(abortAt, "开了 abort 开关却没发 token_stream_aborted 事件").toBeGreaterThan(-1);

      const before = events.slice(0, abortAt).filter((e) => e.name === "token").length;
      const after = events.slice(abortAt + 1).filter((e) => e.name === "token").length;
      expect(before, "作废前应当已经吐过一批 token").toBeGreaterThan(0);
      expect(after, "作废后还在吐 token，违反「不重流」").toBe(0);

      // 作废不等于失败：权威报告照常到位，界面靠它渲染
      expect(events.some((e) => e.name === "final_report"), "作废后没有 final_report").toBe(true);
      expect(events.at(-1)?.name, "流没有正常收尾").toBe("done");
    },
  );

  it("token_usage 的 by_node 每项是字典，不是裸数字", { timeout: 60_000 }, async () => {
    const events = await runFull();
    const usage = events.find((e) => e.name === "token_usage")?.data as { by_node?: Record<string, unknown> };
    const entries = Object.entries(usage?.by_node ?? {});
    expect(entries.length).toBeGreaterThan(0);
    for (const [node, v] of entries) {
      expect(typeof v, `${node} 不是对象`).toBe("object");
      expect(v as Record<string, unknown>).toHaveProperty("total_tokens");
    }
  });

  /* ---- 下面两条盯的是"同一个概念在两条链路上键名不同"这类分裂 ----
     本轮三个 P0 全是这个形状：mock 照 REST 写、真后端发另一套，于是 mock 一路绿灯。
     所以 mock 的**事件**必须照事件侧的形状，测试也就照事件侧断言。 */

  it("token_usage.total 用 *_tokens 全名（后端 `usage_of` 的口径），不是 prompt/completion/total 短名", { timeout: 60_000 }, async () => {
    const events = await runFull();
    const total = (events.find((e) => e.name === "token_usage")?.data as { total?: Record<string, unknown> })?.total;
    expect(total, "没有 token_usage 事件").toBeTruthy();
    for (const key of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
      expect(total, `${key} 缺失`).toHaveProperty(key);
    }
    // 后端**不发**这三个短名 —— 读它们会拿到 undefined，成本卡显示 NaN
    for (const ghost of ["prompt", "completion", "total"]) {
      expect(total, `${ghost} 不该出现在 total 里`).not.toHaveProperty(ghost);
    }
  });

  it("final_report.citations 是事件侧形状：paper_id + 数字 chunk_id，不是 REST 的 arxiv_id", { timeout: 60_000 }, async () => {
    const events = await runFull();
    const citations = (events.find((e) => e.name === "final_report")?.data as { citations?: Array<Record<string, unknown>> })
      ?.citations;
    expect(citations?.length, "final_report 没带 citations").toBeGreaterThan(0);
    for (const c of citations ?? []) {
      expect(c).toHaveProperty("paper_id");
      expect(c, "事件里的引用不叫 arxiv_id（那是 REST 的键名）").not.toHaveProperty("arxiv_id");
      expect(typeof c.chunk_id, "chunk_id 在事件里是数字").toBe("number");
      expect(["verified" in c ? typeof c.verified : "undefined"]).toContain("boolean");
      expect(Object.keys(c).sort()).toEqual(["claim", "chunk_id", "locator", "paper_id", "quote", "source", "verified"].sort());
    }
  });
});

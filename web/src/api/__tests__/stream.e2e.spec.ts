/**
 * 真实端到端流式断言：`token` 事件在**真后端**上到底长什么样。
 *
 * 与 `contract.spec.ts` 同一套可达性口径（那里是范本，改动要同步看那边）：
 * - 本地：后端不可达就整体 skip —— 跳过比假装通过诚实；
 * - CI：**绝不允许 skip**。探测不到后端就在收集阶段直接 throw，让流水线红。
 *   本文件此前用 `describe.skipIf(process.env.PAPERPILOT_E2E !== "1")` 做闸门，
 *   而 CI 不会设这个变量 → 永远 1 skipped → CI 里那三条边界（800+ 条要批处理、
 *   作废要清预览、node_started 晚于第一个 token）其实一次都没被执行过。
 *
 * 唯一的本地额外出口是 `PAPERPILOT_E2E=0`：这条用例会**真跑一次研究**
 * （本地后端如果是 ollama/云端模型，要几分钟并烧 token），显式关掉它是给自己
 * 一个"今天不跑"的选择，不是给 CI 的 —— `CI=true` 时这个开关一律无效（见下面）。
 *
 * 断言的都是**只有真实后端才能证伪**的约定 —— mock replay 本来就是照这些约定写的，
 * 用 mock 测 mock 等于自己给自己盖章。
 */

import { describe, expect, it } from "vitest";

const API = process.env.PAPERPILOT_API ?? "http://127.0.0.1:8000";

/** 后端可达性只探测一次（探针与 contract.spec.ts 一致，避免两处判据漂移） */
const backendUp = await fetch(`${API}/healthz`, { signal: AbortSignal.timeout(2000) })
  .then((r) => r.ok)
  .catch(() => false);

const explicitlyOff = process.env.PAPERPILOT_E2E === "0";

if (!backendUp && process.env.CI) {
  throw new Error(
    `流式端到端测试在 CI 里不允许被跳过：后端探针 ${API}/healthz 不可达。` +
      `请确认 CI 已用 mock provider 在 ${API} 起好后端并等待 /healthz 就绪后再跑 vitest。`,
  );
}
if (explicitlyOff && process.env.CI) {
  throw new Error(
    "PAPERPILOT_E2E=0 只是本地逃生口：CI 里关掉这条 = 静默跳过，正是本文件要消灭的那个隐患。",
  );
}

const skip = !backendUp || explicitlyOff;

interface Ev {
  name: string;
  data: Record<string, any>;
}

async function* sse(path: string, body: unknown): AsyncGenerator<Ev> {
  const resp = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}: ${(await resp.text()).slice(0, 300)}`);

  const reader = resp.body!.getReader();
  const dec = new TextDecoder();
  let buf = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i = buf.search(/\r?\n\r?\n/);
      while (i !== -1) {
        const frame = buf.slice(0, i);
        buf = buf.slice(i).replace(/^\r?\n\r?\n/, "");
        let name = "message";
        let data = "";
        for (const line of frame.split(/\r?\n/)) {
          if (line.startsWith("event: ")) name = line.slice(7).trim();
          else if (line.startsWith("data: ")) data += line.slice(6);
        }
        if (data) yield { name, data: JSON.parse(data) as Record<string, any> };
        i = buf.search(/\r?\n\r?\n/);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
  }
}

describe.skipIf(skip)("真实流式（token 事件）", () => {
  it(
    "writer 真的在发 token；payload 形状、顺序与「预览 ≠ 最终报告」三条边界成立",
    // 实测（本机 Ollama qwen2.5:7b，同一条问题）：144s / 244s / 437s 都过过，
    // 而两个研究同时抢一个 Ollama 时 >600s 被掐死过一次。
    // 这条断言测的是"流式对不对"，不是"跑得快不快"，所以预算按最慢观测的 2 倍给。
    { timeout: 900_000 },
    async () => {
      // ---- 第一段：跑到 HITL 中断点 A
      let runId = "";
      for await (const e of sse("/api/v1/research/stream", { question: "RAG 减少幻觉的方法有哪些？" })) {
        if (e.name === "run_started") runId = String(e.data.run_id ?? "");
        if (e.name === "done") break;
      }
      expect(runId, "第一段没拿到 run_id").toBeTruthy();

      // ---- 第二段：恢复后一路到 final_report
      const events: Ev[] = [];
      for await (const e of sse(`/api/v1/research/${runId}/resume`, {})) {
        events.push(e);
        if (e.name === "done") break;
      }

      const idxOf = (pred: (e: Ev, i: number) => boolean) => events.findIndex(pred);
      const firstToken = idxOf((e) => e.name === "token");
      const writeStarted = idxOf((e) => e.name === "node_started" && e.data.node === "write");

      expect(firstToken, "真实模式下没有任何 token 事件 —— 后端流式没上线或已回退").toBeGreaterThan(-1);
      expect(writeStarted, "没看到 write 节点的 node_started").toBeGreaterThan(-1);

      // ★ §8 边界③：`updates` 通道让 node_started(write) 晚于第一个 token。
      // 前端因此不能拿"节点开始"当流式起点。哪天顺序反转了这条会红，提醒前端改判据。
      expect(firstToken, "第一个 token 应当早于 node_started(write)").toBeLessThan(writeStarted);

      // ★ payload 形状：`app/observability/events.py` 的 `token()` 是字段唯一定义处
      //（`{node, delta}` + `schema_version`），改成别的键名前端就是静默拿到空串。
      const tokenEvents = events.filter((e) => e.name === "token");
      for (const t of tokenEvents) {
        expect(t.data.node, "token.node 必须是 write（前端 EventTimeline 按它分段归组）").toBe("write");
        expect(typeof t.data.delta, "token.delta 必须是字符串增量").toBe("string");
        expect(String(t.data.schema_version ?? ""), "每帧 data 要带 schema_version").toBe("1");
      }

      // ★ §8 边界①：流里只有模型生成的正文，⚠️ 节 / 参考文献 / 页脚是代码追加的，
      // 所以 streamed 拼起来必然短于 final_report.markdown。
      const streamed = tokenEvents.map((e) => String(e.data.delta ?? "")).join("");
      const report = events.find((e) => e.name === "final_report")?.data.markdown ?? "";

      expect(streamed.length, "token 增量拼起来是空的").toBeGreaterThan(0);
      expect(report.length, "final_report 是空的").toBeGreaterThan(0);
      expect(
        report.length,
        `final_report 应比 streamed 长（多出代码追加的节）；streamed=${streamed.length} report=${report.length}`,
      ).toBeGreaterThan(streamed.length);

      // ★ §8 边界②：「宁少流、不重流」—— 一旦声明作废，后面不得再有 token。
      const abortAt = idxOf((e) => e.name === "error" && e.data.code === "token_stream_aborted");
      if (abortAt !== -1) {
        const tokensAfterAbort = events.slice(abortAt + 1).filter((e) => e.name === "token").length;
        expect(tokensAfterAbort, "作废之后又吐了 token，违反「不重流」").toBe(0);
      }

      // ★ 同一条流上顺手钉两颗"键名钉子"（这类 bug 全是 mock 照 REST 写、真后端发另一套）：
      // ① `token_usage.total` 用 *_tokens 全名（`_common.usage_of` 的口径），
      //    前端读短名 `prompt` / `total` 会拿到 undefined；
      const total = (events.find((e) => e.name === "token_usage")?.data ?? {}).total as Record<string, unknown>;
      if (total) {
        for (const key of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
          expect(total, `${key} 缺失`).toHaveProperty(key);
        }
        for (const ghost of ["prompt", "completion", "total"]) {
          expect(total, `${ghost} 不该出现在 total 里`).not.toHaveProperty(ghost);
        }
      }
      // ② `final_report.citations` 用 `paper_id`（`app/agents/nodes/writer.py`），
      //    而 REST `/research/{id}/citations` 用 `arxiv_id` —— 两条路键名不同一套，
      //    前端按哪一个取数就必须按哪一个断言。
      const citations = ((events.find((e) => e.name === "final_report")?.data ?? {}).citations ?? []) as Array<
        Record<string, unknown>
      >;
      for (const c of citations) {
        expect(c).toHaveProperty("paper_id");
        expect(c, "事件里的引用不叫 arxiv_id").not.toHaveProperty("arxiv_id");
        if (c.chunk_id !== undefined && c.chunk_id !== null) {
          expect(typeof c.chunk_id, "事件里的 chunk_id 是数字，不是 REST 那种串").toBe("number");
        }
      }
    },
  );
});

/**
 * 后端契约回归测试。
 *
 * 这些断言全部打在**真实运行中的后端**上（默认 `http://127.0.0.1:8000`，可用
 * `PAPERPILOT_API` 覆盖）。后端不在时就整体 skip —— 跳过比假装通过诚实。
 *
 * 为什么是契约测试而不是组件测试：本轮修掉的三个 P0（filters 422、verification
 * 读了后端不发的键、trend 点形状错）全都是"字段名对不上"，
 * 组件测试要 mock 一份响应，而 mock 一旦写错就正好把 bug 又复制一遍 —— 当初就是这么漏的。
 * 直接对真后端打一次，才是能拦住这类问题的断言。
 */

import { describe, expect, it } from "vitest";

const API = process.env.PAPERPILOT_API ?? "http://127.0.0.1:8000";

/** 后端可达性只探测一次 */
const backendUp = await fetch(`${API}/healthz`, { signal: AbortSignal.timeout(2000) })
  .then((r) => r.ok)
  .catch(() => false);

/**
 * CI 开关（本文件唯一一处后端不可达处理逻辑）。
 *
 * 本地：后端不在就整体 skip 是诚实的（跳过比假装通过诚实）。
 * CI：绝对不能 skip —— "后端没起来 → 8 条全跳 → 流水线照样绿"会把这层契约保护变成假的。
 * 所以在 CI 环境下探测失败就让本文件在收集阶段直接失败，而不是走到 skipIf。
 * `process.env.CI` 由 CI 预设（GitHub Actions 会置为 "true"）。
 */
if (!backendUp && process.env.CI) {
  throw new Error(
    `契约测试在 CI 里不允许被跳过：后端探针 ${API}/healthz 不可达。` +
      `请确认 CI 已用 mock provider 在 ${API} 起好后端并等待 /healthz 就绪后再跑 vitest。`,
  );
}

const skip = backendUp ? false : true;

async function json(path: string): Promise<any> {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

describe("POST /api/v1/research/stream 的请求体契约", () => {
  it("接受后端 ResearchFilters 的真实字段名（extra=forbid，多一个键就整请求 422）", { timeout: 30_000 }, async () => {
    const controller = new AbortController();
    const resp = await fetch(`${API}/api/v1/research/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: "RAG 有哪些方法",
        filters: { categories: ["cs.CL"], year_gte: 2020, year_lte: null, has_code: null, arxiv_ids: [] },
      }),
      signal: controller.signal,
    });

    // ★ 只读第一个数据块就断开。`await resp.text()` 会一直读到流结束，
    // 那等于真跑完一次研究（几分钟），而这里要验的只是请求体能不能过校验。
    let head = "";
    try {
      const reader = resp.body?.getReader();
      if (reader) {
        const { value } = await reader.read();
        head = new TextDecoder().decode(value ?? new Uint8Array());
        await reader.cancel();
      }
    } finally {
      controller.abort();
    }

    expect(resp.status, `首帧: ${head.slice(0, 300)}`).toBe(200);
    expect(head).toContain("event: run_started");
  });

  it.skipIf(skip)("前端旧写法（cat/since/hasCode）必须被拒 —— 这正是当初的 422 现场", async () => {
    const resp = await fetch(`${API}/api/v1/research/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "RAG 有哪些方法", filters: { cat: "cs.CL", since: "2020", hasCode: false } }),
    });
    expect(resp.status).toBe(422);
    const body = await resp.json();
    expect(JSON.stringify(body)).toContain("extra_forbidden");
  });

  it.skipIf(skip)("categories 传空数组合法；传空字符串会被白名单校验拒掉", async () => {
    const ok = await fetch(`${API}/api/v1/research/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "q", filters: { categories: [] } }),
    });
    expect(ok.status).toBe(200);
    // 不消费 body 会让服务端那条 run 一直挂着，读一帧就取消
    await ok.body?.cancel().catch(() => undefined);

    const bad = await fetch(`${API}/api/v1/research/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "q", filters: { categories: [""] } }),
    });
    expect(bad.status).toBe(422);
    await bad.body?.cancel().catch(() => undefined);
  });
});

describe("GET /api/v1/research/{id} 的 verification 契约", () => {
  it.skipIf(skip)("只发 passed/score/unsupported/checked/revision_count 这五个键；剔除数 = unsupported.length", async () => {
    const list = await json("/api/v1/research?limit=30");
    const withCites = list.items.find((r: any) => r.citation_count > 0) ?? list.items[0];
    expect(withCites, "语料里一次运行都没有，无法验这条").toBeTruthy();

    const run = await json(`/api/v1/research/${withCites.run_id}`);
    const v = run.verification;
    expect(v, "verification 为空对象：该 run 没跑到 verify 节点").not.toEqual({});

    // 后端**不发**这几个键 —— 前端读它们就会拿到 undefined，指标卡显示 NaN
    for (const ghost of ["total", "supported", "dropped", "dropped_claims"]) {
      expect(v, `${ghost} 不该出现在后端响应里`).not.toHaveProperty(ghost);
    }
    expect(Array.isArray(v.unsupported)).toBe(true);
    expect(typeof v.checked).toBe("number");
    expect(v.score).toBeGreaterThanOrEqual(0);
    expect(v.score).toBeLessThanOrEqual(1);
  });
});

describe("GET /api/v1/papers/{id}/citation-trend 的契约", () => {
  it.skipIf(skip)("采样点是 {on, cited_by_count}；单点必须给 note 且 verdict=insufficient_data", async () => {
    const list = await json("/api/v1/papers?limit=5");
    const id = list.items[0]?.arxiv_id;
    expect(id).toBeTruthy();

    const t = await json(`/api/v1/papers/${id}/citation-trend`);
    expect(Array.isArray(t.points)).toBe(true);
    expect(["rising", "flat", "falling", "insufficient_data"]).toContain(t.verdict);

    for (const p of t.points) {
      expect(p).toHaveProperty("on");
      expect(p).toHaveProperty("cited_by_count");
      // 前端曾按 {date, citations} 读，两个键都不存在，于是趋势图恒空
      expect(p).not.toHaveProperty("citations");
    }
    if (t.points.length < 2) {
      expect(t.verdict).toBe("insufficient_data");
      expect(typeof t.note).toBe("string");
      expect(t.note.length, "单点必须带说明文案，前端不能自己画斜率").toBeGreaterThan(0);
    }
  });
});

describe("POST /api/v1/papers/{id}/ask 的契约", () => {
  it.skipIf(skip)("context_chunks 是数字不是数组；token_usage 用 *_tokens 全名", { timeout: 120_000 }, async () => {
    const list = await json("/api/v1/papers?limit=3");
    const id = list.items[0]?.arxiv_id;
    const r = await fetch(`${API}/api/v1/papers/${id}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: "它怎么决定何时检索" }),
    });
    expect(r.status).toBe(200);
    const d = await r.json();

    // 前端曾把它声明成数组，真实是 chunk 计数
    expect(typeof d.context_chunks, "context_chunks 必须是数字").toBe("number");
    expect(Array.isArray(d.context_chunks)).toBe(false);

    expect(d.token_usage).toHaveProperty("prompt_tokens");
    expect(d.token_usage).toHaveProperty("completion_tokens");
    expect(d.token_usage).not.toHaveProperty("prompt");

    // retrieval 的取值集合是前端分支的依据，出现新值要能被发现
    expect(["targeted", "fallback_db", "no_chunks", "llm_failed"]).toContain(d.retrieval);
    expect(typeof d.supported).toBe("boolean");
    // supported=false 时后端自己会给一句"不足以回答"，前端不得替它补写
    if (d.supported === false) expect(d.answer.length).toBeGreaterThan(0);
  });
});

describe("token_by_node 的形状", () => {
  it.skipIf(skip)("每项是 {prompt_tokens, completion_tokens, total_tokens, calls}，不是裸数字", async () => {
    const list = await json("/api/v1/research?limit=30");
    const target = list.items.find((r: any) => r.citation_count > 0) ?? list.items[0];
    const run = await json(`/api/v1/research/${target.run_id}`);
    const byNode = run.token_by_node;
    if (!byNode || !Object.keys(byNode).length) return; // 该 run 没跑到 finalize，跳过而不是判过

    for (const [node, v] of Object.entries<any>(byNode)) {
      expect(typeof v, `${node} 必须是对象`).toBe("object");
      expect(v).not.toBeNull();
      expect(typeof v.total_tokens, `${node}.total_tokens 必须是数字`).toBe("number");
      expect(typeof v.calls).toBe("number");
    }
  });
});

describe("GET /api/v1/research/{id}/citations 的 verified 三态", () => {
  it.skipIf(skip)("verified 是 true|false|null，不是布尔；null 表示当时没校验", async () => {
    const list = await json("/api/v1/research?limit=30");
    const target = list.items.find((r: any) => r.citation_count > 0);
    if (!target) return;

    const bundle = await json(`/api/v1/research/${target.run_id}/citations`);
    for (const c of bundle.citations) {
      expect([true, false, null]).toContain(c.verified);
    }
    // verified_rate 的分子只数 true。若前端把 null 也归进"无支撑"，
    // 界面统计就会和这个数、以及导出的 .md 互相矛盾。
    if (bundle.verified_rate !== null && bundle.count > 0) {
      const trues = bundle.citations.filter((c: any) => c.verified === true).length;
      expect(bundle.verified_rate).toBeCloseTo(trues / bundle.count, 3);
    }
  });
});

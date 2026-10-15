/**
 * `client.ts` 的**请求动词与路径**回归测试（不需要后端）。
 *
 * 起因是一次真实报障：配置页点保存回 "Method Not Allowed"。
 * 后端 `PUT /api/v1/config` 好好的，坏的是前端 —— `api.config.put()` 里
 * 手滑调了 `postJson`，于是 POST 打到一个只注册了 GET/PUT 的路径上，拿回 405。
 *
 * 为什么契约测试拦不住它：契约测试自己发请求，不经过 `client.ts` 这一层；
 * 而 tsc 也拦不住 —— `postJson` 的签名完全合法，错的只是语义。
 * 所以这一层单独钉：把 fetch 换成探针，断言每个方法**实际发出去的动词和路径**。
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/api/client";
import { env } from "@/config/env";

interface Sent {
  path: string;
  method: string;
  body: unknown;
}

const sent: Sent[] = [];

beforeEach(() => {
  sent.length = 0;
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const body = typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : init?.body;
    sent.push({
      path: typeof input === "string" ? input : String(input),
      method: init?.method ?? "GET",
      body,
    });
    return Promise.resolve(
      new Response(JSON.stringify({ ok: true }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function only(): Sent {
  expect(sent).toHaveLength(1);
  return sent[0];
}

/** 让 fetch 固定回一个带 `detail` 的错误响应（错误渲染用例共用） */
function stubError(status: number, detail: unknown) {
  vi.stubGlobal(
    "fetch",
    () =>
      Promise.resolve(
        new Response(JSON.stringify({ detail }), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
      ),
  );
}

describe("api.config 发出的请求形状", () => {
  it("vitest 跑的是真实模式（否则下面所有断言都会打在 mock 分支上，等于没测）", () => {
    // `web/.env.development` 才开 mock；vitest 的 mode 是 "test"，不加载它。
    // 这条断言是前面那些用例的前提，前提没了就该红，而不是静默全绿。
    expect(env.useMock).toBe(false);
  });

  it("写配置用 PUT，不是 POST", async () => {
    await api.config.put({ llm_model: "deepseek-chat" });
    const req = only();
    expect(req.method).toBe("PUT");
    expect(req.path).toBe("/api/v1/config");
    expect(req.body).toEqual({ values: { llm_model: "deepseek-chat" } });
  });

  it("读配置用 GET 同一个路径（读写共用 /config，靠动词区分）", async () => {
    await api.config.get();
    const req = only();
    expect(req.method).toBe("GET");
    expect(req.path).toBe("/api/v1/config");
  });

  it("连通性探测是 POST 到子路径 /config/test", async () => {
    await api.config.test({ provider: "ollama", ollama_base_url: "http://localhost:11434" });
    const req = only();
    expect(req.method).toBe("POST");
    expect(req.path).toBe("/api/v1/config/test");
    expect(req.body).toEqual({ provider: "ollama", ollama_base_url: "http://localhost:11434" });
  });
});

describe("结构化错误 detail 的渲染", () => {
  // 后端把对外错误从"异常原文"改成 `{code, reason, hint}` 之后，
  // 前端把 detail 整体 stringify 就等于把 `{"code":"not_resumable",...}` 糊在红条上。
  it("对象 detail 取 reason 给人看，而不是 JSON 残骸", async () => {
    stubError(409, { code: "not_resumable", run_id: "a".repeat(16), status: "completed", reason: "该 run 已经跑完，不能再次续跑" });
    await expect(api.papers.get("2401.00001")).rejects.toThrow("该 run 已经跑完，不能再次续跑");
  });

  it("hint 一起给出，但绝不出现裸 code JSON", async () => {
    stubError(404, { code: "paper_not_found", reason: "论文 2401.00001 不在库中（尚未入库或 id 写错）" });
    await expect(api.papers.get("2401.00001")).rejects.toThrow(/不在库中/);
    await expect(api.papers.get("2401.00001")).rejects.not.toThrow(/^\{"/);
  });

  it("字符串 detail 保持原样（历史接口仍在用）", async () => {
    stubError(400, "prompt_injection_detected");
    await expect(api.papers.get("2401.00001")).rejects.toThrow("prompt_injection_detected");
  });

  it("pydantic 422 的错误列表转成一行中文，且不回显用户刚贴进去的长文本", async () => {
    // 后端 422 的每一项都带 `input`（用户原文）。整坨 stringify 会把用户输入广播回界面上。
    const long = "a".repeat(2100);
    stubError(422, [
      { type: "string_too_long", loc: ["body", "question"], msg: "String should have at most 2000 characters", input: long },
    ]);
    const err = await api.papers.get("2401.00001").catch((e) => e);
    // 整句钉死：`body` 是 pydantic 的内部层级名，对用户没有信息量，必须滤掉
    expect(err.message).toBe("输入不符合要求 —— question：String should have at most 2000 characters");
    expect(err.message, "把用户原文回显到红条上").not.toContain(long);
  });

  it("422 里没有 loc 的项也不产出 undefined 文案", async () => {
    stubError(422, [{ type: "missing", msg: "Field required" }]);
    const err = await api.papers.get("2401.00001").catch((e) => e);
    expect(err.message).toContain("Field required");
    expect(err.message).not.toContain("undefined");
  });
});

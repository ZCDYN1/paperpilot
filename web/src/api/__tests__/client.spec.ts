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

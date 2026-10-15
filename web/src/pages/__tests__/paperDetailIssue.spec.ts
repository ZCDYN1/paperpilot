/**
 * `PapersPage` 详情加载失败分码的**纯逻辑**回归测试（不起后端、不渲染组件）。
 *
 * 起因：`.catch(() => setDetail(null))` 曾把错误整个吞掉，404 时界面只剩一片空白，
 * 后端那句"论文 xxx 不在库中（尚未入库或 id 写错）"完全不可见。
 * 现在 toDetailIssue 按 ApiError.status 分码（不匹配文案字符串），并把 client.ts
 * 已经收成一句中文的 message 如实带进 reason —— 这一层钉住三件事：
 *   ① 404 → warn 级、lead 里同时给出"id 写错 or 未入库"两条可能、后端 reason 原样保留；
 *   ② 连不上后端（fetch 的 TypeError，没有 status）→ 说"服务没起"而不是"论文不存在"；
 *   ③ 5xx → error 级，其它 HTTP 码至少 lead 里有状态码。
 * extractReasonHint 与 toUploadIssue 共用：这里用"detail 还是 JSON 字符串"的用例
 * 把旧链路的解析路径一并钉住（toUploadIssue 本身未导出，不在本文件覆盖范围）。
 */

import { describe, expect, it } from "vitest";
import { ApiError } from "@/api/client";
import { toDetailIssue } from "@/pages/PapersPage";

describe("toDetailIssue：详情加载失败按 status 分码", () => {
  it("404：warn 级，lead 里点明 id 写错 / 未入库两种可能，后端 reason 原样保留", () => {
    // client.ts 的 detailToText 已把 {code,reason,hint} 收成一句中文，这里模拟其产物
    const reason = "论文 2399.00001 不在库中（尚未入库或 id 写错）";
    const issue = toDetailIssue(new ApiError(404, reason), "2399.00001");
    expect(issue.tone).toBe("warn");
    expect(issue.lead).toContain("404");
    expect(issue.lead).toContain("2399.00001");
    expect(issue.lead).toContain("id 写错");
    expect(issue.lead).toContain("还没入库");
    expect(issue.reason).toBe(reason);
  });

  it("404 且 detail 还是 JSON 字符串（旧链路）：reason/hint 被解回来，lead 仍是 404 口径", () => {
    const issue = toDetailIssue(
      new ApiError(404, JSON.stringify({ code: "paper_not_found", reason: "不在库中", hint: "先上传入库" })),
      "1234.56789",
    );
    expect(issue.tone).toBe("warn");
    expect(issue.lead).toContain("404");
    expect(issue.reason).toBe("不在库中");
    expect(issue.hint).toBe("先上传入库");
  });

  it("连不上后端（fetch 的 TypeError，没有 status）：说服务问题，不说论文不存在", () => {
    const issue = toDetailIssue(new TypeError("Failed to fetch"), "2310.11511");
    expect(issue.tone).toBe("error");
    expect(issue.lead).toContain("连不上后端");
    expect(issue.lead).not.toContain("404");
    expect(issue.lead).not.toContain("不在库中");
  });

  it("其它 HTTP 码（如 500）：error 级，lead 带状态码，后端原因照抄", () => {
    const issue = toDetailIssue(new ApiError(500, "服务器内部错误"), "2310.11511");
    expect(issue.tone).toBe("error");
    expect(issue.lead).toContain("500");
    expect(issue.reason).toBe("服务器内部错误");
  });

  it("非 404 的 4xx（如 403）：warn 级而不是红色大告警，lead 带状态码", () => {
    const issue = toDetailIssue(new ApiError(403, "没有权限读取该篇"), "2310.11511");
    expect(issue.tone).toBe("warn");
    expect(issue.lead).toContain("403");
    expect(issue.reason).toBe("没有权限读取该篇");
  });

  it("完全没见过的异常对象：兜底也给出可读 lead，不产生 undefined 文案", () => {
    const issue = toDetailIssue({ weird: true }, "2310.11511");
    expect(issue.tone).toBe("error");
    expect(issue.lead.length).toBeGreaterThan(0);
    expect(issue.lead).not.toContain("undefined");
  });
});

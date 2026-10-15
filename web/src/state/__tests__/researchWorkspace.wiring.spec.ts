/**
 * 接线守卫：SSE 流**不许**住在页面组件里。
 *
 * 起因是真实报障：点「开始研究」后切到别的页面，回来就变成初始状态。
 * 根因不是渲染问题，而是 `useResearchStream()` 被 `ConsolePage` 直接持有 ——
 * 换路由会卸载该组件，卸载回调里 `abort()` 掉 fetch，于是：
 * ① 界面回到空状态（用户以为研究没了）；② 后端看到客户端断开，把 run 落成
 * `cancelled`，HITL 中断点从此续跑不了。一次研究要跑几分钟还要等人确认，
 * "顺手看一眼论文库"不该把它掐死。
 *
 * 修法是把流提到路由之上的 `ResearchWorkspaceProvider`。这个文件守的是**修法不被改回去**：
 * 页面里再直接调 `useResearchStream`、或 provider 不再包住 `<Routes>`，都会让故障原样复发，
 * 而这种复发在单测里看不出来（用例不渲染整棵路由树），所以这里直接对源码断言。
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const src = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), "utf8");

describe("研究流的归属：属于工作会话，不属于页面", () => {
  it("ConsolePage 不再自己持有流", () => {
    const page = src("../../pages/ConsolePage.tsx");
    expect(page, "页面自己持有流 → 切页会 abort，后端把 run 落成 cancelled").not.toMatch(/useResearchStream\s*\(/);
    expect(page).toContain("useResearchWorkspace");
  });

  it("provider 包住整棵路由树，而不是只包住某一个页面", () => {
    const app = src("../../App.tsx");
    const opened = app.indexOf("<ResearchWorkspaceProvider>");
    // 必须是**行首**的 JSX 标签：App.tsx 的注释里也会提到 `<Routes>`，
    // 用裸 indexOf 会先命中注释，得出"provider 在 Routes 之后"的假结论。
    const routes = app.search(/^\s*<Routes>/m);
    expect(opened, "App 里找不到 provider").toBeGreaterThan(-1);
    expect(routes, "App 里找不到 <Routes> 标签行").toBeGreaterThan(-1);
    expect(opened, "provider 必须在 <Routes> 之前开启，否则换路由照样卸载它").toBeLessThan(routes);
    expect(app).toContain("</ResearchWorkspaceProvider>");
  });

  it("卸载时 abort 的那段逻辑仍然留在流钩子里（它现在只在应用销毁时生效）", () => {
    const hook = src("../../hooks/useResearchStream.ts");
    expect(hook, "abort 是「关页面就该停」的正确行为，不该被删；被删的是页面对它的持有").toMatch(/abortRef\.current\?\.abort\(\)/);
  });
});

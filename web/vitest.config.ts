import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { mergeConfig } from "vite";
import { defineConfig } from "vitest/config";

/**
 * Vitest 专用配置（vitest.config.* 优先于 vite.config.ts 被读取，
 * 且 vite.config.ts 里本来也没有 `test:` 块 —— 全局 testTimeout 一直是缺省的 5000ms）。
 *
 * 为什么单独开这个文件而不是往 vite.config.ts 里塞 `test:` 块：
 * Vitest 的解析顺序是 **vitest.config.* 优先，存在即整体接管**。今天把 `test:` 写进
 * vite.config.ts 也能生效，但哪天有人新建了 vitest.config.ts，那份超时预算会被
 * **静默忽略** —— 全局退回 5000ms，本次要治的机器忙时假红原样复发，而且比之前更难
 * 发现（配置看着还在，实际已失效）。测试配置放测试自己的文件里，归属也更干净。
 *
 * ⚠️ 连带效果：本文件一旦存在，vite.config.ts 对 vitest **不再自动生效**（不是合并），
 * 所以 spec 依赖的公共件（`@` → src 别名、react 插件）必须在这里同步 ——
 * 与 vite.config.ts 保持逐字一致，改别名时两处都要动。
 * 这里只搬这两件：server/proxy、build 等只服务 dev 与打包，测试进程根本不经过。
 */
export default defineConfig(
  mergeConfig(
    {
      plugins: [react()],
      resolve: {
        alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
      },
    },
    {
      test: {
        /**
         * 全局 testTimeout：5000（vitest 默认）→ 15000。
         *
         * 治的是什么：contract / stream.e2e 两个 spec 打的是**真实 uvicorn 后端**，
         * vitest 默认各 spec 文件并行 —— e2e 会在本机把 mock LLM + 整张 graph 跑满，
         * contract 里同为真实查询。机器忙时一次普通本地 REST 往返就可能超 5s，
         * 于是出现"整轮 306s 时红 1 条、重跑 56s 全绿"这类纯超时假红
         * （contract 里 GET 列表/详情/422 那几条此前全靠缺省的 5000ms 兜着）。
         *
         * 为什么是 15000：
         * - 快路径用例正常 <1s，15s 已是 10 倍余量；
         * - 慢调用**不靠全局兜底**，保留各自的显式预算：contract 发起真实研究的 30s、
         *   ask 需要完整报告的 120s、e2e 全链路的 900s，均 > 全局值，互不干扰；
         * - 再往上抬只会拖慢"真挂了"的用例暴露的时间，对假红没有额外收益 ——
         *   超时预算要贴着用例的真实需求，不是越大越安心。
         *
         * ★ 不开 retry / repeats：掩盖不稳定比不稳定更糟。
         *   本次改动只动"超时预算"，没有任何断言被放宽、没有用例如实变假绿。
         */
        testTimeout: 15_000,
        /**
         * hookTimeout：5000 → 15000。
         * contract / e2e 在**文件加载阶段**就 `await fetch(healthz)` 探活（探针自身有
         * AbortSignal 2s 上限），且 vitest 对"用例文件导入/收集"阶段的等待同样归 hook
         * 级超时管 —— 机器忙时 worker 排队启动，不该因为钩子预算比用例小而先炸。
         */
        hookTimeout: 15_000,
      },
    },
  ),
);

import { fileURLToPath, URL } from "node:url";
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/**
 * 代理目标只从环境变量读取，源码里不出现任何 localhost / IP。
 * 真实后端跑在别处时改 `web/.env.local` 的 VITE_PROXY_TARGET 即可。
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_PROXY_TARGET || "http://127.0.0.1:8000";

  return {
    base: env.VITE_BASE_PATH || "/",
    plugins: [react()],
    resolve: {
      alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
    },
    server: {
      port: Number(env.VITE_DEV_PORT) || 5173,
      host: true,
      proxy: {
        "/api": { target, changeOrigin: true },
        // 后端的两个根路径探针不在 /api 前缀下（`/healthz` 是容器 healthcheck 的约定，
        // `/resilience` 是熔断器快照），所以必须逐个列出来 —— 漏一个，开发模式下
        // 该请求就会打到 Vite 自己身上拿到 404，而生产是同源、反而看不出问题。
        "/healthz": { target, changeOrigin: true },
        "/resilience": { target, changeOrigin: true },
      },
    },
    build: {
      outDir: "dist",
      sourcemap: mode !== "production",
      chunkSizeWarningLimit: 1200,
      rollupOptions: {
        output: {
          /**
           * ECharts 与 React 运行时单独成 chunk。
           * 不拆的话首屏那一个 JS 会带上 ~1MB 的图表引擎，而只有 `/analytics`
           * 这一个页面用得到它；拆了之后控制台首页不用下载 ECharts。
           */
          manualChunks: {
            react: ["react", "react-dom", "react-router-dom"],
            echarts: ["echarts", "echarts-for-react"],
            motion: ["framer-motion"],
          },
        },
      },
    },
  };
});

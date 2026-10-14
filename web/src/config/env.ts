/// <reference types="vite/client" />

/**
 * 环境变量的唯一出口。组件里禁止直接写 `import.meta.env.*`，
 * 也禁止出现任何字面量地址 —— 所有配置都从 `web/.env.example` 来。
 */

declare module "vite" {
  interface ImportMetaEnv {
    readonly VITE_USE_MOCK?: string;
    readonly VITE_API_BASE?: string;
    readonly VITE_APP_TITLE?: string;
    readonly VITE_BASE_PATH?: string;
    readonly VITE_PROXY_TARGET?: string;
    readonly VITE_DEV_PORT?: string;
  }
}

const raw = import.meta.env;

function bool(v: string | undefined, fallback: boolean): boolean {
  if (v === undefined || v === "") return fallback;
  return v === "1" || v.toLowerCase() === "true";
}

export const env = {
  /**
   * true = 全部数据走内置 mock，不需要 Python 后端。
   *
   * ⚠️ 代码级默认必须是 **false（真实后端）**。这个开关只该由 `web/.env.development`
   * 显式打开；如果默认值是 true，那么任何"读不到 env 文件"的场景都会静默产出
   * 一份全是演示数据的构建 —— 容器里（`.dockerignore` 排掉了 `web/.env*`）、
   * 或者 clone 后没复制 env 文件的人，看到的界面一切正常，只是没有一条数据是真的。
   * "默认走真实、mock 需要显式开启"是唯一安全的方向。
   */
  useMock: bool(raw.VITE_USE_MOCK, false),
  /** 空串 = 与页面同源（FastAPI 单端口挂载的场景） */
  apiBase: (raw.VITE_API_BASE ?? "").replace(/\/+$/, ""),
  appTitle: raw.VITE_APP_TITLE || "PaperPilot",
  basePath: raw.VITE_BASE_PATH || "/",
} as const;

export const isDev = raw.DEV;

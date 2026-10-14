/**
 * JS 侧读取 Design Token。
 *
 * 为什么用 getComputedStyle 而不是在 TS 里再抄一份色值：
 * 那样就有两个真相源，改 CSS 时图表必然忘记同步（ECharts 的渐变尤其明显）。
 * 代价是首次读取要等样式落地，所以这里做了缓存 + 主题切换时手动 invalidate。
 */

const cache = new Map<string, string>();

export function token(name: string, fallback = ""): string {
  const hit = cache.get(name);
  if (hit !== undefined) return hit;
  const value =
    typeof window === "undefined"
      ? fallback
      : getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  cache.set(name, value || fallback);
  return value || fallback;
}

export function invalidateTokens(): void {
  cache.clear();
}

/** 把 `#00f0ff` / `rgb(0,240,255)` 统一成带 alpha 的 rgba()，供图表渐变用 */
export function withAlpha(color: string, alpha: number): string {
  const hex = color.trim();
  const m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(hex);
  if (!m) return hex;
  let h = m[1];
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  const n = parseInt(h, 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** 语义色名 → 实际色值。图表配置只允许用这些名字取色。 */
export const palette = {
  get cyan() {
    return token("--accent-cyan", "#00f0ff");
  },
  get purple() {
    return token("--accent-purple", "#8b5cf6");
  },
  get green() {
    return token("--accent-green", "#10b981");
  },
  get greenSoft() {
    return token("--accent-green-soft", "#6ee7b7");
  },
  get red() {
    return token("--accent-red", "#ef4444");
  },
  get amber() {
    return token("--accent-amber", "#f59e0b");
  },
  get textPrimary() {
    return token("--text-primary", "#f3f4f6");
  },
  get textSecondary() {
    return token("--text-secondary", "#9ca3af");
  },
  get textMuted() {
    return token("--text-muted", "#6b7280");
  },
  get hairline() {
    return token("--border-hairline", "rgba(255,255,255,0.07)");
  },
  get panel() {
    return token("--bg-panel", "#0d1322");
  },
} as const;

export const chartFont = {
  mono: token("--font-mono", "monospace"),
  sans: token("--font-sans", "sans-serif"),
};

/** 多系列循环色板：青 → 紫 → 绿 → 琥珀 */
export const seriesColors: readonly string[] = [
  palette.cyan,
  palette.purple,
  palette.green,
  palette.amber,
];

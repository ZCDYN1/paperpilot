/**
 * 颜色的**唯一真相在 `src/styles/globals.css` 的 CSS 变量里**，这里只做名字映射。
 * 换主题只改那一个文件，组件与图表都会跟着变。
 */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        base: "var(--bg-base)",
        raised: "var(--bg-raised)",
        panel: "var(--bg-panel)",
        hairline: "var(--border-hairline)",
        ink: {
          DEFAULT: "var(--text-primary)",
          soft: "var(--text-secondary)",
          faint: "var(--text-muted)",
        },
        neon: {
          cyan: "var(--accent-cyan)",
          purple: "var(--accent-purple)",
          green: "var(--accent-green)",
          red: "var(--accent-red)",
          amber: "var(--accent-amber)",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        mono: ["var(--font-mono)"],
        display: ["var(--font-display)"],
      },
      boxShadow: {
        glass: "var(--shadow-glass)",
        glow: "0 0 12px rgba(0, 240, 255, 0.35)",
        "glow-lg": "0 0 28px rgba(0, 240, 255, 0.45)",
      },
      backdropBlur: { glass: "12px" },
      screens: { "3xl": "1780px" },
    },
  },
  plugins: [],
};

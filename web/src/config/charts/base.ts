import type { EChartsOption } from "echarts";
import { chartFont, palette, seriesColors, withAlpha } from "@/config/theme";

// 全局图表底：颜色只来自 palette，换主题只改 CSS 变量，这里永远不出现 hex 字面量。

const tooltipGlass = {
  backgroundColor: withAlpha(palette.panel, 0.92),
  borderColor: palette.hairline,
  borderWidth: 1,
  padding: [8, 12] as [number, number],
  textStyle: {
    color: palette.textPrimary,
    fontSize: 12,
    fontFamily: chartFont.sans,
  },
  extraCssText: `backdrop-filter: blur(12px); border-radius: 10px; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), 0 0 0 1px ${withAlpha(palette.cyan, 0.08)};`,
};

export function chartBase(overrides?: Partial<EChartsOption>): EChartsOption {
  return {
    backgroundColor: "transparent",
    color: [...seriesColors],
    animationDuration: 640,
    animationDurationUpdate: 420,
    animationEasing: "cubicOut",
    textStyle: {
      fontFamily: chartFont.sans,
      color: palette.textSecondary,
      fontSize: 12,
    },
    tooltip: { trigger: "axis", ...tooltipGlass },
    grid: { left: 16, right: 16, top: 40, bottom: 12, containLabel: true },
    ...overrides,
  };
}

export const axisCommon = {
  axisLine: { lineStyle: { color: palette.hairline } },
  axisTick: { lineStyle: { color: palette.hairline } },
  axisLabel: {
    color: palette.textMuted,
    fontSize: 11,
    fontFamily: chartFont.mono,
  },
  splitLine: {
    show: true,
    lineStyle: { color: palette.hairline, type: [4, 4] as [number, number], width: 1 },
  },
};

export const valueAxisBase = {
  type: "value" as const,
  ...axisCommon,
  axisLine: { show: false },
  axisTick: { show: false },
};

export const categoryAxisBase = {
  type: "category" as const,
  ...axisCommon,
  axisTick: { show: false },
  splitLine: { show: false },
};

export { tooltipGlass };

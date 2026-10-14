import type { EChartsOption } from "echarts";
import { graphic } from "echarts";
import { chartFont, palette, withAlpha } from "@/config/theme";
import { categoryAxisBase, chartBase, tooltipGlass, valueAxisBase } from "./base";

// 累计=发光面积（存量），区间新增=紫条（脉冲）；单点不画线，一条点线会伪装出"趋势"。

export interface TrendPoint {
  /** ISO 日期，如 "2026-01-05" */
  on: string;
  cited_by_count: number;
}

export interface TrendInput {
  points: TrendPoint[];
}

const finiteNum = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function citationTrendConfig({ points }: TrendInput): EChartsOption {
  const sorted = (points ?? [])
    .filter((p) => p && Number.isFinite(p.cited_by_count) && typeof p.on === "string")
    .slice()
    .sort((a, b) => a.on.localeCompare(b.on));

  if (sorted.length === 0) return chartBase();

  const categories = sorted.map((p) => p.on);
  const cumulative = sorted.map((p) => p.cited_by_count);
  const incremental = sorted.map((p, i) =>
    i === 0 ? p.cited_by_count : Math.max(0, p.cited_by_count - sorted[i - 1].cited_by_count),
  );

  const xAxis = { ...categoryAxisBase, boundaryGap: sorted.length > 1, data: categories };
  const yAxis = { ...valueAxisBase, minInterval: 1 };

  if (sorted.length === 1) {
    return {
      ...chartBase(),
      xAxis: { ...xAxis, boundaryGap: true },
      yAxis,
      tooltip: { ...tooltipGlass, trigger: "item" },
      series: [
        {
          name: "被引总数",
          type: "scatter" as const,
          symbolSize: 14,
          itemStyle: {
            color: palette.cyan,
            shadowBlur: 16,
            shadowColor: withAlpha(palette.cyan, 0.6),
          },
          label: {
            show: true,
            position: "right" as const,
            color: palette.textSecondary,
            fontFamily: chartFont.mono,
            formatter: (p: unknown) => String((p as { value?: number }).value ?? "—"),
          },
          data: [finiteNum(cumulative[0])],
        },
      ],
    };
  }

  return {
    ...chartBase(),
    xAxis,
    yAxis,
    legend: {
      data: ["累计被引", "区间新增"],
      top: 6,
      icon: "roundRect",
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { color: palette.textSecondary, fontSize: 11 },
    },
    tooltip: {
      ...tooltipGlass,
      trigger: "axis",
      valueFormatter: (v) => (typeof v === "number" && Number.isFinite(v) ? `${v} 次` : "—"),
    },
    grid: { left: 16, right: 16, top: 44, bottom: 12, containLabel: true },
    series: [
      {
        name: "累计被引",
        type: "line" as const,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { width: 2, color: palette.cyan, shadowBlur: 10, shadowColor: withAlpha(palette.cyan, 0.4) },
        itemStyle: { color: palette.cyan, shadowBlur: 8, shadowColor: withAlpha(palette.cyan, 0.5) },
        areaStyle: {
          color: new graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: withAlpha(palette.cyan, 0.4) },
            { offset: 1, color: withAlpha(palette.cyan, 0) },
          ]),
        },
        emphasis: { focus: "series" as const },
        data: cumulative.map(finiteNum),
      },
      {
        name: "区间新增",
        type: "bar" as const,
        barMaxWidth: 14,
        itemStyle: { color: withAlpha(palette.purple, 0.55), borderRadius: [3, 3, 0, 0] },
        emphasis: {
          itemStyle: { color: palette.purple, shadowBlur: 10, shadowColor: withAlpha(palette.purple, 0.5) },
        },
        data: incremental.map(finiteNum),
      },
    ],
  };
}

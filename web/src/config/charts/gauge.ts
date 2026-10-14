import type { EChartsOption } from "echarts";
import { graphic } from "echarts";
import { chartFont, palette, withAlpha } from "@/config/theme";
import { chartBase } from "./base";

// 仪表盘走"雷达刻度"语言：明显的 tick + 发光指针，detail 用等宽数字避免跳动。

export interface GaugeThreshold {
  /** 0~1 的占比位置（不是绝对值） */
  upto: number;
  color: string;
}

export interface GaugeInput {
  value: number;
  max?: number;
  name?: string;
  thresholds?: GaugeThreshold[];
}

export function gaugeConfig({ value, max = 100, name, thresholds }: GaugeInput): EChartsOption {
  const safeMax = Number.isFinite(max) && max > 0 ? max : 100;
  const safeValue = Number.isFinite(value) ? Math.min(Math.max(value, 0), safeMax) : 0;
  const ratio = safeValue / safeMax;

  const stops: Array<[number, string]> =
    thresholds && thresholds.length
      ? (() => {
          const sorted = [...thresholds].sort((a, b) => a.upto - b.upto);
          let prev = 0;
          return sorted.map((t) => {
            const upto = Math.min(Math.max(t.upto, prev), 1);
            prev = upto;
            return [upto, t.color] as [number, string];
          });
        })()
      : [
          [0.55, palette.cyan],
          [0.8, palette.purple],
          [1, palette.amber],
        ];

  const colorAt =
    stops.find(([upto]) => ratio <= upto)?.[1] ?? stops[stops.length - 1]?.[1] ?? palette.cyan;

  return {
    ...chartBase({ grid: undefined, tooltip: { show: false } }),
    series: [
      {
        type: "gauge" as const,
        center: ["50%", "56%"],
        radius: "88%",
        startAngle: 210,
        endAngle: -30,
        min: 0,
        max: safeMax,
        splitNumber: 5,
        axisLine: {
          lineStyle: {
            width: 10,
            color: stops,
          },
        },
        progress: {
          show: true,
          width: 10,
          roundCap: true,
          itemStyle: {
            color: new graphic.LinearGradient(0, 0, 1, 0, [
              { offset: 0, color: withAlpha(colorAt, 0.5) },
              { offset: 1, color: colorAt },
            ]),
            shadowBlur: 12,
            shadowColor: withAlpha(colorAt, 0.6),
          },
        },
        pointer: {
          icon: "path://M2.5,0,L5,18,L-5,18,Z",
          length: "58%",
          width: 5,
          offsetCenter: [0, "2%"],
          itemStyle: {
            color: colorAt,
            shadowBlur: 14,
            shadowColor: withAlpha(colorAt, 0.85),
          },
        },
        anchor: {
          show: true,
          size: 12,
          showAbove: true,
          itemStyle: {
            color: palette.panel,
            borderColor: colorAt,
            borderWidth: 2,
            shadowBlur: 10,
            shadowColor: withAlpha(colorAt, 0.5),
          },
        },
        axisTick: {
          distance: -16,
          splitNumber: 5,
          lineStyle: { color: withAlpha(palette.textMuted, 0.6), width: 1 },
          length: 5,
        },
        splitLine: {
          distance: -20,
          length: 12,
          lineStyle: { color: withAlpha(palette.textSecondary, 0.55), width: 2 },
        },
        axisLabel: {
          distance: -8,
          color: palette.textMuted,
          fontSize: 10,
          fontFamily: chartFont.mono,
          formatter: (v: number) => `${Math.round(v)}`,
        },
        detail: {
          valueAnimation: true,
          offsetCenter: [0, "34%"],
          fontSize: 30,
          fontWeight: 600,
          fontFamily: chartFont.mono,
          color: colorAt,
          formatter: (v: number) => (Number.isFinite(v) ? `${Math.round(v)}` : "—"),
        },
        title: {
          offsetCenter: [0, "58%"],
          color: palette.textSecondary,
          fontSize: 12,
        },
        data: [{ value: safeValue, name: name ?? "" }],
      },
    ],
  };
}

import type { EChartsOption } from "echarts";
import { graphic } from "echarts";
import { chartFont, palette, withAlpha } from "@/config/theme";
import { categoryAxisBase, chartBase, tooltipGlass, valueAxisBase } from "./base";

// 青→紫渐变 + 圆角是"HUD 数据条"的视觉语言；hover 加辉光而不是换颜色，保持语义稳定。

export interface BarSeriesInput {
  name: string;
  values: Array<number | null | undefined>;
  /** 单系列时用内置青→紫渐变；传 color 则整条用该色渐变 */
  color?: string;
}

export interface BarChartOpts {
  horizontal?: boolean;
  stack?: boolean;
  barMaxWidth?: number;
  valueFormatter?: (v: number) => string;
  /**
   * 柱顶标数值。指标对比类图（如 85% vs 93%）在 0 基线下柱子几乎一样高，
   * 图就等于没说话；宁可标数字，也不靠截断 y 轴去放大差异 —— 那会反过来误导。
   */
  valueLabel?: boolean;
}

export interface BarChartInput {
  categories: string[];
  series: BarSeriesInput[];
  opts?: BarChartOpts;
}

const finite = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function barChartConfig({ categories, series, opts }: BarChartInput): EChartsOption {
  const horizontal = opts?.horizontal === true;
  const band = opts?.stack ? undefined : 0.65;

  const gradOf = (c: string | undefined, idx: number) => {
    const stops = c
      ? [
          { offset: 0, color: withAlpha(c, 0.55) },
          { offset: 1, color: c },
        ]
      : [
          { offset: 0, color: withAlpha(palette.cyan, Math.max(0.4, 0.85 - idx * 0.12)) },
          { offset: 1, color: palette.purple },
        ];
    // 横向条从左往右渐变，纵向条从下往上：光源永远顺着条的生长方向走
    return horizontal
      ? new graphic.LinearGradient(0, 0, 1, 0, stops)
      : new graphic.LinearGradient(0, 1, 0, 0, stops);
  };

  const categoryAxis = {
    ...categoryAxisBase,
    data: categories,
    axisLabel: {
      ...categoryAxisBase.axisLabel,
      interval: "auto" as const,
      ...(horizontal ? {} : { rotate: categories.length > 8 ? 30 : 0 }),
      hideOverlap: true,
    },
  };
  const measureAxis = {
    ...valueAxisBase,
    axisLabel: {
      ...valueAxisBase.axisLabel,
      formatter: opts?.valueFormatter ? (v: number) => opts.valueFormatter!(v) : undefined,
    },
    min: (value: { min: number }) => (value.min < 0 ? value.min : 0),
  };

  return {
    ...chartBase(),
    tooltip: {
      ...tooltipGlass,
      trigger: "axis",
      axisPointer: { type: "shadow", shadowStyle: { color: withAlpha(palette.cyan, 0.06) } },
      valueFormatter: (v) =>
        typeof v === "number" && Number.isFinite(v)
          ? `${opts?.valueFormatter ? opts.valueFormatter(v) : v}`
          : "—",
    },
    xAxis: horizontal ? measureAxis : categoryAxis,
    yAxis: horizontal ? categoryAxis : measureAxis,
    series: series.map((s, i) => ({
      name: s.name,
      type: "bar" as const,
      barMaxWidth: opts?.barMaxWidth ?? 26,
      barCategoryGap: band,
      stack: opts?.stack ? "total" : undefined,
      itemStyle: {
        borderRadius: horizontal ? [0, 6, 6, 0] : [6, 6, 0, 0],
        color: gradOf(s.color, i),
      },
      emphasis: {
        itemStyle: {
          color: s.color ?? palette.cyan,
          shadowBlur: 14,
          shadowColor: withAlpha(s.color ?? palette.cyan, 0.55),
        },
      },
      data: (s.values ?? []).map(finite),
      label: opts?.valueLabel
        ? {
            show: true,
            position: horizontal ? "right" : "top",
            distance: 4,
            color: palette.textSecondary,
            fontFamily: chartFont.mono,
            fontSize: 10,
            formatter: (p: { value?: unknown }) => {
              const v = p.value;
              if (typeof v !== "number" || !Number.isFinite(v)) return "";
              return opts.valueFormatter ? opts.valueFormatter(v) : String(v);
            },
          }
        : undefined,
    })),
  };
}

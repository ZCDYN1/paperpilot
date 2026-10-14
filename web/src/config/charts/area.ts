import type { EChartsOption } from "echarts";
import { graphic } from "echarts";
import { seriesColors, withAlpha } from "@/config/theme";
import { categoryAxisBase, chartBase, tooltipGlass, valueAxisBase } from "./base";

// 面积从上到下霓虹渐隐，让曲线像发光的仪表盘描线而不是普通折线。

export interface AreaSeriesInput {
  name: string;
  values: Array<number | null | undefined>;
  color?: string;
}

export interface AreaChartOpts {
  stack?: boolean;
  smooth?: boolean;
  yFormatter?: (v: number) => string;
  tooltipUnit?: string;
  hideGrid?: boolean;
}

export interface AreaChartInput {
  categories: string[];
  series: AreaSeriesInput[];
  opts?: AreaChartOpts;
}

const finite = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

export function areaChartConfig({ categories, series, opts }: AreaChartInput): EChartsOption {
  const smooth = opts?.smooth !== false;
  const list = series.length ? series : [];

  return {
    ...chartBase(),
    xAxis: { ...categoryAxisBase, boundaryGap: false, data: categories },
    yAxis: {
      ...valueAxisBase,
      axisLabel: {
        ...categoryAxisBase.axisLabel,
        formatter: opts?.yFormatter ? (v: number) => opts.yFormatter!(v) : undefined,
      },
      splitLine: opts?.hideGrid ? { show: false } : categoryAxisBase.splitLine,
    },
    tooltip: {
      ...tooltipGlass,
      trigger: "axis",
      valueFormatter: (v) =>
        typeof v === "number" && Number.isFinite(v)
          ? `${opts?.yFormatter ? opts.yFormatter(v) : v}${opts?.tooltipUnit ?? ""}`
          : "—",
    },
    series: list.map((s, i) => {
      const c = s.color ?? seriesColors[i % seriesColors.length];
      return {
        name: s.name,
        type: "line" as const,
        smooth,
        symbol: "circle",
        symbolSize: 5,
        showSymbol: categories.length <= 40,
        connectNulls: false,
        stack: opts?.stack ? "total" : undefined,
        lineStyle: { width: 2, color: c, shadowBlur: 10, shadowColor: withAlpha(c, 0.35) },
        itemStyle: { color: c, shadowBlur: 8, shadowColor: withAlpha(c, 0.55) },
        emphasis: { focus: "series" as const },
        areaStyle: {
          opacity: 1,
          color: new graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: withAlpha(c, opts?.stack ? 0.55 : 0.4) },
            { offset: 1, color: withAlpha(c, 0) },
          ]),
        },
        data: (s.values ?? []).map(finite),
      };
    }),
  };
}

import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { chartFont, invalidateTokens, palette, withAlpha } from "@/config/theme";
import { cn } from "@/lib/cn";

// 挂载时 invalidateTokens()：主题切换后 CSS 变量已变，不能拿 theme.ts 的缓存色画旧主题。

function tokenLoadingOption() {
  return {
    text: "加载中…",
    color: palette.cyan,
    textColor: palette.textSecondary,
    maskColor: withAlpha(palette.panel, 0.72),
    zlevel: 0,
    font: `12px ${chartFont.mono}`,
  };
}

export interface ChartProps {
  option: EChartsOption;
  height?: number | string;
  loading?: boolean;
  onEvents?: Record<string, (payload: unknown) => void>;
  className?: string;
}

export function Chart({ option, height = 280, loading, onEvents, className }: ChartProps) {
  const [loadingOption, setLoadingOption] = useState(tokenLoadingOption);

  useEffect(() => {
    invalidateTokens();
    setLoadingOption(tokenLoadingOption());
  }, []);

  return (
    <div className={cn("w-full", className)} style={{ height }}>
      <ReactECharts
        option={option}
        notMerge
        lazyUpdate
        opts={{ renderer: "canvas" }}
        style={{ height: "100%", width: "100%" }}
        showLoading={!!loading}
        loadingOption={loadingOption}
        onEvents={onEvents}
      />
    </div>
  );
}

export default Chart;

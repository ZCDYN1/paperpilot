import type { EChartsOption } from "echarts";
import { chartFont, palette, seriesColors, withAlpha } from "@/config/theme";
import { chartBase, tooltipGlass } from "./base";

// 扇区之间用"与面板同色的描边"制造缝隙：透明背景上不行，所以缝隙靠 borderWidth + panel 色伪造。

export interface RingItem {
  name: string;
  value: number;
  color?: string;
}

export interface RingInput {
  items: RingItem[];
  /** 中心大字下方的说明文字 */
  centerLabel?: string;
}

export function ringConfig({ items, centerLabel = "总计" }: RingInput): EChartsOption {
  const clean = items.filter((it) => Number.isFinite(it.value) && it.value > 0);
  const total = clean.reduce((acc, it) => acc + it.value, 0);
  const colorAt = (i: number) => seriesColors[i % seriesColors.length];

  return {
    ...chartBase({ grid: undefined }),
    tooltip: {
      ...tooltipGlass,
      trigger: "item",
      formatter: (p: unknown) => {
        const d = p as { name?: string; value?: number; percent?: number; marker?: string };
        const v = typeof d.value === "number" ? d.value : 0;
        const pc = typeof d.percent === "number" ? `${d.percent.toFixed(1)}%` : "—";
        return `${d.marker ?? ""}<b style="font-family:${chartFont.mono}">${esc(d.name ?? "")}</b><br/>${v.toLocaleString()} · ${pc}`;
      },
    },
    title: {
      text: total.toLocaleString(),
      subtext: total > 0 ? centerLabel : "暂无数据",
      left: "center",
      top: "center",
      textStyle: {
        color: palette.textPrimary,
        fontSize: 26,
        fontWeight: 700,
        fontFamily: chartFont.mono,
      },
      subtextStyle: { color: palette.textMuted, fontSize: 12 },
    },
    series: [
      {
        type: "pie" as const,
        radius: ["58%", "78%"],
        center: ["50%", "50%"],
        avoidLabelOverlap: true,
        padAngle: total > 0 ? 2 : 0,
        itemStyle: {
          borderColor: palette.panel,
          borderWidth: 2,
          borderRadius: 6,
        },
        label: {
          color: palette.textSecondary,
          fontSize: 11,
          formatter: "{b}\n{d}%",
          lineHeight: 15,
        },
        labelLine: { lineStyle: { color: palette.hairline }, length: 8, length2: 10 },
        emphasis: {
          scaleSize: 6,
          itemStyle: { shadowBlur: 18, shadowColor: withAlpha(palette.cyan, 0.35) },
          label: { color: palette.textPrimary, fontWeight: 600 },
        },
        data: clean.map((it, i) => ({
          name: it.name,
          value: it.value,
          itemStyle: {
            color: it.color ?? colorAt(i),
            shadowBlur: 8,
            shadowColor: withAlpha(it.color ?? colorAt(i), 0.3),
          },
        })),
      },
    ],
  };
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => `&${{ "&": "amp", "<": "lt", ">": "gt", '"': "quot" }[c]};`);
}

import type { EChartsOption } from "echarts";
import { chartFont, palette, withAlpha } from "@/config/theme";
import { chartBase, tooltipGlass } from "./base";
import { tierColor } from "./graph";

// 时间轴讲"谁补了谁的坑"：箭头永远从新年份指向被改进的旧论文；被引数缺失画最小点而不是 0（成对规则）。

export interface TimelineGapRef {
  paper_id: string;
  title?: string;
}

export interface TimelineItemInput {
  arxiv_id: string;
  title?: string;
  year?: number | null;
  venue?: string;
  venue_tier?: string;
  cited_by_count?: number | null;
  core_idea?: string;
  limitation?: string;
  addresses_gap?: TimelineGapRef[];
}

const LANES = 3;

export function methodTimelineConfig({ items }: { items: TimelineItemInput[] }): EChartsOption {
  const plottable = items
    .filter((it) => typeof it.year === "number" && Number.isFinite(it.year))
    .sort((a, b) => (a.year as number) - (b.year as number) || a.arxiv_id.localeCompare(b.arxiv_id));

  const years = plottable.map((it) => it.year as number);
  const minYear = years.length ? Math.min(...years) - 1 : 2018;
  const maxYear = years.length ? Math.max(...years) + 1 : Math.max(minYear + 4, 2025);
  const maxCite = Math.max(
    1,
    ...plottable.map((it) => (typeof it.cited_by_count === "number" && it.cited_by_count > 0 ? it.cited_by_count : 0)),
  );

  const pos = new Map<string, { x: number; y: number }>();
  const nodeData = plottable.map((it, idx) => {
    const lane = idx % LANES;
    pos.set(it.arxiv_id, { x: it.year as number, y: lane });
    const cite = typeof it.cited_by_count === "number" && it.cited_by_count > 0 ? it.cited_by_count : null;
    const size = cite == null ? 9 : 11 + 20 * Math.sqrt(cite / maxCite);
    return {
      name: it.arxiv_id,
      value: [it.year as number, lane] as [number, number],
      symbolSize: size,
      itemStyle: {
        color: tierColor(it.venue_tier, true),
        borderColor: withAlpha(palette.panel, 0.9),
        borderWidth: 1.5,
        shadowBlur: 10,
        shadowColor: withAlpha(tierColor(it.venue_tier, true), 0.5),
      },
      meta: it,
    };
  });

  const gapEdges = plottable.flatMap((it) =>
    (it.addresses_gap ?? [])
      .map((g) => {
        const from = pos.get(it.arxiv_id);
        const to = pos.get(g.paper_id);
        if (!from || !to) return null;
        return {
          coords: [
            [from.x, from.y],
            [to.x, to.y],
          ] as Array<[number, number]>,
          lineStyle: {
            color: withAlpha(palette.purple, 0.65),
            width: 1.4,
            curveness: 0.22,
          },
        };
      })
      .filter((x): x is NonNullable<typeof x> => x !== null),
  );

  const spine = nodeData.map((n) => n.value);

  return {
    ...chartBase({ grid: { left: 24, right: 24, top: 28, bottom: 24, containLabel: true } }),
    tooltip: {
      ...tooltipGlass,
      trigger: "item",
      formatter: (p: unknown) => {
        const d = p as { dataType?: string; data?: { meta?: TimelineItemInput } };
        const m = d.data?.meta;
        if (!m) return "";
        const rows = [
          `<b style="font-family:${chartFont.mono}">${esc(String(m.year ?? "—"))}</b> · ${esc(m.title ?? m.arxiv_id)}`,
          m.venue ? `发表：${esc(m.venue)}` : "发表：—",
          m.cited_by_count == null
            ? "被引：未记录（缺截至日期，宁缺毋滥）"
            : `被引：${m.cited_by_count}`,
          m.addresses_gap?.length ? `补的坑：${esc(m.addresses_gap.map((g) => g.title ?? g.paper_id).join("、"))}` : "",
        ].filter(Boolean);
        return rows.join("<br/>");
      },
    },
    xAxis: {
      type: "value" as const,
      min: minYear,
      max: maxYear,
      interval: 1,
      axisLine: { lineStyle: { color: palette.hairline } },
      axisTick: { show: false },
      axisLabel: {
        color: palette.textMuted,
        fontSize: 11,
        fontFamily: chartFont.mono,
        formatter: (v: number) => `${Math.round(v)}`,
      },
      splitLine: {
        show: true,
        lineStyle: { color: palette.hairline, type: [4, 4] as [number, number] },
      },
    },
    yAxis: { type: "value" as const, show: false, min: -0.6, max: LANES - 0.4 },
    series: [
      {
        name: "演进链",
        type: "lines" as const,
        polyline: true,
        symbol: ["none", "none"],
        silent: true,
        z: 1,
        lineStyle: { color: withAlpha(palette.textMuted, 0.35), width: 1, type: "dashed", opacity: 1 },
        data: spine.length > 1 ? [{ coords: spine }] : [],
      },
      {
        name: "引用改进",
        type: "lines" as const,
        symbol: ["none", "arrow"],
        symbolSize: 8,
        z: 2,
        data: gapEdges,
      },
      {
        name: "论文",
        type: "scatter" as const,
        z: 3,
        data: nodeData,
        label: {
          show: true,
          position: "top" as const,
          distance: 8,
          color: palette.textSecondary,
          fontSize: 10,
          fontFamily: chartFont.mono,
          formatter: (p: unknown) => {
            const m = (p as { data?: { meta?: TimelineItemInput } }).data?.meta;
            const t = (m?.title ?? m?.arxiv_id ?? "").trim();
            return t.length > 16 ? `${t.slice(0, 16)}…` : t;
          },
        },
        emphasis: { scale: 1.15 },
      },
    ],
  };
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => `&${{ "&": "amp", "<": "lt", ">": "gt", '"': "quot" }[c]};`);
}

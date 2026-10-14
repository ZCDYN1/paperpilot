import type { EChartsOption } from "echarts";
import { chartFont, palette, withAlpha } from "@/config/theme";
import { chartBase, tooltipGlass } from "./base";

// 语义色即图例：在库 & 顶会=青、在库核心=紫、预印本=绿、库外=灰、琥珀留给"高被引"锚点，边方向=引用方向。

export type VenueTier = "top" | "core" | "preprint" | "unknown";

export interface GraphNodeInput {
  id: string;
  name?: string;
  /** 是否已在本地库（F16：库里论文 vs 只作为引用节点出现） */
  inLibrary?: boolean;
  tier?: VenueTier | string;
  /** 被引数：映射节点大小；缺失传 null，会画成最小号点而不是 0 */
  citedBy?: number | null;
  year?: number | null;
}

export interface GraphLinkInput {
  source: string;
  target: string;
  label?: string;
}

export interface CitationGraphInput {
  nodes: GraphNodeInput[];
  links: GraphLinkInput[];
}

export function tierColor(tier: string | undefined, inLibrary: boolean): string {
  if (!inLibrary) return palette.textMuted;
  switch (tier) {
    case "top":
      return palette.cyan;
    case "core":
      return palette.purple;
    case "preprint":
      return palette.greenSoft;
    default:
      // 分档未知时用中性灰，不用琥珀色：告警色留给"真的有问题"的状态，
      // 而"没分档"只是信息缺失，染成红色系会让人以为数据出错了。
      return palette.textSecondary;
  }
}

const shortName = (id: string) => (id.length > 18 ? `${id.slice(0, 18)}…` : id);

export function citationGraphConfig({ nodes, links }: CitationGraphInput): EChartsOption {
  const ids = new Set(nodes.map((n) => n.id));
  const safeLinks = links.filter((l) => ids.has(l.source) && ids.has(l.target) && l.source !== l.target);
  const degree = new Map<string, number>();
  safeLinks.forEach((l) => {
    degree.set(l.source, (degree.get(l.source) ?? 0) + 1);
    degree.set(l.target, (degree.get(l.target) ?? 0) + 1);
  });
  const maxCite = Math.max(
    1,
    ...nodes.map((n) => (typeof n.citedBy === "number" && Number.isFinite(n.citedBy) ? n.citedBy : 0)),
  );

  return {
    ...chartBase({ grid: undefined }),
    tooltip: {
      ...tooltipGlass,
      trigger: "item",
      formatter: (p: unknown) => {
        const d = p as { dataType?: string; data?: Record<string, unknown>; name?: string };
        if (d.dataType === "edge") {
          const lbl = (d.data?.label ?? "") as string;
          return `<b>引用关系</b>${lbl ? `<br/>${esc(String(lbl))}` : ""}`;
        }
        const meta = (d.data?.meta ?? {}) as Record<string, unknown>;
        const parts = [
          `<b style="font-family:${chartFont.mono}">${esc(String(d.name ?? ""))}</b>`,
          meta.inLibrary ? "在库" : "库外引用",
          meta.tier ? `档：${esc(String(meta.tier))}` : "",
          meta.citedBy == null ? "被引：未记录" : `被引：${meta.citedBy}`,
          meta.year ? `年份：${meta.year}` : "",
        ].filter(Boolean);
        return parts.join("<br/>");
      },
    },
    series: [
      {
        type: "graph" as const,
        layout: "force" as const,
        roam: true,
        draggable: true,
        zoom: 1.08,
        force: {
          // 面板是宽扁的（约 1900×440），默认斥力会让 8 个节点缩成中间一小簇，
          // 门面页看着像没画完。放大斥力与边长、压低重力让它铺开。
          repulsion: 460,
          edgeLength: [130, 280],
          gravity: 0.018,
          friction: 0.6,
        },
        emphasis: {
          focus: "adjacency" as const,
          lineStyle: { width: 2, color: palette.cyan },
        },
        label: {
          show: true,
          position: "bottom" as const,
          distance: 6,
          color: palette.textMuted,
          fontSize: 10,
          fontFamily: chartFont.mono,
          formatter: (p: unknown) => shortName(String((p as { name?: string }).name ?? "")),
        },
        lineStyle: {
          color: withAlpha(palette.textSecondary, 0.55),
          width: 1,
          curveness: 0.14,
          opacity: 0.7,
        },
        edgeSymbol: ["none", "arrow"],
        edgeSymbolSize: 7,
        data: nodes.map((n) => {
          const color = tierColor(n.tier, n.inLibrary !== false);
          const cite = typeof n.citedBy === "number" && Number.isFinite(n.citedBy) ? n.citedBy : null;
          const size = cite == null ? 10 : 12 + 26 * Math.sqrt(cite / maxCite);
          return {
            id: n.id,
            name: n.name ?? n.id,
            value: size,
            symbolSize: size,
            meta: { inLibrary: n.inLibrary !== false, tier: n.tier, citedBy: cite, year: n.year },
            itemStyle: {
              color,
              borderColor: withAlpha(color, 0.85),
              borderWidth: 1,
              shadowBlur: 10 + Math.min(14, (degree.get(n.id) ?? 0) * 2),
              shadowColor: withAlpha(color, 0.5),
            },
          };
        }),
        links: safeLinks.map((l) => ({
          source: l.source,
          target: l.target,
          label: { show: false, formatter: l.label ?? "" },
        })),
      },
    ],
  };
}

function esc(s: string): string {
  return s.replace(/[&<>"]/g, (c) => `&${{ "&": "amp", "<": "lt", ">": "gt", '"': "quot" }[c]};`);
}

// 发光用 token 现成的 *-dim / *-line 变量做 text-shadow，green 没有 dim 变量就退回中性白叠。
import type { CSSProperties } from "react";
import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/cn";
import { CountUp } from "./CountUp";
import { GlassCard } from "./GlassCard";
import { Sparkline } from "./Sparkline";
import { NavIcon } from "@/components/layout/iconMap";

type Accent = "cyan" | "purple" | "green" | "amber";

const glowStyle: Record<Accent, CSSProperties> = {
  cyan: { textShadow: "0 0 14px var(--accent-cyan-dim)" },
  purple: { textShadow: "0 0 14px var(--accent-purple-dim)" },
  green: { textShadow: "0 0 14px rgba(255, 255, 255, 0.1)" },
  amber: { textShadow: "0 0 14px rgba(255, 255, 255, 0.1)" },
};

const sparkColor: Record<Accent, string> = {
  cyan: "var(--accent-cyan)",
  purple: "var(--accent-purple)",
  green: "var(--accent-green)",
  amber: "var(--accent-amber)",
};

export type StatCardProps = {
  label: string;
  value: number;
  unit?: string;
  /** 环比百分比原值，正数绿↑负数红↓ */
  delta?: number;
  /** 底部微型趋势折线的数据，可为空 */
  series?: number[];
  /** iconMap 里的图标名，如 "zap" / "activity" */
  icon?: string;
  accent?: Accent;
  decimals?: number;
  className?: string;
};

export function StatCard({
  label,
  value,
  unit,
  delta,
  series,
  icon,
  accent = "cyan",
  decimals = 0,
  className,
}: StatCardProps) {
  const hasDelta = typeof delta === "number" && Number.isFinite(delta);
  const up = hasDelta && delta >= 0;

  return (
    <GlassCard hover className={cn("flex flex-col", className)}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-xs font-medium text-ink-soft">{label}</span>
        {icon && <NavIcon name={icon} className="h-4 w-4 shrink-0 text-ink-faint" />}
      </div>

      <div className="mt-3 flex items-baseline gap-1.5">
        <span
          className="text-3xl font-semibold leading-none text-ink"
          style={glowStyle[accent]}
        >
          <CountUp value={value} decimals={decimals} />
        </span>
        {unit && <span className="text-sm text-ink-faint">{unit}</span>}
      </div>

      <div className="mt-4 flex items-end justify-between gap-3">
        {hasDelta && (
          <span
            className={cn(
              "tabular inline-flex items-center gap-1 font-mono text-xs",
              up ? "text-neon-green" : "text-neon-red",
            )}
          >
            {up ? (
              <ArrowUpRight size={14} aria-hidden={true} />
            ) : (
              <ArrowDownRight size={14} aria-hidden={true} />
            )}
            {`${up ? "+" : "-"}${Math.abs(delta).toFixed(1)}%`}
          </span>
        )}
        <Sparkline data={series ?? []} color={sparkColor[accent]} showArea className="ml-auto" />
      </div>
    </GlassCard>
  );
}

export default StatCard;

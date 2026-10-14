import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export type SegmentedOption = {
  value: string;
  label: ReactNode;
};

export type SegmentedProps = {
  options: readonly SegmentedOption[];
  value: string;
  onChange: (value: string) => void;
  /** 同一页面多个切换器时给不同的 name，用作 aria 分组标签 */
  ariaLabel?: string;
  className?: string;
};

export function Segmented({
  options,
  value,
  onChange,
  ariaLabel = "切换视图",
  className,
}: SegmentedProps) {
  return (
    <div
      role="radiogroup"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex items-center gap-1 rounded-[10px] border border-[color:var(--glass-border)] bg-white/[0.03] p-1",
        className,
      )}
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "rounded-[7px] px-3 py-1 text-xs font-medium transition-colors duration-[var(--dur-fast)]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)]",
              active
                ? "border border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)] text-neon-cyan shadow-glow"
                : "border border-transparent text-ink-soft hover:bg-white/[0.05] hover:text-ink",
            )}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

export default Segmented;

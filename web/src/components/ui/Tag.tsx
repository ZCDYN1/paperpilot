// green/red/amber 在 token 表里没有 *-dim 背景变量，按约定用中性白色叠加代替，不新增 token。
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

type TagVariant = "cyan" | "purple" | "green" | "red" | "amber" | "muted";

const variantClass: Record<TagVariant, string> = {
  cyan: "border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)] text-neon-cyan",
  purple:
    "border-[color:var(--accent-purple-dim)] bg-[var(--accent-purple-dim)] text-neon-purple",
  green: "border-[color:var(--accent-green)] bg-white/[0.05] text-neon-green",
  red: "border-[color:var(--accent-red)] bg-white/[0.05] text-neon-red",
  amber: "border-[color:var(--accent-amber)] bg-white/[0.05] text-neon-amber",
  muted: "border-[color:var(--border-hairline)] bg-white/[0.05] text-ink-soft",
};

export type TagProps = {
  variant?: TagVariant;
  /** 等宽字体：给 F 编号 / 数字类徽章用 */
  mono?: boolean;
  className?: string;
  children?: ReactNode;
};

export function Tag({ variant = "muted", mono = false, className, children }: TagProps) {
  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-[6px] border px-1.5 py-0.5 text-[11px] leading-4",
        mono && "font-mono tabular",
        variantClass[variant],
        className,
      )}
    >
      {children}
    </span>
  );
}

export default Tag;

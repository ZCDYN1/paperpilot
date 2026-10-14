import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { NavIcon } from "@/components/layout/iconMap";

export type EmptyStateProps = {
  /** iconMap 里的图标名，默认 inbox */
  icon?: string;
  title: ReactNode;
  description?: ReactNode;
  /** 可选操作：一般塞一个 NeonButton */
  action?: ReactNode;
  className?: string;
};

export function EmptyState({
  icon = "inbox",
  title,
  description,
  action,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        "flex min-h-[180px] flex-col items-center justify-center gap-2 p-6 text-center",
        className,
      )}
    >
      <span className="mb-1 flex h-12 w-12 items-center justify-center rounded-full border border-[color:var(--glass-border-strong)] bg-white/[0.04]">
        <NavIcon name={icon} className="h-5 w-5 text-ink-faint" />
      </span>
      <p className="text-sm font-medium text-ink-soft">{title}</p>
      {description && <p className="max-w-sm text-xs leading-relaxed text-ink-faint">{description}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}

export default EmptyState;

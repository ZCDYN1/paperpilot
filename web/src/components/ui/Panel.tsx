import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export type PanelProps = {
  title: ReactNode;
  subtitle?: ReactNode;
  /** 标题栏右侧插槽（Segmented / Tag / 按钮等） */
  extra?: ReactNode;
  bodyClassName?: string;
  className?: string;
  children?: ReactNode;
};

export function Panel({
  title,
  subtitle,
  extra,
  bodyClassName,
  className,
  children,
}: PanelProps) {
  return (
    <section className={cn("glass panel flex min-h-0 flex-col", className)}>
      <header className="hairline-b flex shrink-0 items-center gap-3 px-4 py-3">
        <div className="min-w-0 flex-1">
          <h3 className="truncate text-sm font-semibold text-ink">{title}</h3>
          {subtitle && (
            <p className="mt-0.5 truncate font-mono text-[11px] text-ink-faint">{subtitle}</p>
          )}
        </div>
        {extra && <div className="flex shrink-0 items-center gap-2">{extra}</div>}
      </header>
      <div className={cn("min-h-0 flex-1 p-4", bodyClassName)}>{children}</div>
    </section>
  );
}

export default Panel;

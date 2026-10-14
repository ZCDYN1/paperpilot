import type { ReactNode } from "react";
import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import { cn } from "@/lib/cn";

// 图表卡只管"标题 + 状态占位 + 插槽"，数据获取留给主控：组件永不自己发请求。

export type ChartCardState = "loading" | "empty" | "error" | undefined;

export interface ChartCardProps {
  title: string;
  subtitle?: string;
  /** 标题栏右侧的操作区（刷新按钮 / 时间范围切换等） */
  extra?: ReactNode;
  height?: number | string;
  state?: ChartCardState;
  emptyHint?: string;
  errorHint?: string;
  className?: string;
  children?: ReactNode;
}

function Skeleton() {
  return (
    <div className="flex h-full w-full flex-col justify-end gap-3 px-6 pb-6" aria-hidden>
      {[42, 66, 30, 80, 54].map((w, i) => (
        <div
          key={i}
          className="h-3 animate-pulse rounded-full bg-hairline"
          style={{ width: `${w}%`, animationDelay: `${i * 90}ms` }}
        />
      ))}
    </div>
  );
}

export function ChartCard({
  title,
  subtitle,
  extra,
  height = 320,
  state,
  emptyHint = "暂无数据",
  errorHint = "图表渲染失败",
  className,
  children,
}: ChartCardProps) {
  return (
    <section className={cn("glass panel overflow-hidden", className)}>
      <header className="hairline-b flex items-start justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-ink">{title}</h3>
          {subtitle ? <p className="mt-0.5 truncate text-xs text-ink-faint">{subtitle}</p> : null}
        </div>
        {extra ? <div className="flex shrink-0 items-center gap-2">{extra}</div> : null}
      </header>

      <div className="relative w-full" style={{ height }}>
        {state === "loading" ? (
          <Skeleton />
        ) : state === "empty" ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-ink-faint">
            <Inbox size={22} strokeWidth={1.5} />
            <p className="text-xs">{emptyHint}</p>
          </div>
        ) : state === "error" ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-neon-red/80">
            <AlertTriangle size={22} strokeWidth={1.5} />
            <p className="text-xs">{errorHint}</p>
          </div>
        ) : (
          children ?? (
            <div className="flex h-full items-center justify-center text-xs text-ink-faint">
              <Loader2 size={14} className="mr-2 animate-spin" />
              等待图表配置
            </div>
          )
        )}
      </div>
    </section>
  );
}

export default ChartCard;

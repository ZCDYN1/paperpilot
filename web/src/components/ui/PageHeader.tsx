import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { Tag } from "./Tag";

export type PageHeaderProps = {
  title: ReactNode;
  description?: ReactNode;
  /** 功能编号串，如 "F4 · F5 · F8"，会自动拆成一排徽章 */
  features?: string;
  /** 右侧操作区插槽 */
  actions?: ReactNode;
  className?: string;
};

export function PageHeader({
  title,
  description,
  features,
  actions,
  className,
}: PageHeaderProps) {
  const badges = features
    ? features
        .split(/[·,，]/)
        .map((s) => s.trim())
        .filter(Boolean)
    : [];

  return (
    <div className={cn("mb-5 flex flex-wrap items-start justify-between gap-3", className)}>
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          {/* 刻意不加 tracking-tight：负字距会让相邻中文字形挤压，看上去像笔画被切了一道 */}
          <h1 className="text-xl font-semibold text-ink">{title}</h1>
          {badges.map((badge) => (
            <Tag key={badge} variant="cyan" mono className="mt-0.5">
              {badge}
            </Tag>
          ))}
        </div>
        {description && (
          <p className="mt-1 max-w-3xl text-sm leading-relaxed text-ink-soft">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

export default PageHeader;

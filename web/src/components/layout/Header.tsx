// 搜索框只做视觉展示（聚焦发光 + 快捷键提示），真实命令面板由后续需求再接。
import { env } from "@/config/env";
import { cn } from "@/lib/cn";
import { NavIcon } from "./iconMap";

export type HeaderStatus = "ok" | "error" | "idle";

const statusDotClass: Record<HeaderStatus, string> = {
  ok: "",
  error: "is-error",
  idle: "is-idle",
};

export function Header({
  onToggleSidebar,
  status = "idle",
  statusLabel,
}: {
  onToggleSidebar: () => void;
  status?: HeaderStatus;
  statusLabel?: string;
}) {
  return (
    <header className="glass panel relative z-30 mx-3 mt-3 flex h-[var(--header-h)] shrink-0 items-center gap-2 rounded-[12px] px-3 sm:gap-3 sm:px-4">
      <button
        type="button"
        onClick={onToggleSidebar}
        aria-label="切换侧栏"
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] text-ink-soft transition-colors duration-[var(--dur-fast)] hover:bg-white/[0.06] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)]"
      >
        <NavIcon name="panel-left-close" className="h-[18px] w-[18px]" />
      </button>

      <div className="flex min-w-0 shrink-0 items-center gap-2.5">
        <span className="flex h-8 w-8 items-center justify-center rounded-[8px] border border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)]">
          <NavIcon name="zap" className="glow-cyan h-4 w-4" />
        </span>
        <span
          className="glow-cyan truncate font-display text-[15px] font-semibold tracking-[0.08em]"
          title={env.appTitle}
        >
          {env.appTitle}
        </span>
      </div>

      {/* 全局搜索：<768px 收起成一个图标按钮 */}
      <div className="hidden min-w-0 flex-1 items-center justify-center px-2 md:flex">
        <div className="group relative w-full max-w-[420px]">
          <NavIcon
            name="search"
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
          />
          <input
            type="search"
            aria-label="全局搜索"
            placeholder="搜索论文 / 报告 / 会话…"
            className="panel h-9 w-full border border-[color:var(--glass-border)] bg-white/[0.03] pl-9 pr-16 text-sm text-ink placeholder:text-ink-faint transition-[border-color,box-shadow] duration-[var(--dur-base)] focus:outline-none focus:border-[color:var(--accent-cyan-line)] focus:shadow-glow"
          />
          <kbd className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded-[6px] border border-[color:var(--glass-border)] bg-white/[0.05] px-1.5 py-0.5 font-mono text-[10px] text-ink-faint transition-opacity duration-[var(--dur-fast)] group-focus-within:opacity-0">
            Ctrl K
          </kbd>
        </div>
      </div>
      <button
        type="button"
        aria-label="打开搜索"
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] text-ink-soft hover:bg-white/[0.06] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)] md:hidden"
      >
        <NavIcon name="search" className="h-[18px] w-[18px]" />
      </button>

      <div className="ml-auto flex items-center gap-2 sm:gap-4">
        <button
          type="button"
          aria-label="消息通知"
          className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-[8px] text-ink-soft hover:bg-white/[0.06] hover:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)]"
        >
          <NavIcon name="bell" className="h-[18px] w-[18px]" />
          <span className="absolute right-1.5 top-1.5 h-[6px] w-[6px] rounded-full bg-[var(--accent-amber)]" />
        </button>

        <div className="hidden items-center gap-2 rounded-[8px] border border-[color:var(--border-hairline)] bg-white/[0.03] px-2.5 py-1.5 sm:flex">
          <span
            aria-hidden={true}
            className={cn("status-dot shrink-0", statusDotClass[status])}
          />
          <span className="whitespace-nowrap text-xs text-ink-soft">
            系统运行状态
            {statusLabel && (
              <span className="tabular ml-1.5 font-mono text-[11px] text-neon-green">
                {statusLabel}
              </span>
            )}
          </span>
        </div>

        <span
          aria-label="当前用户"
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-[color:var(--glass-border-strong)] bg-[linear-gradient(135deg,var(--accent-cyan-dim),var(--accent-purple-dim))] font-display text-sm text-ink select-none"
        >
          {env.appTitle.charAt(0).toUpperCase()}
        </span>
      </div>
    </header>
  );
}

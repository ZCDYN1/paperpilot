// 菜单项全部来自 config/nav.ts；NavLink 自带 isActive，不需要手动比对 pathname。
import { NavLink } from "react-router-dom";
import type { NavItem } from "@/config/nav";
import { navGroups } from "@/config/nav";
import { cn } from "@/lib/cn";
import { NavIcon } from "./iconMap";

function itemBaseClass(collapsed: boolean): string {
  return cn(
    "group relative flex shrink-0 items-center rounded-[10px] text-sm transition-colors duration-[var(--dur-fast)]",
    collapsed ? "justify-center px-0 py-2.5" : "gap-3 px-3 py-2",
  );
}

function renderInner(item: NavItem, collapsed: boolean, isActive: boolean) {
  return (
    <>
      {isActive && (
        <span
          aria-hidden={true}
          className="absolute left-0 top-1/2 h-[55%] w-[3px] -translate-y-1/2 rounded-full bg-[var(--accent-cyan)] shadow-glow"
        />
      )}
      <NavIcon
        name={item.icon}
        className={cn(
          "h-[18px] w-[18px] shrink-0 transition-colors",
          isActive ? "text-neon-cyan" : "text-ink-faint group-hover:text-ink-soft",
        )}
      />
      {!collapsed && (
        <span className="min-w-0 flex-1">
          <span
            className={cn(
              "block truncate leading-5",
              isActive ? "text-ink" : "text-ink-soft group-hover:text-ink",
            )}
          >
            {item.label}
          </span>
          <span className="block truncate font-mono text-[10px] leading-4 text-ink-faint">
            {item.features}
          </span>
        </span>
      )}
      {!collapsed && item.comingSoon && (
        <span className="shrink-0 rounded-[6px] border border-[color:var(--border-hairline)] bg-white/[0.05] px-1.5 py-0.5 text-[10px] text-ink-faint">
          敬请期待
        </span>
      )}
    </>
  );
}

function MenuList({ collapsed }: { collapsed: boolean }) {
  return (
    <nav className="no-scrollbar flex-1 overflow-y-auto px-2.5 py-3" aria-label="主导航">
      {navGroups.map((group, gi) => (
        <section key={group.title} className={cn(gi > 0 && "mt-4")}>
          {collapsed ? (
            gi > 0 && (
              <div
                aria-hidden={true}
                className="mx-auto mb-3 h-px w-6 bg-[var(--border-hairline)]"
              />
            )
          ) : (
            // 分组标题是中文，0.18em 的疏排会把「知识库」拉成三个字各自飞开
            <h3 className="mb-1.5 px-3 font-mono text-[10px] uppercase tracking-[0.06em] text-ink-faint">
              {group.title}
            </h3>
          )}
          <ul className="space-y-1">
            {group.items.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  title={collapsed ? `${item.label} · ${item.features}` : undefined}
                  aria-disabled={item.comingSoon || undefined}
                  onClick={
                    item.comingSoon
                      ? (event) => {
                          event.preventDefault();
                        }
                      : undefined
                  }
                  className={({ isActive }) =>
                    cn(
                      itemBaseClass(collapsed),
                      isActive && !item.comingSoon
                        ? "bg-gradient-to-r from-[var(--accent-cyan-dim)] to-transparent text-ink"
                        : "text-ink-soft hover:bg-white/[0.04]",
                      item.comingSoon &&
                        "cursor-not-allowed opacity-45 hover:bg-transparent",
                      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)]",
                    )
                  }
                >
                  {({ isActive }) => renderInner(item, collapsed, isActive)}
                </NavLink>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </nav>
  );
}

export function Sidebar({
  collapsed,
  mobileOpen,
  onCloseMobile,
}: {
  collapsed: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}) {
  return (
    <>
      {/* >=1024px：常驻侧栏，可折叠成图标条 */}
      <aside
        className={cn(
          "glass panel relative z-20 hidden shrink-0 flex-col overflow-hidden border-r border-y-0 border-l-0 rounded-none backdrop-blur-glass lg:flex",
          collapsed ? "w-[var(--sidebar-w-collapsed)]" : "w-[var(--sidebar-w)]",
        )}
      >
        <MenuList collapsed={collapsed} />
      </aside>

      {/* <1024px：遮罩 + 抽屉 */}
      <div
        aria-hidden={!mobileOpen}
        onClick={onCloseMobile}
        className={cn(
          "fixed inset-0 z-40 bg-black/60 transition-opacity duration-[var(--dur-base)] lg:hidden",
          mobileOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        aria-label="导航抽屉"
        aria-hidden={!mobileOpen}
        className={cn(
          "glass panel fixed inset-y-0 left-0 z-50 flex w-[var(--sidebar-w)] flex-col transition-transform duration-[var(--dur-base)] ease-[var(--ease-out-expo)] lg:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <MenuList collapsed={false} />
      </aside>
    </>
  );
}

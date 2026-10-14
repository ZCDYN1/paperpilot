// body 已 overflow:hidden，主内容区必须是唯一滚动容器；折叠态持久化到 localStorage。
import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Header } from "./Header";
import { Sidebar } from "./Sidebar";
import { useSystemHealth } from "@/hooks/useSystemHealth";

const SIDEBAR_COLLAPSED_KEY = "sidebar-collapsed";
const MOBILE_BREAKPOINT = 1024;

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_COLLAPSED_KEY) === "1";
  } catch {
    return false;
  }
}

export function AppShell() {
  const [collapsed, setCollapsed] = useState<boolean>(readCollapsed);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { pathname } = useLocation();
  const health = useSystemHealth();

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? "1" : "0");
    } catch {
      /* 隐私模式等场景写不进去就算了，不影响功能 */
    }
  }, [collapsed]);

  // 换页时收起移动端抽屉，否则跳完页还盖在主内容上
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  const handleToggleSidebar = () => {
    if (window.innerWidth < MOBILE_BREAKPOINT) {
      setMobileOpen((open) => !open);
    } else {
      setCollapsed((prev) => !prev);
    }
  };

  return (
    <div className="relative z-10 flex h-full min-h-0 flex-col">
      <Header onToggleSidebar={handleToggleSidebar} status={health.status} statusLabel={health.label} />
      <div className="flex min-h-0 flex-1">
        <Sidebar
          collapsed={collapsed}
          mobileOpen={mobileOpen}
          onCloseMobile={() => setMobileOpen(false)}
        />
        <main className="min-h-0 min-w-0 flex-1 overflow-y-auto">
          <div className="flex min-h-full flex-col p-4 md:p-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}

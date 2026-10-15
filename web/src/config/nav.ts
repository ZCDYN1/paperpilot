/**
 * 菜单结构 = 配置，不写在组件 JSX 里。
 * 增删页面 / 改文案 / 调分组只动这个文件；Sidebar 只负责渲染它。
 *
 * `icon` 是图标名字符串，由 `components/layout/iconMap.tsx` 映射到实际组件，
 * 这样加菜单项不需要 import 新图标。
 */

export interface NavItem {
  path: string;
  label: string;
  /** 对应产品定义里的功能编号，直接显示在菜单里，方便对照验收 */
  features: string;
  icon: string;
  /** 该页所需接口尚未就绪时置 true：菜单可见但禁用 */
  comingSoon?: boolean;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const navGroups: NavGroup[] = [
  {
    title: "研究",
    items: [
      {
        path: "/console",
        label: "研究控制台",
        features: "F1 · F2 · F7",
        icon: "terminal",
      },
      { path: "/report", label: "报告与溯源", features: "F4 · F5 · F6 · F8", icon: "file-search" },
      { path: "/session", label: "会话与记忆", features: "F9 · F10 · F11", icon: "messages-square" },
    ],
  },
  {
    title: "知识库",
    items: [
      { path: "/papers", label: "论文库", features: "F13 · F14 · F15", icon: "library" },
      { path: "/analytics", label: "文献计量大屏", features: "F12 · F16 · F19", icon: "share-2" },
    ],
  },
  {
    title: "运维",
    items: [
      { path: "/system", label: "系统状态", features: "健康度 · 熔断 · 成本", icon: "activity" },
      {
        path: "/settings",
        label: "配置中心",
        features: "模型 / 接口 / 密钥",
        icon: "settings",
      },
    ],
  },
];

export const DEFAULT_PATH = navGroups[0].items[0].path;

// nav.ts 里的 icon 是字符串，这里做唯一一次名字 → 组件映射，新增菜单项不用碰 Sidebar 的 import。
import type { LucideIcon } from "lucide-react";
import {
  Activity,
  Bell,
  CircleAlert,
  CircleCheck,
  FileSearch,
  Inbox,
  LayoutGrid,
  Library,
  Loader,
  LoaderCircle,
  MessagesSquare,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  Settings,
  Share2,
  Terminal,
  User,
  Zap,
} from "lucide-react";

export const iconMap: Record<string, LucideIcon> = {
  terminal: Terminal,
  "file-search": FileSearch,
  "messages-square": MessagesSquare,
  library: Library,
  "share-2": Share2,
  activity: Activity,
  search: Search,
  bell: Bell,
  user: User,
  zap: Zap,
  settings: Settings,
  "panel-left-close": PanelLeftClose,
  "panel-left-open": PanelLeftOpen,
  "circle-alert": CircleAlert,
  "circle-check": CircleCheck,
  loader: Loader,
  "loader-circle": LoaderCircle,
  inbox: Inbox,
};

const FALLBACK_ICON: LucideIcon = LayoutGrid;

export function NavIcon({
  name,
  className,
  size,
}: {
  name: string;
  className?: string;
  size?: number;
}) {
  const Icon = iconMap[name] ?? FALLBACK_ICON;
  return <Icon className={className} size={size} aria-hidden={true} />;
}

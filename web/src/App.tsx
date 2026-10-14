import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { DEFAULT_PATH } from "@/config/nav";
import { Skeleton } from "@/components/ui/Skeleton";

/**
 * 页面全部懒加载。
 *
 * 只有 `/analytics`、`/system`、`/report` 用得到 ECharts（1MB 那块），
 * 静态 import 的话控制台首页也得把它下载完才能出图 —— 拆包就白拆了。
 * 按路由切之后，首屏只下载 react + 当前这一页。
 */
const ConsolePage = lazy(() => import("@/pages/ConsolePage").then((m) => ({ default: m.ConsolePage })));
const ReportPage = lazy(() => import("@/pages/ReportPage").then((m) => ({ default: m.ReportPage })));
const SessionPage = lazy(() => import("@/pages/SessionPage").then((m) => ({ default: m.SessionPage })));
const PapersPage = lazy(() => import("@/pages/PapersPage").then((m) => ({ default: m.PapersPage })));
const AnalyticsPage = lazy(() => import("@/pages/AnalyticsPage").then((m) => ({ default: m.AnalyticsPage })));
const SystemPage = lazy(() => import("@/pages/SystemPage").then((m) => ({ default: m.SystemPage })));

function RouteFallback() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton height={54} />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Skeleton height={128} />
        <Skeleton height={128} />
        <Skeleton height={128} />
        <Skeleton height={128} />
      </div>
      <Skeleton height={280} />
    </div>
  );
}

export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<Navigate to={DEFAULT_PATH} replace />} />
        <Route
          path="/console"
          element={
            <Suspense fallback={<RouteFallback />}>
              <ConsolePage />
            </Suspense>
          }
        />
        <Route
          path="/report"
          element={
            <Suspense fallback={<RouteFallback />}>
              <ReportPage />
            </Suspense>
          }
        />
        <Route
          path="/session"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SessionPage />
            </Suspense>
          }
        />
        <Route
          path="/papers"
          element={
            <Suspense fallback={<RouteFallback />}>
              <PapersPage />
            </Suspense>
          }
        />
        <Route
          path="/analytics"
          element={
            <Suspense fallback={<RouteFallback />}>
              <AnalyticsPage />
            </Suspense>
          }
        />
        <Route
          path="/system"
          element={
            <Suspense fallback={<RouteFallback />}>
              <SystemPage />
            </Suspense>
          }
        />
        <Route path="*" element={<Navigate to={DEFAULT_PATH} replace />} />
      </Route>
    </Routes>
  );
}

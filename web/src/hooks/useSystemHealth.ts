import { useEffect, useState } from "react";
import { api } from "@/api/client";

export type SystemStatus = "ok" | "error" | "idle";

/**
 * 顶栏呼吸指示灯的数据源。
 *
 * 轮询而不是只查一次：熔断器状态会变，"只在加载时查一次"的指示灯
 * 会在后端挂掉后继续亮绿，那比没有指示灯更糟。
 */
export function useSystemHealth(intervalMs = 15_000) {
  const [status, setStatus] = useState<SystemStatus>("idle");
  const [label, setLabel] = useState("检测中");

  useEffect(() => {
    let alive = true;

    async function check() {
      try {
        const health = await api.system.health();
        if (!alive) return;
        // 熔断器快照挂在 `resilience.breakers` 下，不是顶层 `breakers`
        const breakers =
          (health as { resilience?: { breakers?: Record<string, { state?: string }> } }).resilience
            ?.breakers ?? {};
        const tripped = Object.entries(breakers).find(([, b]) => b?.state === "open");
        if (health.status === "ok" && !tripped) {
          setStatus("ok");
          setLabel(`${health.llm_provider ?? "ready"} · 正常`);
        } else {
          setStatus("error");
          setLabel(tripped ? `${tripped[0]} 熔断` : "降级");
        }
      } catch {
        if (!alive) return;
        setStatus("error");
        setLabel("后端不可达");
      }
    }

    void check();
    const timer = setInterval(check, intervalMs);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return { status, label };
}

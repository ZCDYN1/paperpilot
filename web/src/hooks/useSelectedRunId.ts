import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/api/client";
import { MOCK_RUN_ID } from "@/mock/fixtures";

/**
 * 决定报告页 / 会话页展示哪一次运行。
 *
 * 优先级：URL 的 `?run=` > 最近一次运行 > 空。
 * 没有"最近一次"这一层的话，真实模式下这两个页面要求用户**先知道 run_id 才能看到东西**，
 * 而 run_id 是一串 16 位十六进制，只能去后端日志里抄 —— 那等于 F5/F8/F9 全都不可达。
 */
export function useSelectedRunId(): { runId: string; resolving: boolean; error: string | null } {
  const [params] = useSearchParams();
  const fromUrl = params.get("run");

  const [fallback, setFallback] = useState("");
  const [resolving, setResolving] = useState(!fromUrl);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (fromUrl) {
      setFallback(fromUrl);
      setResolving(false);
      return;
    }
    if (api.useMock) {
      setFallback(MOCK_RUN_ID);
      setResolving(false);
      return;
    }

    let alive = true;
    setResolving(true);
    api.research
      .listRuns(20)
      .then((res) => {
        if (!alive) return;
        const items = res.items;
        // 默认落点刻意跳过 `done`：那是早期脚本写进去的测试行，走完整应用的运行才会写
        // `completed`（见 `app/api/v1/research.py` 的 `_final_status`）。
        // 按"最近一条"取会正好落在这种半成品上 —— 它有引用、但九列对比表全空，
        // 用户第一眼看到的就是一整片「—」，像产品坏了。
        const pick =
          items.find((r) => r.status === "completed") ??
          items.find((r) => r.citation_count > 0) ??
          items[0];
        if (pick) setFallback(pick.run_id);
        else setError("还没有跑过任何一次研究，先去研究控制台提一个问题。");
      })
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setResolving(false));

    return () => {
      alive = false;
    };
  }, [fromUrl]);

  return { runId: fromUrl ?? fallback, resolving, error };
}

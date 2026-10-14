import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { api } from "@/api/client";
import type { HealthResponse, OverviewStats, TokenByNode } from "@/api/domain";
import { tokenByNodeTotal } from "@/api/domain";
import { mockTokenByNode } from "@/mock/fixtures";
import { useSelectedRunId } from "@/hooks/useSelectedRunId";
import { PageHeader } from "@/components/ui/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { StatCard } from "@/components/ui/StatCard";
import { Tag } from "@/components/ui/Tag";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { NeonButton } from "@/components/ui/NeonButton";
import { Stagger, StaggerItem } from "@/components/ui/Stagger";
import { ChartCard } from "@/components/charts/ChartCard";
import { Chart } from "@/components/charts/Chart";
import { gaugeConfig } from "@/config/charts/gauge";
import { barChartConfig } from "@/config/charts/bar";
import { palette } from "@/config/theme";

// 运维大屏读的是 health 的布尔开关 + 熔断计数：熔断打开代表系统「降级」而非崩，
// 这一块把话说清楚，避免运维对着红灯误判成宕机。

// 运维大屏的字段一律走 domain.ts 里照抄真实 /healthz 的 `HealthResponse`，
// 不再在页面里另立一份本地形状 —— 上一版就是因为各写一份，
// 把 breakers 当成顶层字段、阈值写成 `threshold`、还读了不存在的 `index_ready`。

const STATE_LABEL: Record<string, { text: string; variant: "green" | "amber" | "red" }> = {
  closed: { text: "闭合 · 正常放行", variant: "green" },
  half_open: { text: "半开 · 试探恢复", variant: "amber" },
  open: { text: "打开 · 已降级", variant: "red" },
};

/**
 * 取某个节点的 token 用量。
 *
 * ⚠️ `domain.ts` 把 `token_by_node` 声明成 `Record<string, number>`，但真实
 * `GET /research/{run_id}` 返回的每一项是 `{prompt_tokens, completion_tokens, total_tokens, calls}`
 * 这样的 dict（实测 run e9db744226d94d06 就是这个形状）。所以两种形状都要认：
 * 只认数字会把真实数据全判成"没有上报"，只认 dict 又会在 mock 那侧失效。
 * 这里取后端给的 `total_tokens`，不自己拿 prompt+completion 再加一遍。
 */
function nodeTokenUsage(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (value && typeof value === "object") {
    const total = (value as Record<string, unknown>).total_tokens;
    if (typeof total === "number" && Number.isFinite(total)) return total;
  }
  return null;
}

/**
 * 离线检索消融结果，非实时接口。数值逐条照抄 `eval_results/ablation_24papers_bge_rerank.md`
 * （24 篇语料，paper 级口径：chunk 检索 → paper 聚合 → 再算指标）。
 *
 * ★ 为什么必须点名是哪一份报告：仓库里有两份消融结果，重排器不同 ——
 * `ablation_24papers.md` 那次用的是 `lexical` 词面降级重排，它的 nDCG@10 相对
 * 不重排**是下降的**（0.8777 → 0.7851）；只有这份 BGE 运行才是真交叉编码器的增益。
 * 拿错那一份就等于用降级实验冒充真实提升。
 *
 * 延迟列同样重要：重排把单条查询从 ~131ms 抬到 ~5.7s，这是这笔收益的代价，
 * 只报指标不报延迟的图表没有参考价值。
 */
const ABLATION = {
  categories: ["① 纯向量", "④ +BM25 RRF", "⑤ +BGE 重排", "⑥ +metadata 过滤"],
  recall5: [85.42, 92.5, 93.33, 93.33],
  ndcg10: [84.18, 87.77, 92.22, 93.14],
  latencyMs: [136.95, 130.97, 5734.45, 3113.51],
};

function Switch({ on, labelOn, labelOff }: { on: boolean; labelOn: string; labelOff: string }) {
  return (
    <div className="flex items-center gap-2.5">
      <span className={`status-dot ${on ? "" : "is-idle"}`} aria-hidden />
      <span className="text-sm text-ink">{on ? labelOn : labelOff}</span>
    </div>
  );
}

export function SystemPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState("");
  const [stats, setStats] = useState<OverviewStats | null>(null);
  const [statsLoading, setStatsLoading] = useState(true);
  const [statsError, setStatsError] = useState("");
  const [tokenByNode, setTokenByNode] = useState<TokenByNode | null>(null);
  const [tokenState, setTokenState] = useState<{ phase: "loading" | "empty" | "error" | undefined; hint?: string }>({
    phase: "loading",
  });
  const { runId, resolving } = useSelectedRunId();

  const loadHealth = useCallback(() => {
    setHealthLoading(true);
    setHealthError("");
    api.system
      .health()
      .then((h) => setHealth(h))
      .catch((e: unknown) => {
        setHealth(null);
        setHealthError(e instanceof Error ? e.message : "健康检查失败 —— 后端不可达");
      })
      .finally(() => setHealthLoading(false));
  }, []);

  useEffect(() => {
    loadHealth();
    setStatsLoading(true);
    setStatsError("");
    let alive = true;
    api.system
      .overview()
      .then((s) => alive && setStats(s))
      .catch((e: unknown) => {
        if (!alive) return;
        // 顶部四张卡都来自这份 overview。它挂了却照常显示 0，和"真的是 0 次研究"
        // 在屏幕上长得一样 —— 所以记下原因，由顶部那条横幅说明。
        setStats(null);
        setStatsError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => alive && setStatsLoading(false));
    return () => {
      alive = false;
    };
  }, [loadHealth]);

  const view = health;

  /**
   * 按节点 token 用量：真实模式只认 `GET /api/v1/research/{run_id}` 的 `token_by_node`。
   *
   * ★ 之前这张图无条件画 `mockTokenByNode`（read 1104.5K 那一组）：
   * 后端已经起来了、页面其余部分都是真数据，唯独这里显示的是 fixture 里烧出来的假钱，
   * 而且它比接口数据"好看"，很容易被当成实测结论。取不到就画空态，不留假数。
   */
  useEffect(() => {
    if (api.useMock) {
      setTokenByNode(mockTokenByNode);
      setTokenState({ phase: undefined });
      return;
    }
    if (!runId) {
      setTokenByNode(null);
      setTokenState({
        phase: resolving ? "loading" : "empty",
        hint: resolving
          ? undefined
          : "还没有跑过研究，也就没有 GET /api/v1/research/{run_id} 可取 —— 这里不画占位数字。",
      });
      return;
    }
    let alive = true;
    setTokenState({ phase: "loading" });
    api.research
      .getRun(runId)
      .then((r) => {
        if (!alive) return;
        const raw = (r.token_by_node ?? {}) as Record<string, unknown>;
        const byNode: Record<string, number> = {};
        Object.entries(raw).forEach(([node, v]) => {
          const n = nodeTokenUsage(v);
          if (n !== null && n > 0) byNode[node] = n;
        });
        const has = Object.keys(byNode).length > 0;
        setTokenByNode(has ? byNode : null);
        setTokenState(
          has
            ? { phase: undefined }
            : {
                phase: "empty",
                hint: `这次运行（run ${runId}）没有上报可用的 token_by_node —— 早期脚本跑出来的行不带这个字段，这里不补占位数字。`,
              },
        );
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setTokenByNode(null);
        setTokenState({
          phase: "error",
          hint: e instanceof Error ? e.message : String(e),
        });
      });
    return () => {
      alive = false;
    };
  }, [runId, resolving]);

  const tokenOption = useMemo(() => {
    // total 用 tokenByNodeTotal 取：后端每项是 {prompt_tokens, completion_tokens, total_tokens, calls}
    const entries = Object.entries(tokenByNode ?? {})
      .map(([node, raw]) => ({ node, total: tokenByNodeTotal(raw) }))
      .filter((e): e is { node: string; total: number } => e.total !== null)
      .sort((a, b) => a.total - b.total);
    return barChartConfig({
      categories: entries.map((e) => e.node),
      // 保留一位小数：真实运行的单节点用量可能只有几百 token，
      // 取整到 K 会全变成 0K，图看着像"没花钱"。
      series: [{ name: "Token (K)", values: entries.map((e) => Math.round((e.total / 1000) * 10) / 10) }],
      opts: { horizontal: true, valueFormatter: (v) => `${v}K` },
    });
  }, [tokenByNode]);

  const tokenEntries = Object.keys(tokenByNode ?? {}).length;

  const ablationOption = useMemo(
    () =>
      barChartConfig({
        categories: ABLATION.categories,
        series: [
          { name: "Recall@5", values: ABLATION.recall5 },
          { name: "nDCG@10", values: ABLATION.ndcg10 },
        ],
        opts: { valueFormatter: (v) => `${v}%`, valueLabel: true },
      }),
    [],
  );

  const breakers = view?.resilience?.breakers ?? {};
  const breakerKeys = Object.keys(breakers);

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="系统状态"
        description="运行时健康度 · 熔断器 · Token 成本 · 离线评测 —— 一眼看出现在到底能不能干活"
        features="健康度 · 熔断 · 成本"
        actions={
          <NeonButton variant="secondary" loading={healthLoading} onClick={loadHealth}>
            <RefreshCw size={15} aria-hidden /> 刷新
          </NeonButton>
        }
      />

      {healthError ? (
        <div className="rounded-lg border border-neon-red/40 bg-white/[0.04] px-4 py-3 text-sm text-neon-red">
          {healthError}
        </div>
      ) : null}

      {statsError ? (
        // 顶部四张卡全来自 GET /stats/overview。它挂了还照常显示 0，
        // 用户分不清"真的是 0 次研究"和"没取到数"，所以这句话写在卡片区上方。
        <div className="rounded-lg border border-neon-amber/40 bg-white/[0.04] px-4 py-3 text-sm text-neon-amber">
          <code className="font-mono">GET /api/v1/stats/overview</code> 加载失败：{statsError} ——
          下方「研究次数 / 平均引用支撑率 / Token 总消耗」此刻没有数据来源，显示的是 0 而非真实值。
        </div>
      ) : null}

      {/* 核心运行指标 */}
      {statsLoading && !stats ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {[0, 1, 2, 3].map((i) => (
            <Skeleton key={i} height={132} />
          ))}
        </div>
      ) : (
        <Stagger className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <StaggerItem>
            <StatCard
              label={
                // 这两个数不是同一口径：向量索引里现在有几篇 ≠ 论文表里有几行
                // （实测 24 vs 32，索引还没把新入库的论文补进去）。
                // 不写清口径，用户会拿它跟文献计量页的「语料篇数（在库）」对不上号。
                view?.index?.papers != null ? "语料篇数 · 向量索引内" : "语料篇数 · /stats/overview"
              }
              value={view?.index?.papers ?? stats?.papers ?? 0}
              unit="篇"
              icon="library"
              accent="cyan"
            />
          </StaggerItem>
          <StaggerItem>
            <StatCard label="研究次数" value={stats?.runs ?? 0} unit="次" icon="activity" accent="purple" />
          </StaggerItem>
          <StaggerItem>
            <StatCard
              label={
                stats && stats.avg_verified_rate === null
                  ? "平均引用支撑率 · 后端给的是 null"
                  : "平均引用支撑率"
              }
              value={(stats?.avg_verified_rate ?? 0) * 100}
              unit="%"
              decimals={1}
              icon="shield-check"
              accent="green"
            />
          </StaggerItem>
          <StaggerItem>
            <StatCard
              label="Token 总消耗"
              value={(stats?.total_tokens ?? 0) / 1000}
              unit="K"
              decimals={0}
              icon="zap"
              accent="amber"
            />
          </StaggerItem>
        </Stagger>
      )}

      {stats && stats.avg_verified_rate === null ? (
        // 口径来自后端：null = 这次统计里没有引用可校验，**不等于 0%**。
        // 只把数字写成 0 会让人以为系统一文不值，所以补一句它到底意味着什么。
        <p className="-mt-2 text-xs leading-relaxed text-ink-faint">
          「平均引用支撑率」上方显示 0.0% 是因为 <code className="font-mono">/stats/overview</code>{" "}
          返回的 <code className="font-mono">avg_verified_rate</code> 为 null：库里还没有带引用的运行，
          这不是「0% 的引用被验证」。
        </p>
      ) : null}

      {/* 运行时开关 */}
      <Panel title="运行时" subtitle={view ? `版本 ${view.version ?? "—"} · 状态 ${view.status ?? "—"}` : "探测中"}>
        {healthLoading ? (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[0, 1, 2, 3].map((i) => (
              <Skeleton key={i} height={72} />
            ))}
          </div>
        ) : !view ? (
          <EmptyState
            icon="circle-alert"
            title="无法获取运行时状态"
            description="后端不可达。下面的熔断与按节点成本图拿不到数据时一律显示空态，不会保留占位数字。"
          />
        ) : (
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <div className="flex flex-col gap-2 rounded-lg border border-hairline bg-white/[0.02] p-4">
              <span className="text-xs text-ink-faint">LLM Provider / 模型</span>
              <div className="flex flex-wrap items-center gap-1.5">
                <Tag variant="cyan" mono>
                  {view.llm_provider ?? "—"}
                </Tag>
                <Tag variant="muted" mono>
                  {view.llm_model ?? "—"}
                </Tag>
              </div>
            </div>
            <div className="flex flex-col gap-2 rounded-lg border border-hairline bg-white/[0.02] p-4">
              <span className="text-xs text-ink-faint">Embedding</span>
              <div className="flex flex-wrap items-center gap-1.5">
                <Tag variant="purple" mono>
                  {view.embedding_provider ?? "—"}
                </Tag>
                <Switch on={Boolean(view.index?.ready)} labelOn="索引已就绪" labelOff="索引未就绪" />
              </div>
            </div>
            <div className="flex flex-col gap-2 rounded-lg border border-hairline bg-white/[0.02] p-4">
              <span className="text-xs text-ink-faint">Rerank 重排</span>
              <Switch on={Boolean(view.rerank_enabled)} labelOn="已开启" labelOff="已关闭" />
            </div>
            <div className="flex flex-col gap-2 rounded-lg border border-hairline bg-white/[0.02] p-4">
              <span className="text-xs text-ink-faint">Input Guard 输入防护</span>
              <Switch on={Boolean(view.input_guard_enabled)} labelOn="已开启" labelOff="已关闭" />
            </div>
          </div>
        )}
      </Panel>

      {/* 熔断器 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        {breakerKeys.length === 0 ? (
          <Panel title="熔断器" subtitle="未上报熔断状态">
            <EmptyState title="没有熔断器数据" description="resilience 未开启或后端未返回 breakers。" />
          </Panel>
        ) : (
          breakerKeys.map((key) => {
            const b = breakers[key] ?? {};
            const state = b.state;
            const meta = state
              ? STATE_LABEL[state] ?? { text: state, variant: "amber" as const }
              : { text: "后端未上报状态", variant: "muted" as const };
            const failures = typeof b.failures === "number" ? b.failures : 0;
            /**
             * 阈值只认后端的 `failure_threshold`。之前这里缺字段就兜成 5，
             * 仪表盘于是画出一个"0/5"——一个谁都没给的阈值配上真实失败数，
             * 看上去完全合理，实际是拼出来的。缺字段就画空态。
             */
            const threshold =
              typeof b.failure_threshold === "number" && b.failure_threshold > 0
                ? b.failure_threshold
                : null;
            const shortCircuited =
              typeof b.short_circuited === "number" ? b.short_circuited : null;
            return (
              <ChartCard
                key={key}
                title={`熔断器 · ${key.toUpperCase()}`}
                subtitle="连续失败达阈值即打开；打开 = 走降级路径，不是系统崩溃"
                height={240}
                state={threshold === null ? "empty" : undefined}
                emptyHint={`后端没有在 ${key} 这一项里给出 failure_threshold，占比画不出来 —— 不补一个默认阈值冒充配置值。`}
                extra={
                  <div className="flex items-center gap-2">
                    {shortCircuited != null && shortCircuited > 0 ? (
                      <Tag variant="muted" mono>
                        直接拦掉 {shortCircuited} 次
                      </Tag>
                    ) : null}
                    <Tag variant={meta.variant}>{meta.text}</Tag>
                  </div>
                }
              >
                <div className="flex h-full items-center gap-2">
                  <div className="h-full flex-1">
                    <Chart
                      height="100%"
                      option={gaugeConfig({
                        value: failures,
                        max: threshold ?? 1,
                        name: `失败 ${failures} / 阈值 ${threshold}`,
                        thresholds: [
                          { upto: 0.4, color: palette.green },
                          { upto: 0.8, color: palette.amber },
                          { upto: 1, color: palette.red },
                        ],
                      })}
                    />
                  </div>
                  <div className="flex w-40 flex-col gap-2 pr-2 text-xs">
                    <p className="text-ink-soft">
                      当前失败计数
                      <span className="ml-1 font-mono text-ink">
                        {failures}/{threshold}
                      </span>
                    </p>
                    <p className="leading-relaxed text-ink-faint">
                      {state === "open"
                        ? "处于打开态：请求被快速失败并回退到降级链路，恢复窗口到达后自动转半开试探。"
                        : state === "half_open"
                          ? "半开：放行少量探测请求，成功则闭合、失败则重新打开。"
                          : state === "closed"
                            ? "闭合：链路健康，请求正常通过。"
                            : "后端没给这一项的 state，这里不猜它现在是通还是断。"}
                    </p>
                  </div>
                </div>
              </ChartCard>
            );
          })
        )}
      </div>

      {/* 成本 + 评测 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Token 成本 · 按节点"
          subtitle={
            api.useMock
              ? "mock 模式：这组数字来自内置 fixture，只用于演示版式"
              : `来源：GET /api/v1/research/${runId || "—"} 的 token_by_node.total_tokens —— 只覆盖这一次运行，不是全库累计`
          }
          height={320}
          state={tokenState.phase ?? (tokenEntries === 0 ? "empty" : undefined)}
          emptyHint={tokenState.hint ?? "这次运行没有上报按节点的 token 用量，这里不画占位数字。"}
          errorHint={tokenState.hint ? `token_by_node 取数失败：${tokenState.hint}` : "token_by_node 取数失败"}
          extra={
            !api.useMock && tokenEntries > 0 ? (
              <Tag variant="muted" mono>
                {tokenEntries} 个节点
              </Tag>
            ) : null
          }
        >
          <Chart option={tokenOption} height="100%" />
        </ChartCard>

        <ChartCard
          title="检索消融 · Recall@5 与 nDCG@10"
          subtitle="离线评测结果，非接口、非实时：数值逐条照抄 eval_results/ablation_24papers_bge_rerank.md（24 篇语料 · paper 级口径 · BGE 交叉编码重排）"
          height={320}
          extra={
            <div className="flex items-center gap-2">
              <Tag variant="muted">离线文件 · 非接口数据</Tag>
              <Tag variant="amber" mono>
                重排后单条查询 131ms → 5734ms
              </Tag>
            </div>
          }
        >
          <Chart option={ablationOption} height="100%" />
        </ChartCard>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { api } from "@/api/client";
import type { SSEEvent } from "@/api/events";
import type { OverviewStats, ResearchRequest, TokenByNode } from "@/api/domain";
import { tokenByNodeTotal } from "@/api/domain";
import { PageHeader } from "@/components/ui/PageHeader";
import { StatCard } from "@/components/ui/StatCard";
import { Stagger } from "@/components/ui/Stagger";
import { Panel } from "@/components/ui/Panel";
import { MarkdownBlock } from "@/components/ui/MarkdownBlock";
import { NeonButton } from "@/components/ui/NeonButton";
import { Tag } from "@/components/ui/Tag";
import { Skeleton } from "@/components/ui/Skeleton";
import { EventTimeline } from "@/components/business/EventTimeline";
import { SubQuestionEditor } from "@/components/business/SubQuestionEditor";
import { Chart } from "@/components/charts/Chart";
import { barChartConfig } from "@/config/charts/bar";
import {
  CATEGORY_ALL,
  CATEGORY_CHOICES,
  DEFAULT_TOP_K,
  HAS_CODE_VALUE,
  TOP_K_MAX,
  TOP_K_MIN,
  YEAR_MAX,
  YEAR_MIN,
  useResearchWorkspace,
  type FilterForm,
  type HasCodeChoice,
} from "@/state/researchWorkspace";
import { env } from "@/config/env";

/* ============================================================ 表单 → 请求体
   后端 `ResearchRequest` / `ResearchFilters` 都是 `extra="forbid"`
   （`app/schemas/research.py`），多一个键、或者键名写错（cat/since/hasCode 这类
   UI 命名）就是整请求 422。所以界面上的 state 与送出去的请求体**必须是两套形状**，
   中间由 `buildRequest` 单向转换，转换之后的值还要过 `parse*` 的范围校验 ——
   越界（年份、top_k）同样会被 pydantic 的 ge/le 拒掉。
   （表单 state 与 SSE 流本身住在 `@/state/researchWorkspace`，不在本页：
   切页面不该掐掉一次正在跑的研究，理由写在那个字段的注释里。） */

type Parsed<T> = { value: T; problem: string | null };

/** 空 = 不过滤（`null`）；填了就必须是范围内的整数，否则送出去一定 422 */
function parseYear(raw: string, label: string): Parsed<number | null> {
  const text = raw.trim();
  if (!text) return { value: null, problem: null };
  const n = Number(text);
  if (!Number.isInteger(n)) return { value: null, problem: `${label} 必须是整数年份` };
  if (n < YEAR_MIN || n > YEAR_MAX) {
    return { value: null, problem: `${label} 需要在 ${YEAR_MIN}~${YEAR_MAX} 之间` };
  }
  return { value: n, problem: null };
}

function parseTopK(raw: string): Parsed<number> {
  const text = raw.trim();
  if (!text) return { value: DEFAULT_TOP_K, problem: null }; // 留空 = 用后端默认值
  const n = Number(text);
  if (!Number.isInteger(n)) return { value: DEFAULT_TOP_K, problem: "保留论文数必须是整数" };
  if (n < TOP_K_MIN || n > TOP_K_MAX) {
    return { value: DEFAULT_TOP_K, problem: `保留论文数需要在 ${TOP_K_MIN}~${TOP_K_MAX} 之间` };
  }
  return { value: n, problem: null };
}

/**
 * 界面 state → 后端请求体。返回的 `request` 就是最终发出去的 JSON（除了 question），
 * `problems` 非空时按钮必须禁用 —— 这些都是后端会 422 的形状。
 */
function buildRequest(form: FilterForm): { request: Omit<ResearchRequest, "question" | "session_id">; problems: string[] } {
  const yearFrom = parseYear(form.yearFrom, "起始年份");
  const yearTo = parseYear(form.yearTo, "截止年份");
  const topK = parseTopK(form.topK);

  const problems = [yearFrom.problem, yearTo.problem, topK.problem].filter((p): p is string => p !== null);
  // 后端有 `_check_year_range`：gte > lte 也是一条 422，界面上先自己拦掉
  if (yearFrom.value !== null && yearTo.value !== null && yearFrom.value > yearTo.value) {
    problems.push("起始年份不能大于截止年份");
  }

  return {
    request: {
      language: form.language,
      top_k: topK.value,
      rerank: form.rerank,
      filters: {
        // "全部" 映射成空数组；绝对不能传 "" ，白名单校验会直接拒
        categories: form.category === CATEGORY_ALL ? [] : [form.category],
        year_gte: yearFrom.value,
        year_lte: yearTo.value,
        has_code: HAS_CODE_VALUE[form.hasCode],
      },
    },
    problems,
  };
}

/** 控件统一的边框 / 聚焦样式：颜色只走 Tailwind token 类名，不写死十六进制 */
const FIELD_CLS =
  "rounded-md border border-hairline bg-panel px-2 py-1 font-mono text-xs text-ink outline-none " +
  "transition-[border-color,box-shadow] duration-200 focus:border-neon-cyan/60 focus:shadow-[0_0_0_3px_var(--accent-cyan-dim)]";

/** class 只做拼接，不引依赖 */
function cls(...parts: Array<string | false | null | undefined>) {
  return parts.filter(Boolean).join(" ");
}

const FIELD_OPTIONS = {
  category: [
    { value: CATEGORY_ALL, label: "全部" },
    ...CATEGORY_CHOICES.map((c) => ({ value: c, label: c })),
  ],
  hasCode: [
    { value: "all", label: "全部" },
    { value: "only_with", label: "仅有代码" },
    { value: "only_without", label: "仅无代码" },
  ],
  language: [
    { value: "zh", label: "中文" },
    { value: "en", label: "English" },
  ],
} as const;

/** 一个带标题与说明的控件槽位 */
function Field({
  label,
  hint,
  wide,
  children,
}: {
  label: string;
  hint?: string;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <label className={cls("flex flex-col gap-1", wide ? "w-32" : "w-28")}>
      <span className="text-xs text-ink-soft">{label}</span>
      {children}
      {hint && <span className="text-xs text-ink-faint">{hint}</span>}
    </label>
  );
}

export function ConsolePage() {
  const [stats, setStats] = useState<OverviewStats | null>(null);
  // 流与表单都来自路由之上的工作区：本页可以被随时卸载，研究不能被打断
  const { run, question, setQuestion, form, setForm } = useResearchWorkspace();

  useEffect(() => {
    let alive = true;
    api.system
      .overview()
      .then((s) => alive && setStats(s))
      .catch(() => alive && setStats(null));
    return () => {
      alive = false;
    };
  }, []);

  const running = run.status === "running";
  const tokenByNode = lastTokenByNode(run.events);

  /** 请求体 + 挡住 422 的校验信息，一次算好；表单没改就别重建 */
  const { request, problems } = useMemo(() => buildRequest(form), [form]);

  const canStart = question.trim().length > 0 && problems.length === 0;

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="研究控制台"
        description="拆解问题 → 多路并行检索 → 精读抽取 → 引用校验 → 带溯源的报告"
        features="F1 · F2 · F7"
        actions={
          env.useMock ? (
            <Tag variant="amber">MOCK 数据 · 未连接后端</Tag>
          ) : (
            <Tag variant="green">已连接后端</Tag>
          )
        }
      />

      <Stagger className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="语料规模"
          value={stats?.papers ?? 0}
          unit="篇"
          icon="library"
          series={stats?.by_day.map((d) => d.runs)}
          accent="cyan"
        />
        <StatCard
          label="研究次数"
          value={stats?.runs ?? 0}
          unit="次"
          delta={stats ? Math.round(((stats.runs_completed / stats.runs) * 100 - 100) * 10) / 10 : undefined}
          icon="activity"
          series={stats?.by_day.map((d) => d.runs)}
          accent="purple"
        />
        {/* ★ 支撑率这张卡原来的 sparkline 是一串写死的 [88.2, 89.1, ...]，
           真实模式下它会冒充"支撑率的历史趋势"。`GET /api/v1/stats/overview` 的 `by_day`
           只有 runs / tokens，**没有**按天支撑率 —— 没有真数据可画就不画
           （StatCard 拿到空 series 只留一条占位虚线），而不是再编一组好看的数字。 */}
        <StatCard
          label="平均引用支撑率"
          value={(stats?.avg_verified_rate ?? 0) * 100}
          unit="%"
          decimals={1}
          icon="shield-check"
          series={undefined}
          accent="green"
        />
        <StatCard
          label="Token 总消耗"
          value={(stats?.total_tokens ?? 0) / 1000}
          unit="K"
          decimals={0}
          icon="zap"
          series={stats?.by_day.map((d) => d.tokens / 1000)}
          accent="amber"
        />
      </Stagger>

      <Panel title="发起一次研究" subtitle="子问题拆解后会在下方等你确认，改完再往下跑">
        <div className="flex flex-col gap-4">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            placeholder="例如：检索增强生成在减少大模型幻觉方面有哪些主流方法，各自的代价是什么？"
            className="w-full resize-y rounded-lg border border-hairline bg-base/60 px-4 py-3 text-sm leading-6 text-ink outline-none transition-[border-color,box-shadow] duration-200 placeholder:text-ink-faint focus:border-neon-cyan/60 focus:shadow-[0_0_0_3px_var(--accent-cyan-dim)]"
          />

          {/* 每一项都直接对应后端的一个字段；名字与形状的翻译只发生在 buildRequest 里 */}
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
            <Field label="分类 categories">
              <select
                value={form.category}
                onChange={(e) => setForm((f) => ({ ...f, category: e.target.value }))}
                className={FIELD_CLS}
              >
                {FIELD_OPTIONS.category.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="起始年份 year_gte">
              <input
                type="number"
                value={form.yearFrom}
                onChange={(e) => setForm((f) => ({ ...f, yearFrom: e.target.value }))}
                placeholder="不限"
                className={cls(FIELD_CLS, "w-24")}
              />
            </Field>

            <Field label="截止年份 year_lte">
              <input
                type="number"
                value={form.yearTo}
                onChange={(e) => setForm((f) => ({ ...f, yearTo: e.target.value }))}
                placeholder="不限"
                className={cls(FIELD_CLS, "w-24")}
              />
            </Field>

            <Field label="代码情况 has_code" hint="不筛=不过滤(null)，不是 false">
              <select
                value={form.hasCode}
                onChange={(e) => setForm((f) => ({ ...f, hasCode: e.target.value as HasCodeChoice }))}
                className={FIELD_CLS}
              >
                {FIELD_OPTIONS.hasCode.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="报告语言 language">
              <select
                value={form.language}
                onChange={(e) => setForm((f) => ({ ...f, language: e.target.value as "zh" | "en" }))}
                className={FIELD_CLS}
              >
                {FIELD_OPTIONS.language.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="保留论文 top_k">
              <input
                type="number"
                value={form.topK}
                min={TOP_K_MIN}
                max={TOP_K_MAX}
                onChange={(e) => setForm((f) => ({ ...f, topK: e.target.value }))}
                className={cls(FIELD_CLS, "w-20")}
              />
            </Field>

            <Field label="交叉编码重排 rerank">
              <input
                type="checkbox"
                checked={form.rerank}
                onChange={(e) => setForm((f) => ({ ...f, rerank: e.target.checked }))}
                className="h-3.5 w-3.5 accent-[var(--accent-cyan)]"
              />
            </Field>

            <div className="ml-auto flex items-center gap-2">
              {running && (
                <NeonButton variant="secondary" onClick={run.stop}>
                  中断
                </NeonButton>
              )}
              <NeonButton
                loading={running}
                disabled={!canStart}
                onClick={() => run.start(question, request)}
              >
                开始研究
              </NeonButton>
            </div>
          </div>

          {/* 形状不对就是 422，与其让后端整批静默拒掉，不如在点之前说清是哪一项 */}
          {problems.length > 0 && (
            <div className="rounded-lg border border-neon-amber/40 bg-neon-amber/10 px-4 py-3 text-xs leading-5 text-neon-amber">
              {problems.map((p) => (
                <p key={p}>{p}</p>
              ))}
            </div>
          )}

          {run.error && (
            <div className="rounded-lg border border-neon-red/40 bg-neon-red/10 px-4 py-3 text-sm text-neon-red">
              {run.error}
            </div>
          )}
        </div>
      </Panel>

      {run.pendingSubQuestions.length > 0 && (
        <Panel title="确认子问题" subtitle="这里的修改会真的影响后续检索与精读范围">
          <SubQuestionEditor
            items={run.pendingSubQuestions}
            onCancel={run.stop}
            onResume={(items) => run.resume(items)}
          />
        </Panel>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <Panel
          title="Agent 执行过程"
          subtitle={
            running
              ? "流式进行中"
              : run.status === "awaiting_input"
                ? "已暂停，等你确认子问题"
                : run.status === "done"
                  ? "已完成"
                  : "等待发起"
          }
          extra={<span className="font-mono text-xs text-ink-faint">{run.events.length} events</span>}
          bodyClassName="max-h-[560px] overflow-y-auto"
        >
          {run.events.length === 0 ? (
            <div className="flex flex-col gap-3 p-1">
              <Skeleton height={44} />
              <Skeleton height={44} width="78%" />
              <Skeleton height={44} width="62%" />
              <p className="pt-2 text-center text-xs text-ink-faint">
                发起一次研究后，16 种 SSE 事件会在这里逐条落地
              </p>
            </div>
          ) : (
            <EventTimeline events={run.events} />
          )}
        </Panel>

        <Panel
          title="Token 成本分布"
          subtitle={tokenByNode ? "本次运行按节点细分" : "近 7 天按天汇总"}
        >
          <Chart
            option={
              tokenByNode
                ? barChartConfig({ ...fromNodeMap(tokenByNode), opts: { horizontal: true } })
                : stats
                  ? barChartConfig(fromTokenStats(stats))
                  : barChartConfig({ categories: [], series: [] })
            }
            height={320}
          />
        </Panel>
      </div>

      <StreamPreview
        streamed={run.streamed}
        finalReport={run.finalReport}
        previewVoid={run.previewVoid}
        voidReason={run.error}
        running={running}
      />
    </div>
  );
}

/* ============================================================ 流式预览区
   权威边界（架构 §token 事件实现 / useResearchStream）：
   - `streamed` 只有模型生成的正文；⚠️ 未支撑一节 / 参考文献 / 页脚是代码追加的、不进流，
     所以它**永远不等于** `final_report.markdown` —— 预览必须标成预览，别让人当最终报告。
   - `node_started(write)` 晚于第一个 token（updates 通道固有顺序），
     所以这里不看任何事件，只看 `streamed` 本身有没有内容。
   - `previewVoid` 为真时 reducer 已经清空 `streamed` **并把 token 事件一起摘掉**
     （时间线那边也是这段预览的展示面，只清这一边等于没清），界面上给作废状态。
   ★ 性能：流式期 token ~100ms 批量落地一次，此时**绝不跑 markdown 解析**，
     纯文本 `<pre>` 渲染；只有 `finalReport`（权威、一次性到达）才用 `MarkdownBlock`。 */

type StreamPreviewProps = {
  streamed: string;
  finalReport: string | null;
  previewVoid: boolean;
  /** reducer 记下的错误原文（`code: message`），作废时给用户一个可查的原因 */
  voidReason: string | null;
  running: boolean;
};

function StreamPreview({ streamed, finalReport, previewVoid, voidReason, running }: StreamPreviewProps) {
  const hasFinal = finalReport !== null;
  const boxRef = useRef<HTMLDivElement>(null);

  // 预览期跟随到最新 token 落点（真实增量驱动，不是定时器假动画）
  useEffect(() => {
    if (hasFinal || previewVoid) return;
    const el = boxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [streamed, hasFinal, previewVoid]);

  // 没有任何值得看的东西（没跑过、没预览、没作废、没报告）就不挂面板
  if (!hasFinal && !previewVoid && streamed.length === 0 && !running) return null;

  if (hasFinal) {
    return (
      <Panel
        title="最终研究报告"
        subtitle="final_report 事件的权威正文（含 ⚠️ 未支撑一节 / 参考文献 / 页脚，这些不在流里）"
        extra={<span className="font-mono text-xs text-ink-faint">{finalReport.length.toLocaleString()} 字符</span>}
        bodyClassName="max-h-[640px] overflow-y-auto"
      >
        <MarkdownBlock source={finalReport} />
      </Panel>
    );
  }

  return (
    <Panel
      title="流式预览"
      subtitle="仅模型生成正文 · 不含 ⚠️ 未支撑一节 / 参考文献 / 页脚，非最终报告；权威内容以 final_report 为准"
      extra={
        <span className="flex items-center gap-2 font-mono text-xs text-ink-faint">
          {running && <span className="text-neon-cyan">● 流式中</span>}
          <span>{streamed.length.toLocaleString()} 字符</span>
        </span>
      }
      bodyClassName="max-h-[480px] overflow-y-auto"
    >
      {previewVoid ? (
        <div className="rounded-lg border border-neon-amber/40 bg-neon-amber/10 px-4 py-3 text-sm leading-6 text-neon-amber">
          <p>
            这段流式预览已作废，界面上不留半截文本 —— 权威内容只认 <code className="font-mono">final_report</code>。
          </p>
          <p className="mt-1 text-xs leading-5 text-ink-soft">
            作废有两种来源：后端声明 <code className="font-mono">token_stream_aborted</code>
            （吐过字又失败，重试轮按「宁少流、不重流」不再重流），或这次运行终止于 error 而报告始终没到。
            {voidReason ? <span className="mt-1 block break-all">后端原文：{voidReason}</span> : null}
          </p>
        </div>
      ) : (
        <div ref={boxRef} className="max-h-full overflow-y-auto">
          {streamed.length === 0 ? (
            <p className="text-xs leading-6 text-ink-faint">
              等待 writer 的第一批 token…（流式不依赖 node_started(write) 到达，没有它预览照样会动）
            </p>
          ) : (
            <pre className="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-ink-soft">
              {streamed}
              {running && <span className="ml-0.5 inline-block animate-pulse text-neon-cyan">▍</span>}
            </pre>
          )}
        </div>
      )}
    </Panel>
  );
}

/**
 * SSE `token_usage` 事件的 `by_node`。
 * ⚠️ 每项是 `{prompt_tokens, completion_tokens, total_tokens, calls}` 字典
 * （`app/agents/nodes/finalize.py` 的 `meter.snapshot()`），不是数字 ——
 * 当成数字做算术会整条图变 NaN，而类型检查罩不住（事件 data 是 unknown）。
 */
function lastTokenByNode(events: SSEEvent[]): TokenByNode | null {
  for (let i = events.length - 1; i >= 0; i--) {
    if (events[i].name !== "token_usage") continue;
    const byNode = (events[i].data as { by_node?: TokenByNode }).by_node;
    if (byNode && Object.keys(byNode).length) return byNode;
  }
  return null;
}

function fromNodeMap(map: TokenByNode) {
  // 横向条形图从下往上画，所以按升序排，最大值才落在顶部
  const entries = Object.entries(map)
    .map(([node, raw]) => ({ node, total: tokenByNodeTotal(raw) }))
    .filter((e): e is { node: string; total: number } => e.total !== null)
    .sort((a, b) => a.total - b.total);
  return {
    categories: entries.map((e) => e.node),
    // 保留一位小数：真实单节点可能只有几百 token，取整到 K 会全成 0K，图看着像没花钱
    series: [{ name: "Token (K)", values: entries.map((e) => Math.round((e.total / 1000) * 10) / 10) }],
  };
}

function fromTokenStats(stats: OverviewStats) {
  return {
    categories: stats.by_day.map((d) => d.date),
    series: [{ name: "Token (K)", values: stats.by_day.map((d) => Math.round(d.tokens / 1000)) }],
  };
}

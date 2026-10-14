import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Camera } from "lucide-react";
import { api } from "@/api/client";
import type {
  CitationGraph,
  PaperDetail,
  SnapshotResult,
  TimelineItem,
  TrendResult,
} from "@/api/domain";
import { useSelectedRunId } from "@/hooks/useSelectedRunId";
import { PageHeader } from "@/components/ui/PageHeader";
import { StatCard } from "@/components/ui/StatCard";
import { Segmented } from "@/components/ui/Segmented";
import { Stagger, StaggerItem } from "@/components/ui/Stagger";
import { Tag } from "@/components/ui/Tag";
import { NeonButton } from "@/components/ui/NeonButton";
import { ChartCard } from "@/components/charts/ChartCard";
import { Chart } from "@/components/charts/Chart";
import { citationGraphConfig } from "@/config/charts/graph";
import { methodTimelineConfig } from "@/config/charts/timeline";
import { citationTrendConfig } from "@/config/charts/trend";
import { ringConfig } from "@/config/charts/ring";
import { palette } from "@/config/theme";

// 大屏的取色/图例全靠图表配置与 Design Token，页面里不出现任何色值字面量；
// 引用图与时间线在真实模式下依赖 references/build，缺数据时以空态文案说明设计意图而非留白。

type TierCat = "top" | "core" | "preprint" | "unknown";

const TOP_VENUES = new Set(["NeurIPS", "ICML", "ICLR", "ACL", "SIGIR", "AAAI"]);
const CORE_VENUES = new Set(["NAACL", "EMNLP", "COLING", "CIKM", "WWW"]);

/**
 * 只当兜底用：后端对**库外引用邻居**不给 `venue_tier`（那个字段只在 `GET /papers`
 * 的在库论文上有值），这种点没得可取，才拿 venue 字符串猜一下。
 * ⚠️ 在库论文的分档一律走后端的 `venue_tier.tier` —— 用这份白名单猜过一回，
 * 真实 venue 是 "ACM SIGIR Conference" 这种全名，精确匹配永不命中，A 类占比恒为 0%。
 */
function venueTierOf(venue?: string | null): TierCat {
  if (!venue) return "preprint";
  if (TOP_VENUES.has(venue)) return "top";
  if (CORE_VENUES.has(venue)) return "core";
  return "unknown";
}

type RangeKey = "all" | "s2020" | "s2023";
const RANGES: ReadonlyArray<{ value: RangeKey; label: string; from: number }> = [
  { value: "all", label: "全部年份", from: 0 },
  { value: "s2020", label: "2020 起", from: 2020 },
  { value: "s2023", label: "2023 起", from: 2023 },
];

type LoadPhase = "loading" | "empty" | "error" | undefined;
/** 加载态 + 一句人话原因。只留 `state="error"` 的话，用户只知道"挂了"，不知道挂了啥。 */
interface Load {
  phase: LoadPhase;
  hint?: string;
}

const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * 趋势结论是后端 `app/references/trend.py` 算好的，前端只翻译成人话，不重算一遍。
 * `insufficient_data` 刻意不给结论：少于 2 个采样点时两点间的随机波动就足以
 * 画出"稳步上升"，那种图在学术场景里是有害的 —— 所以这里走空态 + 原样显示 note。
 */
const VERDICT_META: Record<
  TrendResult["verdict"],
  { text: string; short: string; variant: "green" | "amber" | "red" | "muted" }
> = {
  rising: { text: "后端判定 · 上升", short: "上升", variant: "green" },
  flat: { text: "后端判定 · 持平", short: "持平", variant: "muted" },
  falling: { text: "后端判定 · 下降", short: "下降", variant: "red" },
  insufficient_data: {
    text: "采样不足 · 后端不给结论",
    short: "采样不足",
    variant: "amber",
  },
};

const CONCLUSIVE: ReadonlyArray<TrendResult["verdict"]> = ["rising", "flat", "falling"];

/**
 * `POST /papers/citations/snapshot` 的返回按后端实际形状逐字显示（不重命名）。
 * 快照返回的四个统计键。domain.ts 的 SnapshotResult 已与后端对齐
 * （真实键是 recorded / no_match，不是 written / skipped）。
 * `recorded` / `no_match` 是 `app/references/trend.py` 真正返回的键 ——
 * 两边都列进来，页面就不会把后端没给的数字显示成 0。
 */
/** 后端真实键就是这四个（`app/references/edges.py` 的 snapshot 返回） */
const SNAPSHOT_KEYS: ReadonlyArray<{ key: keyof SnapshotResult; label: string }> = [
  { key: "requested", label: "提交" },
  { key: "recorded", label: "已记录" },
  { key: "no_match", label: "未匹配" },
  { key: "errors", label: "失败" },
];

/** 后端 MAX_IDS_PER_REQUEST：一次最多 50 篇，超了是 422，所以这里先截断再说清。 */
const SNAPSHOT_BATCH = 50;

export function AnalyticsPage() {
  const [range, setRange] = useState<RangeKey>("all");
  const [graph, setGraph] = useState<CitationGraph | null>(null);
  const [graphState, setGraphState] = useState<Load>({ phase: "loading" });
  const [timeline, setTimeline] = useState<TimelineItem[]>([]);
  const [timelineState, setTimelineState] = useState<Load>({ phase: "loading" });
  const [trend, setTrend] = useState<TrendResult | null>(null);
  const [trendState, setTrendState] = useState<Load>({ phase: "loading" });
  const [ids, setIds] = useState<string[]>([]);
  /** 语料库论文（带后端算好的 venue_tier）；分档统计用它，不在前端猜 venue 字符串 */
  const [corpus, setCorpus] = useState<PaperDetail[]>([]);
  const [corpusState, setCorpusState] = useState<Load>({ phase: "loading" });
  const [trendId, setTrendId] = useState("");
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [snapshot, setSnapshot] = useState<SnapshotResult | null>(null);
  const [snapshotError, setSnapshotError] = useState("");
  const { runId } = useSelectedRunId();

  /**
   * 先拿语料库，再由它推导后面三个请求的入参。
   *
   * 之前这里写死 `getGraph([])` / `getTimeline(MOCK_RUN_ID)` / `getTrend("2310.11511")`：
   * 空 ids 会被后端闸门判 422，mock 的 run id 在真实库里不存在是 404，
   * 写死的 arxiv_id 也不保证在语料里。mock 模式下三个都"正常"，一切真实后端整页全挂。
   */
  useEffect(() => {
    let alive = true;
    api.papers
      .list({ limit: 60 })
      .then((res) => {
        if (!alive) return;
        setIds(res.items.map((p) => p.arxiv_id).filter(Boolean));
        setCorpus(res.items);
        setCorpusState(res.items.length ? { phase: undefined } : { phase: "empty", hint: "语料库里还没有论文。" });
        // 被引趋势挑被引最多的那篇，而不是随便取第一条
        const best = [...res.items].sort((a, b) => (b.cited_by_count ?? 0) - (a.cited_by_count ?? 0))[0];
        setTrendId(best?.arxiv_id ?? "");
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setIds([]);
        setCorpus([]);
        setCorpusState({ phase: "error", hint: msg(e) });
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!ids.length) {
      setGraphState({
        phase: "empty",
        hint:
          corpusState.phase === "error"
            ? `语料库列表加载失败：${corpusState.hint ?? "未知原因"}`
            : "语料库为空，没有可分析的论文。",
      });
      return;
    }
    let alive = true;
    api.papers
      .getGraph(ids)
      .then((g) => {
        if (!alive) return;
        setGraph(g);
        setGraphState(g.nodes.length ? { phase: undefined } : { phase: "empty" });
      })
      .catch((e: unknown) => alive && setGraphState({ phase: "error", hint: msg(e) }));
    return () => {
      alive = false;
    };
  }, [ids]);

  useEffect(() => {
    if (!runId) {
      setTimelineState({ phase: "empty", hint: "还没有跑过研究，先去研究控制台提一个问题。" });
      return;
    }
    let alive = true;
    api.research
      .getTimeline(runId)
      .then((r) => {
        if (!alive) return;
        setTimeline(r.entries ?? []);
        setTimelineState(r.entries?.length ? { phase: undefined } : { phase: "empty" });
      })
      .catch((e: unknown) => alive && setTimelineState({ phase: "error", hint: msg(e) }));
    return () => {
      alive = false;
    };
  }, [runId]);

  /** 抽成 useCallback：记录完今天的快照后要立刻重取一次趋势，让用户看见点数变了。 */
  const trendSeq = useRef(0);
  const loadTrend = useCallback(() => {
    if (!trendId) {
      trendSeq.current += 1;
      setTrend(null);
      setTrendState({ phase: "empty", hint: "语料库里没有可追踪被引数的论文。" });
      return;
    }
    // 快照按钮会立刻再取一次，两次响应回来的顺序不保证：
    // 用一个自增序号丢弃过期结果，否则旧的一次覆盖掉新写入的那个采样点。
    const seq = ++trendSeq.current;
    setTrendState({ phase: "loading" });
    api.papers
      .getTrend(trendId)
      .then((r) => {
        if (seq !== trendSeq.current) return;
        setTrend(r);
        // 结论与 note 都由后端给；这里只区分"取到了"和"取不到"，
        // 够不够画趋势交给下面的 trendPhase 按 verdict 判，绝不在前端补一条斜率。
        setTrendState({ phase: undefined });
      })
      .catch((e: unknown) => {
        if (seq !== trendSeq.current) return;
        setTrend(null);
        setTrendState({ phase: "error", hint: msg(e) });
      });
  }, [trendId]);

  useEffect(() => {
    // 语料库还在路上时 trendId 必然是空串 —— 那一刻不该闪一句"语料库里没有论文"，
    // 而是继续停在骨架屏上，等真正取完列表再判空。
    if (corpusState.phase === "loading") return;
    loadTrend();
  }, [loadTrend, corpusState.phase]);

  /**
   * 记录今天被引数（F19 趋势的唯一数据来源）。
   *
   * 趋势要**跨天**才成立：这个按钮今天点一次只多出一天，明天再点才有第二个采样点。
   * 界面上必须把这句话讲出来，否则用户点完看到还是"采样不足"，会以为按钮坏了。
   */
  const takeSnapshot = useCallback(() => {
    if (!ids.length || snapshotLoading) return;
    const batch = Array.from(new Set([trendId, ...ids])).filter(Boolean).slice(0, SNAPSHOT_BATCH);
    setSnapshotLoading(true);
    setSnapshotError("");
    setSnapshot(null);
    api.papers
      .snapshotCitations(batch)
      .then((r) => {
        setSnapshot(r);
        loadTrend();
      })
      .catch((e: unknown) => setSnapshotError(msg(e)))
      .finally(() => setSnapshotLoading(false));
  }, [ids, snapshotLoading, trendId, loadTrend]);

  const fromYear = RANGES.find((r) => r.value === range)?.from ?? 0;

  const view = useMemo(() => {
    const g = graph ?? { nodes: [], edges: [], papers: 0 };
    // 年份过滤：节点按自身年份筛，边只保留两端都还在的，剩下的画布才不会悬空引用。
    const nodes = g.nodes.filter((n) => (n.year ?? 9999) >= fromYear);
    const keep = new Set(nodes.map((n) => n.arxiv_id));
    const edges = g.edges.filter((l) => keep.has(l.source) && keep.has(l.target));

    const inCorpus = nodes.filter((n) => n.in_corpus);
    const years = nodes.map((n) => n.year).filter((y): y is number => typeof y === "number");
    const span = years.length ? Math.max(...years) - Math.min(...years) + 1 : 0;

    /**
     * 分档一律采用后端给的 `venue_tier.tier`，不在前端拿 venue 字符串去匹配白名单。
     * 之前那么做的结果是 A 类占比恒为 0%：真实 venue 是 "ACM SIGIR Conference" 这种全名，
     * 而白名单里写的是 "SIGIR"，精确匹配永远不命中 —— 一个看起来合理、实际全错的数字。
     */
    const counts: Record<TierCat, number> = { top: 0, core: 0, preprint: 0, unknown: 0 };
    const inCorpusWindow = corpus.filter((p) => (Number(p.published?.slice(0, 4)) || 9999) >= fromYear);
    inCorpusWindow.forEach((p) => {
      const t = p.venue_tier?.tier as TierCat | undefined;
      counts[t && t in counts ? t : "unknown"] += 1;
    });
    const aShare = inCorpusWindow.length
      ? (counts.top / inCorpusWindow.length) * 100
      : 0;

    // 有至少一条边的节点才算"连上了"。孤立节点数才是用户眼里的"为什么这么散"，
    // 用 边数/节点数 判断会在 15 边 / 43 节点时误判成"不稀疏"，而实际 26 个点在漂。
    const connected = new Set<string>();
    edges.forEach((l) => {
      connected.add(l.source);
      connected.add(l.target);
    });

    return {
      nodes,
      edges,
      edgeCount: edges.length,
      connectedCount: connected.size,
      papers: inCorpus.length,
      span,
      aShare,
      counts,
    };
  }, [graph, corpus, fromYear]);

  const graphOption = useMemo(() => {
    // 分档取后端给的 venue_tier.tier。之前在这里调 venueTierOf(n.venue) 自己猜，
    // 真实 venue 是全名（"ACM SIGIR Conference"）匹配不上白名单，
    // 结果大量节点落进 default 分支被染成琥珀色，看着像"报警"其实只是没猜中。
    const tierByArxiv = new Map(
      corpus.map((p) => [p.arxiv_id, (p.venue_tier?.tier as TierCat | undefined) ?? "unknown"] as const),
    );
    return citationGraphConfig({
      nodes: view.nodes.map((n) => ({
        id: n.arxiv_id,
        name: n.title,
        inLibrary: n.in_corpus,
        tier: tierByArxiv.get(n.arxiv_id) ?? venueTierOf(n.venue),
        citedBy: n.cited_by_count,
        year: n.year,
      })),
      links: view.edges.map((l) => ({ source: l.source, target: l.target })),
    });
  }, [view, corpus]);

  const timelineOption = useMemo(() => {
    const labelByArxiv = new Map((graph?.nodes ?? []).map((n) => [n.arxiv_id, n.title] as const));
    const items = timeline
      .filter((it) => (it.year ?? 9999) >= fromYear)
      .map((it) => ({
        arxiv_id: it.arxiv_id,
        title: it.title || labelByArxiv.get(it.arxiv_id) || it.arxiv_id,
        year: it.year,
        venue: it.venue ?? undefined,
        // 后端已经算过分档，直接用它给的，不在前端再拿 venue 字符串猜一遍
        venue_tier: (it.venue_tier as TierCat) || venueTierOf(it.venue),
        cited_by_count: it.cited_by_count,
        // addresses_gap 的元素本来就是 {paper_id, title, edge_source}，无需再映射
        addresses_gap: it.addresses_gap,
      }));
    return methodTimelineConfig({ items });
  }, [timeline, graph, fromYear]);

  /**
   * 采样点形状就是后端的 `{on, cited_by_count}`（`on` 不是 `date`，
   * `cited_by_count` 不是 `citations`）—— 之前读了两个不存在的键，
   * 结果 `NaN` 一路传进图表，趋势图在任何后端上都是恒空。
   */
  const trendWindow = useMemo(
    () => (trend?.points ?? []).filter((p) => Number(String(p.on).slice(0, 4)) >= fromYear),
    [trend, fromYear],
  );

  /**
   * 画不画由后端的 `verdict` 决定，不由前端数点决定：
   * `insufficient_data` 走空态 + 原样显示 `note`，一条连线都不画。
   * 结论成立但当前年份范围筛得不足 2 点时也一样不画 —— 那属于"范围切太窄"，
   * 说清楚就好，不能拿 1 个点连出一条假装出来的斜率。
   */
  const trendVerdict = trend?.verdict ?? null;
  const trendConclusive = !!trendVerdict && CONCLUSIVE.includes(trendVerdict);
  const trendHidden = !trend || !trendConclusive || trendWindow.length < 2;
  const trendEmptyHint = !trend
    ? trendState.hint ?? "被引趋势取数失败。"
    : trend.verdict === "insufficient_data"
      ? // ★ 后端刻意不给结论，这里把那句话原样搬上来，不改写、不补结论
        trend.note ??
        `只有 ${trend.points.length} 个采样点，无法判断趋势。点下方「记录今天被引数」才会多出一天。`
      : `后端在整条序列（${trend.points.length} 个采样点）上判定为「${
          VERDICT_META[trend.verdict].short
        }」，但当前年份范围内只剩 ${trendWindow.length} 个采样点，画不出趋势 —— 把范围切回「全部年份」试试。`;

  const trendOption = useMemo(
    () => citationTrendConfig({ points: trendWindow }),
    [trendWindow],
  );

  const snapshotStats = useMemo(
    () => (snapshot ? SNAPSHOT_KEYS.map(({ key, label }) => ({ label, value: snapshot[key] })) : []),
    [snapshot],
  );

  const ringOption = useMemo(
    () =>
      ringConfig({
        centerLabel: "在库论文",
        items: [
          { name: "顶会 (A)", value: view.counts.top, color: palette.cyan },
          { name: "核心 (B)", value: view.counts.core, color: palette.purple },
          { name: "预印本", value: view.counts.preprint, color: palette.greenSoft },
          { name: "未分档", value: view.counts.unknown, color: palette.textSecondary },
        ],
      }),
    [view],
  );

  const graphEmpty = graphState.phase === undefined && view.edgeCount === 0;
  /** 连通覆盖率低于 60% 就要说明原因 —— 否则满屏散点会被读成"图坏了" */
  const sparse = view.nodes.length > 0 && view.connectedCount < view.nodes.length * 0.6;
  const isolated = Math.max(0, view.nodes.length - view.connectedCount);

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="文献计量大屏"
        description="方法演进时间线 · 引用关系图 · 被引趋势 · 会议分档 —— 全部来自引用边，不用年份先后编故事"
        features="F12 · F16 · F19"
        actions={
          <Segmented
            ariaLabel="时间范围"
            value={range}
            onChange={(v) => setRange(v as RangeKey)}
            options={RANGES.map((r) => ({ value: r.value, label: r.label }))}
          />
        }
      />

      {corpusState.phase === "error" ? (
        // 取数挂了的时候顶部四张卡会一律显示 0 —— 那和"真的是 0 篇"在屏幕上长得一样。
        // 所以这句话必须写在卡片区上方，而不是只留在分档图的空态里。
        <div className="rounded-lg border border-neon-red/40 bg-white/[0.04] px-4 py-3 text-sm text-neon-red">
          语料库列表 <code className="font-mono">GET /api/v1/papers</code> 加载失败：
          {corpusState.hint ?? "未知原因"} —— 本页指标卡与分档图此刻没有数据来源，显示的是 0 而非真实值。
        </div>
      ) : null}

      <Stagger className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StaggerItem>
          <StatCard label="语料篇数（在库）" value={view.papers} unit="篇" icon="library" accent="cyan" />
        </StaggerItem>
        <StaggerItem>
          <StatCard label="引用边数" value={view.edgeCount} unit="条" icon="share-2" accent="purple" />
        </StaggerItem>
        <StaggerItem>
          <StatCard label="覆盖年份跨度" value={view.span} unit="年" icon="activity" accent="green" />
        </StaggerItem>
        <StaggerItem>
          <StatCard
            label="A 类会议占比"
            value={view.aShare}
            unit="%"
            decimals={1}
            icon="circle-check"
            accent="amber"
          />
        </StaggerItem>
      </Stagger>

      <ChartCard
        className="xl:col-span-2"
        title="引用关系力导向图"
        subtitle="节点色：在库顶会=青 / 核心=紫 / 预印本=浅绿 / 未分档与库外引用=灰；箭头=引用方向，可缩放拖拽、悬停高亮邻接"
        height={440}
        state={graphState.phase ?? (graphEmpty ? "empty" : undefined)}
        emptyHint={
          "引用边为空：`addresses_gap` 与引用关系只有在跑过 POST /papers/references/build 之后才有数据。" +
          "后端刻意不用年份先后去猜「谁补了谁的坑」，所以没跑构建接口时这里就该是空的，而不是编一张图。"
        }
        errorHint={graphState.hint ? `引用关系图加载失败：${graphState.hint}` : "引用关系图加载失败"}
        extra={
          graphState.phase === undefined ? (
            <Tag variant="muted" mono>
              {view.nodes.length} 节点 / {view.edgeCount} 边 / {view.connectedCount} 连通
            </Tag>
          ) : null
        }
      >
        {sparse ? (
          // 只有零星几条边时，力导向会把节点互相推开成一片散点，看上去就像图坏了。
          // 与其让人猜，不如把"为什么连不起来"写在图上：这是数据边界，不是渲染问题。
          <div className="mb-2 rounded-md border border-neon-amber/40 bg-white/[0.04] px-3 py-2 text-xs leading-relaxed text-ink-soft">
            <span className="text-neon-amber">连通性不足：</span>
            {view.nodes.length} 个节点里只有 {view.connectedCount} 个连上了引用边
            （{isolated} 个孤立），所以节点明显是散开的。
            引用边只来自 OpenAlex 的 <code className="font-mono">referenced_works</code> ——
            语料里较新的 arXiv 论文往往还没被收录，解析不出被引方就不建边。
            后端不会用年份先后去补这些边，所以这里就该是散的。
          </div>
        ) : null}
        <Chart option={graphOption} height={sparse ? "calc(100% - 4.5rem)" : "100%"} />
      </ChartCard>

      <ChartCard
        title="方法演进时间线"
        subtitle="横轴年份，节点大小映射被引数，紫色箭头 = 有引用边证据支撑的「补了谁的坑」"
        height={360}
        state={timelineState.phase ?? (timeline.filter((it) => (it.year ?? 9999) >= fromYear).length === 0 ? "empty" : undefined)}
        emptyHint={timelineState.hint ?? "当前时间范围内没有可绘制的方法节点"}
        errorHint={timelineState.hint ? `时间线加载失败：${timelineState.hint}` : "时间线加载失败"}
      >
        <Chart option={timelineOption} height="100%" />
      </ChartCard>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <ChartCard
            title="被引趋势"
            subtitle={
              trendConclusive && !trendHidden
                ? "累计=发光面积（存量），紫条=区间新增（脉冲）"
                : "少于 2 个采样点时后端刻意不给结论 —— 这里就不画，前端不替它编一条斜率"
            }
            height={320}
            state={trendState.phase ?? (trendHidden ? "empty" : undefined)}
            emptyHint={trendEmptyHint}
            errorHint={trendState.hint ? `被引趋势加载失败：${trendState.hint}` : "被引趋势加载失败"}
            extra={
              <div className="flex items-center gap-2">
                {trend ? (
                  <Tag variant={VERDICT_META[trend.verdict].variant}>
                    {VERDICT_META[trend.verdict].text}
                  </Tag>
                ) : null}
                {/* delta / span_days 是后端算好的，这里只显示，不自已再拿首尾两点求差 */}
                {trendConclusive && trend?.delta != null ? (
                  <Tag variant="muted" mono>
                    Δ {trend.delta > 0 ? "+" : ""}
                    {trend.delta} / {trend.span_days ?? 0} 天
                  </Tag>
                ) : trend ? (
                  <Tag variant="muted" mono>
                    {trend.points.length} 个采样点 · {trend.arxiv_id}
                  </Tag>
                ) : null}
              </div>
            }
          >
            <Chart option={trendOption} height="100%" />
          </ChartCard>

          {/* 快照按钮放在趋势图正下方：它俩是同一件事的"因"和"果"，
              分开摆用户点完找不到变化发生在哪。空态时图表整块被 ChartCard 换成了占位，
              所以这块提示必须待在卡片外面，否则空态下正好看不见。 */}
          <div className="rounded-lg border border-hairline bg-white/[0.02] px-4 py-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0 flex-1 text-xs leading-relaxed text-ink-soft">
                <p>
                  <span className="text-ink">记录今天被引数</span> —— 趋势要靠
                  <span className="text-ink">跨天</span>累积：今天点一次只多出「今天」这一个采样点，
                  <span className="text-neon-amber">明天再点一次才会有第二个点</span>，后端才会开始给趋势结论。
                </p>
                <p className="mt-1 text-ink-faint">
                  数据源 <code className="font-mono">POST /api/v1/papers/citations/snapshot</code> →{" "}
                  <code className="font-mono">citation_snapshots</code> 表 · 同一天重复点是幂等的 ·{" "}
                  单次上限 {SNAPSHOT_BATCH} 篇（语料 {ids.length} 篇，本次提交{" "}
                  {Math.min(ids.length, SNAPSHOT_BATCH)} 篇）
                </p>
                {snapshotStats.length ? (
                  <p className="mt-2 flex flex-wrap items-center gap-1.5">
                    <span className="text-ink">本次返回：</span>
                    {snapshotStats.map(({ label, value }) => (
                      <Tag key={label} variant="muted" mono>
                        {label} {value}
                      </Tag>
                    ))}
                    {typeof snapshot?.no_match === "number" && snapshot.no_match > 0 ? (
                      <span className="text-ink-faint">
                        （未匹配 = OpenAlex 标题+作者交叉校验没过，宁可不记也不记到别的论文头上）
                      </span>
                    ) : null}
                  </p>
                ) : null}
                {snapshotError ? (
                  <p className="mt-2 text-neon-red">记录失败：{snapshotError}</p>
                ) : null}
              </div>
              <NeonButton
                variant="secondary"
                className="shrink-0 px-3 py-1.5 text-xs"
                loading={snapshotLoading}
                disabled={!ids.length}
                onClick={takeSnapshot}
              >
                <Camera size={14} aria-hidden /> 记录今天被引数
              </NeonButton>
            </div>
          </div>
        </div>

        <ChartCard
          title="会议分档占比"
          subtitle="分档取自后端 venue_tier.tier（GET /papers 返回），不在前端按 venue 字符串猜"
          height={320}
          state={corpusState.phase ?? (view.counts.top + view.counts.core + view.counts.preprint + view.counts.unknown === 0 ? "empty" : undefined)}
          emptyHint={corpusState.hint ?? "当前时间范围内没有在库论文。"}
          errorHint={corpusState.hint ? `分档统计加载失败：${corpusState.hint}` : "分档统计加载失败"}
        >
          <Chart option={ringOption} height="100%" />
        </ChartCard>
      </div>
    </div>
  );
}

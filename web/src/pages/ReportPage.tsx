import { useCallback, useEffect, useMemo, useState } from "react";
import { Download } from "lucide-react";
import { api } from "@/api/client";
import { COMPARISON_COLUMNS } from "@/api/domain";
import type { Citation, CitationBundle, ComparisonRow, RunDetail } from "@/api/domain";
import { useSelectedRunId } from "@/hooks/useSelectedRunId";
import { palette } from "@/config/theme";
import { PageHeader } from "@/components/ui/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { Pager } from "@/components/ui/Pager";
import { StatCard } from "@/components/ui/StatCard";
import { Stagger } from "@/components/ui/Stagger";
import { Segmented } from "@/components/ui/Segmented";
import { NeonButton } from "@/components/ui/NeonButton";
import { Tag } from "@/components/ui/Tag";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { MarkdownBlock } from "@/components/ui/MarkdownBlock";
import { ComparisonTable } from "@/components/business/ComparisonTable";
import { CitationCard } from "@/components/business/CitationCard";
import { VerificationSummary } from "@/components/business/VerificationSummary";
import { Chart } from "@/components/charts/Chart";
import { gaugeConfig } from "@/config/charts/gauge";

// 报告页的唯一取数入口是 api.research.*；缺失字段一律原样传 null 给展示组件，页面绝不预先补 0/"N/A"，否则"缺字段留空"这条对外承诺就被前端糊掉了。

type Lang = "zh" | "en";

const EN_HEADER: Record<(typeof COMPARISON_COLUMNS)[number]["key"], string> = {
  method_name: "Method",
  core_idea: "Core idea",
  datasets: "Datasets",
  improvement: "Gain vs baseline",
  cost: "Cost",
  limitation: "Limitation (authors' own words)",
  code_url: "Code",
  venue: "Venue",
  cited_by_count: "Cited by",
};

const T: Record<Lang, Record<string, string>> = {
  zh: {
    title: "报告与溯源",
    desc: "每一条论断都能追回原文；原文没给的字段留空，绝不用 0 或猜测填满。",
    compare: "九列结构化对比",
    compareSub: "口径固定九列 · 缺失即「—」",
    report: "报告正文",
    reportSub: "点击正文里的 [n] 可跳到对应证据",
    evidence: "引用证据链",
    evidenceSub: "按校验结果分组",
    rate: "支撑率仪表盘",
    noRate: "这次研究没有引用",
    noRateDesc: "没有引用时支撑率是 null，不是 0% —— 仪表盘不画。",
    exportMd: "导出报告",
    exportRelated: "导出相关工作",
    langZh: "中文",
    langEn: "英文",
    supported: "有原文支撑",
    unsupported: "无支撑 · 已剔除",
    unchecked: "未校验",
    all: "全部论文",
    filterPaper: "按论文筛选",
    cardsCitations: "引用总数",
    cardsRate: "支撑率",
    cardsDropped: "被剔除论断",
    cardsRevision: "修订次数",
    noRun: "未指定研究编号",
    noRunDesc: "真实模式下用 /report?run=<run_id> 打开，或先在研究控制台跑一次。",
    loadFail: "数据加载失败：",
    emptyEvidence: "这次研究没有留下任何引用记录。",
    note: "校验说明",
  },
  en: {
    title: "Report & Provenance",
    desc: "Every claim links back to source text; fields the paper never states stay blank, never filled with 0 or guesses.",
    compare: "Nine-column comparison",
    compareSub: "Fixed schema · blank means missing",
    report: "Report body",
    reportSub: "Click [n] inline to jump to its evidence",
    evidence: "Citation evidence chain",
    evidenceSub: "Grouped by verification",
    rate: "Support-rate gauge",
    noRate: "This run cited nothing",
    noRateDesc: "With no citations the support rate is null, not 0% — so no gauge is drawn.",
    exportMd: "Export report",
    exportRelated: "Export related work",
    langZh: "中文",
    langEn: "English",
    supported: "Supported by source",
    unsupported: "Unsupported · dropped",
    unchecked: "Not verified",
    all: "All papers",
    filterPaper: "Filter by paper",
    cardsCitations: "Citations",
    cardsRate: "Support rate",
    cardsDropped: "Dropped claims",
    cardsRevision: "Revisions",
    noRun: "No run selected",
    noRunDesc: "Open /report?run=<run_id> in live mode, or start a run in the console first.",
    loadFail: "Failed to load data: ",
    emptyEvidence: "This run produced no citations.",
    note: "Verifier note",
  },
};

/**
 * 把报告按 `## ` 小节切成分页用的段落。
 *
 * 为什么按小节而不是按固定行数：这份正文是 `app/export/markdown.py` 的模板产出的，
 * 结构稳定（`## 总体概览` / `## 一、…` / `## 参考文献` / `## 成本与耗时`），
 * 按小节切每一页都是一个完整语义单元；按行数切会把一段论证拦腰截断，
 * 而"引用角标 [n] 在正文里、它的证据在右栏"这种跨栏对应关系最怕截断。
 *
 * 第一个 `## ` 之前的内容（报告大标题 + 引言）单独算一页，不丢。
 */
export function splitReportSections(markdown: string): { title: string; body: string }[] {
  const text = (markdown || "").trim();
  if (!text) return [];

  const sections: { title: string; body: string }[] = [];
  let current: { title: string; lines: string[] } | null = null;

  for (const line of text.split("\n")) {
    const heading = /^##\s+(.+?)\s*$/.exec(line);
    if (heading) {
      if (current) sections.push({ title: current.title, body: current.lines.join("\n") });
      current = { title: heading[1], lines: [line] };
    } else if (current) {
      current.lines.push(line);
    } else {
      // 还没有遇到任何 `## `：这些行属于开头（大标题 / 引言）
      current = { title: "", lines: [line] };
    }
  }
  if (current) sections.push({ title: current.title, body: current.lines.join("\n") });

  return sections.map((s, i) => ({
    title: s.title || (i === 0 ? "开头" : "未命名小节"),
    body: s.body,
  }));
}

/** 导航条上的小节名：太长会挤成一团，截断即可（完整标题在正文里看得到） */
function shortLabel(title: string, limit = 12): string {
  const clean = title.replace(/[⚠️→·]/g, " ").replace(/\s+/g, " ").trim();
  return clean.length > limit ? `${clean.slice(0, limit)}…` : clean;
}

export function ReportPage() {
  const { runId } = useSelectedRunId();

  const [run, setRun] = useState<RunDetail | null>(null);
  const [bundle, setBundle] = useState<CitationBundle | null>(null);
  const [loading, setLoading] = useState(Boolean(runId));
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<Lang>("zh");
  const [paperFilter, setPaperFilter] = useState("all");
  const [activeCite, setActiveCite] = useState<number | null>(null);
  // 正文分页：一次研究的报告实测能到 5000px 高，而右栏证据链只有 ~700px，
  // 两栏完全对不齐；而且长内容内层滚动 + 页面滚动套在一起会"抢滚轮"。
  const [sectionPage, setSectionPage] = useState(1);
  const [readWhole, setReadWhole] = useState(false);

  const t = T[lang];

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setLoading(true);
    setError(null);
    // 换一条 run 就是把另一篇报告放进来了，分页位置必须回到第一页 ——
    // 停在第 4 页会让用户以为新报告"只有三节"。
    setSectionPage(1);
    setReadWhole(false);
    Promise.all([api.research.getRun(runId), api.research.getCitations(runId)])
      .then(([r, c]) => {
        if (!alive) return;
        setRun(r);
        setBundle(c);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  const citations = bundle?.citations ?? [];

  const sections = useMemo(() => splitReportSections(run?.draft ?? ""), [run?.draft]);
  const pageCount = sections.length;
  const safePage = Math.min(Math.max(sectionPage, 1), Math.max(pageCount, 1));

  const papers = useMemo(
    () => Array.from(new Set(citations.map((c) => c.arxiv_id).filter(Boolean))),
    [citations],
  );

  const visible = useMemo(
    () =>
      citations
        .map((c, i) => ({ c, index: i + 1 }))
        .filter(({ c }) => paperFilter === "all" || c.arxiv_id === paperFilter),
    [citations, paperFilter],
  );
  // ★ verified 三态：只有 false 算"无支撑·已剔除"，null=当时根本没校验，单独一档。
  // 判据与 writer.py / export/markdown.py / /citations 的 verified_rate 同源，不许各写一套。
  const supported = visible.filter(({ c }) => c.verified === true);
  const unsupported = visible.filter(({ c }) => c.verified === false);
  const unchecked = visible.filter(({ c }) => c.verified === null);

  /** 正文角标可能指向当前被筛选隐藏的引用，先放开筛选再滚 */
  const jumpToCitation = useCallback(
    (n: number) => {
      setPaperFilter("all");
      setActiveCite(n);
      window.setTimeout(() => {
        document.getElementById(`cite-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 60);
      window.setTimeout(() => setActiveCite((cur) => (cur === n ? null : cur)), 2400);
    },
    [],
  );

  const table = useMemo(() => {
    const cols = COMPARISON_COLUMNS.map((c) => ({
      key: c.key,
      header: lang === "zh" ? c.header : EN_HEADER[c.key],
    }));
    const rows: Array<Record<string, string | number | null>> = (run?.comparison ?? []).map((r: ComparisonRow) =>
      // 只做 key → 表头的换名，null 保持 null
      Object.fromEntries(cols.map((c) => [c.header, r[c.key] ?? null])),
    );
    return { columns: cols.map((c) => c.header), rows };
  }, [run, lang]);

  const rate = bundle?.verified_rate ?? null;
  // verification 只认后端五键（verifier.py）：剔除数 = unsupported.length，检查数 = checked，通过率 = score。
  // ★ 不再拿 citations 二次统计剔除数 —— unsupported 才是 Verifier 认定的那一份，两套口径迟早对不上。
  const verification = run?.verification ?? null;
  const droppedCount = verification ? verification.unsupported.length : null;
  const droppedClaims = verification?.unsupported ?? [];
  const checkedCount = verification?.checked ?? null;

  if (!runId) {
    return (
      <div className="flex flex-col gap-5">
        <PageHeader title={t.title} description={t.desc} features="F4 · F5 · F6 · F8" />
        <Panel title={t.noRun}>
          <EmptyState icon="file-search" title={t.noRun} description={t.noRunDesc} />
        </Panel>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title={t.title}
        description={t.desc}
        features="F4 · F5 · F6 · F8"
        actions={
          <>
            <Segmented
              ariaLabel="报告语言"
              value={lang}
              onChange={(v) => setLang(v === "en" ? "en" : "zh")}
              options={[
                { value: "zh", label: t.langZh },
                { value: "en", label: t.langEn },
              ]}
            />
            <NeonButton
              variant="secondary"
              onClick={() => window.open(api.research.exportUrl(runId), "_blank", "noopener")}
            >
              <Download size={15} aria-hidden={true} />
              {t.exportMd}
            </NeonButton>
            {/* F18 相关工作章节：同样是纯链接下载，不走 fetch */}
            <NeonButton
              variant="secondary"
              onClick={() => window.open(api.research.exportUrl(runId, "related-work"), "_blank", "noopener")}
            >
              <Download size={15} aria-hidden={true} />
              {t.exportRelated}
            </NeonButton>
          </>
        }
      />

      {error && (
        <div className="glass panel rounded-xl px-4 py-3 text-sm text-neon-red">
          {t.loadFail}
          {error}
        </div>
      )}

      <Stagger className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label={t.cardsCitations}
          value={bundle?.count ?? Number.NaN}
          icon="file-search"
          accent="cyan"
        />
        <StatCard
          label={t.cardsRate}
          value={rate == null ? Number.NaN : rate * 100}
          unit={rate == null ? undefined : "%"}
          decimals={1}
          icon="shield-check"
          accent="green"
        />
        <StatCard
          label={t.cardsDropped}
          value={droppedCount ?? Number.NaN}
          icon="circle-alert"
          accent="amber"
        />
        <StatCard
          label={t.cardsRevision}
          value={run?.revision_count ?? Number.NaN}
          icon="activity"
          accent="purple"
        />
      </Stagger>

      <Panel title={t.compare} subtitle={t.compareSub}>
        {loading ? (
          <div className="flex flex-col gap-2">
            <Skeleton height={38} />
            <Skeleton height={38} width="92%" />
            <Skeleton height={38} width="84%" />
          </div>
        ) : (
          <ComparisonTable columns={table.columns} rows={table.rows} maxHeight={420} />
        )}
      </Panel>

      {/* 两栏严格等高：正文分页之后每一栏都是有限高度，
          不再出现"左边 5000px、右边 700px"那种一边倒 */}
      <div className="grid grid-cols-1 items-stretch gap-4 xl:h-[760px] xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
        <Panel
          title={t.report}
          subtitle={t.reportSub}
          className="min-h-0 xl:h-full"
          bodyClassName="flex min-h-0 flex-col gap-3 overflow-hidden"
          extra={
            pageCount > 1 ? (
              <button
                type="button"
                onClick={() => setReadWhole((v) => !v)}
                title={readWhole ? "改回按小节翻页" : "一次性看完整篇（方便 Ctrl+F 全文查找）"}
                className="rounded-lg border border-hairline px-2.5 py-1 text-[11px] text-ink-soft transition hover:border-neon-cyan/50 hover:text-neon-cyan"
              >
                {readWhole ? "按小节翻页" : "看完整篇"}
              </button>
            ) : null
          }
        >
          {loading ? (
            <div className="flex flex-col gap-3">
              <Skeleton height={22} width="55%" />
              <Skeleton height={90} />
              <Skeleton height={22} width="42%" />
              <Skeleton height={120} />
            </div>
          ) : (
            <>
              <div className="min-h-0 flex-1 overflow-y-auto pr-1">
                <MarkdownBlock
                  source={readWhole ? (run?.draft ?? "") : (sections[safePage - 1]?.body ?? "")}
                  onCiteClick={jumpToCitation}
                />
              </div>
              {/* 「看完整篇」时不再摆分页条：那时正文是一整篇，翻页控件点了也没意义 */}
              {!readWhole && pageCount > 1 ? (
                <Pager
                  page={safePage}
                  pages={pageCount}
                  onChange={setSectionPage}
                  labels={sections.map((s) => shortLabel(s.title))}
                  unit="节"
                  ariaLabel="报告小节导航"
                  className="shrink-0 border-t border-hairline pt-3"
                />
              ) : null}
            </>
          )}
        </Panel>

        <Panel
          title={t.evidence}
          subtitle={t.evidenceSub}
          className="min-h-0 xl:h-full"
          bodyClassName="min-h-0 flex-1 overflow-y-auto"
          extra={
            <label className="flex items-center gap-2 text-[11px] text-ink-soft">
              <span className="hidden sm:inline">{t.filterPaper}</span>
              <select
                value={paperFilter}
                onChange={(e) => setPaperFilter(e.target.value)}
                className="max-w-[168px] rounded-md border border-hairline bg-panel px-2 py-1 font-mono text-xs text-ink outline-none focus:border-neon-cyan/60"
              >
                <option value="all">{t.all}</option>
                {papers.map((id) => (
                  <option key={id} value={id}>
                    {id}
                  </option>
                ))}
              </select>
            </label>
          }
        >
          {loading ? (
            <div className="flex flex-col gap-2">
              <Skeleton height={56} />
              <Skeleton height={56} width="90%" />
              <Skeleton height={56} width="80%" />
            </div>
          ) : visible.length === 0 ? (
            <EmptyState icon="inbox" title={t.emptyEvidence} />
          ) : (
            <div className="flex flex-col gap-4">
              {supported.length > 0 && (
                <CitationGroup
                  heading={t.supported}
                  tone="green"
                  items={supported}
                  activeCite={activeCite}
                  noteLabel={t.note}
                />
              )}
              {unsupported.length > 0 && (
                <CitationGroup
                  heading={t.unsupported}
                  tone="red"
                  items={unsupported}
                  activeCite={activeCite}
                  noteLabel={t.note}
                />
              )}
              {unchecked.length > 0 && (
                <CitationGroup
                  heading={t.unchecked}
                  tone="amber"
                  items={unchecked}
                  activeCite={activeCite}
                  noteLabel={t.note}
                />
              )}
            </div>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
        <Panel title={t.rate}>
          {rate == null ? (
            <EmptyState
              icon="shield-check"
              title={t.noRate}
              description={t.noRateDesc}
              className="min-h-[240px]"
            />
          ) : (
            <Chart
              option={gaugeConfig({
                value: Math.round(rate * 1000) / 10,
                max: 100,
                name: t.cardsRate,
                thresholds: [
                  { upto: 0.6, color: palette.red },
                  { upto: 0.85, color: palette.amber },
                  { upto: 1, color: palette.green },
                ],
              })}
              height={240}
            />
          )}
        </Panel>

        <VerificationSummary
          passedCount={checkedCount == null ? Number.NaN : Math.max(0, checkedCount - (droppedCount ?? 0))}
          droppedCount={droppedCount ?? Number.NaN}
          droppedClaims={droppedClaims}
          score={verification ? verification.score : null}
          revisionCount={run?.revision_count ?? 0}
          passed={verification ? verification.passed : null}
        />
      </div>
    </div>
  );
}

function CitationGroup({
  heading,
  tone,
  items,
  activeCite,
  noteLabel,
}: {
  heading: string;
  tone: "green" | "red" | "amber";
  items: Array<{ c: Citation; index: number }>;
  activeCite: number | null;
  noteLabel: string;
}) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <span
          className={
            tone === "green"
              ? "h-2 w-2 rounded-full bg-neon-green/80"
              : tone === "red"
                ? "h-2 w-2 rounded-full bg-neon-red"
                : "h-2 w-2 rounded-full bg-neon-amber"
          }
          aria-hidden={true}
        />
        <h4 className="text-xs font-semibold text-ink">{heading}</h4>
        <Tag variant={tone} mono>
          {items.length}
        </Tag>
      </div>
      {items.map(({ c, index }) => (
        <CitationRow
          key={`${c.arxiv_id}-${index}`}
          citation={c}
          index={index}
          active={activeCite === index}
          noteLabel={noteLabel}
        />
      ))}
    </div>
  );
}

function CitationRow({
  citation: c,
  index,
  active,
  noteLabel,
}: {
  citation: Citation;
  index: number;
  active: boolean;
  noteLabel: string;
}) {
  return (
    <div
      id={`cite-${index}`}
      className={
        active
          ? "rounded-lg ring-1 ring-neon-cyan/70 transition-shadow duration-[var(--dur-base)]"
          : "rounded-lg transition-shadow duration-[var(--dur-base)]"
      }
    >
      <CitationCard
        claim={c.claim}
        arxiv_id={c.arxiv_id}
        section={c.locator ?? undefined}
        chunk_text={c.quote ?? undefined}
        supported={c.verified}
        source={c.source}
        resolution_how={c.resolution_how}
      />
      {c.note ? (
        <p className="mt-1 pl-3 text-[11px] leading-relaxed text-neon-red/85">
          <span className="font-mono text-ink-faint">{noteLabel} · </span>
          {c.note}
        </p>
      ) : null}
    </div>
  );
}

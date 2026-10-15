import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";
import { ApiError, api } from "@/api/client";
import { COMPARISON_COLUMNS } from "@/api/domain";
import type { AskResult, BuildEdgesResult, Citation, ComparisonRow, PaperDetail } from "@/api/domain";
import { PageHeader } from "@/components/ui/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { Pager } from "@/components/ui/Pager";
import { Tag } from "@/components/ui/Tag";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { NeonButton } from "@/components/ui/NeonButton";
import { FadeIn } from "@/components/ui/FadeIn";
import { ComparisonTable } from "@/components/business/ComparisonTable";
import { CitationCard } from "@/components/business/CitationCard";

// 列表接口 /ask/upload 全部只经 api.client 取数：真实模式下没有 GET /papers，
// 这个失败态必须被如实渲染成一句人话，而不是转圈或空白。

const TITLES: Record<string, string> = {
  "2310.11511": "Self-RAG",
  "2401.15884": "CRAG",
  "2112.01488": "ColBERTv2",
  "2402.03218": "BGE M3",
};

// ⚠️ PAPER_COMPARISON / TITLES 是写死的演示数据，**只在 api.useMock 为真时使用**。
// 真实模式的对比行一律由 detailToComparisonRow() 从 GET /papers/{id} 的真实字段拼装，
// 拿不到的列留空显示「—」—— 用户勾的是真论文，对比表不能是编的（九列口径：缺失留空不编造）。
const PAPER_COMPARISON: Record<string, ComparisonRow> = {
  "2310.11511": {
    method_name: "Self-RAG",
    core_idea: "训练模型按需发出反思 token，自行决定何时检索并批判性评估召回",
    datasets: "PopQA / TriviaQA / ARC-Challenge / ASQA",
    improvement: "ASQA 事实性 +5.1%；7B 追平 ChatGPT 级系统",
    cost: "需额外训练反思 token，7B/13B 各一次全参微调",
    limitation: "反思 token 阈值需按任务调；长文生成会累积错误",
    code_url: "https://github.com/AkariAsai/self-rag",
    venue: "ICML 2024",
    cited_by_count: 412,
  },
  "2401.15884": {
    method_name: "CRAG (Corrective RAG)",
    core_idea: "用轻量检索评估器给每条结果打质量分，低于阈值时改走联网搜索",
    datasets: "PopQA / Biography / PubHealth",
    improvement: "三数据集一致提升，与所用 LLM 无关",
    cost: "额外一次评估器推理 + 失败时多一路 web 调用",
    limitation: null,
    code_url: null,
    venue: "2024 arXiv",
    cited_by_count: 138,
  },
  "2112.01488": {
    method_name: "ColBERTv2 压缩检索",
    core_idea: "多向量 late-interaction 表示做残差压缩，把索引体积压到 1.7GB",
    datasets: "BEIR / MS MARCO",
    improvement: "nDCG@10 相比 BM25 +12.3；索引比 v1 小 6~10 倍",
    cost: "late-interaction 打分开销高于单向量稠密检索",
    limitation: "压缩后长尾领域召回下降，需重排补偿",
    code_url: "https://github.com/stanford-futuredata/ColBERT",
    venue: "NAACL 2022",
    cited_by_count: 986,
  },
  "2402.03218": {
    method_name: "BGE M3-Embedding",
    core_idea: "单模型同时输出稠密 / 稀疏 / 多向量三种表示，跨语言跨粒度",
    datasets: "MIRACL / MKQA",
    improvement: null,
    cost: null,
    limitation: null,
    code_url: null,
    venue: null,
    cited_by_count: null,
  },
};

function yearOf(p: PaperDetail): string {
  const y = (p.published ?? "").slice(0, 4);
  return /^\d{4}$/.test(y) ? y : "—";
}

function rowToTable(r: ComparisonRow): Record<string, string | number | null> {
  const out: Record<string, string | number | null> = {};
  for (const c of COMPARISON_COLUMNS) {
    const v = r[c.key];
    out[c.header] = typeof v === "number" ? v : v == null ? null : String(v);
  }
  return out;
}

/**
 * retrieval 四种取值语义互斥，必须让用户一眼看出"这还算不算单篇深挖"：
 * fallback_db 已经不是单篇，必须标红级提示（Tag amber），否则用户会以为答案出自这篇。
 */
const RETRIEVAL_VIEW: Record<string, { label: string; tone: "green" | "amber" | "red" | "muted"; hint: string }> = {
  targeted: {
    label: "targeted · 只在这篇的正文里答",
    tone: "green",
    hint: "检索范围锁定本篇已入库片段，是真正的单篇深挖。",
  },
  fallback_db: {
    label: "fallback_db · 已退回全库检索",
    tone: "amber",
    hint: "注意：本篇内容没有命中检索，后端退回了整个语料库 —— 这已经不是「单篇深挖」，答案可能出自其它论文。",
  },
  no_chunks: {
    label: "no_chunks · 这篇没有正文",
    tone: "red",
    hint: "库里没有这篇的任何正文片段（可能只抓到了元数据），没有可依据的内容。",
  },
  llm_failed: {
    label: "llm_failed · 模型调用失败",
    tone: "red",
    hint: "LLM 调用失败或没按格式产出，后端选择不编答案，答案栏那句话是原文照抄。",
  },
};

/** 详情接口的引文定位键是 section，列表/Citation 契约里叫 locator —— 两边都兼容，取有值的那个 */
function citationSection(c: Citation): string | undefined {
  const section = (c as Citation & { section?: string | null }).section;
  return c.locator ?? section ?? undefined;
}

const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v : null);
const num = (v: unknown): number | null => (typeof v === "number" && !Number.isNaN(v) ? v : null);

/**
 * 真实模式对比行：只从 GET /papers/{id} 的真实字段取数。
 * 核心思路/数据集/提升/代价/局限 只有研究运行精读落库的 comparison_rows 才有；
 * 没有就留 null（ComparisonTable 会渲染成「—」），绝不拿演示数据顶替。
 */
function detailToComparisonRow(d: PaperDetail): ComparisonRow {
  const raw = (d as PaperDetail & { comparison_rows?: unknown }).comparison_rows;
  const deep = (Array.isArray(raw) && raw.length ? (raw[raw.length - 1] as Record<string, unknown>) : {}) as Record<
    string,
    unknown
  >;
  const codeCol =
    str(deep.code_url) ??
    (d.has_code === true
      ? "接口确认有仓库链接（未返回具体 URL）"
      : d.has_code === false
        ? "未见仓库链接"
        : null);
  return {
    method_name: str(deep.method_name) ?? d.title,
    core_idea: str(deep.core_idea),
    datasets: str(deep.datasets),
    improvement: str(deep.improvement),
    cost: str(deep.cost),
    limitation: str(deep.limitation),
    code_url: codeCol,
    venue: str(deep.venue) ?? str(d.venue),
    cited_by_count: num(deep.cited_by_count) ?? num(d.cited_by_count),
  };
}

/**
 * 后端的 4xx/5xx `detail` 是 `{code, reason, hint?}` dict。
 * 新链路里 client.ts 的 `detailToText` 已把对象收成一句中文（`reason（hint）`），
 * detail 本来就是 JSON 字符串的旧链路仍要在这里解回 reason/hint 分开渲染。
 * 上传和详情加载共用这一份解析 —— 分码只按 ApiError.status，不匹配文案字符串。
 */
function extractReasonHint(message: string): { reason?: string; hint?: string } {
  try {
    const d = JSON.parse(message) as { reason?: unknown; hint?: unknown };
    return {
      reason: str(d?.reason) ?? undefined,
      hint: str(d?.hint) ?? undefined,
    };
  } catch {
    return { reason: message || undefined }; // detail 本来就是字符串的端点：整句当原因显示
  }
}

interface Issue {
  lead: string;
  reason?: string;
  hint?: string;
}

function toUploadIssue(e: unknown): Issue {
  if (e instanceof ApiError) {
    const { reason, hint } = extractReasonHint(e.message);
    switch (e.status) {
      case 415:
        return {
          lead: "415 · 格式不支持：后端只接受 .pdf / .txt / .md，请换一个文件重新选择。",
          reason,
          hint,
        };
      case 422:
        return {
          lead: "422 · 文档没能入库：内容为空 / 无法抽取正文（含加密 PDF）/ 超过大小上限。请按后端给出的具体原因处理。",
          reason,
          hint,
        };
      case 501:
        return {
          lead: '501 · 服务器没装 PDF 解析器（pypdf 可选依赖）。这不是故障，照抄下面原因里的安装命令（pip install -e ".[pdf]"）即可修复；不想装依赖也可以把正文另存为 .txt / .md 走零依赖通道重传。',
          reason,
          hint,
        };
      case 400:
        return { lead: "400 · 入库失败（其它原因），后端给出的原始原因如下。", reason, hint };
      default:
        return { lead: `请求失败（HTTP ${e.status}）。`, reason, hint };
    }
  }
  return { lead: e instanceof Error ? `上传失败：${e.message}` : "上传失败" };
}

/**
 * 详情加载失败的分码文案 —— 与 toUploadIssue 同一口径（按 status 分码 + 共用 extractReasonHint），
 * 不再往页面里塞第二套解析。导出仅供纯逻辑用例覆盖。
 *
 * tone 只给详情面板用：404 是 warn（琥珀级提示，不是故障），后端 5xx / 连不上才是 error（红）。
 * 呈现位置的取舍见渲染处注释。
 */
export type DetailIssueTone = "warn" | "error";

export interface DetailIssue extends Issue {
  tone: DetailIssueTone;
}

export function toDetailIssue(e: unknown, arxivId: string): DetailIssue {
  if (e instanceof ApiError) {
    const { reason, hint } = extractReasonHint(e.message);
    if (e.status === 404) {
      return {
        lead: `404 · 库里没有 id 为 ${arxivId} 的论文 —— 大概率是 id 写错了，或者这篇还没入库。`,
        reason,
        hint,
        tone: "warn",
      };
    }
    return {
      lead: `详情请求失败（HTTP ${e.status}），后端给出的原始原因如下。`,
      reason,
      hint,
      tone: e.status >= 500 ? "error" : "warn",
    };
  }
  const msg = e instanceof Error ? e.message : "";
  // fetch 连不上后端时 reject 的是 TypeError（"Failed to fetch" 等），没有 status ——
  // 这是服务没起 / 网络断了，不是"这篇论文不存在"，措辞必须分开，否则用户会去白找论文。
  if (e instanceof TypeError || /failed to fetch|networkerror|err_network|net::/i.test(msg)) {
    return {
      lead: "连不上后端：网络请求失败（不是这篇论文不存在）。请确认后端服务已启动后再重试。",
      tone: "error",
    };
  }
  return { lead: "详情加载失败（未识别的异常）。", reason: msg || undefined, tone: "error" };
}

type ListState = "loading" | "ready" | "error";

/**
 * 论文列表每页几条。
 *
 * 定这个数不是为了"看起来清爽"，而是为了**和右栏对齐**：右栏（单篇深挖 + 定向对比）
 * 正常高度在 600~800px，左栏 32 条一次铺开是 2500px 以上，于是右栏下面出现一大片空白，
 * 而且滚轮滚到列表里就再也滚不动页面。8 条 ≈ 640px，两栏基本同高。
 */
const PAPERS_PAGE_SIZE = 8;

export function PapersPage() {
  const [papers, setPapers] = useState<PaperDetail[]>([]);
  const [total, setTotal] = useState(0);
  const [listState, setListState] = useState<ListState>("loading");
  const [listError, setListError] = useState("");
  const [listPage, setListPage] = useState(1);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const listPages = Math.max(1, Math.ceil(papers.length / PAPERS_PAGE_SIZE));
  const safeListPage = Math.min(Math.max(listPage, 1), listPages);
  const pagePapers = papers.slice((safeListPage - 1) * PAPERS_PAGE_SIZE, safeListPage * PAPERS_PAGE_SIZE);
  // 列表内容换了（搜索、筛选、重新加载、刚入库一篇）就把页码带回第一页：
  // 停在第 4 页看一份只剩 6 条的结果集，界面会显示成"什么都没有"
  useEffect(() => {
    setListPage(1);
  }, [papers]);
  const [detail, setDetail] = useState<PaperDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailIssue, setDetailIssue] = useState<DetailIssue | null>(null);
  // "重试这篇"要让同一个 id 的 effect 重新跑一遍：setSelectedId(同一个值) 不会触发，得靠这个计数器进依赖
  const [detailReloadTick, setDetailReloadTick] = useState(0);

  const [askQ, setAskQ] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [ask, setAsk] = useState<AskResult | null>(null);
  const [askError, setAskError] = useState("");

  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [showCompare, setShowCompare] = useState(false);
  const [compareDetails, setCompareDetails] = useState<Record<string, PaperDetail>>({});
  const [compareLoading, setCompareLoading] = useState(false);

  const fileRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState("");
  const [titleHint, setTitleHint] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<{ arxiv_id: string; chunks: number } | null>(null);
  const [uploadIssue, setUploadIssue] = useState<Issue | null>(null);

  const [refsBuilding, setRefsBuilding] = useState(false);
  const [refsBatches, setRefsBatches] = useState(0);
  const [refsResult, setRefsResult] = useState<BuildEdgesResult | null>(null);
  const [refsError, setRefsError] = useState("");

  const loadList = useCallback(() => {
    setListState("loading");
    api.papers
      .list({ limit: 50 })
      .then((res) => {
        setPapers(res.items);
        setTotal(res.total);
        setListState("ready");
      })
      .catch((e: unknown) => {
        setListError(
          e instanceof Error
            ? `列表加载失败：${e.message}`
            : "列表加载失败",
        );
        setListState("error");
      });
  }, []);

  useEffect(() => {
    loadList();
  }, [loadList]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setDetailIssue(null);
      return;
    }
    let alive = true;
    setDetailLoading(true);
    setDetailIssue(null);
    setAsk(null);
    setAskError("");
    const id = selectedId;
    api.papers
      .get(id)
      .then((d) => alive && setDetail(d))
      // 之前这里 `.catch(() => setDetail(null))` 把错误整个吞掉：404 的"论文不在库中"
      // 变成了界面上的一片空白，用户无从知道是 id 写错还是没入库。现在如实分码显示。
      .catch((e: unknown) => {
        if (!alive) return;
        setDetail(null);
        setDetailIssue(toDetailIssue(e, id));
      })
      .finally(() => alive && setDetailLoading(false));
    return () => {
      alive = false;
    };
  }, [selectedId, detailReloadTick]);

  const toggleCompare = (id: string) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 3) return prev; // 定向对比最多 3 篇
      return [...prev, id];
    });
  };

  const runAsk = () => {
    if (!selectedId || !askQ.trim()) return;
    setAskLoading(true);
    setAskError("");
    api.papers
      .ask(selectedId, askQ.trim())
      .then((r) => {
        // supported=false 时 answer 里那句"内容不足以回答"是后端原文，原样显示，不补任何推测
        setAsk({ ...r, citations: Array.isArray(r.citations) ? r.citations : [] });
      })
      .catch((e: unknown) => setAskError(e instanceof Error ? e.message : "追问失败"))
      .finally(() => setAskLoading(false));
  };

  const submitUpload = () => {
    const file = fileRef.current?.files?.[0];
    if (!file) {
      setUploadIssue({ lead: "请先选择一个文件（支持 .pdf / .txt / .md）" });
      return;
    }
    setUploading(true);
    setUploadIssue(null);
    setUploadResult(null);
    api.papers
      .upload(file, titleHint.trim() || undefined)
      .then((r) => setUploadResult(r))
      .catch((e: unknown) => setUploadIssue(toUploadIssue(e)))
      .finally(() => setUploading(false));
  };

  /**
   * F16 抓引用边。后端一次最多 50 篇（MAX_IDS_PER_REQUEST），串行限速打 OpenAlex，
   * 24 篇约 30 秒 —— 分批必须**串行**跑（reduce 链），并发会撞对方限速。
   */
  const runBuildRefs = () => {
    if (refsBuilding) return;
    const ids = papers.map((p) => p.arxiv_id);
    if (!ids.length) return;
    const batches: string[][] = [];
    for (let i = 0; i < ids.length; i += 50) batches.push(ids.slice(i, i + 50));
    setRefsBuilding(true);
    setRefsBatches(batches.length);
    setRefsError("");
    setRefsResult(null);
    const agg: BuildEdgesResult = {
      requested: 0,
      works_found: 0,
      edges_new: 0,
      edges_known: 0,
      refs_unresolved: 0,
      errors: 0,
    };
    batches
      .reduce(
        (chain, batch) =>
          chain.then(() =>
            api.papers.buildReferences(batch).then((r) => {
              agg.requested += r.requested;
              agg.works_found += r.works_found;
              agg.edges_new += r.edges_new;
              agg.edges_known += r.edges_known;
              agg.refs_unresolved += r.refs_unresolved;
              agg.errors += num(r.errors) ?? 0;
            }),
          ),
        Promise.resolve(),
      )
      .then(() => setRefsResult({ ...agg }))
      .catch((e: unknown) => setRefsError(e instanceof Error ? e.message : "抓取引用关系失败"))
      .finally(() => setRefsBuilding(false));
  };

  // 真实模式的对比数据按需拉详情：mock 模式没有 GET /papers/{id} 的真九列，沿用演示数据。
  useEffect(() => {
    if (api.useMock || !showCompare || compareIds.length === 0) return;
    const missing = compareIds.filter((id) => !compareDetails[id]);
    if (!missing.length) return;
    let alive = true;
    setCompareLoading(true);
    Promise.all(
      missing.map((id) =>
        api.papers
          .get(id)
          .then((d) => [id, d] as const)
          .catch(() => null),
      ),
    )
      .then((pairs) => {
        if (!alive) return;
        setCompareDetails((prev) => ({
          ...prev,
          ...Object.fromEntries(pairs.filter((p): p is readonly [string, PaperDetail] => p !== null)),
        }));
      })
      .finally(() => alive && setCompareLoading(false));
    return () => {
      alive = false;
    };
  }, [showCompare, compareIds, compareDetails]);

  const compareRows = useMemo(
    () =>
      compareIds
        .map((id) => {
          if (api.useMock) return PAPER_COMPARISON[id] ?? { method_name: TITLES[id] ?? id };
          // 真实行：详情优先，列表行兜底；都拿不到就只有 id，其余列留空（九列口径：缺失留空不编造）
          const d = compareDetails[id] ?? papers.find((p) => p.arxiv_id === id);
          return d ? detailToComparisonRow(d) : { method_name: id };
        })
        .map(rowToTable),
    [compareIds, compareDetails, papers],
  );

  const citedAts = useMemo(
    () =>
      Array.from(
        new Set(
          compareIds
            .map((id) => str((compareDetails[id] as PaperDetail & { cited_at?: unknown })?.cited_at))
            .filter((v): v is string => v !== null),
        ),
      ),
    [compareIds, compareDetails],
  );

  const selectedPaper = papers.find((p) => p.arxiv_id === selectedId) ?? detail;

  const retrievalView = ask
    ? RETRIEVAL_VIEW[ask.retrieval] ?? {
        label: `retrieval=${String(ask.retrieval)}（后端返回了未预置的值）`,
        tone: "muted" as const,
        hint: "",
      }
    : null;

  // ⚠️ 口径别混：GET /papers/{id} 返回 chunks **数组**（这里数长度），
  // GET /papers 列表返回的才是 chunk_count **数字**。
  const detailChunks = (detail as (PaperDetail & { chunks?: unknown }) | null)?.chunks;
  const detailChunkCount = Array.isArray(detailChunks) ? detailChunks.length : null;

  // ask 的 context_chunks 真实后端是**数字**（进入上下文的片段数）；domain 里标成数组是契约瑕疵，两边兼容。
  const ccRaw = ask?.context_chunks as unknown;
  const ccCount = typeof ccRaw === "number" ? ccRaw : Array.isArray(ccRaw) ? ccRaw.length : null;

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="论文库"
        description="单篇深挖问答 · 定向九列对比 · 上传自己的 PDF/TXT/MD · 抓引用关系喂时间线 —— 数据一律经 api.client，缺接口的地方如实报错"
        features="F13 · F14 · F15"
        actions={
          api.useMock ? <Tag variant="amber">MOCK 数据 · 未连接后端</Tag> : <Tag variant="green">已连接后端</Tag>
        }
      />

      <div className="grid grid-cols-1 items-start gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
        {/* 左：列表 + 上传 */}
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="论文列表"
            subtitle={`共 ${total} 篇 · 第 ${safeListPage}/${listPages} 页 · 本页 ${pagePapers.length} 篇 · 勾选 2~3 篇可定向对比`}
            extra={
              <div className="flex items-center gap-2">
                {compareIds.length >= 2 ? (
                  <NeonButton variant="secondary" className="px-3 py-1 text-xs" onClick={() => setShowCompare(true)}>
                    对比选中 ({compareIds.length})
                  </NeonButton>
                ) : null}
                <NeonButton
                  variant="secondary"
                  className="px-3 py-1 text-xs"
                  loading={refsBuilding}
                  disabled={refsBuilding || papers.length === 0}
                  onClick={runBuildRefs}
                  title="向 OpenAlex 逐篇抓 referenced_works 落库；不跑这一步，F12 时间线的 addresses_gap 永远是空的"
                >
                  抓取引用关系
                </NeonButton>
              </div>
            }
          >
            {listState === "loading" ? (
              <div className="flex flex-col gap-3">
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} height={64} />
                ))}
              </div>
            ) : listState === "error" ? (
              <EmptyState
                icon="circle-alert"
                title="暂时取不到论文列表"
                description={listError}
                action={
                  <NeonButton variant="secondary" onClick={loadList}>
                    重试
                  </NeonButton>
                }
              />
            ) : papers.length === 0 ? (
              <EmptyState title="库里还没有论文" description="上传一篇 PDF，或先跑一次研究把论文收进来。" />
            ) : (
              <>
              <ul className="flex flex-col gap-2">
                {pagePapers.map((p) => {
                  const active = p.arxiv_id === selectedId;
                  const tier = p.venue_tier?.tier;
                  return (
                    <li
                      key={p.arxiv_id}
                      className={`rounded-lg border px-3 py-2.5 transition-colors ${
                        active
                          ? "border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)]"
                          : "border-hairline hover:bg-white/[0.03]"
                      }`}
                    >
                      <div className="flex items-start gap-2.5">
                        <input
                          type="checkbox"
                          aria-label={`选中 ${p.title}`}
                          checked={compareIds.includes(p.arxiv_id)}
                          disabled={!compareIds.includes(p.arxiv_id) && compareIds.length >= 3}
                          onChange={() => toggleCompare(p.arxiv_id)}
                          className="mt-1 h-3.5 w-3.5 shrink-0 accent-[var(--accent-cyan)]"
                        />
                        <button
                          type="button"
                          onClick={() => setSelectedId(p.arxiv_id)}
                          className="min-w-0 flex-1 text-left"
                        >
                          <p className="line-clamp-2 text-sm font-medium text-ink" title={p.title}>
                            {p.title}
                          </p>
                          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                            <span className="font-mono text-[11px] text-neon-cyan/80">{p.arxiv_id}</span>
                            <span className="font-mono text-[11px] text-ink-faint">{yearOf(p)}</span>
                            {(p.categories ?? []).slice(0, 2).map((c) => (
                              <Tag key={c} variant="muted" mono>
                                {c}
                              </Tag>
                            ))}
                            {(() => {
                              // 真实档位是 top|core|preprint|unknown（见 venue_tiers.lookup），
                              // 不是字母档；preprint/unknown 归到"未分档"而不是给它染上顶会的青色。
                              const label = p.venue ? `${p.venue}${p.venue_tier?.label ? ` · ${p.venue_tier.label}` : ""}` : "未分档";
                              const strong = tier === "top" || tier === "core";
                              return (
                                <Tag variant={tier === "top" ? "cyan" : tier === "core" ? "purple" : "muted"}>
                                  {strong ? label : "未分档"}
                                </Tag>
                              );
                            })()}
                          </div>
                        </button>
                      </div>
                    </li>
                  );
                })}
              </ul>
              <Pager
                page={safeListPage}
                pages={listPages}
                onChange={setListPage}
                ariaLabel="论文列表分页"
                className="mt-3 border-t border-hairline pt-3"
              />
              </>
            )}
            {(refsBuilding || refsResult || refsError) && (
              <div className="mt-3 flex flex-col gap-2 rounded-lg border border-hairline bg-white/[0.03] px-3 py-2.5">
                {refsBuilding && (
                  <p className="text-xs text-ink-soft">
                    正在逐篇抓 referenced_works{refsBatches > 1 ? `（共 ${refsBatches} 批，串行）` : ""}…
                    这是限速的外部调用（24 篇约 30 秒），请勿重复点击。
                  </p>
                )}
                {refsError && <p className="text-xs text-neon-red">抓取引用关系失败：{refsError}</p>}
                {refsResult && (
                  <>
                    <p className="text-xs font-medium text-ink">
                      引用关系抓取完成{refsBatches > 1 ? `（${refsBatches} 批合计）` : ""}：
                    </p>
                    <ul className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] text-ink-soft sm:grid-cols-3">
                      <li>请求 {refsResult.requested} 篇</li>
                      <li>OpenAlex 收录命中 {refsResult.works_found} 篇</li>
                      <li>新增引用边 {refsResult.edges_new}</li>
                      <li>库中已存在的边 {refsResult.edges_known}</li>
                      <li>参考文献未解析回 arXiv id：{refsResult.refs_unresolved} 条</li>
                      <li>单篇抓取失败 {refsResult.errors} 篇</li>
                    </ul>
                    <p className="text-[11px] leading-relaxed text-ink-faint">
                      {refsResult.requested < total
                      ? `本批只抓了已载入的 ${refsResult.requested} 篇（语料共 ${total} 篇）—— 界面一次只从接口取 50 篇，剩下的要等载入后再跑一批，和下面的分页条无关。`
                      : ""}
                      诚实提示：works_found 远小于 requested 多半是因为 OpenAlex
                      <b>还没收录这么新的 arXiv 论文</b>，不是抓取失败。抓回来的边是 F12 时间线
                      addresses_gap（&quot;谁补了谁的坑&quot;）的唯一数据来源 —— 不跑这一步，时间线永远没有引用证据。
                    </p>
                  </>
                )}
              </div>
            )}
          </Panel>

          <Panel title="上传入库 (F15)" subtitle="选择本地 PDF / TXT / MD，解析切块后写入语料库">
            <div className="flex flex-col gap-3">
              <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-dashed border-hairline px-4 py-4 text-sm text-ink-soft transition-colors hover:border-[color:var(--accent-cyan-line)] hover:bg-white/[0.02]">
                <UploadCloud size={18} className="shrink-0 text-neon-cyan" />
                <span className="min-w-0 flex-1 truncate">
                  {fileName || "点击选择文件（.pdf / .txt / .md）"}
                </span>
                <input
                  ref={fileRef}
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
                  className="hidden"
                  onChange={(e) => {
                    setFileName(e.target.files?.[0]?.name ?? "");
                    setUploadIssue(null);
                    setUploadResult(null);
                  }}
                />
              </label>
              <div className="flex flex-col gap-1">
                <label className="text-xs text-ink-soft" htmlFor="title-hint">
                  标题提示（可选）—— 文件首页标题抽不准时用它兜底
                </label>
                <input
                  id="title-hint"
                  value={titleHint}
                  onChange={(e) => setTitleHint(e.target.value)}
                  placeholder="例如：My Chunking Notes（最多 300 字符）"
                  maxLength={300}
                  className="rounded-lg border border-hairline bg-base/60 px-3 py-2 text-sm text-ink outline-none transition-[border-color,box-shadow] focus:border-neon-cyan/60 focus:shadow-[0_0_0_3px_rgba(0,240,255,0.08)]"
                />
              </div>
              <div className="flex items-center gap-2">
                <NeonButton loading={uploading} disabled={!fileName} onClick={submitUpload}>
                  提交入库
                </NeonButton>
                {uploadResult && (
                  <span className="font-mono text-xs text-neon-green">
                    已入库 {uploadResult.arxiv_id} · {uploadResult.chunks} chunks
                  </span>
                )}
              </div>
              {uploadIssue && (
                <div className="flex flex-col gap-1.5 rounded-lg border border-neon-red/40 bg-white/[0.03] px-3 py-2">
                  <p className="text-xs font-medium leading-relaxed text-neon-red">{uploadIssue.lead}</p>
                  {uploadIssue.reason && (
                    <p className="select-all break-words font-mono text-[11px] leading-relaxed text-ink-soft">
                      后端原因：{uploadIssue.reason}
                    </p>
                  )}
                  {uploadIssue.hint && <p className="text-[11px] leading-relaxed text-ink-faint">{uploadIssue.hint}</p>}
                </div>
              )}
              {uploadResult && (
                <div className="rounded-lg border border-neon-amber/40 bg-white/[0.04] px-3 py-2 text-xs leading-relaxed text-neon-amber">
                  注意：新入库的论文需要<b>重建索引</b>后才会被检索到 —— 不是传完立刻生效，否则检索命中的仍是旧语料。
                </div>
              )}
            </div>
          </Panel>
        </div>

        {/* 右：单篇深挖 (F13) + 定向对比 (F14) */}
        {/* sticky：左栏列表比右栏高时，右栏跟着滚到视口顶就停住 ——
            否则读完一篇想追问，得先滚回一大片空白下面找那个面板 */}
        <div className="flex min-w-0 flex-col gap-4 lg:sticky lg:top-4">
          <Panel title="单篇深挖 (F13)" subtitle={selectedPaper ? selectedPaper.title : "从左侧选择一篇论文"}>
            {!selectedId ? (
              <EmptyState icon="file-search" title="未选择论文" description="点击左侧任一篇论文，查看摘要并就该篇追问。" />
            ) : detailLoading ? (
              <div className="flex flex-col gap-3">
                <Skeleton height={20} width="60%" />
                <Skeleton height={80} />
                <Skeleton height={40} />
              </div>
            ) : detailIssue ? (
              /*
               * 呈现位置的取舍：错误只出现在详情面板这一块，不给整页盖红色大告警 ——
               * 404（id 写错 / 没入库）在浏览论文库时是常态，用户随时可以点列表里别的论文，
               * 所以列表不受影响、面板内给琥珀级提示 + 一个明确出口（回到列表挑一篇）。
               * 后端故障（5xx / 连不上）才升级成红色，但同样只在面板内，附重试入口。
               */
              <div
                className={`flex flex-col gap-1.5 rounded-lg border bg-white/[0.03] px-3 py-2.5 ${
                  detailIssue.tone === "error" ? "border-neon-red/40" : "border-neon-amber/40"
                }`}
              >
                <p
                  className={`text-xs font-medium leading-relaxed ${
                    detailIssue.tone === "error" ? "text-neon-red" : "text-neon-amber"
                  }`}
                >
                  {detailIssue.lead}
                </p>
                {detailIssue.reason && (
                  <p className="select-all break-words font-mono text-[11px] leading-relaxed text-ink-soft">
                    后端原因：{detailIssue.reason}
                  </p>
                )}
                {detailIssue.hint && <p className="text-[11px] leading-relaxed text-ink-faint">{detailIssue.hint}</p>}
                <div className="mt-1 flex gap-2">
                  <NeonButton variant="secondary" className="px-3 py-1 text-xs" onClick={() => setSelectedId(null)}>
                    回论文库列表挑一篇
                  </NeonButton>
                  <NeonButton variant="secondary" className="px-3 py-1 text-xs" onClick={() => setDetailReloadTick((t) => t + 1)}>
                    重试这篇
                  </NeonButton>
                </div>
              </div>
            ) : !detail ? (
              <EmptyState icon="circle-alert" title="详情加载失败" description="后端未返回该篇详情。" />
            ) : (
              <FadeIn className="flex flex-col gap-4">
                <div className="flex flex-col gap-2">
                  <span className="font-mono text-xs text-neon-cyan">{detail.arxiv_id}</span>
                  <p className="text-sm font-semibold leading-relaxed text-ink">{detail.title}</p>
                  {detail.abstract ? (
                    <p className="text-xs leading-relaxed text-ink-soft">{detail.abstract}</p>
                  ) : null}
                  {detail.authors?.length ? (
                    <p className="text-xs text-ink-faint">作者：{detail.authors.join("、")}</p>
                  ) : null}
                  {detailChunkCount !== null && (
                    <p className="text-xs text-ink-faint">
                      正文片段 {detailChunkCount} 段
                      {detailChunkCount === 0 ? " —— 本篇没有正文，单篇追问只会得到「不足以回答」" : ""}
                    </p>
                  )}
                </div>

                <div className="flex flex-col gap-2">
                  <label className="text-xs font-medium text-ink-soft" htmlFor="ask-input">
                    就该篇追问
                  </label>
                  <div className="flex gap-2">
                    <input
                      id="ask-input"
                      value={askQ}
                      onChange={(e) => setAskQ(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && runAsk()}
                      placeholder="例如：它的反思 token 是怎么训练的？"
                      className="min-w-0 flex-1 rounded-lg border border-hairline bg-base/60 px-3 py-2 text-sm text-ink outline-none transition-[border-color,box-shadow] focus:border-neon-cyan/60 focus:shadow-[0_0_0_3px_rgba(0,240,255,0.08)]"
                    />
                    <NeonButton loading={askLoading} disabled={!askQ.trim()} onClick={runAsk}>
                      提问
                    </NeonButton>
                  </div>
                  {askError && <p className="text-xs text-neon-red">{askError}</p>}
                </div>

                {ask ? (
                  <div className="flex flex-col gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <Tag variant={retrievalView!.tone} mono>
                        {retrievalView!.label}
                      </Tag>
                      {!ask.supported && (
                        <Tag variant="red" mono>
                          supported=false
                        </Tag>
                      )}
                      {ccCount !== null && (
                        <span className="font-mono text-[11px] text-ink-faint">上下文片段 {ccCount}</span>
                      )}
                    </div>
                    {retrievalView!.hint && (
                      <p className="text-[11px] leading-relaxed text-ink-faint">{retrievalView!.hint}</p>
                    )}
                    {/* supported=false 时 answer 本身就是后端那句"内容不足以回答"，原样显示，不补推测 */}
                    <p
                      className={`rounded-lg border px-3 py-2.5 text-sm leading-relaxed text-ink ${
                        ask.supported ? "border-hairline bg-white/[0.02]" : "border-neon-red/40 bg-white/[0.03]"
                      }`}
                    >
                      {ask.answer}
                    </p>
                    {ask.note && (
                      <p className="rounded-lg border border-hairline bg-white/[0.03] px-3 py-2 text-[11px] leading-relaxed text-ink-soft">
                        后端附注：{ask.note}
                      </p>
                    )}
                    {ask.citations.length > 0 && (
                      <div className="flex flex-col gap-2">
                        <p className="font-mono text-[11px] text-ink-faint">引用证据（{ask.citations.length}）</p>
                        {ask.citations.map((c, i) => (
                          <div key={`${c.chunk_id ?? "c"}-${i}`} className="flex flex-col gap-1">
                            <CitationCard
                              claim={c.claim ?? ""}
                              // ★ verified 三态：CitationCard 的 supported 只有布尔一档，
                              // 只有明确 false 才允许渲染成"无支撑"；null/缺键（单篇问答接口的引文
                              // 本就只放逐字比对通过的，不带 verified 键）绝不能归进 false。
                              supported={c.verified !== false}
                              arxiv_id={c.arxiv_id}
                              section={citationSection(c)}
                              chunk_text={c.quote ?? undefined}
                              resolution_how={c.resolution_how ?? null}
                            />
                            {c.verified !== true && c.verified !== false && (
                              <p className="pl-1 text-[10px] leading-relaxed text-ink-faint">
                                该条未记录 verified 标记（= 当时没校验）—— 这不是「无支撑」。
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ) : null}
              </FadeIn>
            )}
          </Panel>

          {showCompare && (
            <Panel
              title="定向对比 (F14)"
              subtitle={`九列口径 · 已选 ${compareIds.length} 篇${api.useMock ? " · MOCK 演示数据" : ""}`}
              extra={
                <NeonButton variant="secondary" className="px-3 py-1 text-xs" onClick={() => setShowCompare(false)}>
                  收起
                </NeonButton>
              }
            >
              {compareLoading && !api.useMock ? (
                <Skeleton height={120} />
              ) : (
                <ComparisonTable
                  columns={COMPARISON_COLUMNS.map((c) => c.header)}
                  rows={compareRows}
                  rowKey="方法"
                />
              )}
              {!api.useMock && (
                <p className="mt-2 text-[11px] leading-relaxed text-ink-faint">
                  真实模式：每行取自 GET /papers/{"{id}"} 的真实字段（标题 / 发表会议 / 被引数{citedAts.length ? `，截至 ${citedAts.join("、")}` : ""} /
                  有无仓库链接）。「核心思路、数据集、提升、代价、局限」只有研究运行精读后落库的
                  comparison_rows 才有 —— 该篇没有历史精读记录时这些列留空显示「—」，不会拿演示数据顶替。
                </p>
              )}
            </Panel>
          )}
        </div>
      </div>
    </div>
  );
}

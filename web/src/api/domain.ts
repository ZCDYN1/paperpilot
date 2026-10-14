/**
 * 后端数据契约的 TS 镜像。字段名与 `app/api/v1/*.py` 的返回体逐字对齐 ——
 * 前端不做 rename，改名只发生在展示层，这样接口出错时能一眼看出是哪一层的问题。
 */

/** F4 九列对比表的对外口径，顺序即列顺序（源：`app/export/markdown.py` 的 `_TABLE_COLUMNS`） */
export const COMPARISON_COLUMNS = [
  { header: "方法", key: "method_name" },
  { header: "核心思路", key: "core_idea" },
  { header: "数据集", key: "datasets" },
  { header: "相对 baseline 提升", key: "improvement" },
  { header: "代价", key: "cost" },
  { header: "局限（原文自述）", key: "limitation" },
  { header: "代码", key: "code_url" },
  { header: "发表会议", key: "venue" },
  { header: "被引", key: "cited_by_count" },
] as const;

export type ComparisonRow = Partial<Record<(typeof COMPARISON_COLUMNS)[number]["key"], string | number | null>>;

/** F5 引用证据链（源：`GET /api/v1/research/{run_id}/citations`） */
export interface Citation {
  claim: string;
  arxiv_id: string;
  chunk_id: string | null;
  quote: string | null;
  locator: string | null;
  /** 证据强度：`html`=正文原文，`abstract`=摘要降级，`pdf`=上传件。F5 的核心承诺之一 */
  source: string | null;
  /**
   * ★ 三态，不是布尔（DB 列是 `Boolean | None`，见 `app/db/models.py`）。
   * `true`=有原文支撑；`false`=被判无支撑并已剔除；`null`=**当时根本没校验**。
   * 全产品口径是"只有 false 才算无支撑并剔除，null 保留"，
   * 把 null 归进 false 会让界面与导出的 .md、与 `verified_rate` 三者互相矛盾。
   */
  verified: boolean | null;
  note: string | null;
  /** exact | quote | swapped_chunk | verify_reanchored；迁移前的历史行为 null */
  resolution_how: string | null;
  verify_reanchored: boolean | null;
}

export interface CitationBundle {
  run_id: string;
  count: number;
  /** 引用准确率：verified=True 的占比；无引用时为 null（不是 0） */
  verified_rate: number | null;
  citations: Citation[];
}

/**
 * 子问题。字段与 `app/schemas/research.py` 的 `SubQuestion` 逐字对齐，
 * 且后端是 `extra="forbid"` —— 多带一个键就会被 422 拒掉，所以这里不能自作主张加字段。
 * `keywords` 必须是**英文**术语：BM25 是词面匹配，中文关键词会让召回接近随机。
 */
export interface SubQuestion {
  id: string;
  text: string;
  keywords: string[];
  category: string | null;
}

/**
 * F6 校验结果。字段照抄 `app/agents/nodes/verifier.py` 的 `verification` dict。
 *
 * ⚠️ 后端**没有** `total` / `supported` / `dropped` / `dropped_claims` 这几个键。
 * 被剔除数就是 `unsupported.length`，不要再拿 citations 二次统计 ——
 * `unsupported` 才是 Verifier 认定的那一份，两套口径迟早对不上。
 */
export interface Verification {
  /** 全部通过才算 passed；后端刻意不做"提高通过率"的宽松处理 */
  passed: boolean;
  /** 0~1 的通过率，展示时自己乘 100 */
  score: number;
  /** 被判定无支撑的论断原文 */
  unsupported: string[];
  checked: number;
  revision_count: number;
}

/** `GET /api/v1/research/{run_id}` */
/**
 * `research_runs.status`。
 *
 * `done` 与 `pending` 不是笔误：早期脚本跑出来的行写的是 `done`，
 * 而 `pending` 是列默认值（还没开跑）。前端必须认得它们，
 * 否则这 22 次历史运行在列表里会显示成空白状态。
 */
export type RunStatus =
  | "pending"
  | "running"
  | "awaiting_input"
  | "completed"
  | "done"
  | "failed"
  | "cancelled";

export interface RunDetail {
  run_id: string;
  session_id: string;
  question: string;
  status: RunStatus;
  plan_edited: boolean | null;
  stage: string | null;
  sub_questions: SubQuestion[];
  verification: Verification;
  revision_count: number;
  token_usage: Record<string, number>;
  token_by_node: TokenByNode;
  draft: string;
  citations: Citation[];
  comparison: ComparisonRow[];
  memory_context: Array<Record<string, unknown>>;
}

/** F19 会议/期刊分档。真实形状见 `app/analysis/venue_tiers.py` 的 `lookup()`。 */
export interface VenueTier {
  /** top | core | preprint | unknown —— 注意不是 A/B 字母档 */
  tier: string;
  label: string;
  evidence?: Record<string, unknown>;
}

export interface PaperDetail {
  arxiv_id: string;
  title: string;
  abstract?: string;
  authors?: string[];
  published?: string | null;
  primary_category?: string | null;
  categories?: string[];
  venue?: string | null;
  cited_by_count?: number | null;
  /** 被引数必须带截至日期，否则会被当成实时值来比较 */
  cited_at?: string | null;
  has_code?: boolean | null;
  chunk_count?: number;
  /** 分档是独立字段，不进 F4 的九列口径 */
  venue_tier?: VenueTier | null;
  [k: string]: unknown;
}

/** `GET /api/v1/papers` */
export interface PaperListResult {
  total: number;
  limit: number;
  offset: number;
  items: PaperDetail[];
}

/** `GET /api/v1/research` 的一行 */
export interface RunListItem {
  run_id: string;
  session_id: string;
  question: string;
  status: RunStatus;
  started_at: string | null;
  ended_at: string | null;
  citation_count: number;
  verified_rate: number | null;
}

export interface RunListResult {
  total: number;
  limit: number;
  offset: number;
  items: RunListItem[];
}

/**
 * F16 引用关系图。字段照抄 `app/references/edges.py` 的 `get_reference_graph()`：
 * 边叫 `edges`（不是 `links`），节点主键叫 `arxiv_id`（不是 `id`）。
 */
export interface GraphNode {
  arxiv_id: string;
  title: string;
  year: number | null;
  venue: string | null;
  cited_by_count: number | null;
  /** false = 库里没有这篇，点开它没有正文证据可溯源，F5 对它不成立 */
  in_corpus: boolean;
  /** true = 在请求的 ids 里，而不是只作为引用邻居出现 */
  focused: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  /** 引用边数据来源（Crossref / OpenAlex 等） */
  evidence: string | null;
}

export interface CitationGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  /** 请求里有多少个 id 被展开成了图 */
  papers: number;
}

/** F12 方法演进时间线的一条（`app/analysis/timeline.py`） */
export interface TimelineItem {
  arxiv_id: string;
  title: string;
  year: number | null;
  venue: string | null;
  venue_tier: string;
  venue_evidence: Record<string, unknown> | null;
  core_idea: string;
  limitation: string;
  cited_by_count: number | null;
  cited_at: string | null;
  /**
   * ⚠️ 只有 citation_edges 里有引用边证据时才非空。
   * 后端明确禁止用年份先后推断"谁补了谁的坑" —— 那是编故事，不是发现。
   */
  addresses_gap: Array<{ paper_id: string; title: string; edge_source: string }>;
}

/** `GET /api/v1/research/{run_id}/timeline` 的整体返回 */
export interface TimelineResult {
  entries: TimelineItem[];
  paper_ids: string[];
  /** false = 本次没取到任何引用边，`addresses_gap` 必然全空 */
  edge_evidence_available: boolean;
  notes: string;
  run_id: string;
}

/** F19 单篇被引趋势的一个采样点（`app/references/trend.py`） */
export interface TrendPoint {
  /** 采样日期 ISO，字段名是 `on` 不是 `date` */
  on: string;
  /** 字段名是 `cited_by_count`，不是 `citations` */
  cited_by_count: number;
}

/**
 * `GET /api/v1/papers/{arxiv_id}/citation-trend` 的整体返回。
 *
 * ★ `verdict === "insufficient_data"` 时必须原样显示 `note` 那句话，
 * 不要画一条从原点到今天的假斜率 —— 后端刻意不给结论，前端不能替它编。
 */
export interface TrendResult {
  arxiv_id: string;
  title: string;
  points: TrendPoint[];
  verdict: "rising" | "flat" | "falling" | "insufficient_data";
  delta: number | null;
  span_days: number | null;
  note?: string;
}

export interface OverviewStats {
  papers: number;
  chunks: number;
  runs: number;
  runs_completed: number;
  avg_verified_rate: number | null;
  total_tokens: number;
  by_day: Array<{ date: string; runs: number; tokens: number }>;
}

/** 单个熔断器快照（`app/core/resilience.py` 的 `breaker_snapshot()`） */
export interface BreakerState {
  state: "closed" | "half_open" | "open";
  failures: number;
  /** 字段名是 failure_threshold，不是 threshold —— 少一个词就会让仪表盘显示成 0/0 */
  failure_threshold: number;
  recovery_timeout: number;
  /** 被熔断直接拦掉的调用次数（没真打出去） */
  short_circuited: number;
  transitions: number;
  last_transition: string | null;
}

export interface HealthResponse {
  status: string;
  version: string;
  llm_provider: string;
  llm_model?: string;
  embedding_provider?: string;
  embedding_model?: string;
  mock_mode?: boolean;
  llm_configured?: boolean;
  rerank_enabled?: boolean;
  input_guard_enabled?: boolean;
  tracing_enabled?: boolean;
  embedding?: { ready: boolean; provider?: string; hint?: string | null };
  index?: { ready: boolean; compatible?: boolean; papers?: number; chunks?: number; hint?: string | null };
  /** 熔断器快照挂在 resilience 下，不是顶层 */
  resilience?: { enabled: boolean; input_guard?: boolean; breakers?: Record<string, BreakerState> };
  [k: string]: unknown;
}

/** `GET /resilience` —— 比 /healthz 更窄、适合 watch 的运维视图 */
export interface ResilienceSnapshot {
  enabled: boolean;
  timeout_budget: number;
  attempts: number;
  breakers: Record<string, BreakerState>;
}

/* ============================================================ 请求体
   顶层与 filters 都是 `extra="forbid"`（`app/schemas/research.py`）——
   多带任何一个键就整请求 422，而且是静默整批拒，所以这里一个字段都不要加。 */

export interface ResearchFilters {
  /** arXiv 分类白名单。空时要传 [] 或整个键不传，传 "" 会被白名单校验拒掉 */
  categories?: string[];
  /** 必须是 int | null，"2020" 这种字符串 pydantic 直接拒 */
  year_gte?: number | null;
  year_lte?: number | null;
  /**
   * ★ 三态：true=只召回确认有仓库链接的；false=只召回**未见**链接的；
   * null / 不传 = 不过滤。界面上"只看有代码"不勾时必须传 null，
   * 传 false 语义完全相反（那是"我要看没有代码的论文"）。
   */
  has_code?: boolean | null;
  /** F14 定向对比；会被并进 filters */
  arxiv_ids?: string[];
}

export interface ResearchRequest {
  question: string;
  session_id?: string;
  language?: "zh" | "en";
  /** 最终留几篇，不是每路召回量 */
  top_k?: number;
  rerank?: boolean;
  arxiv_ids?: string[];
  filters?: ResearchFilters;
}

/** `POST /api/v1/papers/{arxiv_id}/ask` */
export interface AskResult {
  arxiv_id: string;
  question: string;
  answer: string;
  supported: boolean;
  citations: Citation[];
  /** targeted=只在这篇上检索；fallback_db=退回全库；no_chunks=这篇没正文；llm_failed=模型调用失败 */
  retrieval: "targeted" | "fallback_db" | "no_chunks" | "llm_failed";
  /** 用作上下文的 chunk **数量**（实测是 int，不是数组）；正文列表在 `GET /papers/{id}` 里 */
  context_chunks: number;
  token_usage: Record<string, number>;
  note?: string | null;
}

/** `POST /api/v1/papers/references/build` */
export interface BuildEdgesResult {
  requested: number;
  works_found: number;
  edges_new: number;
  edges_known: number;
  refs_unresolved: number;
  errors: number;
}

/**
 * `token_usage` / `token_by_node` 里每个节点的用量。
 *
 * ⚠️ 后端给的是**对象**不是数字（`app/agents/nodes/finalize.py` 按节点各落一行）。
 * 曾把它声明成 `Record<string, number>`，于是成本图把所有节点都算成 NaN。
 * `number` 分支保留是为了兼容历史 checkpoint 里可能存在的裸数字。
 */
export interface NodeTokenUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  calls?: number;
}

export type TokenByNodeEntry = NodeTokenUsage | number;

export type TokenByNode = Record<string, TokenByNodeEntry>;

/** 取一个节点条目的总 token 数；两种历史形状都认，取不到返回 null（而不是 0） */
export function tokenByNodeTotal(entry: TokenByNodeEntry | undefined): number | null {
  if (typeof entry === "number") return Number.isFinite(entry) ? entry : null;
  if (entry && typeof entry === "object") {
    const v = entry.total_tokens ?? entry.prompt_tokens;
    return typeof v === "number" && Number.isFinite(v) ? v : null;
  }
  return null;
}

/** `POST /api/v1/papers/citations/snapshot` 的真实返回（`recorded` / `no_match`） */
export interface SnapshotResult {
  requested: number;
  /** 成功写入今天这一行的篇数 */
  recorded: number;
  /** OpenAlex 没匹配上、或被交叉校验挡掉的篇数 */
  no_match: number;
  errors: number;
}

/** `POST /api/v1/terms/unknown` */
export interface UnknownTermsResult {
  unknown: string[];
  [k: string]: unknown;
}

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

/* ============================================================ 配置中心
   源：`app/api/v1/config.py`。密钥**永不回传明文**，只回"是否已设置 + 掩码"。 */

/** 表单能改的字段。空字符串对密钥字段表示"保持原值不变"，不是"清空"。 */
export interface ConfigValues {
  llm_provider?: "openai" | "ollama" | "mock";
  llm_model?: string;
  llm_temperature?: number;
  /**
   * 结构化输出的实现方式。`auto` = 不干预，用各端点自己的默认。
   *
   * ★ 接 OpenAI 兼容网关（DeepSeek 等）时这个值决定**能不能跑通**：
   * 那些端点会对默认的 json_schema 方式回
   * `400 This response_format type is unavailable now`，
   * 症状是"每个结构化节点都失败、而普通对话一切正常"。
   */
  llm_structured_method?: "auto" | "function_calling" | "json_mode" | "json_schema";
  /** true = 结构化调用时关掉思考模式（思考模式端点不接受 function_calling 所需的 tool_choice） */
  llm_disable_thinking?: boolean;
  openai_base_url?: string;
  openai_api_key?: string;
  ollama_base_url?: string;
  ollama_model?: string;
  embedding_provider?: "local" | "api" | "mock";
  embedding_model?: string;
  embedding_api_base_url?: string;
  embedding_api_key?: string;
  rerank_provider?: "local" | "lexical";
  /** HF 仓库名或本地目录；权重缺失时后端不崩，会降级 lexical 并在日志给出获取命令 */
  rerank_model?: string;
  min_corpus?: number;
}

/** 密钥字段的展示形态：只给"设过了没有"和掩码，绝不给明文 */
export interface SecretState {
  set: boolean;
  /** 形如 `sk-…3f9a`；未设置时为空串 */
  masked: string;
}

/**
 * 表单能改的字段名。
 *
 * ★ 为什么把 `sources` / `restart_required` 的键收成这个类型而不是
 * `Record<string, ...>` / `string[]`：这个页面唯一的真相源就是字段名，
 * 而 mock 里的字段名是**手抄**的。之前 mock 的 `sources` 漏了
 * `embedding_api_base_url` / `embedding_api_key`，界面就少渲染两个来源标签，
 * 而 `Record<string, X>` 让这件事在编译期完全无声。收成联合类型后抄漏/抄错立刻红。
 */
export type ConfigField = keyof ConfigValues;

/** 这个字段当前生效的值是从哪一层来的（优先级：env > .env > 代码默认值） */
export type ConfigSource = "dotenv" | "env" | "default";

export interface ConfigResponse {
  values: ConfigValues;
  secrets: { openai_api_key: SecretState; embedding_api_key: SecretState };
  /** 每个可编辑字段的取值来源 —— 后端按 `EDITABLE` 全量返回，所以这里要求键齐全 */
  sources: Record<ConfigField, ConfigSource>;
  /** 后端是否允许写配置（关掉时界面只读并说明原因） */
  writable: boolean;
  /** 改了这些字段要重启才生效 —— 必须显示给用户，不能让他以为存了就完事 */
  restart_required: ConfigField[];
  /** 每个 provider 取值的合法集合，界面下拉照它渲染，不在前端硬编码 */
  options: {
    llm_provider: string[];
    llm_structured_method: string[];
    embedding_provider: string[];
    rerank_provider: string[];
  };
}

export interface ConfigWriteResult {
  ok: boolean;
  updated: ConfigField[];
  /** 需要重启才生效的字段 */
  restart_required: ConfigField[];
  /** 写入前自动备份的文件名，界面上要显示 —— 用户要知道改坏了能退回去 */
  backup: string | null;
  /** 被拒绝的字段及原因（如 provider 取值不在白名单） */
  rejected: Array<{ field: string; reason: string }>;
  /**
   * 已写入 `.env`、但**当前进程里被同名环境变量压住所以不生效**的字段。
   *
   * ★ 必须显示：这是"保存成功但值没变"的唯一解释。`sources` 只在 GET 里给，
   * 而保存后界面展示的是这份返回体，少了它就只能让用户自己去猜。
   */
  shadowed: ConfigField[];
}

/**
 * 连通性探测的请求体（`POST /api/v1/config/test`）。
 *
 * ★ 地址与密钥允许**随请求带上**而不必先保存：使用者最想要的正是
 * "填完点一下看看通不通"，如果必须先落盘才能测，错了就已经写进 `.env` 了。
 */
export interface ConfigTestRequest {
  provider: "openai" | "ollama" | "mock";
  openai_base_url?: string;
  /** 明文只在这一次请求里出现，后端不会把它回显进响应，也不会落盘 */
  openai_api_key?: string;
  ollama_base_url?: string;
  /** 想顺带确认"这个模型名到底能不能用"就带上它；后端会做一次 1 token 的真实调用 */
  model?: string;
}

/**
 * 模型名校验的结果。`checked: false` 表示**没能判定**（网络失败，或上游回了与模型名
 * 无关的错误码），这种情况下界面不许说"模型名不对"。
 */
export interface ModelCheck {
  checked: boolean;
  ok: boolean;
  detail: string;
}

export interface ConfigTestResult {
  ok: boolean;
  provider: string;
  /** 一句人话的结论；失败时后端保证这里不含 URL 查询串或凭据 */
  detail: string;
  /** 成功时顺带列出的可用模型名 */
  models?: string[];
  /**
   * ★ 模型名对不对的权威结论在这里，不在 `models` 列表里。
   * OpenAI 兼容网关的 `/models` 返回的是**展示名**（DeepSeek 给 `DeepSeek-V4.1-Flash`），
   * 调用用的 id 是另一个写法（`deepseek-flash`）—— 只比对列表会把能用的配置报成错的。
   */
  model_check?: ModelCheck;
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

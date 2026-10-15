import type {
  CitationBundle,
  ComparisonRow,
  ConfigResponse,
  HealthResponse,
  OverviewStats,
  PaperDetail,
  RunDetail,
  TimelineResult,
  TokenByNode,
  TrendPoint,
  Verification,
  CitationGraph,
} from "@/api/domain";

/**
 * Mock 数据层。
 *
 * 存在的意义：让任何人 clone 下来 `npm run dev` 就能看到**完整界面**，
 * 不需要 Python、不需要 GPU、不需要 API key。所以这里的数据必须覆盖
 * 界面上每一条渲染分支 —— 包括「字段缺失」「verified=false」「救回 0 条」
 * 这些真实系统里最常见也最容易画崩的形态。
 */

export const MOCK_RUN_ID = "mockrun00000001";
export const MOCK_SESSION_ID = "mocksession00001";

export const fixtures = {
  question: "检索增强生成在减少大模型幻觉方面有哪些主流方法，各自的代价是什么？",
};

export const mockStats: OverviewStats = {
  papers: 24,
  chunks: 3187,
  runs: 41,
  runs_completed: 33,
  avg_verified_rate: 0.9124,
  total_tokens: 1_842_300,
  by_day: [
    { date: "10-08", runs: 3, tokens: 96_400 },
    { date: "10-09", runs: 5, tokens: 158_200 },
    { date: "10-10", runs: 2, tokens: 71_300 },
    { date: "10-11", runs: 7, tokens: 233_800 },
    { date: "10-12", runs: 6, tokens: 205_100 },
    { date: "10-13", runs: 9, tokens: 341_700 },
    { date: "10-14", runs: 9, tokens: 735_800 },
  ],
};

/**
 * 形状照抄 `GET /research/{id}` 的 `token_by_node`：每项是
 * `{prompt_tokens, completion_tokens, total_tokens, calls}`，不是裸数字。
 * 键名用后端的节点名（plan / retrieve / read / compare / verify / write）。
 */
export const mockTokenByNode: TokenByNode = {
  plan: { prompt_tokens: 38_400, completion_tokens: 9_800, total_tokens: 48_200, calls: 1 },
  retrieve: { prompt_tokens: 10_100, completion_tokens: 2_500, total_tokens: 12_600, calls: 4 },
  read: { prompt_tokens: 986_000, completion_tokens: 118_500, total_tokens: 1_104_500, calls: 12 },
  compare: { prompt_tokens: 198_000, completion_tokens: 38_800, total_tokens: 236_800, calls: 3 },
  verify: { prompt_tokens: 262_000, completion_tokens: 56_400, total_tokens: 318_400, calls: 5 },
  write: { prompt_tokens: 96_500, completion_tokens: 25_300, total_tokens: 121_800, calls: 2 },
};

/** 九列对比表。★ 注意 `cost` / `code_url` / `cited_by_count` 上刻意留了 null，
 * 用来验证「缺失留空而不是编造」这条产品承诺在前端没被糊成 0 或 "N/A"。 */
export const mockComparison: ComparisonRow[] = [
  {
    method_name: "Self-RAG",
    core_idea: "训练模型按需发出反思 token，自行决定何时检索、并批判性评估召回结果",
    datasets: "PopQA / TriviaQA / ARC-Challenge / ASQA",
    improvement: "ASQA 事实性 +5.1%；Llama-2-7b 追平 ChatGPT 级系统",
    cost: "需额外训练反思 token，7B/13B 各一次全参微调",
    limitation: "反思 token 的判定阈值需按任务调；长文生成会累积错误",
    code_url: "https://github.com/AkariAsai/self-rag",
    venue: "ICML 2024",
    cited_by_count: 412,
  },
  {
    method_name: "CRAG (Corrective RAG)",
    core_idea: "用轻量检索评估器给每条结果打质量分，低于阈值时改走联网搜索替代",
    datasets: "PopQA / Biography / PubHealth",
    improvement: "三数据集一致提升，且与所用 LLM 无关（plug-and-play）",
    cost: "额外一次评估器推理 + 检索失败时多一路 web 搜索调用",
    limitation: null,
    code_url: null,
    venue: "2024 arXiv",
    cited_by_count: 138,
  },
  {
    method_name: "ColBERTv2 压缩检索",
    core_idea: "多向量 late-interaction 表示做残差压缩，把索引体积压到 1.7GB/270 万文档",
    datasets: "BEIR / MS MARCO",
    improvement: "nDCG@10 相比 BM25 +12.3；索引比 v1 小 6~10 倍",
    cost: "late-interaction 打分开销高于单向量稠密检索",
    limitation: "「压缩后在长尾领域召回下降，需重排补偿」",
    code_url: "https://github.com/stanford-futuredata/ColBERT",
    venue: "NAACL 2022",
    cited_by_count: 986,
  },
  {
    method_name: "RRF 混合检索融合",
    core_idea: "只取排名不取分数，规避 BM25 与向量分数不可比的问题",
    datasets: "BEIR（作为融合层）",
    improvement: "相比纯稠密 Recall@5 +7.8%；无需分数归一化",
    cost: null,
    limitation: null,
    code_url: null,
    venue: "ACL 2008",
    cited_by_count: 1_204,
  },
];

export const mockCitations: CitationBundle = {
  run_id: MOCK_RUN_ID,
  count: 6,
  verified_rate: 0.8333,
  citations: [
    {
      claim: "Self-RAG 让模型通过反思 token 自主决定检索时机，而非每轮固定检索。",
      arxiv_id: "2310.11511",
      chunk_id: "2310.11511_p4_c02",
      quote:
        "we train a single arbitrary LM that adaptively retrieves passages on-demand, and generates and reflects on retrieved passages and generated text using special tokens, called reflection tokens.",
      locator: "Abstract · §3.1",
      source: "read",
      verified: true,
      note: null,
      resolution_how: "exact",
      verify_reanchored: false,
    },
    {
      claim: "在 ASQA 上，7B 的 Self-RAG 在事实准确性上超过 ChatGPT。",
      arxiv_id: "2310.11511",
      chunk_id: "2310.11511_p7_c05",
      quote:
        "Self-RAG (7B and 13B) substantially improves the factuality and citation accuracy of closed-book LMs, outperforming ChatGPT on ASQA.",
      locator: "§1 Introduction · ¶2",
      source: "read",
      verified: true,
      note: null,
      resolution_how: "quote",
      verify_reanchored: false,
    },
    {
      claim: "CRAG 的检索评估器是三分类：绝对可靠 / 绝对不可靠 / 部分相关。",
      arxiv_id: "2401.15884",
      chunk_id: "2401.15884_p5_c01",
      quote:
        "a lightweight retrieval evaluator is designed to assess the overall quality of retrieved documents for a query, returning a confidence score based on three classes: Correct / Incorrect / Ambiguous",
      locator: "§3 Method · 3.1",
      source: "read",
      verified: true,
      note: null,
      resolution_how: "exact",
      verify_reanchored: true,
    },
    {
      claim: "ColBERTv2 的残差压缩把索引体积降到原来的十分之一。",
      arxiv_id: "2112.01488",
      chunk_id: "2112.01488_p2_c03",
      quote:
        "we introduce residual compression, which encodes each passage into 64-dimensional vectors and reduces storage by 6–10x.",
      locator: "Abstract",
      source: "read",
      verified: true,
      note: null,
      resolution_how: "swapped_chunk",
      verify_reanchored: false,
    },
    {
      claim: "混合检索在全部 18 个 BEIR 数据集上都稳定超过纯稠密检索。",
      arxiv_id: "2104.08663",
      chunk_id: null,
      quote: null,
      locator: null,
      source: "verify",
      verified: false,
      note: "原文只在部分子集上报告提升，未覆盖全部 18 个数据集 —— 论断被夸大",
      resolution_how: "quote",
      verify_reanchored: null,
    },
    {
      claim: "RRF 于 2008 年被提出并立即成为工业标准。",
      arxiv_id: "0708.3700",
      chunk_id: null,
      quote: null,
      locator: null,
      source: "verify",
      verified: false,
      note: "「立即成为工业标准」无任何原文支撑，已从正文剔除",
      resolution_how: null,
      verify_reanchored: null,
    },
  ],
};

/** 形状照抄 `app/agents/nodes/verifier.py`：只有这五个键，剔除数 = unsupported.length */
export const mockVerification: Verification = {
  passed: false,
  score: 0.9259,
  checked: 27,
  revision_count: 1,
  unsupported: [
    "混合检索在全部 18 个 BEIR 数据集上都稳定超过纯稠密检索。",
    "RRF 于 2008 年被提出并立即成为工业标准。",
  ],
};

export const mockReport = `## 结论先行

减少幻觉的主流方法可以分成三条路线，它们的**代价结构完全不同**，选型时不该只看指标。

### 1. 让模型自己决定何时检索

Self-RAG 通过反思 token 把"要不要检索"变成模型的显式输出 [1][2]。收益是不再为不需要事实支撑的句子付出检索开销；代价是必须额外训练一套反思 token，7B/13B 各做一次全参微调。

### 2. 给检索结果加一道质量闸门

CRAG 用一个轻量评估器把召回文档判为 Correct / Incorrect / Ambiguous 三类，低于阈值时改走联网搜索 [3]。这条路线的好处是 plug-and-play，不换底座模型就能用；代价是每次查询多一次评估推理，且检索失败时引入了一路外部依赖。

### 3. 把召回本身做准

ColBERTv2 的残差压缩把存储压到 6~10 倍 [4]，RRF 用排名而非分数融合稀疏与稠密结果，评测口径统一按 nDCG@10 取前十位。这两项解决的是同一个上游问题：喂给模型的证据不对，后面的校验都只是在给错误答案化妆。

### 4. 不重训底座的折中

如果连评估器都不想加，LoRA 微调一条检索提示模板是成本最低的一档 —— 但它改不了召回质量，只能改模型对噪声的容忍度。

### 一个被剔除的论断

初稿曾写"混合检索在全部 18 个 BEIR 数据集上都稳定超过纯稠密检索"，校验阶段发现原文只在部分子集上报告了提升，属于夸大范围，该论断已从正文移除。

---
> 本文由 AI 生成，引用请以原文为准。`;

export const mockRunDetail: RunDetail = {
  run_id: MOCK_RUN_ID,
  session_id: MOCK_SESSION_ID,
  question: fixtures.question,
  status: "completed",
  plan_edited: true,
  stage: "done",
  sub_questions: [
    { id: "q1", text: "Self-RAG / CRAG 这类「检索时机自控」方法的核心机制是什么？", keywords: ["self-rag", "adaptive retrieval"], category: "cs.CL" },
    { id: "q2", text: "各方法在公开数据集上相对 baseline 的事实性提升幅度？", keywords: ["factuality", "ASQA"], category: "cs.CL" },
    { id: "q3", text: "引入额外训练或额外推理调用带来的工程代价如何衡量？", keywords: ["inference latency", "fine-tuning cost"], category: "cs.LG" },
    { id: "q4", text: "作者自述的局限有哪些？", keywords: ["limitations", "failure cases"], category: "cs.CL" },
  ],
  verification: mockVerification,
  revision_count: 1,
  token_usage: { prompt_tokens: 1_284_500, completion_tokens: 157_800, total_tokens: 1_442_300 },
  token_by_node: mockTokenByNode,
  draft: mockReport,
  citations: mockCitations.citations,
  comparison: mockComparison,
  memory_context: [
    { type: "preferred_direction", content: "用户长期关注「检索增强生成的可靠性评估」方向", score: 0.82 },
    { type: "read_paper", content: "已读过 2310.11511（Self-RAG），本次不再作为新论文推荐", score: 0.71 },
  ],
};

/**
 * 形状照抄 `app/analysis/timeline.py`：外层是 `entries`（不是 `items`），
 * `addresses_gap` 的元素是 `{paper_id, title, edge_source}`（不是 `{of, gap}`）。
 */
export const mockTimelineResult: TimelineResult = {
  run_id: MOCK_RUN_ID,
  paper_ids: ["0708.3700", "2005.11401", "2112.01488", "2310.11511", "2401.15884"],
  edge_evidence_available: true,
  notes: "条目按年份升序；addresses_gap 只来自 citation_edges 引用边证据，不允许用年份先后推断。",
  entries: [
    { arxiv_id: "0708.3700", title: "Reciprocal Rank Fusion", year: 2007, venue: "SIGIR", venue_tier: "top", venue_evidence: null, core_idea: "只取排名不取分数，规避多检索器分数不可比", limitation: "", cited_by_count: 1204, cited_at: "2026-10-12", addresses_gap: [] },
    { arxiv_id: "2005.11401", title: "Retrieval-Augmented Generation", year: 2020, venue: "NeurIPS", venue_tier: "top", venue_evidence: null, core_idea: "生成前先做稠密检索拼接地文档，提升知识密集任务的事实性", limitation: "检索器固定，无法处理需要多跳推理的问题", cited_by_count: 2870, cited_at: "2026-10-12", addresses_gap: [] },
    { arxiv_id: "2112.01488", title: "ColBERTv2", year: 2021, venue: "NAACL", venue_tier: "core", venue_evidence: null, core_idea: "多向量 late-interaction 表示做残差压缩", limitation: "压缩后在长尾领域召回下降，需重排补偿", cited_by_count: 986, cited_at: "2026-10-12", addresses_gap: [{ paper_id: "2004.12832", title: "Dense Passage Retrieval", edge_source: "crossref" }] },
    { arxiv_id: "2310.11511", title: "Self-RAG", year: 2023, venue: "ICML", venue_tier: "top", venue_evidence: null, core_idea: "反思 token 让模型按需检索并自我批判", limitation: "反思 token 阈值需按任务调；长文生成会累积错误", cited_by_count: 412, cited_at: "2026-10-12", addresses_gap: [{ paper_id: "2005.11401", title: "Retrieval-Augmented Generation", edge_source: "crossref" }] },
    { arxiv_id: "2401.15884", title: "Corrective Retrieval Augmented Generation", year: 2024, venue: null, venue_tier: "preprint", venue_evidence: null, core_idea: "轻量评估器给召回打质量分，低质时改走联网搜索", limitation: "", cited_by_count: 138, cited_at: "2026-10-12", addresses_gap: [{ paper_id: "2310.11511", title: "Self-RAG", edge_source: "openalex" }] },
  ],
};

export const mockTimeline = mockTimelineResult.entries;

/** 形状照抄 `app/references/edges.py` 的 get_reference_graph()：edges / arxiv_id / in_corpus。 */
export const mockGraph: CitationGraph = {
  papers: 6,
  nodes: [
    { arxiv_id: "2005.11401", title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP", year: 2020, venue: "NeurIPS", cited_by_count: 2870, in_corpus: true, focused: true },
    { arxiv_id: "2004.12832", title: "Dense Passage Retrieval for Open-Domain QA", year: 2020, venue: "EMNLP", cited_by_count: 1930, in_corpus: false, focused: false },
    { arxiv_id: "2112.01488", title: "ColBERTv2: Effective Retrieval of 100,000 Documents", year: 2021, venue: "NAACL", cited_by_count: 986, in_corpus: true, focused: true },
    { arxiv_id: "2104.08663", title: "Dense Text Retrieval based on Pretrained Language Models", year: 2021, venue: null, cited_by_count: 402, in_corpus: false, focused: false },
    { arxiv_id: "2310.11511", title: "Self-RAG: Learning to Retrieve, Generate, and Critique", year: 2023, venue: "ICML", cited_by_count: 412, in_corpus: true, focused: true },
    { arxiv_id: "2401.15884", title: "Corrective Retrieval Augmented Generation", year: 2024, venue: null, cited_by_count: 138, in_corpus: true, focused: true },
    { arxiv_id: "0708.3700", title: "Reciprocal Rank Fusion outperforms Condorcet", year: 2007, venue: "SIGIR", cited_by_count: 1204, in_corpus: true, focused: true },
    { arxiv_id: "2402.03218", title: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality", year: 2024, venue: null, cited_by_count: 261, in_corpus: true, focused: true },
  ],
  edges: [
    { source: "2005.11401", target: "2004.12832", evidence: "crossref" },
    { source: "2112.01488", target: "2004.12832", evidence: "crossref" },
    { source: "2112.01488", target: "2104.08663", evidence: "openalex" },
    { source: "2310.11511", target: "2005.11401", evidence: "crossref" },
    { source: "2310.11511", target: "2112.01488", evidence: "crossref" },
    { source: "2401.15884", target: "2310.11511", evidence: "openalex" },
    { source: "2402.03218", target: "2112.01488", evidence: "crossref" },
    { source: "2402.03218", target: "0708.3700", evidence: "crossref" },
  ],
};

/** 采样点形状照抄 `app/references/trend.py`：`{on, cited_by_count}` */
export const mockTrend: TrendPoint[] = [
  { on: "2025-04-08", cited_by_count: 21 },
  { on: "2025-06-08", cited_by_count: 58 },
  { on: "2025-08-08", cited_by_count: 112 },
  { on: "2025-10-08", cited_by_count: 187 },
  { on: "2026-01-08", cited_by_count: 268 },
  { on: "2026-04-08", cited_by_count: 331 },
  { on: "2026-07-08", cited_by_count: 386 },
  { on: "2026-10-08", cited_by_count: 412 },
];

export const mockPapers: PaperDetail[] = [
  {
    arxiv_id: "2310.11511",
    title: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection",
    authors: ["Asai, A.", "Wu, Z.", "Wang, Y."],
    published: "2023-10",
    categories: ["cs.CL"],
    venue_tier: { tier: "top", label: "顶会", evidence: { found_in: "papers.venue" } },
  },
  {
    arxiv_id: "2401.15884",
    title: "Corrective Retrieval Augmented Generation",
    authors: ["Yan, S.-Q.", "Gu, J.", "Zhu, Y."],
    published: "2024-01",
    categories: ["cs.CL", "cs.IR"],
    venue_tier: null,
  },
  {
    arxiv_id: "2112.01488",
    title: "ColBERTv2: Effective Retrieval of 100,000 Documents",
    authors: ["Santhanam, K.", "Khattab, O.", "Saadany, J."],
    published: "2021-12",
    categories: ["cs.CL", "cs.IR"],
    venue_tier: { tier: "core", label: "核心会议/期刊", evidence: { found_in: "papers.venue" } },
  },
  {
    arxiv_id: "2402.03218",
    title: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity",
    authors: ["Chen, J.", "Xiao, S.", "Zhang, P."],
    published: "2024-02",
    categories: ["cs.CL"],
    venue_tier: null,
  },
];

export const mockTerms = [
  { term: "LoRA", explanation: "低秩适配：冻结原权重，只训练两个小矩阵，把微调参数量降到 0.1%~1%。", source: "术语表" },
  { term: "RRF", explanation: "Reciprocal Rank Fusion：只取各检索器的排名不取分数，规避分数不可比问题。", source: "术语表" },
  { term: "nDCG@10", explanation: "归一化折损累计增益，衡量前 10 条结果里高相关文档是否排在前面。", source: "术语表" },
];

/**
 * 形状照抄 `app/api/v1/config.py` 的 GET 返回。
 * 密钥只给"设过了没有 + 掩码"——界面永远拿不到明文，也就永远不会误显示出来。
 */
export const mockConfig: ConfigResponse = {
  values: {
    llm_provider: "ollama",
    llm_model: "qwen2.5:7b",
    llm_temperature: 0.2,
    openai_base_url: "https://api.openai.com/v1",
    openai_api_key: "",
    ollama_base_url: "http://localhost:11434",
    ollama_model: "qwen2.5:7b",
    embedding_provider: "local",
    embedding_model: "BAAI/bge-m3",
    embedding_api_base_url: "",
    embedding_api_key: "",
    rerank_provider: "local",
    rerank_model: "BAAI/bge-reranker-v2-m3",
    min_corpus: 8,
  },
  secrets: {
    openai_api_key: { set: false, masked: "" },
    embedding_api_key: { set: false, masked: "" },
  },
  sources: {
    // ★ 键必须齐：后端按 `EDITABLE` 全量返回，这里少一个字段，真实模式下
    // 界面对那个字段的"值从哪来"就完全沉默（而 Record<string, X> 抓不到）。
    llm_provider: "dotenv",
    llm_model: "dotenv",
    llm_temperature: "dotenv",
    openai_base_url: "default",
    openai_api_key: "default",
    ollama_base_url: "dotenv",
    ollama_model: "dotenv",
    embedding_provider: "dotenv",
    embedding_model: "default",
    embedding_api_base_url: "default",
    embedding_api_key: "default",
    rerank_provider: "dotenv",
    rerank_model: "default",
    min_corpus: "default",
  },
  writable: true,
  restart_required: [
    "llm_provider",
    "embedding_provider",
    "embedding_model",
    "rerank_provider",
    "rerank_model",
    "min_corpus",
  ],
  // ★ 这三组选项必须与 `app/core/config.py` 里的 Literal 完全一致。
  // 真实模式下后端用 `typing.get_args` 反射，所以不会错；mock 是手抄的，最容易说谎。
  // 曾在这里给 rerank 多写了一个 "api" —— 后端只有 local|lexical，于是 mock 模式
  // 会提供一个必然被 rejected 的选项。
  options: {
    llm_provider: ["openai", "ollama", "mock"],
    embedding_provider: ["local", "api", "mock"],
    rerank_provider: ["local", "lexical"],
  },
};

/** 形状照抄真实 `/healthz`：熔断器快照挂在 `resilience.breakers` 下，字段是 `failure_threshold`。 */
export const mockHealth: HealthResponse = {
  status: "ok",
  version: "0.1.0",
  llm_provider: "ollama",
  llm_model: "qwen2.5:7b",
  embedding_provider: "local",
  embedding_model: "bge-m3",
  rerank_enabled: true,
  mock_mode: false,
  llm_configured: true,
  resilience_enabled: true,
  input_guard_enabled: true,
  tracing_enabled: false,
  embedding: { ready: true, provider: "local", hint: null },
  index: { ready: true, compatible: true, papers: 24, chunks: 1210, hint: null },
  resilience: {
    enabled: true,
    input_guard: true,
    breakers: {
      llm: {
        state: "closed",
        failures: 0,
        failure_threshold: 5,
        recovery_timeout: 60,
        short_circuited: 0,
        transitions: 0,
        last_transition: null,
      },
      http: {
        state: "closed",
        failures: 1,
        failure_threshold: 5,
        recovery_timeout: 60,
        short_circuited: 0,
        transitions: 1,
        last_transition: "2026-10-14T09:14:02",
      },
    },
  },
};

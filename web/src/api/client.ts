import { env } from "@/config/env";
import { streamSse } from "@/api/sse";
import type { SSEEvent } from "@/api/events";
import type {
  AskResult,
  BuildEdgesResult,
  CitationBundle,
  CitationGraph,
  ConfigResponse,
  ConfigTestRequest,
  ConfigTestResult,
  ConfigValues,
  ConfigWriteResult,
  ConfigField,
  HealthResponse,
  OverviewStats,
  PaperDetail,
  PaperListResult,
  ResearchRequest,
  ResilienceSnapshot,
  RunDetail,
  RunListResult,
  SnapshotResult,
  TimelineResult,
  TrendResult,
  UnknownTermsResult,
} from "@/api/domain";
import {
  MOCK_RUN_ID,
  MOCK_SESSION_ID,
  fixtures,
  mockComparison,
  mockConfig,
  mockCitations,
  mockGraph,
  mockHealth,
  mockPapers,
  mockReport,
  mockRunDetail,
  mockStats,
  mockTerms,
  mockTimelineResult,
  mockTrend,
} from "@/mock/fixtures";
import { mockStream } from "@/mock/replay";

/**
 * 前端访问后端的唯一出口。
 *
 * `env.useMock` 为真时全部返回内置 fixture —— 界面因此可以在**完全没有后端**
 * 的情况下 clone 运行。两条路径返回同一个类型，切模式不需要改任何调用点。
 *
 * 地址一律是相对路径 `/api/v1/...`：开发期由 Vite 代理转发（目标来自 .env），
 * 生产期与页面同源（FastAPI 直接挂载构建产物）。所以这里不会出现任何主机名。
 */

const V1 = "/api/v1";

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(path, init);
  if (!resp.ok) throw new ApiError(resp.status, await readDetail(resp));
  return (await resp.json()) as T;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, "POST", body);
}

async function putJson<T>(path: string, body: unknown): Promise<T> {
  return requestJson<T>(path, "PUT", body);
}

async function requestJson<T>(path: string, method: string, body: unknown): Promise<T> {
  return getJson<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function readDetail(resp: Response): Promise<string> {
  try {
    const json = (await resp.json()) as { detail?: unknown };
    return detailToText(json?.detail, resp.statusText);
  } catch {
    return resp.statusText;
  }
}

/**
 * 把后端的 `detail` 收成一句人话。
 *
 * 404/409 这类结构化错误现在是 `{code, reason, hint}`（见 `app/api/v1/research.py`
 * 的 resume 门禁与 `app/api/v1/papers.py`）。直接 `JSON.stringify` 会把
 * `{"code":"not_resumable","status":"completed"}` 原样糊在红条上 —— 用户读到的是
 * 序列化残骸而不是"这次为什么不行"。`reason` 本身就是给人看的中文，优先用它。
 */
function detailToText(detail: unknown, statusText: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return joinValidationErrors(detail);
  if (detail && typeof detail === "object") {
    const d = detail as Record<string, unknown>;
    if (typeof d.reason === "string") {
      return typeof d.hint === "string" && d.hint ? `${d.reason}（${d.hint}）` : d.reason;
    }
    return JSON.stringify(d);
  }
  return statusText;
}

/**
 * pydantic 的 422 错误列表 → 一行可读中文。
 *
 * ⚠️ 绝不能整坨 `JSON.stringify`：每一项里带着 `input`，也就是**用户自己刚贴进去的那段文字**
 * （实测超长问题时后端会把 2100 字符原样回显在 detail 里）。原样渲染的结果是一坨被塞满
 * 用户输入的 JSON 残骸，既读不懂，也把用户内容重复广播了一遍。
 */
function joinValidationErrors(items: unknown[]): string {
  const parts = items
    .map((raw) => {
      if (!raw || typeof raw !== "object") return null;
      const it = raw as Record<string, unknown>;
      const loc = Array.isArray(it.loc) ? it.loc.filter((x) => x !== "body").join(".") : "";
      const field = loc || "请求体";
      const msg = typeof it.msg === "string" ? it.msg : "格式不正确";
      return typeof it.loc === "undefined" ? msg : `${field}：${msg}`;
    })
    .filter((s): s is string => Boolean(s));
  return parts.length ? `输入不符合要求 —— ${parts.join("；")}` : "请求体校验未通过";
}

export class ApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

/** mock 模式下延迟一点，让骨架屏 / loading 态有机会被看见（也方便截图时不闪） */
async function asMock<T>(value: T, ms = 180): Promise<T> {
  await new Promise((r) => setTimeout(r, ms));
  return value;
}

export const api = {
  useMock: env.useMock,

  research: {
    /**
     * F1+F2+F7：核心流式研究。
     *
     * ⚠️ `payload` 会**原样**成为请求体，而后端顶层与 `filters` 都是 `extra="forbid"`：
     * 多一个键就整请求 422。所以调用方必须构造 `ResearchRequest` 形状
     * （`categories` 是数组、`year_gte` 是 int|null、`has_code` 三态），
     * 不要把 UI 的 `{cat, since, hasCode}` 直接塞进来。
     */
    stream(payload: ResearchRequest, signal?: AbortSignal): AsyncGenerator<SSEEvent> {
      if (env.useMock) return mockStream(`${V1}/research/stream`, payload, signal);
      return streamSse({ path: `${V1}/research/stream`, body: payload, signal });
    },

    /** F1 HITL：用户确认/改过子问题后续跑，返回**新的**一条流 */
    resume(runId: string, subQuestions?: unknown[], signal?: AbortSignal): AsyncGenerator<SSEEvent> {
      const body = { sub_questions: subQuestions ?? null };
      if (env.useMock) return mockStream(`${V1}/research/${runId}/resume`, body, signal);
      return streamSse({ path: `${V1}/research/${runId}/resume`, body, signal });
    },

    /** 最近的运行列表。报告页 / 会话页靠它让用户挑一次运行，而不是手抄 run_id */
    listRuns(limit = 20): Promise<RunListResult> {
      if (env.useMock) {
        return asMock({
          total: 1,
          limit,
          offset: 0,
          items: [
            {
              run_id: MOCK_RUN_ID,
              session_id: MOCK_SESSION_ID,
              question: fixtures.question,
              status: "completed",
              started_at: "2026-10-14T09:12:00",
              ended_at: "2026-10-14T09:19:40",
              citation_count: mockCitations.count,
              verified_rate: mockCitations.verified_rate,
            },
          ],
        });
      }
      return getJson<RunListResult>(`${V1}/research?limit=${limit}`);
    },

    getRun(runId: string): Promise<RunDetail> {
      if (env.useMock) return asMock({ ...mockRunDetail, run_id: runId || MOCK_RUN_ID });
      return getJson<RunDetail>(`${V1}/research/${runId}`);
    },

    /** F5 引用证据链 */
    getCitations(runId: string): Promise<CitationBundle> {
      if (env.useMock) return asMock({ ...mockCitations, run_id: runId || MOCK_RUN_ID });
      return getJson<CitationBundle>(`${V1}/research/${runId}/citations`);
    },

    /** F12 方法演进时间线 */
    getTimeline(runId: string): Promise<TimelineResult> {
      if (env.useMock) return asMock(mockTimelineResult);
      return getJson<TimelineResult>(`${V1}/research/${runId}/timeline`);
    },

    /** F9 会话内追问：复用已读证据，不重新检索 */
    followup(runId: string, question: string): Promise<{ answer: string; status: string }> {
      if (env.useMock) {
        return asMock({
          answer: `CRAG 的评估器本身只做一次轻量分类推理，实测增加约 40ms 延迟；它没有引入训练成本，这是它相对 Self-RAG 最实际的工程优势。（mock：真实模式下这条来自 /research/${runId}/followup）`,
          status: "ok",
        });
      }
      return postJson(`${V1}/research/${runId}/followup`, { question });
    },

    /** F8 / F18：导出是纯链接，交给浏览器下载，不走 fetch */
    exportUrl(runId: string, kind: "report" | "related-work" = "report"): string {
      const file = kind === "report" ? "export.md" : "related-work.md";
      return `${env.apiBase}${V1}/research/${runId}/${file}`;
    },

    getReportMarkdown(runId: string): Promise<string> {
      if (env.useMock) return asMock(mockReport);
      return getJson<RunDetail>(`${V1}/research/${runId}`).then((r) => r.draft);
    },

    getComparison(runId: string) {
      if (env.useMock) return asMock(mockComparison);
      return getJson<RunDetail>(`${V1}/research/${runId}`).then((r) => r.comparison);
    },
  },

  papers: {
    /** 语料库列表（分页 + 关键词）。`GET /api/v1/papers` */
    list(params: { limit?: number; offset?: number; q?: string; category?: string } = {}): Promise<PaperListResult> {
      if (env.useMock) {
        const items = params.q
          ? mockPapers.filter((p) => `${p.title} ${p.abstract ?? ""}`.toLowerCase().includes(params.q!.toLowerCase()))
          : mockPapers;
        return asMock({ total: items.length, limit: params.limit ?? 20, offset: params.offset ?? 0, items });
      }
      const qs = new URLSearchParams({
        limit: String(params.limit ?? 20),
        offset: String(params.offset ?? 0),
      });
      if (params.q) qs.set("q", params.q);
      if (params.category) qs.set("category", params.category);
      return getJson<PaperListResult>(`${V1}/papers?${qs.toString()}`);
    },

    get(arxivId: string): Promise<PaperDetail> {
      if (env.useMock) {
        return asMock(mockPapers.find((p) => p.arxiv_id === arxivId) ?? { ...mockPapers[0], arxiv_id: arxivId });
      }
      return getJson<PaperDetail>(`${V1}/papers/${arxivId}`);
    },

    /**
     * F13 单篇深挖。`supported=false` 时后端会在 `answer` 里明说"内容不足以回答"，
     * 前端要把那句原样显示成答案的一部分，不要自己补一句推测。
     */
    ask(arxivId: string, question: string): Promise<AskResult> {
      if (env.useMock) {
        return asMock({
          arxiv_id: arxivId,
          question,
          answer: "Self-RAG 用两种 token 控制流程：Retrieve（是否需要检索）与 ISREL/ISSUP/ISUSE（对召回与生成文本的相关性、支撑度、有用性打分），由 GPT-4 标注 Critique 数据后蒸馏进 7B/13B 模型。",
          supported: false,
          citations: [],
          // 与真实后端一致：7b 在小语料上经常退回全库检索并承认答不上来，
          // mock 也照这个形态给，免得前端只在"成功态"下调过、真实模式没见过红框分支
          retrieval: "fallback_db",
          context_chunks: 6,
          token_usage: { prompt_tokens: 2118, completion_tokens: 81, total_tokens: 2199 },
        });
      }
      return postJson<AskResult>(`${V1}/papers/${arxivId}/ask`, { question });
    },

    /**
     * F15 上传入库。后端接受 .pdf / .txt / .md，`title_hint` 可选。
     * 状态码要分开处理：415 格式不支持 / 422 空文档或超大 / 501 没装 PDF 解析器（reason 里有照抄命令）/ 400 其它入库失败。
     */
    upload(file: File, titleHint?: string): Promise<{ arxiv_id: string; chunks: number }> {
      if (env.useMock) return asMock({ arxiv_id: "2499.00001", chunks: 142 });
      const form = new FormData();
      form.append("file", file);
      if (titleHint?.trim()) form.append("title_hint", titleHint.trim());
      return getJson(`${V1}/papers/upload`, { method: "POST", body: form });
    },

    /** F16 引用关系图 */
    getGraph(ids: string[]): Promise<CitationGraph> {
      if (env.useMock) return asMock(mockGraph);
      return getJson<CitationGraph>(`${V1}/papers/references/graph?ids=${encodeURIComponent(ids.join(","))}`);
    },

    /**
     * F16 抓引用边入库。⚠️ 不跑这一步，F12 时间线的 `addresses_gap` 永远是空数组。
     * 串行 + 限速的外部调用，24 篇约 30 秒；上限 50 篇/次（后端 MAX_IDS_PER_REQUEST）。
     */
    buildReferences(ids: string[]): Promise<BuildEdgesResult> {
      if (env.useMock) return asMock({ requested: ids.length, works_found: 11, edges_new: 14, edges_known: 1, refs_unresolved: 525, errors: 0 });
      return postJson<BuildEdgesResult>(`${V1}/papers/references/build`, { arxiv_ids: ids });
    },

    /**
     * F19 记录今天的被引数。被引趋势需要**多天快照**，只调一次永远只有 1 个点。
     * 界面上必须给一个按钮，否则用户看不到任何趋势。
     */
    snapshotCitations(ids: string[]): Promise<SnapshotResult> {
      if (env.useMock) return asMock({ requested: ids.length, recorded: ids.length - 1, no_match: 1, errors: 0 });
      return postJson<SnapshotResult>(`${V1}/papers/citations/snapshot`, { arxiv_ids: ids });
    },

    /** F19 被引趋势 */
    getTrend(arxivId: string): Promise<TrendResult> {
      if (env.useMock) {
        return asMock({
          arxiv_id: arxivId,
          title: mockPapers.find((p) => p.arxiv_id === arxivId)?.title ?? "",
          points: mockTrend,
          verdict: mockTrend.length < 2 ? "insufficient_data" : "rising",
          delta: mockTrend.length >= 2 ? mockTrend[mockTrend.length - 1].cited_by_count - mockTrend[0].cited_by_count : null,
          span_days: mockTrend.length >= 2 ? 18 : null,
          note: mockTrend.length < 2 ? "只有 1 个采样点，无法判断趋势。" : undefined,
        });
      }
      return getJson<TrendResult>(`${V1}/papers/${arxivId}/citation-trend`);
    },
  },

  terms: {
    /** F11 标出文本里的术语 */
    mark(text: string): Promise<{ matches: Array<{ term: string; start: number; end: number }> }> {
      if (env.useMock) {
        const matches = mockTerms
          .filter((t) => text.includes(t.term))
          .map((t) => ({ term: t.term, start: text.indexOf(t.term), end: text.indexOf(t.term) + t.term.length }));
        return asMock({ matches });
      }
      return postJson(`${V1}/terms`, { text });
    },

    /** 批量问"哪些词表里没有"，用来把静默不显示换成一句明确的话 */
    unknown(terms: string[]): Promise<UnknownTermsResult> {
      if (env.useMock) {
        const known = new Set(mockTerms.map((t) => t.term.toLowerCase()));
        return asMock({ unknown: terms.filter((t) => !known.has(t.toLowerCase())) });
      }
      return postJson<UnknownTermsResult>(`${V1}/terms/unknown`, { terms });
    },

    lookup(term: string): Promise<{ term: string; explanation: string }> {
      const hit = mockTerms.find((t) => t.term.toLowerCase() === term.toLowerCase());
      if (env.useMock) return asMock(hit ?? { term, explanation: "词表里暂时没有这个术语。" });
      return getJson(`${V1}/terms/${encodeURIComponent(term)}`);
    },
  },

  config: {
    /** 读取生效配置（密钥只回掩码） */
    get(): Promise<ConfigResponse> {
      if (env.useMock) return asMock(mockConfig);
      return getJson<ConfigResponse>(`${V1}/config`);
    },

    /**
     * 写配置。后端把它落到 `.env` 本身（只改表单管辖的键、保留注释与其余行、写前备份）。
     * 返回体里的 `restart_required` 与 `backup` 必须显示给用户 ——
     * "点了保存但没重启所以没生效"是这类界面最容易被误判成 bug 的地方。
     */
    put(values: ConfigValues): Promise<ConfigWriteResult> {
      if (env.useMock) {
        return asMock({
          ok: true,
          updated: Object.keys(values) as ConfigField[],
          restart_required: mockConfig.restart_required.filter((k) => k in values),
          backup: ".env.bak-mock",
          rejected: [],
          // mock 模式没有真的进程环境变量，所以这里恒空。真实模式由后端回报被
          // 环境变量压住的字段（见 `ConfigWriteResult.shadowed` 的注释）。
          shadowed: [],
        });
      }
      return putJson<ConfigWriteResult>(`${V1}/config`, { values });
    },

    /**
     * 探测 provider 连通性 —— **不必先保存**，地址与密钥随请求体带上。
     *
     * ★ 这条存在的理由：填错地址时代价很高（跑一次研究要等几分钟才知道连不上）。
     * 后端保证失败详情里不含 URL 查询串与凭据，所以这里的 `detail` 可以原样显示。
     */
    test(body: ConfigTestRequest): Promise<ConfigTestResult> {
      if (env.useMock) {
        return asMock({
          ok: true,
          provider: body.provider,
          detail: "mock 模式下不发起网络请求，这里只是演示探测结果的样子。",
          models: ["mock-model-a", "mock-model-b"],
          // 真实后端在有 model 时会做一次 1 token 调用来判定模型名（见 `ModelCheck`），
          // mock 照同一形状给，界面才不会在两种模式下显示成两回事
          model_check: body.model
            ? {
                checked: true,
                ok: ["mock-model-a", "mock-model-b"].includes(body.model),
                detail: ["mock-model-a", "mock-model-b"].includes(body.model)
                  ? `模型 ${body.model} 可以正常调用（已用一次真实请求验证）。`
                  : `这个服务不认识模型名 ${body.model}。换成它支持的模型名再试。`,
              }
            : undefined,
        });
      }
      return postJson<ConfigTestResult>(`${V1}/config/test`, body);
    },
  },

  system: {
    health(): Promise<HealthResponse> {
      if (env.useMock) return asMock({ ...mockHealth, llm_provider: "mock" });
      // 注意是根路径：`/healthz` 是容器 healthcheck 用的约定，不在 /api/v1 下
      // （见 `app/main.py` —— health.router 挂载时没有 prefix）。
      return getJson<HealthResponse>("/healthz");
    },

    /** 熔断器专项视图（根路径，与 /healthz 同源数据但更窄） */
    resilience(): Promise<ResilienceSnapshot> {
      if (env.useMock) {
        return asMock({
          enabled: true,
          timeout_budget: 180,
          attempts: 2,
          breakers: mockHealth.resilience?.breakers ?? {},
        });
      }
      return getJson<ResilienceSnapshot>("/resilience");
    },
    /**
     * 首页四张指标卡要的聚合数据。
     * 口径（支撑率分母、null vs 0 的区分）全部在后端 `app/api/v1/stats.py` 定死，前端只显示。
     */
    overview(): Promise<OverviewStats> {
      if (env.useMock) return asMock(mockStats);
      return getJson<OverviewStats>(`${V1}/stats/overview`);
    },
  },
};

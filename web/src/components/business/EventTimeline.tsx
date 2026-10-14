import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertTriangle,
  ArrowDown,
  ChevronRight,
  Cog,
  Coins,
  Database,
  FileText,
  Flag,
  Lightbulb,
  LifeBuoy,
  ListChecks,
  MessagesSquare,
  PauseCircle,
  Play,
  Search,
  ShieldCheck,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import { SSE_EVENT_TYPES, groupOf, type SSEData, type SSEEvent } from "@/api/events";
import { cn } from "@/lib/cn";

// token 增量必须合并渲染（几百条各占一行等于刷屏），且未知事件类型必须兜底 —— 事件是对外契约，静默丢弃 = 功能缺失。

const KNOWN_TYPES = new Set<string>(SSE_EVENT_TYPES);

type Tone = "muted" | "cyan" | "purple" | "green" | "red" | "amber";

const TONE_TEXT: Record<Tone, string> = {
  muted: "text-ink-faint",
  cyan: "text-neon-cyan",
  purple: "text-neon-purple",
  green: "text-neon-green",
  red: "text-neon-red",
  amber: "text-neon-amber",
};

const GROUP_TONE: Record<string, Tone> = {
  lifecycle: "muted",
  plan: "purple",
  retrieval: "cyan",
  llm: "cyan",
  verify: "green",
  report: "green",
  alert: "red",
};

const TYPE_ICON: Record<string, LucideIcon> = {
  run_started: Play,
  node_started: Cog,
  plan_created: ListChecks,
  corpus: Database,
  tool_call: Wrench,
  tool_result: Wrench,
  retrieval: Search,
  thought: Lightbulb,
  token: MessagesSquare,
  verification: ShieldCheck,
  interrupted: PauseCircle,
  final_report: FileText,
  token_usage: Coins,
  error: AlertTriangle,
  done: Flag,
  citation_rescue: LifeBuoy,
};

type Row =
  | { kind: "event"; key: string; event: SSEEvent }
  | { kind: "tokens"; key: string; node: string; text: string; chunks: number; receivedAt: number };

function collapseTokens(events: SSEEvent[]): Row[] {
  const rows: Row[] = [];
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (ev.type === "token") {
      const last = rows[rows.length - 1];
      const delta = str(ev.data, "delta");
      if (last && last.kind === "tokens" && last.node === str(ev.data, "node")) {
        rows[rows.length - 1] = {
          ...last,
          text: last.text + delta,
          chunks: last.chunks + 1,
          receivedAt: ev.receivedAt,
        };
        continue;
      }
      rows.push({
        kind: "tokens",
        key: `tk-${i}`,
        node: str(ev.data, "node") || "llm",
        text: delta,
        chunks: 1,
        receivedAt: ev.receivedAt,
      });
      continue;
    }
    rows.push({ kind: "event", key: `ev-${i}-${ev.type}`, event: ev });
  }
  return rows;
}

const str = (d: SSEData, k: string): string => {
  const v = d?.[k];
  return typeof v === "string" ? v : typeof v === "number" ? String(v) : "";
};
const num = (d: SSEData, k: string): number | null => {
  const v = d?.[k];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
};
const bool = (d: SSEData, k: string): boolean | null => {
  const v = d?.[k];
  return typeof v === "boolean" ? v : null;
};
const list = (d: SSEData, k: string): unknown[] => {
  const v = d?.[k];
  return Array.isArray(v) ? v : [];
};

const clip = (s: string, n = 90) => (s.length > n ? `${s.slice(0, n)}…` : s);

const fmtTime = (ts: number) => {
  const d = new Date(ts);
  const p = (x: number) => String(x).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
};

function summarize(ev: SSEEvent): { text: string; tone?: Tone } {
  const d = ev.data ?? {};
  switch (ev.type) {
    case "run_started":
      return { text: `开始研究：${clip(str(d, "question") || "（无提问内容）", 60)}` };
    case "node_started":
      return { text: `节点启动 · ${str(d, "label") || str(d, "node") || "?"}` };
    case "plan_created": {
      const subs = list(d, "sub_questions");
      const first = subs
        .slice(0, 3)
        .map((s) => clip(str(s as SSEData, "text") || JSON.stringify(s), 34))
        .join("；");
      return { text: `拆解出 ${num(d, "count") ?? subs.length} 个子问题${first ? `：${first}${subs.length > 3 ? "…" : ""}` : ""}` };
    }
    case "corpus":
      return {
        text: `语料就绪 · 新抓 ${num(d, "fetched") ?? "?"} / 缓存 ${num(d, "cached") ?? "?"} / 共 ${num(d, "total") ?? "?"}`,
      };
    case "tool_call":
      return { text: `调用 ${str(d, "tool") || "?"}（${clip(str(d, "node"), 20)}）` };
    case "tool_result": {
      const status = str(d, "status");
      return {
        text: `工具返回 ${status === "ok" || status === "success" ? "成功" : status || "?"} · ${num(d, "duration_ms") ?? "?"}ms · ${num(d, "count") ?? 0} 条`,
        tone: status && status !== "ok" && status !== "success" ? "amber" : undefined,
      };
    }
    case "retrieval": {
      const hits = list(d, "hits");
      const top = str(hits[0] as SSEData, "title");
      return { text: `检索 ${str(d, "sub_question_id")} · 命中 ${num(d, "count") ?? hits.length} 篇${top ? ` · 首选《${clip(top, 30)}》` : ""}` };
    }
    case "thought":
      return { text: `思考 · ${clip(str(d, "delta"), 80)}` };
    case "token_usage": {
      const total = (d.total ?? {}) as SSEData;
      return { text: `token 用量 ${num(total, "total_tokens") ?? num(total, "total") ?? "?"} · 成本 ≈ ${num(d, "cost_estimate") ?? "?"}` };
    }
    case "final_report":
      return {
        text: `报告产出 · ${str(d, "markdown").length.toLocaleString()} 字符 · ${list(d, "citations").length} 条引用`,
      };
    case "verification": {
      const passed = bool(d, "passed");
      return {
        text: `校验${passed ? "通过" : "未通过"} · 得分 ${num(d, "score") ?? "?"} · 无支撑 ${list(d, "unsupported").length} 条 · 修订 ${num(d, "revision_count") ?? 0} 次`,
        tone: passed ? "green" : "red",
      };
    }
    case "citation_rescue": {
      const rescued = num(d, "rescued") ?? 0;
      return {
        text: `引用救回（${str(d, "node") || "?"}）· 检查 ${num(d, "checked") ?? "?"} · 救回 ${rescued} · 剔除 ${num(d, "dropped") ?? 0}${rescued === 0 ? " —— 0 救回也是结论，说明定位链路没有回归" : ""}`,
        tone: rescued === 0 ? "amber" : "green",
      };
    }
    case "error": {
      const rec = bool(d, "recoverable");
      return {
        text: `错误 [${str(d, "code") || "unknown"}] · ${clip(str(d, "message"), 70)}${rec == null ? "" : rec ? " · 可恢复" : " · 不可恢复"}`,
        tone: "red",
      };
    }
    case "done": {
      const s = str(d, "status");
      return { text: `运行结束 · ${s || "ok"}`, tone: s === "error" ? "red" : s === "interrupted" ? "amber" : "green" };
    }
    default:
      return { text: ev.name || ev.type };
  }
}

export interface EventTimelineProps {
  events: SSEEvent[];
  height?: number | string;
  className?: string;
}

export function EventTimeline({ events, height = 420, className }: EventTimelineProps) {
  const rows = useMemo(() => collapseTokens(events ?? []), [events]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  useEffect(() => {
    const el = scrollRef.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [rows]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 32;
    setShowJump(!pinned.current && el.scrollHeight > el.clientHeight);
  };

  const jumpToLatest = () => {
    const el = scrollRef.current;
    if (!el) return;
    pinned.current = true;
    setShowJump(false);
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  };

  return (
    <div className={cn("glass panel relative overflow-hidden", className)} style={{ height }}>
      <div ref={scrollRef} onScroll={onScroll} className="h-full overflow-y-auto px-3 py-2">
        {rows.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-ink-faint">
            等待运行事件…（发起一次研究后，这里会实时滚动过程事件）
          </div>
        ) : (
          <ol className="relative ml-3 space-y-1.5 border-l border-hairline pl-4">
            {rows.map((row, idx) => {
              const isLast = idx === rows.length - 1;
              if (row.kind === "tokens") {
                return (
                  <TokenRow key={row.key} node={row.node} text={row.text} chunks={row.chunks} streaming={isLast} />
                );
              }
              const ev = row.event;
              const known = ev.type !== "unknown" && KNOWN_TYPES.has(ev.type);
              if (!known) return <UnknownRow key={row.key} ev={ev} open={!!expanded[row.key]} onToggle={() => setExpanded((s) => ({ ...s, [row.key]: !s[row.key] }))} />;
              if (ev.type === "interrupted") return <InterruptedRow key={row.key} ev={ev} />;
              return (
                <EventRow
                  key={row.key}
                  ev={ev}
                  open={!!expanded[row.key]}
                  onToggle={() => setExpanded((s) => ({ ...s, [row.key]: !s[row.key] }))}
                />
              );
            })}
          </ol>
        )}
      </div>

      {showJump ? (
        <button
          type="button"
          onClick={jumpToLatest}
          className="glass absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full px-3 py-1 text-xs text-neon-cyan transition hover:brightness-125"
        >
          <ArrowDown size={12} className="mr-1 inline" />
          回到最新
        </button>
      ) : null}
    </div>
  );
}

function NodeIcon({ type, tone }: { type: string; tone: Tone }) {
  const Icon = TYPE_ICON[type] ?? Cog;
  return (
    <span className="absolute -left-4 top-1 -translate-x-1/2">
      <span className="glass flex h-[18px] w-[18px] items-center justify-center rounded-full">
        <Icon size={11} strokeWidth={2} className={TONE_TEXT[tone]} />
      </span>
    </span>
  );
}

function EventRow({ ev, open, onToggle }: { ev: SSEEvent; open: boolean; onToggle: () => void }) {
  const group = groupOf(ev.type);
  const { text, tone: toneOverride } = summarize(ev);
  const tone = toneOverride ?? GROUP_TONE[group] ?? "muted";
  return (
    <motion.li
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className="relative flex items-start"
    >
      <NodeIcon type={ev.type} tone={tone} />
      <button
        type="button"
        onClick={onToggle}
        className="min-w-0 flex-1 rounded-md px-2 py-1 text-left transition hover:bg-white/[0.03]"
      >
        <div className="flex items-baseline gap-2">
          <span className={cn("font-mono text-[10px] uppercase tracking-wider", TONE_TEXT[tone])}>{ev.type}</span>
          <span className="ml-auto shrink-0 font-mono text-[10px] text-ink-faint">{fmtTime(ev.receivedAt)}</span>
          <ChevronRight size={12} className={cn("shrink-0 text-ink-faint transition-transform", open && "rotate-90")} />
        </div>
        <p className={cn("mt-0.5 text-xs leading-relaxed", tone === "red" ? "text-neon-red" : "text-ink-soft")}>{text}</p>
        {open ? (
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-black/20 p-2 font-mono text-[10px] text-ink-faint">
            {JSON.stringify(ev.data, null, 2)}
          </pre>
        ) : null}
      </button>
    </motion.li>
  );
}

function TokenRow({ node, text, chunks, streaming }: { node: string; text: string; chunks: number; streaming: boolean }) {
  return (
    <motion.li initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="relative flex items-start">
      <NodeIcon type="token" tone="cyan" />
      <div className="min-w-0 flex-1 rounded-md px-2 py-1">
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-neon-cyan/70">token · {node}</span>
          <span className="font-mono text-[10px] text-ink-faint">{chunks} 段增量</span>
        </div>
        <p className="mt-0.5 max-h-28 overflow-hidden whitespace-pre-wrap break-words text-xs leading-relaxed text-ink-soft">
          {clip(text, 600)}
          {streaming ? <span className="ml-0.5 inline-block animate-pulse text-neon-cyan">▍</span> : null}
        </p>
      </div>
    </motion.li>
  );
}

function InterruptedRow({ ev }: { ev: SSEEvent }) {
  const d = ev.data ?? {};
  const payload = (d.payload ?? {}) as SSEData;
  const subs = list(payload, "sub_questions");
  return (
    <motion.li initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} className="relative flex items-start">
      <NodeIcon type="interrupted" tone="purple" />
      <div className="min-w-0 flex-1 rounded-lg border border-dashed border-neon-purple/60 bg-[color:var(--accent-purple-dim)] px-3 py-2">
        <div className="flex items-center gap-2">
          <PauseCircle size={14} className="text-neon-purple" />
          <span className="text-xs font-semibold text-neon-purple">已暂停 · 等待你确认子问题</span>
          <span className="ml-auto font-mono text-[10px] text-ink-faint">{fmtTime(ev.receivedAt)}</span>
        </div>
        <p className="mt-1 text-xs leading-relaxed text-ink-soft">
          {str(d, "reason") || "图在断点中断（HITL）。"}
          {str(d, "at") ? <span className="ml-1 font-mono text-[10px] text-ink-faint">@{str(d, "at")}</span> : null}
        </p>
        {subs.length > 0 ? (
          <p className="mt-1 font-mono text-[10px] text-ink-faint">待确认拆解 {subs.length} 项，编辑面板已可填写</p>
        ) : null}
        {str(d, "resume_token") ? (
          <p className="mt-1 break-all font-mono text-[10px] text-ink-faint">resume_token: {str(d, "resume_token")}</p>
        ) : null}
      </div>
    </motion.li>
  );
}

function UnknownRow({ ev, open, onToggle }: { ev: SSEEvent; open: boolean; onToggle: () => void }) {
  return (
    <motion.li initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="relative flex items-start">
      <NodeIcon type="?" tone="amber" />
      <button
        type="button"
        onClick={onToggle}
        className="min-w-0 flex-1 rounded-md border border-neon-amber/30 px-2 py-1 text-left transition hover:bg-white/[0.03]"
      >
        <div className="flex items-baseline gap-2">
          <span className="font-mono text-[10px] uppercase tracking-wider text-neon-amber">未知事件</span>
          <span className="font-mono text-[10px] text-ink-soft">{ev.name || ev.type}</span>
          <span className="ml-auto font-mono text-[10px] text-ink-faint">{fmtTime(ev.receivedAt)}</span>
        </div>
        <p className="mt-0.5 text-xs text-ink-faint">
          前端枚举未覆盖此类型（后端契约可能新增）—— 原始 JSON {open ? "收起" : "点击展开"}，绝不静默丢弃
        </p>
        {open ? (
          <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-all rounded-md bg-black/20 p-2 font-mono text-[10px] text-neon-amber/80">
            {JSON.stringify({ type: ev.type, name: ev.name, data: ev.data }, null, 2)}
          </pre>
        ) : null}
      </button>
    </motion.li>
  );
}

export default EventTimeline;

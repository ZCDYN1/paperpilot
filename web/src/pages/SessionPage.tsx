import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Send } from "lucide-react";
import { api } from "@/api/client";
import type { RunDetail } from "@/api/domain";
import { useSelectedRunId } from "@/hooks/useSelectedRunId";
import { PageHeader } from "@/components/ui/PageHeader";
import { Panel } from "@/components/ui/Panel";
import { Tag } from "@/components/ui/Tag";
import { NeonButton } from "@/components/ui/NeonButton";
import { Skeleton } from "@/components/ui/Skeleton";
import { EmptyState } from "@/components/ui/EmptyState";
import { FadeIn } from "@/components/ui/FadeIn";
import { TermChip } from "@/components/business/TermChip";

// 追问的 status 必须原样落界面：后端复用不了证据时如实返回 no_evidence，前端吞掉它就等于把"没答上来"伪装成"答完了"。

type Entry = {
  id: number;
  question: string;
  answer: string;
  status: string;
  at: string;
};

const STATUS_VIEW: Record<string, { label: string; tone: "green" | "red" | "amber" | "muted" }> = {
  ok: { label: "复用已校验证据作答", tone: "green" },
  no_evidence: { label: "本轮没有可复用的证据，未作答", tone: "red" },
  partial: { label: "只有部分证据可复用", tone: "amber" },
};

export function SessionPage() {
  const { runId } = useSelectedRunId();

  const [run, setRun] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(Boolean(runId));
  const [error, setError] = useState<string | null>(null);

  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [askError, setAskError] = useState<string | null>(null);
  const [history, setHistory] = useState<Entry[]>([]);

  const [termInput, setTermInput] = useState("");
  const [termResult, setTermResult] = useState<{ term: string; explanation: string } | null>(null);
  /** unknown 判定走 POST /terms/unknown：未命中是有效结果（200 + known=false），不是故障，要明说而不是静默留白 */
  const [termStatus, setTermStatus] = useState<"idle" | "loading" | "found" | "unknown" | "error">("idle");
  const [termError, setTermError] = useState("");
  const [unknownTerm, setUnknownTerm] = useState("");

  useEffect(() => {
    if (!runId) return;
    let alive = true;
    setLoading(true);
    setError(null);
    api.research
      .getRun(runId)
      .then((r) => alive && setRun(r))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)))
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, [runId]);

  const plainText = useMemo(() => extractPlainText(run?.draft ?? ""), [run]);

  const [matches, setMatches] = useState<Array<{ term: string; start: number; end: number }>>([]);
  const [defs, setDefs] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!plainText) return;
    let alive = true;
    api.terms
      .mark(plainText)
      .then(async (res) => {
        if (!alive) return;
        const list = res.matches.filter((m) => Number.isFinite(m.start) && Number.isFinite(m.end));
        setMatches(list);
        // 单个 lookup 失败只丢那一个解释，不再把全部标注静默清空
        const looked = await Promise.all(
          list.map((m) => api.terms.lookup(m.term).catch(() => null)),
        );
        if (!alive) return;
        setDefs(
          Object.fromEntries(
            looked
              .filter((r): r is { term: string; explanation: string } => r !== null)
              .map((r) => [r.term, r.explanation]),
          ),
        );
      })
      .catch(() => alive && setMatches([]));
    return () => {
      alive = false;
    };
  }, [plainText]);

  const send = () => {
    const q = input.trim();
    if (!q || !runId || sending) return;
    setSending(true);
    setAskError(null);
    api.research
      .followup(runId, q)
      .then((res) => {
        setHistory((h) => [
          {
            id: h.length + 1,
            question: q,
            answer: res.answer,
            status: res.status,
            at: new Date().toLocaleTimeString("en-GB"),
          },
          ...h,
        ]);
        setInput("");
      })
      .catch((e: unknown) => setAskError(e instanceof Error ? e.message : String(e)))
      .finally(() => setSending(false));
  };

  const queryTerm = () => {
    const term = termInput.trim();
    if (!term) return;
    setTermStatus("loading");
    setTermError("");
    setUnknownTerm("");
    setTermResult(null);
    api.terms
      .unknown([term])
      .then((res) => {
        const list = Array.isArray(res.unknown) ? res.unknown : [];
        if (list.some((t) => t.toLowerCase() === term.toLowerCase())) {
          setUnknownTerm(term);
          setTermStatus("unknown");
          return;
        }
        return api.terms.lookup(term).then((r) => {
          // 兜底：unknown 端点说收录了、lookup 却没给出解释时，仍按「暂无词条」显示，不渲染空白
          if (!r.explanation) {
            setUnknownTerm(r.term || term);
            setTermStatus("unknown");
            return;
          }
          setTermResult(r);
          setTermStatus("found");
        });
      })
      .catch((e: unknown) => {
        setTermError(e instanceof Error ? e.message : String(e));
        setTermStatus("error");
      });
  };

  const memory = run?.memory_context ?? [];

  return (
    <div className="flex flex-col gap-5">
      <PageHeader
        title="会话与记忆"
        description="在同一会话里继续追问，只消费上一轮已经读过并校验过的证据；长期记忆决定哪些论文不再作为新论文推荐。"
        features="F9 · F10 · F11"
        actions={
          runId ? (
            <Tag variant="cyan" mono>
              run {runId.slice(0, 12)}
            </Tag>
          ) : (
            <Tag variant="amber" mono>
              未指定 run
            </Tag>
          )
        }
      />

      {error && (
        <div className="glass panel rounded-xl px-4 py-3 text-sm text-neon-red">{error}</div>
      )}

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="会话内追问"
            subtitle="复用已读证据 · 不重新检索 · 不碰索引"
            extra={
              <span className="font-mono text-[11px] text-ink-faint">
                {history.length} 条追问
              </span>
            }
          >
            <div className="flex flex-col gap-3">
              <div className="flex items-end gap-2">
                <textarea
                  value={input}
                  rows={2}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
                  }}
                  placeholder="就上面报告里的结论继续问，例如：CRAG 的评估器本身有多少延迟？"
                  className="w-full resize-y rounded-lg border border-hairline bg-base/60 px-3 py-2 text-sm leading-6 text-ink outline-none transition-[border-color,box-shadow] duration-200 placeholder:text-ink-faint focus:border-neon-cyan/60 focus:shadow-[0_0_0_3px_rgba(0,240,255,0.08)]"
                />
                <NeonButton
                  loading={sending}
                  disabled={!input.trim() || !runId}
                  onClick={send}
                  className="h-10 shrink-0"
                >
                  {!sending && <Send size={15} aria-hidden={true} />}
                  追问
                </NeonButton>
              </div>

              <p className="rounded-lg border border-hairline bg-white/[0.03] px-3 py-2 text-[11px] leading-relaxed text-ink-soft">
                这条路只在<span className="text-neon-cyan">上一轮已读并通过校验的证据片段</span>
                内作答：不触发检索、不改索引，因此没有新的 token 大头开销。要问一个需要重新检索的新问题，请回研究控制台带上同一个{" "}
                <code className="font-mono text-neon-cyan">session_id</code> 重跑完整链路。
              </p>

              {askError && <p className="text-xs text-neon-red">追问失败：{askError}</p>}

              {history.length === 0 ? (
                <EmptyState
                  icon="messages-square"
                  title="还没有追问"
                  description="追问记录只存在本页 state，刷新即清空；会话真正的事实存在后端的 run 里。"
                />
              ) : (
                <ul className="flex flex-col gap-3">
                  {history.map((item) => {
                    const view = STATUS_VIEW[item.status] ?? {
                      label: "后端返回了未预置的状态",
                      tone: "muted" as const,
                    };
                    return (
                      <FadeIn key={item.id} y={8} duration={0.3}>
                        <li className="glass rounded-lg p-3">
                          <div className="flex flex-wrap items-center gap-2">
                            <span className="font-mono text-[10px] text-ink-faint">{item.at}</span>
                            <p className="min-w-0 flex-1 text-xs font-medium text-ink">
                              {item.question}
                            </p>
                            <Tag variant={view.tone} mono>
                              {item.status}
                            </Tag>
                          </div>
                          <p className="mt-2 text-[11px] text-ink-faint">{view.label}</p>
                          <p className="mt-1.5 text-sm leading-7 text-ink-soft">{item.answer}</p>
                        </li>
                      </FadeIn>
                    );
                  })}
                </ul>
              )}
            </div>
          </Panel>

          <Panel title="术语即时查询" subtitle="F11 · 词表里没有的术语会如实说没有">
            <div className="flex flex-col gap-3">
              <div className="flex items-center gap-2">
                <input
                  value={termInput}
                  onChange={(e) => setTermInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && queryTerm()}
                  placeholder="输入术语，例如 RRF / nDCG@10"
                  className="w-full rounded-lg border border-hairline bg-base/60 px-3 py-2 text-sm text-ink outline-none placeholder:text-ink-faint focus:border-neon-cyan/60"
                />
                <NeonButton
                  variant="secondary"
                  className="h-9 shrink-0"
                  loading={termStatus === "loading"}
                  disabled={!termInput.trim() || termStatus === "loading"}
                  onClick={queryTerm}
                >
                  查
                </NeonButton>
              </div>
              {termStatus === "loading" ? (
                <p className="text-[11px] text-ink-faint">查词表…</p>
              ) : termStatus === "unknown" ? (
                <p className="rounded-lg border border-neon-amber/40 bg-white/[0.04] px-3 py-2 text-xs leading-relaxed text-neon-amber">
                  词表内暂无该词条：「{unknownTerm}」。当前词表是人工维护的，
                  <b>未收录不等于这个术语不存在</b> —— 只是这里给不出解释，不是查询出错。
                </p>
              ) : termStatus === "error" ? (
                <p className="text-xs text-neon-red">术语查询失败：{termError}</p>
              ) : termResult && termStatus === "found" ? (
                <div className="rounded-lg border border-hairline bg-white/[0.03] px-3 py-2 text-xs leading-relaxed">
                  <span className="font-mono text-neon-cyan">{termResult.term}</span>
                  <span className="ml-2 text-ink-soft">{termResult.explanation}</span>
                </div>
              ) : (
                <p className="text-[11px] text-ink-faint">还没查过术语。</p>
              )}
            </div>
          </Panel>
        </div>

        <div className="flex min-w-0 flex-col gap-4">
          <Panel
            title="会话上下文"
            subtitle="本次召回到的长期记忆"
            extra={
              <span className="font-mono text-[11px] text-ink-faint">
                session {run?.session_id?.slice(0, 12) ?? "—"}
              </span>
            }
          >
            {loading ? (
              <div className="flex flex-col gap-2">
                <Skeleton height={30} />
                <Skeleton height={54} width="92%" />
                <Skeleton height={54} width="80%" />
              </div>
            ) : (
              <div className="flex flex-col gap-3">
                <div className="flex flex-wrap items-center gap-2 rounded-lg bg-white/[0.03] px-3 py-2">
                  <span className="text-[11px] text-ink-faint">session_id</span>
                  <code className="font-mono text-xs text-neon-cyan">
                    {run?.session_id || "—"}
                  </code>
                </div>

                <p className="text-[11px] leading-relaxed text-ink-soft">
                  下面的记忆条目是本次检索前召回的结果。它们不会改写报告，只影响
                  <span className="text-ink">推荐什么、不推荐什么</span>
                  —— 比如「为什么推荐列表里没有你已经读过的那些论文」，答案就在这一组召回里。
                </p>

                {memory.length === 0 ? (
                  <EmptyState
                    icon="inbox"
                    title="这次没有召回到长期记忆"
                    description="新会话或该 session 还没有积累偏好与阅读记录。"
                  />
                ) : (
                  <ul className="flex flex-col gap-2">
                    {memory.map((m, i) => {
                      const type = typeof m.type === "string" ? m.type : "memory";
                      const content = typeof m.content === "string" ? m.content : null;
                      const score = typeof m.score === "number" ? m.score : null;
                      return (
                        <li
                          key={i}
                          className="rounded-lg border-l-2 border-l-neon-purple/60 bg-white/[0.03] px-3 py-2"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <Tag variant="purple" mono>
                              {type}
                            </Tag>
                            <span className="tabular font-mono text-[10px] text-ink-faint">
                              {score == null ? "score —" : `score ${score.toFixed(2)}`}
                            </span>
                          </div>
                          <p className="mt-1.5 text-xs leading-relaxed text-ink-soft">
                            {content ?? "—"}
                          </p>
                        </li>
                      );
                    })}
                  </ul>
                )}
              </div>
            )}
          </Panel>

          <Panel
            title="报告术语标注"
            subtitle={`${matches.length} 处命中词表`}
            bodyClassName="max-h-[420px] overflow-y-auto"
          >
            {loading ? (
              <div className="flex flex-col gap-2">
                <Skeleton height={72} />
                <Skeleton height={72} width="88%" />
              </div>
            ) : plainText ? (
              <p className="text-sm leading-7 text-ink-soft">
                {renderWithTerms(plainText, matches, defs)}
              </p>
            ) : (
              <EmptyState icon="file-search" title="没有可标注的正文" description="报告为空时无术语可标。" />
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}

/** 只保留纯文本段落：Markdown 标记会干扰术语词面匹配 */
function extractPlainText(markdown: string): string {
  return markdown
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith("#") && !l.startsWith(">") && !l.startsWith("-") && !/^-{3,}$/.test(l))
    .join(" ")
    .replace(/[*`]/g, "")
    .replace(/\[\d+(?:\s*,\s*\d+)*\]/g, "")
    .slice(0, 900);
}

/** 按 mark 返回的字符区间切片渲染；重叠区间只保留先出现的那个，绝不让一个字符被两个 chip 包住 */
function renderWithTerms(
  text: string,
  matches: Array<{ term: string; start: number; end: number }>,
  defs: Record<string, string>,
): ReactNode[] {
  const sorted = [...matches].sort((a, b) => a.start - b.start);
  const nodes: ReactNode[] = [];
  let cursor = 0;
  sorted.forEach((m, i) => {
    if (m.start < cursor || m.end > text.length || m.end <= m.start) return;
    if (m.start > cursor) nodes.push(text.slice(cursor, m.start));
    const term = text.slice(m.start, m.end);
    nodes.push(
      <TermChip key={`${term}-${i}`} term={term} definition={defs[term]}>
        {term}
      </TermChip>,
    );
    cursor = m.end;
  });
  nodes.push(text.slice(cursor));
  return nodes;
}

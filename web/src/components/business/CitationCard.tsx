import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookmarkCheck, ChevronRight, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";
import { cn } from "@/lib/cn";

// 默认只给"论断 + 出处"，原文 chunk 展开才见 —— 溯源可查，但一屏要能扫几十条。

export interface CitationCardProps {
  claim: string;
  paper_title?: string;
  arxiv_id?: string;
  section?: string;
  chunk_text?: string;
  /**
   * ★ 三态（与后端 DB 列 `Boolean | None` 对齐）：
   * true=有原文支撑；false=无支撑·已剔除；null=当时根本没校验。
   * 判据与 `app/agents/nodes/writer.py`、`app/export/markdown.py` 同一套：只有 false 才算无支撑。
   */
  supported: boolean | null;
  /**
   * 证据强度（产品定义 §4）：`html`=正文原文；
   * `abstract`/`pdf`/缺失一律显示「摘要级证据」—— 后端导出同样宁低不高估。
   */
  source?: string | null;
  /** 后端定位方式（exact/quote/swapped_chunk），有就标出来 */
  resolution_how?: string | null;
  defaultExpanded?: boolean;
  className?: string;
}

function highlight(text: string, keywords: string[]): Array<{ t: string; hit: boolean }> {
  if (!text || keywords.length === 0) return [{ t: text, hit: false }];
  const escaped = keywords.map((k) => k.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const rx = new RegExp(`(${escaped.join("|")})`, "gi");
  const out: Array<{ t: string; hit: boolean }> = [];
  let last = 0;
  for (const m of text.matchAll(rx)) {
    const idx = m.index ?? 0;
    if (idx > last) out.push({ t: text.slice(last, idx), hit: false });
    out.push({ t: m[0], hit: true });
    last = idx + m[0].length;
  }
  if (last < text.length) out.push({ t: text.slice(last), hit: false });
  return out;
}

export function CitationCard({
  claim,
  paper_title,
  arxiv_id,
  section,
  chunk_text,
  supported,
  source,
  resolution_how,
  defaultExpanded = false,
  className,
}: CitationCardProps) {
  const [open, setOpen] = useState(defaultExpanded);

  const keywords = useMemo(() => {
    const words = (claim ?? "")
      .split(/[\s,。；;.!?？、]+/)
      .map((w) => w.trim())
      .filter((w) => (/[a-zA-Z]/.test(w) ? w.length >= 4 : w.length >= 2))
      .slice(0, 24);
    return Array.from(new Set(words));
  }, [claim]);

  const segments = useMemo(
    () => (chunk_text && open ? highlight(chunk_text, keywords) : []),
    [chunk_text, open, keywords],
  );

  // 三态徽章：true=有原文支撑 / false=无支撑·已剔除 / null=未校验（null 绝不写成"无支撑"）
  const verdict =
    supported === true
      ? {
          icon: <ShieldCheck size={11} />,
          label: "有原文支撑",
          cls: "bg-[color:var(--accent-cyan-dim)] text-neon-green",
        }
      : supported === false
        ? {
            icon: <ShieldAlert size={11} />,
            label: "无支撑 · 已剔除",
            cls: "bg-[color:var(--accent-cyan-dim)] text-neon-red",
          }
        : {
            icon: <ShieldQuestion size={11} />,
            label: "未校验",
            cls: "bg-white/[0.05] text-neon-amber",
          };

  return (
    <article
      className={cn(
        "glass rounded-lg border-l-2 px-3 py-2.5 transition-colors",
        supported === true
          ? "border-l-neon-green/70"
          : supported === false
            ? "border-l-neon-red"
            : "border-l-neon-amber/50",
        className,
      )}
    >
      <p className="text-xs leading-relaxed text-ink">{claim || "（空论断）"}</p>

      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-mono text-[10px]",
            verdict.cls,
          )}
        >
          {verdict.icon}
          {verdict.label}
        </span>
        {/* 证据强度：只有 html 才配叫"正文原文"，abstract/pdf/缺失一律"摘要级证据" */}
        <span
          className={cn(
            "rounded-full px-2 py-0.5 font-mono text-[10px]",
            source === "html"
              ? "bg-white/[0.04] text-ink-soft"
              : "bg-white/[0.04] text-neon-amber",
          )}
          title={source === "html" ? "证据来自论文 HTML 正文" : "未取到 HTML 正文：摘要降级 / 上传件 / 来源缺失"}
        >
          {source === "html" ? "正文原文" : "摘要级证据"}
        </span>
        {arxiv_id ? (
          <span className="inline-flex items-center gap-1 rounded-full bg-white/[0.04] px-2 py-0.5 font-mono text-[10px] text-neon-cyan">
            <BookmarkCheck size={11} />
            {arxiv_id}
          </span>
        ) : null}
        {section ? (
          <span className="rounded-full bg-white/[0.04] px-2 py-0.5 text-[10px] text-ink-soft">{section}</span>
        ) : null}
        {resolution_how ? (
          <span className="rounded-full bg-white/[0.04] px-2 py-0.5 font-mono text-[10px] text-ink-faint">
            定位: {resolution_how}
          </span>
        ) : null}
        {paper_title ? (
          <span className="min-w-0 truncate text-[10px] text-ink-faint" title={paper_title}>
            《{paper_title}》
          </span>
        ) : null}

        {chunk_text ? (
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="ml-auto inline-flex shrink-0 items-center gap-0.5 text-[10px] text-neon-cyan/80 hover:text-neon-cyan"
            aria-expanded={open}
          >
            {open ? "收起原文" : "查看原文"}
            <ChevronRight size={11} className={cn("transition-transform", open && "rotate-90")} />
          </button>
        ) : null}
      </div>

      <AnimatePresence initial={false}>
        {open && chunk_text ? (
          <motion.blockquote
            key="chunk"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <p className="mt-2 max-h-48 overflow-y-auto whitespace-pre-wrap break-words rounded-md bg-black/20 p-2 font-mono text-[11px] leading-relaxed text-ink-soft">
              {segments.map((s, i) =>
                s.hit ? (
                  <mark key={i} className="rounded-sm bg-[color:var(--accent-cyan-dim)] text-neon-cyan">
                    {s.t}
                  </mark>
                ) : (
                  <span key={i}>{s.t}</span>
                ),
              )}
            </p>
          </motion.blockquote>
        ) : null}
      </AnimatePresence>
    </article>
  );
}

export default CitationCard;

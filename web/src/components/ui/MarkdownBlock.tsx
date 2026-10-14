import { Fragment, type ReactNode } from "react";
import { cn } from "@/lib/cn";

// 只覆盖本项目导出模板（app/export/markdown.py）真正用到的语法子集，按行切分 + 单次行内 split，不追求 CommonMark 完整实现。

const HEADING_LEVELS = ["#", "##", "###", "####"] as const;

/** 行内只认三种：`**粗体**` / `` `代码` `` / `[1]` `[1,2]` 引用角标 */
const INLINE_RX = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+(?:\s*,\s*\d+)*\])/g;

function isBold(t: string) {
  return t.length >= 4 && t.startsWith("**") && t.endsWith("**");
}
function isCode(t: string) {
  return t.length >= 2 && t.startsWith("`") && t.endsWith("`") && !t.startsWith("``");
}
function citeNums(t: string): number[] | null {
  if (!/^\[\d+(?:\s*,\s*\d+)*\]$/.test(t)) return null;
  return t
    .slice(1, -1)
    .split(",")
    .map((s) => Number(s.trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
}

function parseInline(
  text: string,
  onCiteClick?: (index: number) => void,
): ReactNode[] {
  const parts = text.split(INLINE_RX).filter((p) => p !== "" && p !== undefined);
  return parts.map((tok, i) => {
    if (isBold(tok)) {
      return (
        <strong key={i} className="font-semibold text-ink">
          {parseInline(tok.slice(2, -2), undefined)}
        </strong>
      );
    }
    if (isCode(tok)) {
      return (
        <code
          key={i}
          className="rounded-[4px] border border-hairline bg-black/25 px-1 py-0.5 font-mono text-[0.85em] text-neon-cyan"
        >
          {tok.slice(1, -1)}
        </code>
      );
    }
    const nums = citeNums(tok);
    if (nums) {
      return (
        <span key={i} className="mx-0.5 inline-flex align-super gap-0.5 font-mono text-[10px]">
          {nums.map((n, j) =>
            onCiteClick ? (
              <button
                key={j}
                type="button"
                onClick={() => onCiteClick(n)}
                title={`跳到引用 [${n}] 的证据链`}
                className={cn(
                  "rounded-[4px] border border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)] px-1 leading-4",
                  "text-neon-cyan transition-colors duration-[var(--dur-fast)] hover:shadow-glow",
                )}
              >
                {n}
              </button>
            ) : (
              <span key={j} className="rounded-[4px] bg-white/[0.06] px-1 leading-4 text-ink-soft">
                {n}
              </span>
            ),
          )}
        </span>
      );
    }
    return <Fragment key={i}>{tok}</Fragment>;
  });
}

export type MarkdownBlockProps = {
  source: string;
  /** 点击 `[n]` 角标时的行为（一般是滚到第 n 条引用）；不传则角标只展示不可点 */
  onCiteClick?: (index: number) => void;
  className?: string;
};

export function MarkdownBlock({ source, onCiteClick, className }: MarkdownBlockProps) {
  const lines = (source ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let quote: string[] = [];
  let para: string[] = [];

  const flushList = () => {
    if (!list.length) return;
    const items = list;
    list = [];
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="my-2 list-disc space-y-1 pl-5 text-sm text-ink-soft">
        {items.map((it, i) => (
          <li key={i} className="leading-relaxed">
            {parseInline(it, onCiteClick)}
          </li>
        ))}
      </ul>,
    );
  };

  const flushQuote = () => {
    if (!quote.length) return;
    const qs = quote;
    quote = [];
    blocks.push(
      <blockquote
        key={`bq-${blocks.length}`}
        className="my-3 border-l-2 border-l-neon-cyan/60 bg-white/[0.03] px-3 py-2 text-sm italic leading-relaxed text-ink-soft"
      >
        {qs.map((q, i) => (
          <p key={i}>{parseInline(q, onCiteClick)}</p>
        ))}
      </blockquote>,
    );
  };

  const flushPara = () => {
    if (!para.length) return;
    const text = para.join(" ");
    para = [];
    blocks.push(
      <p key={`p-${blocks.length}`} className="my-2 text-sm leading-7 text-ink-soft">
        {parseInline(text, onCiteClick)}
      </p>,
    );
  };

  const flushAll = () => {
    flushPara();
    flushList();
    flushQuote();
  };

  lines.forEach((raw) => {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushAll();
      return;
    }
    if (/^(-{3,}|\*{3,})$/.test(line.trim())) {
      flushAll();
      blocks.push(<hr key={`hr-${blocks.length}`} className="my-4 border-hairline" />);
      return;
    }
    const level = HEADING_LEVELS.find((h) => line.startsWith(`${h} `));
    if (level) {
      flushAll();
      const text = line.slice(level.length + 1).trim();
      const size =
        level === "#"
          ? "text-lg text-ink"
          : level === "##"
            ? "text-base text-ink"
            : level === "###"
              ? "text-sm text-neon-cyan"
              : "text-sm text-ink";
      blocks.push(
        <p
          key={`h-${blocks.length}`}
          className={cn(
            "mb-1 mt-4 font-semibold tracking-tight first:mt-0",
            level === "###" && "font-display uppercase tracking-[0.06em]",
            size,
          )}
        >
          {parseInline(text, onCiteClick)}
        </p>,
      );
      return;
    }
    if (line.startsWith("> ")) {
      flushPara();
      flushList();
      quote.push(line.slice(2).trim());
      return;
    }
    if (line.startsWith("- ")) {
      flushPara();
      flushQuote();
      list.push(line.slice(2).trim());
      return;
    }
    flushList();
    flushQuote();
    para.push(line.trim());
  });
  flushAll();

  return <div className={cn("max-w-none", className)}>{blocks}</div>;
}

export default MarkdownBlock;

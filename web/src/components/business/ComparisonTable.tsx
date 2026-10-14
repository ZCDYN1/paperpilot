import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { cn } from "@/lib/cn";

// "缺失留空而非编造"是产品核心承诺：null/""/"未提及" 都归一成同一个空态占位（—），绝不用 0 或猜测文案顶替。

export type ComparisonRow = Record<string, string | number | null | undefined>;

export interface ComparisonTableProps {
  columns: string[];
  rows: ComparisonRow[];
  /** 行主键列（默认第一列）；仅用于 React key */
  rowKey?: string;
  maxHeight?: number | string;
  className?: string;
}

const HOLLOW_VALUES = new Set(["", "n/a", "na", "null", "none", "-", "—", "未提及", "未说明", "未知", "不适用", "无"]);

export function isMissing(v: unknown): boolean {
  if (v === null || v === undefined) return true;
  if (typeof v === "number") return Number.isNaN(v);
  if (typeof v !== "string") return false;
  const t = v.trim().toLowerCase();
  return HOLLOW_VALUES.has(t);
}

const LONG_TEXT = 56;

function Cell({ value }: { value: unknown }) {
  const [open, setOpen] = useState(false);
  if (isMissing(value)) {
    return (
      <span className="font-mono text-ink-faint/60" title="原文中未给出该字段（缺失留空，不编造）">
        —
      </span>
    );
  }
  const text = String(value);
  const long = text.length > LONG_TEXT;
  if (!long) return <span className="text-ink-soft">{text}</span>;
  return (
    <button
      type="button"
      onClick={() => setOpen((o) => !o)}
      className="block w-full text-left"
      title={open ? "收起" : "展开全文"}
    >
      <span className={cn("whitespace-pre-wrap break-words text-ink-soft", !open && "line-clamp-2")}>
        {text}
      </span>
      <span className="mt-0.5 flex items-center gap-0.5 text-[10px] text-neon-cyan/70">
        <ChevronDown size={11} className={cn("transition-transform", open && "rotate-180")} />
        {open ? "收起" : "展开"}
      </span>
    </button>
  );
}

export function ComparisonTable({ columns, rows, rowKey, maxHeight = 420, className }: ComparisonTableProps) {
  const cols = columns ?? [];
  const data = rows ?? [];
  const keyOf = (r: ComparisonRow, i: number) => {
    if (rowKey && !isMissing(r[rowKey])) return String(r[rowKey]);
    const first = cols.length ? r[cols[0]] : undefined;
    return isMissing(first) ? `row-${i}` : String(first);
  };

  if (cols.length === 0 || data.length === 0) {
    return (
      <div className={cn("glass panel flex h-40 items-center justify-center text-xs text-ink-faint", className)}>
        对比表尚未生成 —— 精读完成后按九列结构填充（缺字段显示为「—」，不会隐藏行）
      </div>
    );
  }

  return (
    <div
      className={cn("glass panel overflow-auto rounded-xl", className)}
      style={{ maxHeight }}
      role="region"
      aria-label="论文结构化对比表"
    >
      <table className="w-full min-w-max border-collapse text-xs">
        <thead>
          <tr>
            {cols.map((c, ci) => (
              <th
                key={c}
                scope="col"
                className={cn(
                  "hairline-b sticky top-0 z-20 bg-panel px-3 py-2.5 text-left font-semibold whitespace-nowrap text-ink",
                  ci === 0 && "left-0 z-30 border-r border-hairline",
                )}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((r, i) => (
            <tr key={keyOf(r, i)} className="group transition-colors hover:bg-white/[0.03]">
              {cols.map((c, ci) => (
                <td
                  key={c}
                  className={cn(
                    "border-b border-hairline px-3 py-2 align-top",
                    "min-w-[120px] max-w-[280px]",
                    ci === 0 &&
                      "sticky left-0 z-10 border-r border-hairline bg-panel group-hover:shadow-[inset_0_0_0_9999px_rgba(255,255,255,0.03)]",
                  )}
                >
                  <Cell value={r[c]} />
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ComparisonTable;

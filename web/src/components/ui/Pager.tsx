import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";

type PagerProps = {
  /** 1-based */
  page: number;
  pages: number;
  onChange: (page: number) => void;
  /** 每页的标题（报告的小节名 / 列表的页）。给了就额外渲染一行可点导航 */
  labels?: string[];
  /** 量词，默认"页" */
  unit?: string;
  /** 导航条的无障碍名 */
  ariaLabel?: string;
  className?: string;
};

/**
 * 长内容的分页条。
 *
 * 为什么要有它而不是"给容器加个 max-height 内部滚动"：报告正文一次能到 5000px，
 * 论文列表 32 条也一样 —— 内层滚动条和页面滚动条套在一起，滚轮落在哪一层取决于
 * 光标位置，用户会觉得"页面卡住了"。分完页之后每一栏都是有限高度，
 * 左右两栏才能对齐（这也是这两处排版问题的同一个根因）。
 *
 * `labels` 那一行是给报告用的：读者多半想直接跳到"三、训练代价"，
 * 而不是下一页下一页地翻。
 */
export function Pager({ page, pages, onChange, labels, unit = "页", ariaLabel = "分页", className }: PagerProps) {
  if (pages <= 1) return null;
  const safe = Math.min(Math.max(page, 1), pages);

  return (
    <div className={cn("flex flex-col gap-2", className)}>
      {labels && labels.length === pages ? (
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label={ariaLabel}>
          {labels.map((label, i) => (
            <button
              key={`${label}-${i}`}
              type="button"
              role="tab"
              aria-selected={safe === i + 1}
              onClick={() => onChange(i + 1)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-[11px] transition",
                safe === i + 1
                  ? "border-neon-cyan/50 bg-[color:var(--accent-cyan-dim)] text-neon-cyan"
                  : "border-hairline text-ink-faint hover:border-neon-cyan/30 hover:text-ink",
              )}
            >
              {label}
            </button>
          ))}
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => onChange(safe - 1)}
          disabled={safe <= 1}
          aria-label="上一页"
          className="inline-flex items-center gap-1 rounded-lg border border-hairline px-2.5 py-1 text-xs text-ink-soft transition hover:border-neon-cyan/50 hover:text-neon-cyan disabled:cursor-not-allowed disabled:opacity-40"
        >
          <ChevronLeft size={13} aria-hidden="true" /> 上一页
        </button>

        <span className="font-mono text-[11px] text-ink-faint">
          第 {safe} / {pages} {unit}
        </span>

        <button
          type="button"
          onClick={() => onChange(safe + 1)}
          disabled={safe >= pages}
          aria-label="下一页"
          className="inline-flex items-center gap-1 rounded-lg border border-hairline px-2.5 py-1 text-xs text-ink-soft transition hover:border-neon-cyan/50 hover:text-neon-cyan disabled:cursor-not-allowed disabled:opacity-40"
        >
          下一页 <ChevronRight size={13} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}

export default Pager;

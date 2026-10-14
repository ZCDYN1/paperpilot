import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, ShieldCheck, ShieldX } from "lucide-react";
import { cn } from "@/lib/cn";

// 剔除率 = 被砍掉的论断 / 全部论断；这个数字是"引用准确率"的另一面，缺数据时显示「—」而不是 0%。

export interface VerificationSummaryProps {
  /** 校验通过的论断数 */
  passedCount: number;
  /** 无支撑被剔除的论断 */
  droppedCount: number;
  /** 被剔除论断原文（可展开逐条看） */
  droppedClaims?: string[];
  /** 后端 verification 得分（0~1），可缺省 */
  score?: number | null;
  /** 修订轮次 */
  revisionCount?: number;
  /** 报告整体是否通过校验 */
  passed?: boolean | null;
  className?: string;
}

const fmtRatio = (n: number | null | undefined): string =>
  n == null || !Number.isFinite(n) ? "—" : `${Math.round(Math.min(Math.max(n, 0), 1) * 100)}%`;

export function VerificationSummary({
  passedCount,
  droppedCount,
  droppedClaims = [],
  score,
  revisionCount = 0,
  passed,
  className,
}: VerificationSummaryProps) {
  const [open, setOpen] = useState(false);
  const total = Math.max(0, passedCount) + Math.max(0, droppedCount);
  const dropRate = total > 0 ? droppedCount / total : null;

  const stats: Array<{ label: string; value: string; tone: string }> = [
    { label: "通过", value: String(Math.max(0, passedCount) || 0), tone: "text-neon-green" },
    { label: "剔除", value: String(Math.max(0, droppedCount) || 0), tone: "text-neon-red" },
    { label: "剔除率", value: fmtRatio(dropRate), tone: "text-neon-amber" },
    // score 后端给的是 0~1，必须换算成百分比再显示：
    // 原来直接 String(score) 会渲染成"得分 0.3077"，和旁边的"剔除率 33%"两种口径，读的人无从对照。
    { label: "通过率", value: score == null ? "—" : fmtRatio(score), tone: "text-neon-cyan" },
    { label: "修订", value: `${revisionCount} 轮`, tone: "text-ink-soft" },
  ];

  return (
    <section className={cn("glass panel p-4", className)}>
      <header className="flex items-center gap-2">
        {passed === false ? (
          <ShieldX size={16} className="text-neon-red" />
        ) : (
          <ShieldCheck size={16} className="text-neon-green" />
        )}
        <h3 className="text-sm font-semibold text-ink">引用校验结果</h3>
        {passed != null ? (
          <span
            className={cn(
              "ml-auto rounded-full px-2 py-0.5 font-mono text-[10px]",
              passed ? "bg-[color:var(--accent-cyan-dim)] text-neon-green" : "bg-[color:var(--accent-cyan-dim)] text-neon-red",
            )}
          >
            {passed ? "PASSED" : "FAILED"}
          </span>
        ) : null}
      </header>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">
        {stats.map((s) => (
          <div key={s.label} className="rounded-lg bg-white/[0.03] px-3 py-2">
            <p className="text-[10px] text-ink-faint">{s.label}</p>
            <p className={cn("tabular mt-0.5 font-mono text-lg font-semibold", s.tone)}>{s.value}</p>
          </div>
        ))}
      </div>

      <div className="mt-3 flex h-1.5 w-full overflow-hidden rounded-full bg-hairline">
        {total > 0 ? (
          <>
            <div
              className="h-full bg-neon-green/70"
              style={{ width: `${(Math.max(0, passedCount) / total) * 100}%` }}
            />
            <div
              className="h-full bg-neon-red/70"
              style={{ width: `${(Math.max(0, droppedCount) / total) * 100}%` }}
            />
          </>
        ) : null}
      </div>

      {droppedClaims.length > 0 ? (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 text-xs text-neon-cyan/80 hover:text-neon-cyan"
            aria-expanded={open}
          >
            <ChevronDown size={12} className={cn("transition-transform", open && "rotate-180")} />
            被剔除的论断（{droppedClaims.length} 条）
          </button>
          <AnimatePresence initial={false}>
            {open ? (
              <motion.ul
                key="list"
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: "auto", opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.18 }}
                className="mt-2 space-y-1.5 overflow-hidden"
              >
                {droppedClaims.map((c, i) => (
                  <li
                    key={i}
                    className="rounded-md border-l-2 border-l-neon-red/60 bg-black/20 px-2.5 py-1.5 text-xs leading-relaxed text-ink-soft"
                  >
                    {c || "（空论断）"}
                  </li>
                ))}
              </motion.ul>
            ) : null}
          </AnimatePresence>
        </div>
      ) : (
        <p className="mt-3 text-xs text-ink-faint">没有论断被剔除。</p>
      )}
    </section>
  );
}

export default VerificationSummary;

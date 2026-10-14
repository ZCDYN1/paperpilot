import { useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { BookText } from "lucide-react";
import { cn } from "@/lib/cn";

// 气泡用 fixed 定位并显式做视口夹取：靠屏幕边缘/贴底的术语解释不能溢出，也不允许遮挡自身。

export interface TermChipProps {
  term: string;
  definition?: string;
  source?: string;
  children?: ReactNode;
  className?: string;
}

const BUBBLE_W = 272;

type Placement = { side: "top" | "bottom"; left: number; arrow: number };

export function TermChip({ term, definition, source, children, className }: TermChipProps) {
  const anchorRef = useRef<HTMLSpanElement>(null);
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<Placement | null>(null);

  const show = () => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const half = BUBBLE_W / 2;
    let left = r.left + r.width / 2 - half;
    left = Math.max(8, Math.min(left, vw - BUBBLE_W - 8));
    const side: Placement["side"] = r.top > 190 || r.bottom + 160 > vh ? "top" : "bottom";
    setPlace({ side, left, arrow: r.left + r.width / 2 - left - 6 });
    setOpen(true);
  };

  const hide = () => setOpen(false);

  return (
    <>
      <span
        ref={anchorRef}
        tabIndex={0}
        role="button"
        aria-label={`术语 ${term} 的解释`}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onKeyDown={(e) => {
          if (e.key === "Escape") hide();
          if (e.key === "Enter" || e.key === " ") (open ? hide : show)();
        }}
        className={cn(
          "cursor-help rounded-sm underline decoration-dotted decoration-neon-cyan/50 underline-offset-4",
          "text-ink transition-colors hover:text-neon-cyan focus:outline-none focus-visible:ring-1 focus-visible:ring-neon-cyan/60",
          className,
        )}
      >
        {children ?? term}
      </span>

      <AnimatePresence>
        {open && place ? (
          <motion.span
            key="bubble"
            role="tooltip"
            initial={{ opacity: 0, y: place.side === "top" ? 4 : -4, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: place.side === "top" ? 4 : -4, scale: 0.97 }}
            transition={{ duration: 0.14 }}
            className="glass fixed z-50 block rounded-lg p-3 text-xs leading-relaxed"
            style={{
              width: BUBBLE_W,
              left: place.left,
              ...(place.side === "top"
                ? { bottom: window.innerHeight - (anchorRef.current?.getBoundingClientRect().top ?? 0) + 10 }
                : { top: (anchorRef.current?.getBoundingClientRect().bottom ?? 0) + 10 }),
            }}
          >
            <span
              className={cn(
                "pointer-events-none absolute left-6 h-2 w-2 rotate-45 border-hairline bg-panel",
                place.side === "top" ? "-bottom-1 border-b border-r" : "-top-1 border-l border-t",
              )}
              style={{ left: place.arrow }}
              aria-hidden
            />
            <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider text-neon-cyan">
              <BookText size={11} />
              {term}
            </span>
            <span className="mt-1.5 block text-ink-soft">
              {definition || "词表中暂无该术语的释义。"}
            </span>
            {source ? (
              <span className="mt-2 block border-t border-hairline pt-1.5 font-mono text-[10px] text-ink-faint">
                来源：{source}
              </span>
            ) : null}
          </motion.span>
        ) : null}
      </AnimatePresence>
    </>
  );
}

export default TermChip;

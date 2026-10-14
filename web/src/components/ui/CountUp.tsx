// 补间起点取"上一个显示值"而非 0：指标轮询刷新时数字连续滚动，不会每次从 0 猛跳。
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";

export type CountUpProps = {
  value: number;
  duration?: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  className?: string;
};

function prefersReducedMotion(): boolean {
  return (
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
}

export function CountUp({
  value,
  duration = 900,
  decimals = 0,
  prefix = "",
  suffix = "",
  className,
}: CountUpProps) {
  const [display, setDisplay] = useState(0);
  const fromRef = useRef(0);

  useEffect(() => {
    const from = fromRef.current;
    // document.hidden 也要直达终值：页面在后台时 rAF 完全不触发，
    // 补间会永远停在初始的 0，用户切回标签页之前看到的就是「指标全是 0」。
    if (prefersReducedMotion() || duration <= 0 || from === value || document.hidden) {
      fromRef.current = value;
      setDisplay(value);
      return;
    }
    let frame = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const current = from + (value - from) * eased;
      setDisplay(t < 1 ? current : value);
      if (t < 1) frame = requestAnimationFrame(tick);
      else fromRef.current = value;
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value, duration]);

  const text = Number.isFinite(display)
    ? display.toLocaleString("en-US", {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals,
      })
    : "—";

  return (
    <span className={cn("tabular", className)}>
      {prefix}
      {text}
      {suffix}
    </span>
  );
}

export default CountUp;

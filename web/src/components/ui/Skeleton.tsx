import type { CSSProperties } from "react";
import { cn } from "@/lib/cn";

export type SkeletonProps = {
  className?: string;
  style?: CSSProperties;
  /** 便捷尺寸：会被合进 style，数值按 px 处理 */
  width?: number | string;
  height?: number | string;
};

// 复用 globals.css 的 @keyframes scan：从左扫到右的高光条，暗示"这块正在加载"。
export function Skeleton({ className, style, width, height }: SkeletonProps) {
  return (
    <div
      aria-hidden={true}
      style={{ width, height, ...style }}
      className={cn(
        "relative overflow-hidden rounded-[8px] bg-white/[0.05]",
        className,
      )}
    >
      <div className="absolute inset-y-0 left-0 w-1/3 bg-gradient-to-r from-transparent via-white/[0.08] to-transparent [animation:scan_1.8s_linear_infinite]" />
    </div>
  );
}

export default Skeleton;

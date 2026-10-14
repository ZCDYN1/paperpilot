// 纯手写 SVG：Sparkline 尺寸太小，引 ECharts 是杀鸡用牛刀，还能省一次实例化开销。
import { useId } from "react";

export type SparklineProps = {
  data: number[];
  /** 任何合法 CSS 颜色，推荐传 var(--accent-*) */
  color?: string;
  width?: number;
  height?: number;
  showArea?: boolean;
  className?: string;
};

const PAD = 2;

export function Sparkline({
  data,
  color = "var(--accent-cyan)",
  width = 96,
  height = 28,
  showArea = true,
  className,
}: SparklineProps) {
  const gradientId = `spark-${useId().replace(/:/g, "")}`;
  const finite = data.filter((n) => Number.isFinite(n));

  if (finite.length === 0) {
    return (
      <svg width={width} height={height} className={className} aria-hidden={true}>
        <line
          x1={PAD}
          y1={height / 2}
          x2={width - PAD}
          y2={height / 2}
          stroke={color}
          strokeOpacity={0.25}
          strokeDasharray="3 3"
        />
      </svg>
    );
  }

  const rawMin = Math.min(...finite);
  const rawMax = Math.max(...finite);
  // 全相等时给一个虚拟量程，否则 (v-min)/(max-min) 会除零产生 NaN 路径
  const min = rawMax === rawMin ? rawMin - 1 : rawMin;
  const max = rawMax === rawMin ? rawMax + 1 : rawMax;

  const innerW = width - PAD * 2;
  const innerH = height - PAD * 2;
  const step = finite.length > 1 ? innerW / (finite.length - 1) : 0;
  const points = finite.map((v, i) => ({
    x: PAD + (finite.length > 1 ? i * step : innerW / 2),
    y: PAD + innerH - ((v - min) / (max - min)) * innerH,
  }));

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(" ");
  const last = points[points.length - 1];
  const areaPath = `${linePath} L${last.x.toFixed(1)},${height - PAD} L${points[0].x.toFixed(
    1,
  )},${height - PAD} Z`;

  return (
    <svg
      width={width}
      height={height}
      className={className}
      role="img"
      aria-label="趋势迷你图"
    >
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.3} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      {showArea && <path d={areaPath} fill={`url(#${gradientId})`} stroke="none" />}
      <path
        d={linePath}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle
        cx={last.x}
        cy={last.y}
        r={2}
        fill={color}
        style={{ filter: `drop-shadow(0 0 3px ${color})` }}
      />
    </svg>
  );
}

export default Sparkline;

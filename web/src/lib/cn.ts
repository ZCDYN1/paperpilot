import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * 合并 class 并让**后传入的 Tailwind 工具类真正覆盖先传入的**。
 *
 * 为什么不能只用 clsx：Tailwind v3 按数值升序生成工具类，`.px-3` 排在 `.px-4`
 * 之前，所以 `<button className={cn("px-4", override)} />` 里 override 传 `px-3`
 * 是**输**的（样式表顺序决定胜负，与 class 字符串里的先后无关）。
 * 全站大量组件靠"外部传 className 微调"来复用，这个坑会表现为
 * "明明传了小尺寸，按钮还是大号"，且极难定位。
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

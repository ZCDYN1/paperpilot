// 子项不要各自写 transition —— variants 会从 Stagger 容器沿组件树自动传播。
import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/cn";

export type StaggerProps = {
  children: ReactNode;
  /** 第一个子项之前的整体延迟（秒） */
  delay?: number;
  /** 相邻子项的间隔（秒） */
  stagger?: number;
  className?: string;
};

export function Stagger({
  children,
  delay = 0,
  stagger = 0.07,
  className,
}: StaggerProps) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={cn(className)}
      initial={reduce ? false : "hidden"}
      animate="show"
      variants={{
        hidden: {},
        show: { transition: { staggerChildren: stagger, delayChildren: delay } },
      }}
    >
      {children}
    </motion.div>
  );
}

export function StaggerItem({
  children,
  y = 14,
  className,
}: {
  children: ReactNode;
  y?: number;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      variants={{
        hidden: { opacity: 0, y },
        show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] } },
      }}
    >
      {children}
    </motion.div>
  );
}

export default Stagger;

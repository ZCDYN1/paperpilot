import type { ReactNode } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { cn } from "@/lib/cn";

export type FadeInProps = {
  children: ReactNode;
  /** 入场延迟（秒） */
  delay?: number;
  /** 起始下移距离（px） */
  y?: number;
  duration?: number;
  className?: string;
};

export function FadeIn({
  children,
  delay = 0,
  y = 14,
  duration = 0.45,
  className,
}: FadeInProps) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={cn(className)}
      initial={reduce ? false : { opacity: 0, y }}
      animate={{ opacity: 1, y: 0 }}
      transition={
        reduce
          ? { duration: 0 }
          : { duration, delay, ease: [0.16, 1, 0.3, 1] }
      }
    >
      {children}
    </motion.div>
  );
}

export default FadeIn;

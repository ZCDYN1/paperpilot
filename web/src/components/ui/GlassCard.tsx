import type {
  ComponentPropsWithoutRef,
  ElementType,
  ReactNode,
} from "react";
import { cn } from "@/lib/cn";

type Padding = "none" | "sm" | "md" | "lg";

const paddingClass: Record<Padding, string> = {
  none: "",
  sm: "p-3",
  md: "p-4",
  lg: "p-6",
};

// 自有 props 单独拆出来，否则 `Omit<..., keyof GlassCardProps>` 会构成类型别名自引用
interface GlassCardOwnProps<C extends ElementType> {
  as?: C;
  hover?: boolean;
  padding?: Padding;
  className?: string;
  children?: ReactNode;
}

export type GlassCardProps<C extends ElementType = "div"> = GlassCardOwnProps<C> &
  Omit<ComponentPropsWithoutRef<C>, keyof GlassCardOwnProps<C>>;

export function GlassCard<C extends ElementType = "div">(props: GlassCardProps<C>) {
  const { as, hover = false, padding = "md", className, children, ...rest } = props;
  const Tag = (as ?? "div") as ElementType;
  return (
    <Tag
      className={cn("glass panel", paddingClass[padding], hover && "glass-hover", className)}
      {...rest}
    >
      {children}
    </Tag>
  );
}

export default GlassCard;

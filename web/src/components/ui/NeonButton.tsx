import type { ButtonHTMLAttributes, ReactNode } from "react";
import { LoaderCircle } from "lucide-react";
import { cn } from "@/lib/cn";

type Variant = "primary" | "secondary" | "danger";

const variantClass: Record<Variant, string> = {
  primary:
    "border-[color:var(--accent-cyan-line)] bg-[var(--accent-cyan-dim)] text-neon-cyan hover:shadow-glow",
  secondary:
    "border-[color:var(--glass-border-strong)] bg-white/[0.04] text-ink-soft hover:bg-white/[0.08] hover:text-ink",
  danger:
    "border-[color:var(--accent-red)] bg-white/[0.04] text-neon-red hover:bg-white/[0.08]",
};

export type NeonButtonProps = {
  variant?: Variant;
  loading?: boolean;
  children?: ReactNode;
} & ButtonHTMLAttributes<HTMLButtonElement>;

export function NeonButton({
  variant = "primary",
  loading = false,
  disabled,
  className,
  children,
  ...rest
}: NeonButtonProps) {
  return (
    <button
      type="button"
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        "inline-flex select-none items-center justify-center gap-2 rounded-[10px] border px-4 py-2 text-sm font-medium",
        "transition-[background-color,box-shadow,color,transform] duration-[var(--dur-fast)]",
        "active:translate-y-px",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-cyan-line)]",
        "disabled:pointer-events-none disabled:opacity-50",
        variantClass[variant],
        className,
      )}
      {...rest}
    >
      {loading && <LoaderCircle size={15} className="animate-spin" aria-hidden={true} />}
      {children}
    </button>
  );
}

export default NeonButton;

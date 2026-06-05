import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost";

const variantStyles: Record<Variant, string> = {
  primary:
    "bg-accent text-bg hover:bg-accent-glow focus-visible:bg-accent-glow",
  secondary:
    "border border-border bg-bg-raised text-fg hover:border-accent " +
    "hover:text-accent-glow",
  ghost: "text-fg-muted hover:text-fg",
};

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
};

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", ...rest }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center px-5 py-2.5 text-sm " +
          "font-medium transition-colors duration-150 rounded-sm " +
          "disabled:opacity-50 disabled:pointer-events-none",
        variantStyles[variant],
        className,
      )}
      {...rest}
    />
  ),
);
Button.displayName = "Button";

// Link-styled button (since Next.js Link wraps an anchor).
export function LinkButton({
  href,
  variant = "primary",
  className,
  children,
}: {
  href: string;
  variant?: Variant;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      className={cn(
        "inline-flex items-center justify-center px-5 py-2.5 text-sm " +
          "font-medium transition-colors duration-150 rounded-sm",
        variantStyles[variant],
        className,
      )}
    >
      {children}
    </a>
  );
}

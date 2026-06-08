import * as React from "react";
import { cn } from "@/lib/utils";

export function Card({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "bg-bg-raised border border-border rounded-sm p-6",
        "transition-[border-color,transform,box-shadow,background-color] duration-200",
        "hover:border-warm/55 hover:-translate-y-0.5 hover:bg-bg-raised/95",
        "hover:shadow-[0_10px_28px_-14px_rgba(183,146,104,0.22),inset_0_0_0_1px_rgba(183,146,104,0.06)]",
        className,
      )}
      {...rest}
    />
  );
}

export function CardHeader({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("mb-3", className)} {...rest} />;
}

export function CardTitle({
  className,
  ...rest
}: React.HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn("font-serif text-lg text-fg leading-tight", className)}
      {...rest}
    />
  );
}

export function CardDescription({
  className,
  ...rest
}: React.HTMLAttributes<HTMLParagraphElement>) {
  return (
    <p className={cn("text-sm text-fg-muted mt-1", className)} {...rest} />
  );
}

export function CardContent({
  className,
  ...rest
}: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("text-sm text-fg", className)} {...rest} />;
}

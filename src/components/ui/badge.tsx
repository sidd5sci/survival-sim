import type { ReactNode } from "react";

type BadgeProps = {
  children: ReactNode;
  variant?: "default" | "secondary";
};

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function Badge({ children, variant = "default" }: BadgeProps) {
  const tone = variant === "secondary" ? "bg-slate-700 text-slate-100" : "bg-blue-700 text-white";
  return <span className={cx("rounded-full px-2 py-1 text-xs font-medium", tone)}>{children}</span>;
}

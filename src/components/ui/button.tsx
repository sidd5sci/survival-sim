import type { ButtonHTMLAttributes, ReactNode } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  children: ReactNode;
  variant?: "default" | "secondary" | "destructive";
};

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function Button({ children, className, variant = "default", ...props }: ButtonProps) {
  const tone =
    variant === "destructive"
      ? "bg-red-700 hover:bg-red-600 text-white"
      : variant === "secondary"
        ? "bg-slate-700 hover:bg-slate-600 text-slate-100"
        : "bg-blue-700 hover:bg-blue-600 text-white";

  return (
    <button
      type="button"
      className={cx("rounded-lg px-3 py-2 text-sm font-medium transition-colors disabled:opacity-50", tone, className)}
      {...props}
    >
      {children}
    </button>
  );
}

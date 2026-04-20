import type { ReactNode } from "react";

type DivProps = {
  children: ReactNode;
  className?: string;
};

function cx(...classes: Array<string | undefined>) {
  return classes.filter(Boolean).join(" ");
}

export function Card({ children, className }: DivProps) {
  return <div className={className}>{children}</div>;
}

export function CardHeader({ children, className }: DivProps) {
  return <div className={cx("p-5 pb-2", className)}>{children}</div>;
}

export function CardContent({ children, className }: DivProps) {
  return <div className={cx("p-5", className)}>{children}</div>;
}

export function CardTitle({ children, className }: DivProps) {
  return <h2 className={className}>{children}</h2>;
}

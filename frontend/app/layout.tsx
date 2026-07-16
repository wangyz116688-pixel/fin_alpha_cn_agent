import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AlphaAgent · 量化投资助手",
  description: "可解释的 A 股量化选股与策略表现演示系统",
  icons: { icon: "/favicon.svg" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="zh-CN"><body>{children}</body></html>;
}

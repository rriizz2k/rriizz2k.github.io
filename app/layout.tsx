import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Руслан Мирзоев — Data Science и Credit Risk Scoring",
  description: "Портфолио Руслана Мирзоева: кредитный скоринг, работающее ML-демо, данные, метрики и открытый API. Ищу стажировку / Junior Data Scientist.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body className="antialiased">{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "WordRoute — анализ заимствований в русском языке",
  description:
    "NLP-инструмент для выявления и анализа лексических заимствований в русском языке",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ru">
      <body className="min-h-screen" style={{ background: "var(--background)" }}>
        <header
          className="border-b px-6 py-4 flex items-center gap-3"
          style={{ borderColor: "var(--border)", background: "var(--surface)" }}
        >
          <div className="flex items-center gap-2">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
              <circle cx="14" cy="14" r="13" stroke="#4f8ef7" strokeWidth="2" />
              <path
                d="M8 14 L14 8 L20 14 L14 20 Z"
                stroke="#4f8ef7"
                strokeWidth="1.5"
                fill="none"
              />
              <circle cx="14" cy="14" r="2" fill="#4f8ef7" />
            </svg>
            <span
              className="text-lg font-semibold tracking-tight"
              style={{ color: "var(--text-primary)" }}
            >
              WordRoute
            </span>
          </div>
          <span
            className="text-sm ml-1"
            style={{ color: "var(--text-secondary)" }}
          >
            анализ лексических заимствований
          </span>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-8">{children}</main>
        <footer
          className="border-t mt-16 px-6 py-4 text-center text-sm"
          style={{ borderColor: "var(--border)", color: "var(--text-secondary)" }}
        >
          WordRoute · NLP-инструмент для анализа заимствований в русском языке ·{" "}
          <span style={{ color: "var(--accent)" }}>Computational Linguistics Project</span>
        </footer>
      </body>
    </html>
  );
}

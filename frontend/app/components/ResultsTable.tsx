"use client";
import { useState } from "react";
import type { WordResult } from "../../lib/types";

const DONOR_COLORS: Record<string, string> = {
  English: "#4f8ef7",
  French: "#a78bfa",
  German: "#60a5fa",
  "Greek/Latin": "#34d399",
  "Arabic/Persian": "#f59e0b",
  Turkic: "#f87171",
  Italian: "#fb923c",
  Dutch: "#38bdf8",
  Slavic: "#6b7280",
  Unknown: "#4b5563",
};

function ProbBadge({ value }: { value: number }) {
  const color =
    value >= 0.8
      ? "var(--green)"
      : value >= 0.5
      ? "var(--accent)"
      : value >= 0.3
      ? "var(--amber)"
      : "var(--text-secondary)";
  const bg =
    value >= 0.8
      ? "var(--green-dim)"
      : value >= 0.5
      ? "var(--accent-dim)"
      : value >= 0.3
      ? "var(--amber-dim)"
      : "rgba(100,100,120,0.1)";

  return (
    <span
      className="text-xs font-mono px-2 py-0.5 rounded-full font-semibold"
      style={{ color, background: bg }}
    >
      {(value * 100).toFixed(0)}%
    </span>
  );
}

interface Props {
  words: WordResult[];
  onWordClick: (word: WordResult) => void;
}

export default function ResultsTable({ words, onWordClick }: Props) {
  const [filter, setFilter] = useState<"all" | "borrowed" | "native">("all");
  const [sortBy, setSortBy] = useState<"probability" | "word" | "donor">("probability");

  const filtered = words.filter((w) => {
    if (filter === "borrowed") return w.is_loanword;
    if (filter === "native") return !w.is_loanword;
    return true;
  });

  const sorted = [...filtered].sort((a, b) => {
    if (sortBy === "probability") return b.loanword_probability - a.loanword_probability;
    if (sortBy === "word") return a.word.localeCompare(b.word, "ru");
    if (sortBy === "donor") return a.donor_language.localeCompare(b.donor_language);
    return 0;
  });

  return (
    <div>
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <div className="flex gap-1">
          {(["all", "borrowed", "native"] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className="px-3 py-1 rounded-full text-xs font-medium border transition-all"
              style={{
                background: filter === f ? "var(--accent)" : "transparent",
                color: filter === f ? "#fff" : "var(--text-secondary)",
                borderColor: filter === f ? "var(--accent)" : "var(--border)",
              }}
            >
              {f === "all" ? "Все" : f === "borrowed" ? "Заимствования" : "Исконные"}
            </button>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-2 text-xs" style={{ color: "var(--text-secondary)" }}>
          <span>Сортировать:</span>
          {(["probability", "word", "donor"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSortBy(s)}
              className="underline-offset-2"
              style={{
                color: sortBy === s ? "var(--accent)" : "var(--text-secondary)",
                textDecoration: sortBy === s ? "underline" : "none",
              }}
            >
              {s === "probability" ? "вероятность" : s === "word" ? "слово" : "донор"}
            </button>
          ))}
        </div>
      </div>

      {/* Table */}
      <div
        className="rounded-xl overflow-hidden border"
        style={{ borderColor: "var(--border)" }}
      >
        <table className="w-full text-sm">
          <thead>
            <tr
              className="text-xs uppercase tracking-wider"
              style={{ background: "var(--surface-2)", color: "var(--text-secondary)" }}
            >
              <th className="px-4 py-3 text-left">Слово</th>
              <th className="px-4 py-3 text-left">Лемма / POS</th>
              <th className="px-4 py-3 text-left">P(заимств.)</th>
              <th className="px-4 py-3 text-left">Язык-донор</th>
              <th className="px-4 py-3 text-left">Семант. поле</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((w, i) => {
              const donorColor = DONOR_COLORS[w.donor_language] ?? "var(--text-secondary)";
              return (
                <tr
                  key={`${w.word}-${i}`}
                  onClick={() => onWordClick(w)}
                  className="cursor-pointer transition-colors"
                  style={{
                    background: i % 2 === 0 ? "var(--surface)" : "transparent",
                    borderBottom: `1px solid var(--border)`,
                  }}
                  onMouseEnter={(e) =>
                    (e.currentTarget.style.background = "var(--surface-2)")
                  }
                  onMouseLeave={(e) =>
                    (e.currentTarget.style.background =
                      i % 2 === 0 ? "var(--surface)" : "transparent")
                  }
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium" style={{ color: "var(--text-primary)" }}>
                        {w.word}
                      </span>
                      {w.in_seed && (
                        <span
                          className="w-1.5 h-1.5 rounded-full"
                          style={{ background: "var(--green)" }}
                          title="В эталонной базе"
                        />
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <span style={{ color: "var(--text-secondary)" }}>
                      {w.lemma !== w.word ? w.lemma : "—"}
                    </span>
                    {w.pos && (
                      <span
                        className="ml-2 text-xs px-1.5 py-0.5 rounded"
                        style={{ background: "var(--border)", color: "var(--text-secondary)" }}
                      >
                        {w.pos}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <ProbBadge value={w.loanword_probability} />
                  </td>
                  <td className="px-4 py-3">
                    {w.is_loanword ? (
                      <div className="flex items-center gap-1.5">
                        <span
                          className="w-2 h-2 rounded-full"
                          style={{ background: donorColor }}
                        />
                        <span style={{ color: donorColor }}>
                          {w.donor_language_ru ?? w.donor_language}
                        </span>
                      </div>
                    ) : (
                      <span style={{ color: "var(--text-secondary)" }}>исконное</span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                      {w.semantic_domain_ru || w.semantic_domain || "—"}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {sorted.length === 0 && (
          <div
            className="py-12 text-center text-sm"
            style={{ color: "var(--text-secondary)" }}
          >
            Ничего не найдено
          </div>
        )}
      </div>
    </div>
  );
}

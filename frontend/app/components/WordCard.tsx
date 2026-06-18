"use client";
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
  Slavic: "var(--text-secondary)",
  Unknown: "var(--border)",
};

function ProbBar({ value }: { value: number }) {
  const color =
    value >= 0.8
      ? "var(--green)"
      : value >= 0.5
      ? "var(--accent)"
      : value >= 0.3
      ? "var(--amber)"
      : "var(--text-secondary)";

  return (
    <div className="flex items-center gap-2">
      <div className="prob-bar flex-1">
        <div
          className="prob-bar-fill"
          style={{ width: `${value * 100}%`, background: color }}
        />
      </div>
      <span className="text-xs font-mono font-medium" style={{ color }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  );
}

interface Props {
  word: WordResult;
  onClose: () => void;
}

export default function WordCard({ word, onClose }: Props) {
  const donorColor = DONOR_COLORS[word.donor_language] ?? "var(--accent)";

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(0,0,0,0.7)" }}
      onClick={onClose}
    >
      <div
        className="relative max-w-2xl w-full rounded-2xl overflow-y-auto max-h-[90vh]"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="px-6 py-5 border-b flex items-start justify-between"
          style={{ borderColor: "var(--border)" }}
        >
          <div>
            <div className="flex items-center gap-3 mb-1">
              <span className="text-2xl font-bold" style={{ color: "var(--text-primary)" }}>
                {word.word}
              </span>
              {word.is_loanword ? (
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{ background: "var(--green-dim)", color: "var(--green)" }}
                >
                  заимствование
                </span>
              ) : (
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-medium"
                  style={{ background: "rgba(100,100,120,0.2)", color: "var(--text-secondary)" }}
                >
                  исконное
                </span>
              )}
              {word.in_seed && (
                <span
                  className="text-xs px-2 py-0.5 rounded-full"
                  style={{ background: "var(--accent-dim)", color: "var(--accent)" }}
                >
                  в базе
                </span>
              )}
            </div>
            <span className="text-sm" style={{ color: "var(--text-secondary)" }}>
              лемма: <span style={{ color: "var(--text-primary)" }}>{word.lemma}</span>
              {word.pos && (
                <>
                  {" · "}
                  <span>{word.pos}</span>
                </>
              )}
              {word.gender && <> · {word.gender}</>}
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-xl leading-none p-1 rounded hover:bg-white/5"
            style={{ color: "var(--text-secondary)" }}
          >
            ✕
          </button>
        </div>

        <div className="p-6 space-y-5">
          {/* Probability */}
          <div>
            <div className="flex justify-between text-xs mb-2" style={{ color: "var(--text-secondary)" }}>
              <span>Вероятность заимствования</span>
              <span className="font-mono">{word.loanword_probability.toFixed(3)}</span>
            </div>
            <ProbBar value={word.loanword_probability} />
          </div>

          {/* Donor info */}
          {word.is_loanword && (
            <div
              className="rounded-xl p-4 space-y-2"
              style={{ background: "var(--surface-2)" }}
            >
              <div className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ background: donorColor }}
                />
                <span className="font-semibold" style={{ color: donorColor }}>
                  {word.donor_language_ru ?? word.donor_language}
                </span>
                {word.donor_family && (
                  <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                    · {word.donor_family}
                  </span>
                )}
              </div>
              {word.source_word && (
                <div className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  Слово-источник:{" "}
                  <span
                    className="font-mono font-medium px-1.5 py-0.5 rounded"
                    style={{ background: "var(--border)", color: "var(--text-primary)" }}
                  >
                    {word.source_word}
                  </span>
                </div>
              )}
              {word.semantic_domain_ru && (
                <div className="text-sm" style={{ color: "var(--text-secondary)" }}>
                  Семантическое поле:{" "}
                  <span style={{ color: "var(--text-primary)" }}>{word.semantic_domain_ru}</span>
                </div>
              )}
            </div>
          )}

          {/* Glottolog */}
          {word.glottolog && word.is_loanword && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>
                Glottolog
              </div>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                {[
                  ["Код", word.glottolog.glottocode],
                  ["Семья", word.glottolog.family],
                  ["Подгруппа", word.glottolog.subfamily],
                  ["Регион", word.glottolog.macroarea],
                  ["Страна", word.glottolog.country],
                  ["ISO 639", word.glottolog.iso639],
                ]
                  .filter(([, v]) => v)
                  .map(([k, v]) => (
                    <div key={k} className="flex gap-2">
                      <span style={{ color: "var(--text-secondary)" }}>{k}:</span>
                      <span style={{ color: "var(--text-primary)" }}>{v}</span>
                    </div>
                  ))}
              </div>
              {word.glottolog.description && (
                <p className="mt-2 text-xs leading-relaxed" style={{ color: "var(--text-secondary)" }}>
                  {word.glottolog.description}
                </p>
              )}
            </div>
          )}

          {/* Top donors */}
          {word.top_donors.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>
                Вероятные источники
              </div>
              <div className="space-y-2">
                {word.top_donors.slice(0, 3).map((d) => (
                  <div key={d.language} className="flex items-center gap-3">
                    <span
                      className="text-sm w-28 shrink-0"
                      style={{ color: DONOR_COLORS[d.language] ?? "var(--text-primary)" }}
                    >
                      {d.language}
                    </span>
                    <div className="flex-1 h-1.5 rounded-full" style={{ background: "var(--border)" }}>
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${d.probability * 100}%`,
                          background: DONOR_COLORS[d.language] ?? "var(--accent)",
                        }}
                      />
                    </div>
                    <span className="text-xs font-mono w-10 text-right" style={{ color: "var(--text-secondary)" }}>
                      {(d.probability * 100).toFixed(0)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Morphological derivatives */}
          {word.morphological_derivatives.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>
                Морфологическая адаптация
              </div>
              <div className="flex flex-wrap gap-2">
                {word.morphological_derivatives.map((d) => (
                  <span
                    key={d}
                    className="text-sm px-2.5 py-1 rounded-lg font-mono"
                    style={{ background: "var(--surface-2)", color: "var(--text-primary)" }}
                  >
                    {d}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Explanation */}
          {word.explanation.length > 0 && (
            <div>
              <div className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: "var(--text-secondary)" }}>
                Признаки заимствования
              </div>
              <ul className="space-y-1.5">
                {word.explanation.map((r, i) => (
                  <li key={i} className="flex gap-2 text-sm">
                    <span style={{ color: "var(--accent)" }} className="shrink-0 mt-0.5">
                      ·
                    </span>
                    <span style={{ color: "var(--text-secondary)" }}>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

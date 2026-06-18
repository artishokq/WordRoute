"use client";
import { useState } from "react";
import InputForm from "./components/InputForm";
import ResultsTable from "./components/ResultsTable";
import WordCard from "./components/WordCard";
import DonorChart from "./components/DonorChart";
import { analyzeText } from "../lib/api";
import type { AnalyzeResponse, AnalysisMode, WordResult } from "../lib/types";

export default function HomePage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [selectedWord, setSelectedWord] = useState<WordResult | null>(null);

  const handleSubmit = async (input: string, mode: AnalysisMode) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await analyzeText(input, mode);
      setResult(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Ошибка запроса к API");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-10">
      {/* Hero */}
      <div className="max-w-2xl">
        <h1
          className="text-3xl font-bold mb-3 tracking-tight"
          style={{ color: "var(--text-primary)" }}
        >
          Анализ лексических заимствований
        </h1>
        <p className="text-base leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          Введите русское слово, список слов или текст – система автоматически
          выделит заимствования, определит язык-источник и объяснит причины.
        </p>
      </div>

      {/* Input */}
      <div
        className="rounded-2xl p-6 border"
        style={{ background: "var(--surface)", borderColor: "var(--border)" }}
      >
        <InputForm onSubmit={handleSubmit} loading={loading} />
      </div>

      {/* Error */}
      {error && (
        <div
          className="rounded-xl p-4 text-sm border"
          style={{
            background: "var(--red-dim)",
            borderColor: "var(--red)",
            color: "var(--red)",
          }}
        >
          {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-8">
          {/* Charts */}
          <div
            className="rounded-2xl p-6 border"
            style={{ background: "var(--surface)", borderColor: "var(--border)" }}
          >
            <h2
              className="text-base font-semibold mb-5"
              style={{ color: "var(--text-primary)" }}
            >
              Статистика анализа
            </h2>
            <DonorChart stats={result.stats} />
          </div>

          {/* Table */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
                Результаты
                <span
                  className="ml-2 text-sm font-normal"
                  style={{ color: "var(--text-secondary)" }}
                >
                  {result.words.length} слов проанализировано
                </span>
              </h2>
              <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
                Нажмите на слово для подробной карточки
              </span>
            </div>
            <ResultsTable words={result.words} onWordClick={setSelectedWord} />
          </div>
        </div>
      )}

      {/* Empty state */}
      {!result && !loading && !error && (
        <div
          className="rounded-2xl p-12 text-center border"
          style={{ background: "var(--surface)", borderColor: "var(--border)" }}
        >
          <div className="text-4xl mb-4">→</div>
          <p className="text-sm" style={{ color: "var(--text-secondary)" }}>
            Результаты появятся здесь после анализа
          </p>
          <p className="text-xs mt-2" style={{ color: "var(--text-secondary)" }}>
            Попробуйте:{" "}
            <button
              className="underline"
              style={{ color: "var(--accent)" }}
              onClick={() =>
                handleSubmit(
                  "В офисе обсуждали маркетинг и новый дизайн сайта. Менеджер предложил провести брифинг.",
                  "text"
                )
              }
            >
              пример текста
            </button>
          </p>
        </div>
      )}

      {/* Word card modal */}
      {selectedWord && (
        <WordCard word={selectedWord} onClose={() => setSelectedWord(null)} />
      )}
    </div>
  );
}

"use client";
import { useState } from "react";
import type { AnalysisMode } from "../../lib/types";

const EXAMPLES = {
  word: "маркетинг, компьютер, базар, философия",
  text: "В офисе обсуждали маркетинг и новый дизайн сайта. Менеджер предложил провести брифинг с инвесторами.",
  batch: "компьютер\nмаркетинг\nресторан\nфутбол\nокно\nлес",
};

interface Props {
  onSubmit: (input: string, mode: AnalysisMode) => void;
  loading: boolean;
}

export default function InputForm({ onSubmit, loading }: Props) {
  const [mode, setMode] = useState<AnalysisMode>("text");
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) onSubmit(input.trim(), mode);
  };

  const loadExample = () => setInput(EXAMPLES[mode]);

  const modeLabels: Record<AnalysisMode, string> = {
    word: "Слово / список",
    text: "Текст",
    batch: "Batch",
  };

  const modeHints: Record<AnalysisMode, string> = {
    word: "Введите одно слово или несколько через запятую",
    text: "Вставьте любой русский текст — система найдёт все потенциальные заимствования",
    batch: "Введите слова по одному на строке",
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      {/* Mode selector */}
      <div className="flex gap-2">
        {(["word", "text", "batch"] as AnalysisMode[]).map((m) => (
          <button
            key={m}
            type="button"
            onClick={() => setMode(m)}
            className="px-4 py-1.5 rounded-full text-sm font-medium transition-all border"
            style={{
              background: mode === m ? "var(--accent)" : "transparent",
              color: mode === m ? "#fff" : "var(--text-secondary)",
              borderColor: mode === m ? "var(--accent)" : "var(--border)",
            }}
          >
            {modeLabels[m]}
          </button>
        ))}
      </div>

      {/* Text input */}
      <div
        className="rounded-xl border overflow-hidden"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={modeHints[mode]}
          rows={mode === "batch" ? 6 : mode === "text" ? 5 : 2}
          className="w-full px-4 py-3 resize-none bg-transparent outline-none text-sm leading-relaxed"
          style={{ color: "var(--text-primary)" }}
        />
        <div
          className="flex items-center justify-between px-4 py-2 border-t"
          style={{ borderColor: "var(--border)" }}
        >
          <button
            type="button"
            onClick={loadExample}
            className="text-xs underline-offset-2 hover:underline"
            style={{ color: "var(--text-secondary)" }}
          >
            Загрузить пример
          </button>
          <span className="text-xs" style={{ color: "var(--text-secondary)" }}>
            {input.length} символов
          </span>
        </div>
      </div>

      {/* Submit */}
      <button
        type="submit"
        disabled={loading || !input.trim()}
        className="w-full py-3 rounded-xl font-semibold text-sm transition-all disabled:opacity-40"
        style={{
          background: "var(--accent)",
          color: "#fff",
        }}
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
              <path d="M12 2a10 10 0 0 1 10 10" stroke="#fff" strokeWidth="3" strokeLinecap="round" />
            </svg>
            Анализируем...
          </span>
        ) : (
          "Analyze borrowings"
        )}
      </button>
    </form>
  );
}

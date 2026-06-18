import type { AnalyzeResponse, AnalysisMode } from "./types";

const API_BASE = "/api";

export async function analyzeText(
  input: string,
  mode: AnalysisMode
): Promise<AnalyzeResponse> {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, mode }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }

  return res.json();
}

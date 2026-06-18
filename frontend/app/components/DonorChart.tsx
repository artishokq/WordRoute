"use client";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import type { AnalysisStats } from "../../lib/types";

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

const FALLBACK_COLORS = [
  "#4f8ef7","#a78bfa","#34d399","#f59e0b","#f87171","#fb923c","#38bdf8","#6b7280",
];

interface Props {
  stats: AnalysisStats;
}

export default function DonorChart({ stats }: Props) {
  const donorData = Object.entries(stats.donor_distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);

  const semanticData = Object.entries(stats.semantic_distribution)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const customTooltipStyle = {
    background: "var(--surface-2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text-primary)",
    fontSize: 12,
  };

  return (
    <div className="space-y-8">
      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: "Всего слов", value: stats.total_words },
          {
            label: "Заимствований",
            value: stats.borrowings_found,
            color: "var(--green)",
          },
          { label: "Исконных", value: stats.native_words },
          {
            label: "Доля заимств.",
            value: `${(stats.borrowing_rate * 100).toFixed(1)}%`,
            color: "var(--accent)",
          },
        ].map((s) => (
          <div
            key={s.label}
            className="rounded-xl p-4 text-center"
            style={{ background: "var(--surface-2)" }}
          >
            <div
              className="text-2xl font-bold mb-1"
              style={{ color: s.color ?? "var(--text-primary)" }}
            >
              {s.value}
            </div>
            <div className="text-xs" style={{ color: "var(--text-secondary)" }}>
              {s.label}
            </div>
          </div>
        ))}
      </div>

      {donorData.length > 0 && (
        <div className="grid grid-cols-2 gap-6">
          {/* Pie chart */}
          <div>
            <h3
              className="text-sm font-semibold mb-4"
              style={{ color: "var(--text-secondary)" }}
            >
              Языки-доноры
            </h3>
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={donorData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {donorData.map((entry, i) => (
                    <Cell
                      key={entry.name}
                      fill={DONOR_COLORS[entry.name] ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length]}
                    />
                  ))}
                </Pie>
                <Tooltip contentStyle={customTooltipStyle} />
                <Legend
                  formatter={(v) => (
                    <span style={{ color: "var(--text-secondary)", fontSize: 11 }}>{v}</span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Bar chart — semantic domains */}
          {semanticData.length > 0 && (
            <div>
              <h3
                className="text-sm font-semibold mb-4"
                style={{ color: "var(--text-secondary)" }}
              >
                Семантические области
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={semanticData}
                  layout="vertical"
                  margin={{ left: 8, right: 16 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="var(--border)"
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    tick={{ fill: "var(--text-secondary)", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    dataKey="name"
                    type="category"
                    width={80}
                    tick={{ fill: "var(--text-secondary)", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip contentStyle={customTooltipStyle} />
                  <Bar dataKey="value" fill="var(--accent)" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

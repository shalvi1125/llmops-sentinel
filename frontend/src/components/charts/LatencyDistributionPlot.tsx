import React from 'react';

interface LatencyDistributionPlotProps {
  latencies: number[];
  title?: string;
  emptyMessage?: string;
}

export const LatencyDistributionPlot: React.FC<LatencyDistributionPlotProps> = ({
  latencies,
  title = 'Latency Percentile Breakdown',
  emptyMessage = 'No execution timings recorded yet',
}) => {
  if (!latencies || latencies.length === 0) {
    return (
      <div className="w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col items-center justify-center min-h-[180px]">
        {title && <span className="text-xs font-semibold text-slate-300 self-start mb-2">{title}</span>}
        <div className="text-slate-500 text-xs italic my-auto">{emptyMessage}</div>
      </div>
    );
  }

  const sorted = [...latencies].sort((a, b) => a - b);
  const count = sorted.length;

  const min = Math.round(sorted[0]);
  const max = Math.round(sorted[count - 1]);
  const p50 = Math.round(sorted[Math.floor(count * 0.5)]);
  const p90 = Math.round(sorted[Math.floor(count * 0.9)]);
  const p95 = Math.round(sorted[Math.floor(count * 0.95)]);
  const mean = Math.round(sorted.reduce((acc, v) => acc + v, 0) / count);

  // Buckets
  const fast = sorted.filter((l) => l < 500).length;
  const normal = sorted.filter((l) => l >= 500 && l <= 1500).length;
  const slow = sorted.filter((l) => l > 1500).length;

  const fastPct = Math.round((fast / count) * 100);
  const normalPct = Math.round((normal / count) * 100);
  const slowPct = Math.round((slow / count) * 100);

  return (
    <div className="w-full bg-slate-950/60 border border-slate-800/90 rounded-xl p-4 flex flex-col space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-300">{title}</span>
        <span className="text-[10px] font-mono text-slate-500">n = {count} samples</span>
      </div>

      {/* Percentiles Metric Strip */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 text-center">
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">Min</span>
          <span className="text-xs font-bold font-mono text-slate-200">{min} ms</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">P50 (Median)</span>
          <span className="text-xs font-bold font-mono text-sky-400">{p50} ms</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">Mean</span>
          <span className="text-xs font-bold font-mono text-slate-200">{mean} ms</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">P90</span>
          <span className="text-xs font-bold font-mono text-indigo-400">{p90} ms</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">P95</span>
          <span className="text-xs font-bold font-mono text-amber-400">{p95} ms</span>
        </div>
        <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-2">
          <span className="text-[9px] font-mono text-slate-400 block uppercase">Max</span>
          <span className="text-xs font-bold font-mono text-rose-400">{max} ms</span>
        </div>
      </div>

      {/* Latency Range Bar */}
      <div className="space-y-1.5 pt-1">
        <div className="flex items-center justify-between text-[11px] text-slate-400">
          <span>Latency Distribution Tiers</span>
          <span className="font-mono text-[10px]">Fast (&lt;500ms) · Normal (0.5–1.5s) · Heavy (&gt;1.5s)</span>
        </div>
        <div className="h-3 w-full bg-slate-900 rounded-full overflow-hidden flex">
          {fastPct > 0 && (
            <div
              style={{ width: `${fastPct}%` }}
              className="bg-emerald-500 hover:brightness-110 transition-all"
              title={`Fast (<500ms): ${fast} queries (${fastPct}%)`}
            />
          )}
          {normalPct > 0 && (
            <div
              style={{ width: `${normalPct}%` }}
              className="bg-sky-500 hover:brightness-110 transition-all"
              title={`Normal (500-1500ms): ${normal} queries (${normalPct}%)`}
            />
          )}
          {slowPct > 0 && (
            <div
              style={{ width: `${slowPct}%` }}
              className="bg-amber-500 hover:brightness-110 transition-all"
              title={`Heavy (>1500ms): ${slow} queries (${slowPct}%)`}
            />
          )}
        </div>
        <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono">
          <span className="text-emerald-400">⚡ {fastPct}% Fast</span>
          <span className="text-sky-400">⏳ {normalPct}% Normal</span>
          <span className="text-amber-400">🐢 {slowPct}% Heavy</span>
        </div>
      </div>
    </div>
  );
};

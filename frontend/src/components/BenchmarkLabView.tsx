import React, { useState, useEffect } from 'react';
import {
  FlaskConical,
  Play,
  RefreshCw,
  Clock,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { getBenchmark, listBenchmarks, triggerBenchmark } from '../api';
import { BenchmarkRunResponse, BenchmarkSummary } from '../types';

interface BenchmarkLabViewProps {
  apiKey: string;
}

export const BenchmarkLabView: React.FC<BenchmarkLabViewProps> = ({ apiKey }) => {
  const [history, setHistory] = useState<BenchmarkSummary[]>([]);
  const [activeBenchmark, setActiveBenchmark] = useState<BenchmarkRunResponse | null>(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isTriggering, setIsTriggering] = useState(false);
  const [expandedCaseId, setExpandedCaseId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const data = await listBenchmarks(apiKey);
      setHistory(data);
      if (data.length > 0 && !activeBenchmark) {
        // Load the latest run by default
        const latest = await getBenchmark(data[0].benchmark_id, apiKey);
        setActiveBenchmark(latest);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load benchmarks.';
      setError(msg);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [apiKey]);

  // Responsible Polling: Halts automatically upon completed or failed status
  useEffect(() => {
    if (!activeBenchmark || activeBenchmark.status !== 'running') return;

    const interval = setInterval(async () => {
      try {
        const updated = await getBenchmark(activeBenchmark.benchmark_id, apiKey);
        setActiveBenchmark(updated);
        if (updated.status !== 'running') {
          // Refresh historical list when benchmark reaches terminal state
          const newHistory = await listBenchmarks(apiKey);
          setHistory(newHistory);
        }
      } catch {
        // Ignore background polling errors
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [activeBenchmark?.benchmark_id, activeBenchmark?.status, apiKey]);

  const handleStartBenchmark = async () => {
    setIsTriggering(true);
    setError(null);
    try {
      const started = await triggerBenchmark(apiKey);
      setActiveBenchmark(started);
      const newHistory = await listBenchmarks(apiKey);
      setHistory(newHistory);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to launch benchmark.';
      setError(msg);
    } finally {
      setIsTriggering(false);
    }
  };

  const handleSelectRun = async (id: string) => {
    try {
      const data = await getBenchmark(id, apiKey);
      setActiveBenchmark(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to select run.';
      setError(msg);
    }
  };

  // Progress percentage
  const progressPct =
    activeBenchmark && activeBenchmark.total_cases > 0
      ? Math.round((activeBenchmark.completed_cases / activeBenchmark.total_cases) * 100)
      : 0;

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col h-full overflow-hidden">
      {/* Top Header */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-slate-950/40 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-2">
          <FlaskConical className="w-4 h-4 text-sky-400" />
          <span className="font-semibold text-slate-200">RAG Evaluation Laboratory</span>
          <span className="text-slate-400">· Curated Golden Test Dataset Regression</span>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={fetchHistory}
            disabled={isLoadingHistory}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition text-xs border border-slate-700"
          >
            <RefreshCw
              className={`w-3.5 h-3.5 ${isLoadingHistory ? 'animate-spin text-sky-400' : ''}`}
            />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleStartBenchmark}
            disabled={isTriggering || activeBenchmark?.status === 'running'}
            className="flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white font-semibold transition text-xs shadow-md shadow-sky-600/20"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>
              {activeBenchmark?.status === 'running'
                ? 'Benchmark in Progress...'
                : 'Run Benchmark'}
            </span>
          </button>
        </div>
      </div>

      {/* Main Grid: Sidebar Runs & Benchmark Details */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        {/* Left Column: Historical Runs (4 cols) */}
        <div className="lg:col-span-4 border-r border-slate-800/80 p-4 overflow-y-auto space-y-2.5 bg-slate-950/40">
          <div className="flex items-center justify-between text-xs font-semibold text-slate-400 px-1 mb-1">
            <span>Regression Run History</span>
            <span className="text-[11px] font-mono bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">
              {history.length} Runs
            </span>
          </div>

          {history.length === 0 ? (
            <p className="text-xs text-slate-500 p-3 italic bg-slate-900/40 rounded-lg border border-slate-800">
              No benchmarks recorded in history yet.
            </p>
          ) : (
            history.map((h) => {
              const isSelected = activeBenchmark?.benchmark_id === h.benchmark_id;
              const dateStr = new Date(h.started_at).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                month: 'short',
                day: 'numeric',
              });

              return (
                <button
                  key={h.benchmark_id}
                  onClick={() => handleSelectRun(h.benchmark_id)}
                  className={`w-full text-left p-3 rounded-xl border transition text-xs ${
                    isSelected
                      ? 'bg-sky-500/15 border-sky-500/40 text-slate-100 shadow-sm'
                      : 'bg-slate-900/60 border-slate-800 hover:border-slate-700 text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between text-[11px] mb-1.5 font-mono">
                    <span className="font-semibold text-sky-400">{h.benchmark_id}</span>
                    <span
                      className={`px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase ${
                        h.status === 'completed'
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : h.status === 'running'
                          ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse'
                          : 'bg-rose-500/20 text-rose-400 border border-rose-500/30'
                      }`}
                    >
                      {h.status}
                    </span>
                  </div>

                  <div className="text-[11px] text-slate-400 flex items-center justify-between font-mono">
                    <span>{dateStr}</span>
                    {h.relevancy_pass_rate !== null && (
                      <span className="text-emerald-400 font-bold">
                        Pass: {h.relevancy_pass_rate}%
                      </span>
                    )}
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Right Column: Active Benchmark Dashboard (8 cols) */}
        <div className="lg:col-span-8 p-5 overflow-y-auto space-y-4">
          {error && (
            <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
              <span>{error}</span>
            </div>
          )}

          {!activeBenchmark ? (
            <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-slate-400 bg-slate-950/40 border border-slate-800/60 rounded-xl">
              <FlaskConical className="w-10 h-10 text-slate-600 mb-2" />
              <p className="text-xs font-semibold text-slate-300">No benchmark run selected</p>
              <p className="text-[11px] text-slate-500 mt-1">
                Click "Run Benchmark" above to start an automated test run.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Active Progress Bar if Running */}
              {activeBenchmark.status === 'running' && (
                <div className="bg-slate-950/90 border border-sky-500/30 rounded-xl p-4 space-y-2">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="text-sky-400 font-semibold flex items-center space-x-2">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-sky-400" />
                      </span>
                      <span>Benchmark Execution Active...</span>
                    </span>
                    <span className="text-slate-300">
                      {activeBenchmark.completed_cases} / {activeBenchmark.total_cases} Cases ({progressPct}%)
                    </span>
                  </div>
                  <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-sky-500 rounded-full transition-all duration-300"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Summary Cards */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Progress
                  </span>
                  <div className="text-lg font-bold text-slate-100 font-mono">
                    {activeBenchmark.completed_cases} / {activeBenchmark.total_cases}
                  </div>
                  <span className="text-[11px] text-slate-500 capitalize font-mono">
                    {activeBenchmark.status}
                  </span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Mean Latency
                  </span>
                  <div className="text-lg font-bold text-sky-400 font-mono">
                    {activeBenchmark.mean_latency_ms ? `${activeBenchmark.mean_latency_ms} ms` : '—'}
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">FastAPI Round-trip</span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Relevancy Pass
                  </span>
                  <div className="text-lg font-bold text-emerald-400 font-mono">
                    {activeBenchmark.relevancy_pass_rate !== null
                      ? `${activeBenchmark.relevancy_pass_rate}%`
                      : '—'}
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">Semantic Grounding</span>
                </div>

                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
                  <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
                    Safety Pass
                  </span>
                  <div className="text-lg font-bold text-indigo-400 font-mono">
                    {activeBenchmark.safety_pass_rate !== null
                      ? `${activeBenchmark.safety_pass_rate}%`
                      : '—'}
                  </div>
                  <span className="text-[11px] text-slate-500 font-mono">Zero Toxicity</span>
                </div>
              </div>

              {/* Benchmark Cases Table */}
              <div className="space-y-2">
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider block font-mono">
                  Test Cases Execution Breakdown
                </span>

                {activeBenchmark.cases.map((c) => {
                  const isExpanded = expandedCaseId === c.case_id;
                  return (
                    <div
                      key={c.case_id}
                      className="bg-slate-950/80 border border-slate-800 rounded-xl overflow-hidden hover:border-slate-700 transition text-xs"
                    >
                      <div
                        onClick={() => setExpandedCaseId(isExpanded ? null : c.case_id)}
                        className="p-3.5 flex flex-wrap items-center justify-between gap-2 cursor-pointer select-none"
                      >
                        <div className="flex-1 min-w-[200px]">
                          <div className="flex items-center space-x-2 text-[11px] text-slate-400 mb-0.5 font-mono">
                            <span className="font-semibold text-sky-400">{c.case_id}</span>
                            {c.latency_ms && (
                              <span className="flex items-center text-slate-300">
                                <Clock className="w-3 h-3 mr-1 text-slate-500" />
                                {c.latency_ms} ms
                              </span>
                            )}
                          </div>
                          <h5 className="font-medium text-slate-100">{c.question}</h5>
                        </div>

                        <div className="flex items-center space-x-2 font-mono">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-semibold uppercase ${
                              c.status === 'completed'
                                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                                : c.status === 'running'
                                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse'
                                : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }`}
                          >
                            {c.status}
                          </span>

                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-slate-400" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-slate-400" />
                          )}
                        </div>
                      </div>

                      {isExpanded && (
                        <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 space-y-3">
                          <div>
                            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1 font-mono">
                              Reference Ground Truth
                            </span>
                            <p className="text-[11px] text-slate-300 bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
                              {c.reference_answer}
                            </p>
                          </div>

                          <div>
                            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1 font-mono">
                              RAG Generated Answer
                            </span>
                            <p className="text-[11px] text-slate-200 bg-slate-950/90 p-2.5 rounded-lg border border-slate-800">
                              {c.generated_answer || 'Pending evaluation execution...'}
                            </p>
                          </div>

                          {c.relevancy_rationale && (
                            <div className="text-[11px] bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-slate-400">
                              <span className="font-semibold text-slate-300 font-mono">
                                Judge Scorer Evaluation:{' '}
                              </span>
                              {c.relevancy_rationale}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

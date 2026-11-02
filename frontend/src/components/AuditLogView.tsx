import React, { useState, useEffect } from 'react';
import {
  ScrollText,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Search,
  Filter,
  SlidersHorizontal,
} from 'lucide-react';
import { getEvaluationTraces } from '../api';
import { TraceAuditItem } from '../types';

interface AuditLogViewProps {
  apiKey: string;
}

export const AuditLogView: React.FC<AuditLogViewProps> = ({ apiKey }) => {
  const [traces, setTraces] = useState<TraceAuditItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [expandedTraceId, setExpandedTraceId] = useState<string | null>(null);
  const [searchFilter, setSearchFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState<'all' | 'pass' | 'fail' | 'skipped'>('all');
  const [sortBy, setSortBy] = useState<'newest' | 'longest' | 'shortest'>('newest');
  const [error, setError] = useState<string | null>(null);

  const fetchTraces = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getEvaluationTraces(apiKey, 50);
      setTraces(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to load evaluation traces.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchTraces();
  }, [apiKey]);

  // Aggregate stats across traces
  let totalPass = 0;
  let totalFail = 0;
  let totalSkipped = 0;

  traces.forEach((t) => {
    Object.values(t.assessments).forEach((a) => {
      const val = (a.value || '').toLowerCase();
      if (val === 'yes') totalPass++;
      else if (val === 'no') totalFail++;
      else totalSkipped++;
    });
  });

  const totalAssessments = totalPass + totalFail + totalSkipped;
  const passRate = totalAssessments > 0 ? Math.round((totalPass / totalAssessments) * 100) : null;
  const latencies = traces.map((t) => t.execution_time_ms).filter((ms) => ms > 0);
  const meanLatency =
    latencies.length > 0 ? Math.round(latencies.reduce((a, b) => a + b, 0) / latencies.length) : null;

  // Filtering & Sorting
  const filteredTraces = traces
    .filter((t) => {
      // Search text
      if (searchFilter.trim()) {
        const term = searchFilter.toLowerCase();
        const matchesQ = t.question && t.question.toLowerCase().includes(term);
        const matchesA = t.answer && t.answer.toLowerCase().includes(term);
        const matchesId = t.request_id.toLowerCase().includes(term);
        if (!matchesQ && !matchesA && !matchesId) return false;
      }

      // Status filter
      if (statusFilter === 'all') return true;

      const values = Object.values(t.assessments).map((a) => (a.value || '').toLowerCase());
      if (statusFilter === 'pass') {
        return values.includes('yes') && !values.includes('no');
      }
      if (statusFilter === 'fail') {
        return values.includes('no');
      }
      if (statusFilter === 'skipped') {
        return values.some((v) => v === 'skipped' || v === 'null');
      }
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'longest') return b.execution_time_ms - a.execution_time_ms;
      if (sortBy === 'shortest') return a.execution_time_ms - b.execution_time_ms;
      return b.timestamp_ms - a.timestamp_ms; // newest
    });

  const getVerdictBadge = (name: string, valObj?: { value: string; rationale: string | null }) => {
    if (!valObj) {
      return (
        <span className="text-[10px] text-slate-500 bg-slate-900 px-2 py-0.5 rounded border border-slate-800 font-mono">
          {name}: —
        </span>
      );
    }
    const val = valObj.value.toLowerCase();
    if (val === 'yes') {
      return (
        <span
          className="text-[10px] bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={valObj.rationale || ''}
        >
          <CheckCircle2 className="w-2.5 h-2.5 mr-0.5 text-emerald-400" />
          <span>{name}: Yes</span>
        </span>
      );
    }
    if (val === 'no') {
      return (
        <span
          className="text-[10px] bg-rose-500/15 text-rose-400 border border-rose-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={valObj.rationale || ''}
        >
          <XCircle className="w-2.5 h-2.5 mr-0.5 text-rose-400" />
          <span>{name}: No</span>
        </span>
      );
    }
    if (val === 'skipped') {
      return (
        <span
          className="text-[10px] bg-amber-500/15 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={valObj.rationale || ''}
        >
          <AlertTriangle className="w-2.5 h-2.5 mr-0.5 text-amber-400" />
          <span>{name}: Skipped</span>
        </span>
      );
    }
    return (
      <span
        className="text-[10px] bg-slate-800 text-slate-300 border border-slate-700 px-2 py-0.5 rounded font-mono"
        title={valObj.rationale || ''}
      >
        {name}: {valObj.value}
      </span>
    );
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col h-full overflow-hidden">
      {/* Top Bar */}
      <div className="px-5 py-3.5 border-b border-slate-800 bg-slate-950/40 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-2">
          <ScrollText className="w-4 h-4 text-sky-400" />
          <span className="font-semibold text-slate-200">MLflow Evaluation & Trace Audit</span>
          <span className="text-slate-400">· Real SQLite Store Records</span>
        </div>

        <div className="flex items-center space-x-2 sm:space-x-3">
          {/* Refresh Button */}
          <button
            onClick={fetchTraces}
            disabled={isLoading}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition disabled:opacity-50 border border-slate-700"
            title="Refresh Trace Logs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-sky-400' : ''}`} />
            <span>Refresh</span>
          </button>

          {/* External MLflow Link */}
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 border border-sky-500/30 transition font-medium"
          >
            <span>Open in MLflow</span>
            <ExternalLink className="w-3 h-3 ml-0.5" />
          </a>
        </div>
      </div>

      {/* Main Container */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Top Analytics Summary Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
              Recorded Traces
            </span>
            <div className="text-lg font-bold text-slate-100 font-mono">{traces.length}</div>
            <span className="text-[11px] text-slate-500 font-mono">SQLite MLflow Store</span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
              Assessment Pass Rate
            </span>
            <div className="text-lg font-bold text-emerald-400 font-mono">
              {passRate !== null ? `${passRate}%` : '—'}
            </div>
            <span className="text-[11px] text-slate-500 font-mono">
              {totalPass} of {totalAssessments} checks
            </span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
              Failed Assessments
            </span>
            <div className="text-lg font-bold text-rose-400 font-mono">{totalFail}</div>
            <span className="text-[11px] text-slate-500 font-mono">Requires investigation</span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
              Mean Trace Latency
            </span>
            <div className="text-lg font-bold text-sky-400 font-mono">
              {meanLatency !== null ? `${meanLatency} ms` : '—'}
            </div>
            <span className="text-[11px] text-slate-500 font-mono">Measured execution time</span>
          </div>
        </div>

        {/* Filter & Search Bar */}
        <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-3 flex flex-wrap items-center justify-between gap-3 text-xs">
          {/* Search Box */}
          <div className="relative flex-1 min-w-[200px]">
            <Search className="w-3.5 h-3.5 text-slate-500 absolute left-3 top-2.5" />
            <input
              type="text"
              value={searchFilter}
              onChange={(e) => setSearchFilter(e.target.value)}
              placeholder="Search by question, answer, or trace ID..."
              className="w-full bg-slate-900 border border-slate-700/80 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-sans"
            />
          </div>

          {/* Filter Status Pills */}
          <div className="flex items-center space-x-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-500 mr-1" />
            {(['all', 'pass', 'fail', 'skipped'] as const).map((mode) => (
              <button
                key={mode}
                onClick={() => setStatusFilter(mode)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition uppercase font-mono ${
                  statusFilter === mode
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {mode}
              </button>
            ))}
          </div>

          {/* Sort Dropdown */}
          <div className="flex items-center space-x-1.5 text-slate-400">
            <SlidersHorizontal className="w-3.5 h-3.5 text-slate-500" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-slate-900 border border-slate-700/80 text-slate-200 text-xs rounded-lg px-2 py-1 focus:outline-none focus:border-sky-500 font-mono"
            >
              <option value="newest">Newest First</option>
              <option value="longest">Longest Latency</option>
              <option value="shortest">Shortest Latency</option>
            </select>
          </div>
        </div>

        {/* Error notification */}
        {error && (
          <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300">
            {error}
          </div>
        )}

        {/* Traces List */}
        {filteredTraces.length === 0 && !isLoading ? (
          <div className="h-64 flex flex-col items-center justify-center text-center p-6 text-slate-400 bg-slate-950/40 border border-slate-800/60 rounded-xl">
            <ScrollText className="w-10 h-10 text-slate-600 mb-2" />
            <p className="text-xs font-semibold text-slate-300">
              No matching evaluation traces found
            </p>
            <p className="text-[11px] text-slate-500 mt-1">
              Ask questions in the Live Chat to generate traces and trigger judge evaluations.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {filteredTraces.map((trace) => {
              const isExpanded = expandedTraceId === trace.request_id;
              const formattedDate = new Date(trace.timestamp_ms).toLocaleString([], {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              });

              return (
                <div
                  key={trace.request_id}
                  className="bg-slate-950/80 border border-slate-800 rounded-xl overflow-hidden hover:border-slate-700 transition"
                >
                  {/* Collapsed Row Header */}
                  <div
                    onClick={() => setExpandedTraceId(isExpanded ? null : trace.request_id)}
                    className="p-3.5 flex flex-wrap items-center justify-between gap-3 cursor-pointer select-none"
                  >
                    <div className="flex-1 min-w-[240px]">
                      <div className="flex items-center space-x-2 text-[11px] text-slate-400 mb-1 font-mono">
                        <span className="text-sky-400 font-semibold">{trace.request_id}</span>
                        <span>·</span>
                        <span>{formattedDate}</span>
                        <span>·</span>
                        <span className="flex items-center text-slate-300">
                          <Clock className="w-3 h-3 mr-1 text-slate-500" />
                          {trace.execution_time_ms} ms
                        </span>
                      </div>

                      <h4 className="text-xs font-medium text-slate-100 line-clamp-1">
                        {trace.question || '(No recorded question text)'}
                      </h4>
                    </div>

                    {/* Assessment Badges Strip */}
                    <div className="flex flex-wrap items-center gap-1.5">
                      {getVerdictBadge('Relevancy', trace.assessments['AnswerRelevancy'])}
                      {getVerdictBadge('Hallucination', trace.assessments['Hallucination'])}
                      {getVerdictBadge('Safety', trace.assessments['Safety'])}

                      <div className="text-slate-500 ml-1">
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Expanded Details Pane */}
                  {isExpanded && (
                    <div className="px-5 pb-5 pt-3 border-t border-slate-800/80 bg-slate-900/40 text-xs space-y-4">
                      {/* Q&A Full View */}
                      <div className="space-y-1.5">
                        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block font-mono">
                          Question & Synthesized Answer
                        </span>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-3 text-slate-200 leading-relaxed select-text">
                          {trace.answer || 'No answer recorded in root span.'}
                        </div>
                      </div>

                      {/* Rationales Breakdown from Mistral 7B */}
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block font-mono">
                            Judge Assessment Rationales (Mistral 7B)
                          </span>
                          <span className="text-[10px] text-slate-500 font-mono">
                            MLflow Scorer Metadata
                          </span>
                        </div>

                        {Object.keys(trace.assessments).length === 0 ? (
                          <p className="text-[11px] text-slate-500 italic bg-slate-950/50 p-3 rounded-lg border border-slate-800/60">
                            No assessment rationales recorded for this trace yet.
                          </p>
                        ) : (
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                            {Object.entries(trace.assessments).map(([name, item]) => (
                              <div
                                key={name}
                                className="bg-slate-950/90 border border-slate-800 rounded-lg p-3.5 text-[11px] space-y-1.5"
                              >
                                <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
                                  <span className="font-semibold text-slate-200 font-mono">{name}</span>
                                  <span
                                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ${
                                      item.value.toLowerCase() === 'yes'
                                        ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                                        : item.value.toLowerCase() === 'no'
                                        ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                                    }`}
                                  >
                                    {item.value}
                                  </span>
                                </div>
                                <p className="text-slate-400 leading-relaxed max-h-40 overflow-y-auto pt-1 font-sans">
                                  {item.rationale || 'No explanation rationale provided by judge.'}
                                </p>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

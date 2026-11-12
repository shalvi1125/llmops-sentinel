import React, { useState, useEffect } from 'react';
import {
  ScrollText,
  RefreshCw,
  Clock,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  AlertOctagon,
  Ban,
  HelpCircle,
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
  const [statusFilter, setStatusFilter] = useState<
    'all' | 'pass' | 'fail' | 'cancelled' | 'skipped' | 'incomplete'
  >('all');
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

  const EXPECTED_METRICS = ['AnswerRelevancy', 'Hallucination', 'Safety'] as const;

  type AssessmentState = 'pass' | 'fail' | 'error' | 'cancelled' | 'skipped' | 'incomplete';

  const classifyAssessment = (val?: string | null): AssessmentState => {
    const normalized = (val || '').trim().toLowerCase();
    if (normalized === 'yes') return 'pass';
    if (normalized === 'no') return 'fail';
    if (normalized === 'error') return 'error';
    if (normalized === 'cancelled') return 'cancelled';
    if (normalized === 'skipped') return 'skipped';
    return 'incomplete';
  };

  // Aggregate stats across traces (keep pass rate strictly over evaluated yes/no verdicts)
  let totalPass = 0;
  let totalFail = 0;
  let totalError = 0;
  let totalCancelled = 0;
  let totalSkipped = 0;
  let totalIncomplete = 0;

  traces.forEach((t) => {
    const metricKeys = Array.from(
      new Set<string>([...EXPECTED_METRICS, ...Object.keys(t.assessments || {})])
    );
    metricKeys.forEach((key) => {
      const state = classifyAssessment(t.assessments?.[key]?.value);
      if (state === 'pass') totalPass++;
      else if (state === 'fail') totalFail++;
      else if (state === 'error') totalError++;
      else if (state === 'cancelled') totalCancelled++;
      else if (state === 'skipped') totalSkipped++;
      else totalIncomplete++;
    });
  });

  const totalEvaluated = totalPass + totalFail;
  const totalFailed = totalFail + totalError;
  const passRate = totalEvaluated > 0 ? Math.round((totalPass / totalEvaluated) * 100) : null;
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

      const metricKeys = Array.from(
        new Set<string>([...EXPECTED_METRICS, ...Object.keys(t.assessments || {})])
      );
      const states = metricKeys.map((k) => classifyAssessment(t.assessments?.[k]?.value));

      const hasPass = states.includes('pass');
      const hasFail = states.includes('fail');
      const hasError = states.includes('error');
      const hasCancelled = states.includes('cancelled');
      const hasSkipped = states.includes('skipped');
      const hasIncomplete = states.includes('incomplete');

      if (statusFilter === 'pass') {
        return hasPass && !hasFail && !hasError && !hasCancelled && !hasSkipped && !hasIncomplete;
      }
      if (statusFilter === 'fail') {
        return hasFail || hasError || t.evaluation_status === 'failed';
      }
      if (statusFilter === 'cancelled') {
        return t.evaluation_status === 'cancelled' || hasCancelled;
      }
      if (statusFilter === 'skipped') {
        return t.evaluation_status === 'skipped' || hasSkipped;
      }
      if (statusFilter === 'incomplete') {
        return (
          t.evaluation_status === 'pending' ||
          t.evaluation_status === 'running' ||
          hasIncomplete
        );
      }
      return true;
    })
    .sort((a, b) => {
      if (sortBy === 'longest') return b.execution_time_ms - a.execution_time_ms;
      if (sortBy === 'shortest') return a.execution_time_ms - b.execution_time_ms;
      return b.timestamp_ms - a.timestamp_ms; // newest
    });

  const getVerdictBadge = (
    name: string,
    valObj?: { value: string; rationale: string | null },
    evalStatus?: string | null
  ) => {
    const state = classifyAssessment(valObj?.value);
    const title = valObj?.rationale || valObj?.value || 'No assessment recorded';

    if (state === 'pass') {
      return (
        <span
          className="text-[10px] bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={title}
        >
          <CheckCircle2 className="w-2.5 h-2.5 mr-0.5 text-emerald-400" />
          <span>{name}: Yes</span>
        </span>
      );
    }
    if (state === 'fail') {
      return (
        <span
          className="text-[10px] bg-rose-500/15 text-rose-400 border border-rose-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={title}
        >
          <XCircle className="w-2.5 h-2.5 mr-0.5 text-rose-400" />
          <span>{name}: No</span>
        </span>
      );
    }
    if (state === 'error') {
      return (
        <span
          className="text-[10px] bg-orange-500/15 text-orange-300 border border-orange-500/40 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={title}
        >
          <AlertOctagon className="w-2.5 h-2.5 mr-0.5 text-orange-400" />
          <span>{name}: Error</span>
        </span>
      );
    }
    if (state === 'cancelled') {
      return (
        <span
          className="text-[10px] bg-violet-500/15 text-violet-300 border border-violet-500/35 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={title}
        >
          <Ban className="w-2.5 h-2.5 mr-0.5 text-violet-400" />
          <span>{name}: Cancelled</span>
        </span>
      );
    }
    if (state === 'skipped') {
      return (
        <span
          className="text-[10px] bg-amber-500/15 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
          title={title}
        >
          <AlertTriangle className="w-2.5 h-2.5 mr-0.5 text-amber-400" />
          <span>{name}: Skipped</span>
        </span>
      );
    }

    const isPending = evalStatus === 'pending' || evalStatus === 'running';
    return (
      <span
        className="text-[10px] bg-slate-900/90 text-sky-300/80 border border-dashed border-sky-500/30 px-2 py-0.5 rounded font-medium flex items-center space-x-1 font-mono"
        title={isPending ? 'Evaluation still in progress' : title}
      >
        {isPending ? (
          <Clock className="w-2.5 h-2.5 mr-0.5 text-sky-400" />
        ) : (
          <HelpCircle className="w-2.5 h-2.5 mr-0.5 text-slate-400" />
        )}
        <span>{name}: {isPending ? 'Pending' : 'Incomplete'}</span>
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
              {totalPass} of {totalEvaluated} evaluated
            </span>
          </div>

          <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3.5">
            <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1">
              Failed Assessments
            </span>
            <div className="text-lg font-bold text-rose-400 font-mono">{totalFailed}</div>
            <span className="text-[11px] text-slate-500 font-mono">
              {totalFail} fail · {totalError} err · {totalCancelled} cancel · {totalSkipped} skip · {totalIncomplete} inc
            </span>
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
          <div className="flex flex-wrap items-center gap-1.5">
            <Filter className="w-3.5 h-3.5 text-slate-500 mr-1" />
            {(
              [
                { id: 'all', label: 'all' },
                { id: 'pass', label: 'pass' },
                { id: 'fail', label: 'failed' },
                { id: 'cancelled', label: 'cancelled' },
                { id: 'skipped', label: 'skipped' },
                { id: 'incomplete', label: 'incomplete' },
              ] as const
            ).map(({ id, label }) => (
              <button
                key={id}
                onClick={() => setStatusFilter(id)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition uppercase font-mono ${
                  statusFilter === id
                    ? 'bg-sky-500/20 text-sky-300 border border-sky-500/40'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {label}
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
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px] text-slate-400 mb-1 font-mono">
                        <span className="text-sky-400 font-semibold">{trace.request_id}</span>
                        <span>·</span>
                        <span>{formattedDate}</span>
                        <span>·</span>
                        <span className="flex items-center text-slate-300">
                          <Clock className="w-3 h-3 mr-1 text-slate-500" />
                          Total: {trace.execution_time_ms} ms
                        </span>
                        {trace.model && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-sky-300 text-[10px]">
                            {trace.model}
                          </span>
                        )}
                        {trace.prompt_id && (
                          <span className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-indigo-300 text-[10px]">
                            {trace.prompt_id}
                            {trace.prompt_version ? ` v${trace.prompt_version}` : ''}
                          </span>
                        )}
                        {trace.evaluation_status && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] border uppercase ${
                              trace.evaluation_status === 'completed'
                                ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30'
                                : trace.evaluation_status === 'cancelled'
                                ? 'bg-violet-500/10 text-violet-300 border-violet-500/35'
                                : trace.evaluation_status === 'skipped'
                                ? 'bg-amber-500/10 text-amber-300 border-amber-500/30'
                                : trace.evaluation_status === 'failed'
                                ? 'bg-rose-500/10 text-rose-300 border-rose-500/30'
                                : trace.evaluation_status === 'running' || trace.evaluation_status === 'pending'
                                ? 'bg-sky-500/10 text-sky-300 border-sky-500/30'
                                : 'bg-slate-900 text-slate-400 border-slate-700'
                            }`}
                          >
                            Eval: {trace.evaluation_status}
                            {trace.evaluation_duration_ms && trace.evaluation_duration_ms > 0
                              ? ` (${(trace.evaluation_duration_ms / 1000).toFixed(1)}s)`
                              : ''}
                          </span>
                        )}
                      </div>

                      <h4 className="text-xs font-medium text-slate-100 line-clamp-1">
                        {trace.question || '(No recorded question text)'}
                      </h4>
                    </div>

                    {/* Assessment Badges Strip */}
                    <div className="flex flex-wrap items-center gap-1.5">
                      {getVerdictBadge('Relevancy', trace.assessments['AnswerRelevancy'], trace.evaluation_status)}
                      {getVerdictBadge('Hallucination', trace.assessments['Hallucination'], trace.evaluation_status)}
                      {getVerdictBadge('Safety', trace.assessments['Safety'], trace.evaluation_status)}

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
                      {/* Detailed Stage Telemetry & RAG Metadata */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2 text-[10px] font-mono">
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Retrieval</span>
                          <span className="text-sky-300 font-semibold">
                            {trace.retrieval_latency_ms !== null && trace.retrieval_latency_ms !== undefined
                              ? `${trace.retrieval_latency_ms} ms`
                              : '—'}
                          </span>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Generation</span>
                          <span className="text-indigo-300 font-semibold">
                            {trace.generation_latency_ms !== null && trace.generation_latency_ms !== undefined
                              ? `${trace.generation_latency_ms} ms`
                              : '—'}
                          </span>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Throughput</span>
                          <span className="text-emerald-300 font-semibold">
                            {trace.tokens_per_second ? `${trace.tokens_per_second} tok/s` : '—'}
                          </span>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Chunks / MMR</span>
                          <span className="text-slate-200 font-semibold">
                            {trace.unique_chunks_returned ?? '—'}/{trace.candidates_fetched ?? '—'}{' '}
                            {trace.use_mmr === true ? '· MMR' : trace.use_mmr === false ? '· Sim' : ''}
                          </span>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Scope</span>
                          <span className="text-slate-300 font-semibold truncate block">
                            {trace.document_scope || '—'}
                          </span>
                        </div>
                        <div className="bg-slate-950/90 border border-slate-800 rounded-lg p-2">
                          <span className="text-slate-500 block">Judge Job ({trace.evaluation_status || 'pending'})</span>
                          <span className="text-amber-300 font-semibold">
                            {trace.evaluation_duration_ms && trace.evaluation_duration_ms > 0
                              ? `${(trace.evaluation_duration_ms / 1000).toFixed(2)} s`
                              : '—'}
                          </span>
                        </div>
                      </div>

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

                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          {Array.from(
                            new Set<string>([
                              ...EXPECTED_METRICS,
                              ...Object.keys(trace.assessments || {}),
                            ])
                          ).map((name) => {
                            const item = trace.assessments?.[name];
                            const state = classifyAssessment(item?.value);
                            const displayLabel =
                              state === 'pass'
                                ? 'Yes'
                                : state === 'fail'
                                ? 'No'
                                : state === 'error'
                                ? 'Error'
                                : state === 'cancelled'
                                ? 'Cancelled'
                                : state === 'skipped'
                                ? 'Skipped'
                                : trace.evaluation_status === 'running' ||
                                  trace.evaluation_status === 'pending'
                                ? 'Pending'
                                : 'Incomplete';
                            const badgeStyle =
                              state === 'pass'
                                ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                                : state === 'fail'
                                ? 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                                : state === 'error'
                                ? 'bg-orange-500/15 text-orange-300 border border-orange-500/40'
                                : state === 'cancelled'
                                ? 'bg-violet-500/15 text-violet-300 border border-violet-500/35'
                                : state === 'skipped'
                                ? 'bg-amber-500/15 text-amber-300 border border-amber-500/30'
                                : 'bg-slate-900 text-sky-300/80 border border-dashed border-sky-500/30';

                            return (
                              <div
                                key={name}
                                className="bg-slate-950/90 border border-slate-800 rounded-lg p-3.5 text-[11px] space-y-1.5"
                              >
                                <div className="flex items-center justify-between border-b border-slate-800/80 pb-1.5">
                                  <span className="font-semibold text-slate-200 font-mono">{name}</span>
                                  <span
                                    className={`text-[10px] font-mono px-1.5 py-0.5 rounded font-bold uppercase ${badgeStyle}`}
                                  >
                                    {displayLabel}
                                  </span>
                                </div>
                                <p className="text-slate-400 leading-relaxed max-h-40 overflow-y-auto pt-1 font-sans">
                                  {item?.rationale ||
                                    (state === 'incomplete'
                                      ? 'No assessment verdict recorded for this metric on this trace.'
                                      : 'No explanation rationale provided by judge.')}
                                </p>
                              </div>
                            );
                          })}
                        </div>
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

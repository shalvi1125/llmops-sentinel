import React, { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Cpu,
  Database,
  ExternalLink,
  Flame,
  HardDrive,
  Layers,
  Radar,
  RefreshCw,
  ScrollText,
  Server,
  ShieldCheck,
  Terminal,
} from 'lucide-react';
import {
  getObservabilitySummary,
  getPerformanceMetrics,
  getRawPrometheusMetrics,
} from '../api';
import {
  ActiveTab,
  ObservabilitySummaryResponse,
  PerformanceSummaryResponse,
} from '../types';
import { KpiCard } from './common/KpiCard';
import { StatusBadge } from './common/StatusBadge';

interface ObservabilityViewProps {
  apiKey: string;
  onNavigateTab: (tab: ActiveTab) => void;
}

interface ParsedSample {
  name: string;
  labels: string;
  value: string;
}

const GRAFANA_DASHBOARDS = [
  {
    uid: 'sentinel-system-overview',
    slug: 'llmops-sentinel-01-system-overview',
    title: '01 · System Overview',
    focus: 'API request rate, 5xx ratio, P50/P95 /query latency, active requests & dependencies',
    metrics: ['http_requests_total', 'http_request_duration_seconds', 'service_up'],
  },
  {
    uid: 'sentinel-rag-pipeline',
    slug: 'llmops-sentinel-02-rag-pipeline-analytics',
    title: '02 · RAG Pipeline Analytics',
    focus: 'Ingestion rates, embed cache hit %, Chroma search vs MMR rerank latency, duplicate chunks filtered',
    metrics: ['retrieval_duration_seconds', 'query_embedding_duration_seconds', 'retrieval_duplicates_removed_total'],
  },
  {
    uid: 'sentinel-llm-performance',
    slug: 'llmops-sentinel-03-llm-performance-model-comparison',
    title: '03 · LLM Performance & Arena',
    focus: 'Llama 3.2 vs Mistral generation P50/P95, actual vs estimated token throughput & totals',
    metrics: ['llm_generation_duration_seconds', 'llm_tokens_generated_total', 'llm_tokens_per_second'],
  },
  {
    uid: 'sentinel-eval-judge',
    slug: 'llmops-sentinel-04-evaluation-judge-monitoring',
    title: '04 · Evaluation & Judge Lifecycle',
    focus: 'Completed/cancelled/skipped/failed jobs, preemptions by trigger, scorer duration, benchmark pass rates',
    metrics: ['eval_jobs_total', 'eval_preemptions_total', 'eval_scorer_duration_seconds'],
  },
  {
    uid: 'sentinel-infra-health',
    slug: 'llmops-sentinel-05-infrastructure-service-health',
    title: '05 · Infrastructure & Health',
    focus: 'Process RSS RAM, CPU %, backend uptime, Chroma docs/chunks, Ollama models, probe latency',
    metrics: ['process_memory_rss_bytes', 'process_cpu_percent', 'chromadb_documents'],
  },
];

function parsePrometheusLines(rawText: string): ParsedSample[] {
  const out: ParsedSample[] = [];
  for (const line of rawText.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) continue;
    const match = trimmed.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([^\s]+)$/);
    if (!match) continue;
    out.push({
      name: match[1],
      labels: match[2] || '',
      value: match[3],
    });
  }
  return out;
}

export const ObservabilityView: React.FC<ObservabilityViewProps> = ({
  apiKey,
  onNavigateTab,
}) => {
  const [summary, setSummary] = useState<ObservabilitySummaryResponse | null>(null);
  const [perf, setPerf] = useState<PerformanceSummaryResponse | null>(null);
  const [rawMetrics, setRawMetrics] = useState<string>('');
  const [filterText, setFilterText] = useState<string>('llmops_sentinel_');
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const loadObservability = async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const [obsRes, perfRes, promText] = await Promise.all([
        getObservabilitySummary(apiKey),
        getPerformanceMetrics(apiKey, 100).catch(() => null),
        getRawPrometheusMetrics().catch(() => ''),
      ]);
      setSummary(obsRes);
      setPerf(perfRes);
      setRawMetrics(promText);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      setErrorMsg(err instanceof Error ? err.message : 'Could not load observability summary');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadObservability();
  }, [apiKey]);

  const parsedSamples = useMemo(() => {
    const all = parsePrometheusLines(rawMetrics);
    const q = filterText.trim().toLowerCase();
    if (!q) return all.slice(0, 60);
    return all.filter((s) => `${s.name}${s.labels}`.toLowerCase().includes(q)).slice(0, 60);
  }, [rawMetrics, filterText]);

  const prometheusBase = summary?.prometheus_url || 'http://localhost:9090';
  const grafanaBase = summary?.grafana_url || 'http://localhost:3000';
  const mlflowBase = summary?.mlflow_tracking_uri || 'http://localhost:5000';
  const deps = summary?.dependencies || {};

  return (
    <div className="flex-1 min-h-0 flex flex-col space-y-5 overflow-y-auto pr-1">
      {/* Complementary 3-Pillar Architecture Header */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-col space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2">
              <Radar className="w-5 h-5 text-sky-400" />
              <h2 className="text-lg font-bold text-slate-100 tracking-tight">
                Production Observability Architecture
              </h2>
              <StatusBadge
                status={deps.fastapi ? 'Metrics Active' : 'Syncing'}
                variant={deps.fastapi ? 'healthy' : 'neutral'}
                pulse={Boolean(deps.fastapi)}
                size="sm"
              />
            </div>
            <p className="text-xs text-slate-400 mt-1 max-w-3xl">
              Three complementary observability pillars work side by side: MLflow tracks LLM traces &amp;
              judge verdicts, Prometheus scrapes operational time-series from FastAPI{' '}
              <code className="text-sky-300 font-mono">/metrics</code>, and Grafana renders pre-provisioned dashboards.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            {lastUpdated && (
              <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">
                Synced: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={loadObservability}
              disabled={loading}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs transition"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-sky-400' : ''}`} />
              <span>Refresh Scrapes</span>
            </button>
          </div>
        </div>

        {errorMsg && (
          <div className="p-2.5 rounded-lg bg-rose-950/50 border border-rose-800/60 text-xs text-rose-300">
            {errorMsg}
          </div>
        )}

        {/* 3 Systems Comparison Strip */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-sky-300 flex items-center space-x-1.5">
                  <ScrollText className="w-4 h-4" />
                  <span>1. MLflow Tracking</span>
                </span>
                <StatusBadge
                  status={deps.mlflow ? 'Online' : 'Check Server'}
                  variant={deps.mlflow ? 'healthy' : 'degraded'}
                  size="sm"
                />
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Source of truth for query/answer traces, prompt versions, per-trace judge scores
                (Relevancy, Hallucination, Safety), and SQLite audit history (<code className="text-slate-300">mlflow.db</code>).
              </p>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px]">
              <button
                onClick={() => onNavigateTab('eval')}
                className="text-sky-400 hover:text-sky-300 font-medium"
              >
                Open Evaluation &amp; Traces →
              </button>
              <a
                href={mlflowBase}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-300 hover:text-white font-mono"
              >
                <span>:5000</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-amber-300 flex items-center space-x-1.5">
                  <Flame className="w-4 h-4" />
                  <span>2. Prometheus TSDB</span>
                </span>
                <span className="text-[10px] font-mono text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2 py-0.5 rounded-full">
                  {summary?.registered_metric_families.length ?? 0} families
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Scrapes FastAPI <code className="text-slate-300">/metrics</code> every 10s via{' '}
                <code className="text-slate-300">host.docker.internal:8000</code> and evaluates 7 operational alert
                rules (API down, 5xx spike, P95 latency, Ollama/Chroma health, judge failure rate).
              </p>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px]">
              <a
                href={`${prometheusBase}/targets`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-amber-300 hover:text-amber-200 font-medium"
              >
                <span>Targets &amp; Alerts</span>
                <ExternalLink className="w-3 h-3" />
              </a>
              <a
                href="/metrics"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-slate-400 hover:text-white font-mono"
              >
                <span>/metrics</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>

          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between space-y-3">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-emerald-300 flex items-center space-x-1.5">
                  <BarChart3 className="w-4 h-4" />
                  <span>3. Grafana Dashboards</span>
                </span>
                <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                  5 Provisioned
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-2">
                Auto-provisioned via Docker Compose (<code className="text-slate-300">infra/monitoring</code>) with
                persistent volumes covering System Overview, RAG Analytics, LLM Arena, Eval Lifecycle &amp; Health.
              </p>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-slate-800/80 text-[11px]">
              <span className="text-slate-500 font-mono">Folder: LLMOps Sentinel</span>
              <a
                href={grafanaBase}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center space-x-1 text-emerald-300 hover:text-emerald-200 font-mono"
              >
                <span>:3000</span>
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Live Operational Telemetry Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Process RSS Memory"
          value={summary ? summary.process_memory_rss_mb.toFixed(1) : '—'}
          unit="MB"
          subtitle={`CPU: ${summary?.process_cpu_percent ?? 0}% · Uptime: ${Math.round(summary?.process_uptime_seconds ?? 0)}s`}
          icon={<HardDrive className="w-4 h-4" />}
          accent="sky"
          badge={{ text: 'FastAPI Worker', variant: 'info' }}
        />

        <KpiCard
          title="ChromaDB Vector Index"
          value={summary ? summary.chromadb_chunks : '—'}
          unit="chunks"
          subtitle={`${summary?.chromadb_documents ?? 0} indexed docs · cache: ${summary?.embedding_cache_entries ?? 0} queries`}
          icon={<Database className="w-4 h-4" />}
          accent="indigo"
          badge={{
            text: deps.chromadb ? 'Connected' : 'Error',
            variant: deps.chromadb ? 'success' : 'error',
          }}
        />

        <KpiCard
          title="Ollama & Active Eval"
          value={summary ? summary.ollama_models_available : '—'}
          unit="models"
          subtitle={`Active background eval jobs: ${summary?.eval_jobs_active ?? 0}`}
          icon={<Cpu className="w-4 h-4" />}
          accent="emerald"
          badge={{
            text: deps.ollama ? 'Daemon UP' : 'Offline',
            variant: deps.ollama ? 'success' : 'error',
          }}
        />

        <KpiCard
          title="Live Latency Alignment"
          value={perf?.median_query_latency_ms ? Math.round(perf.median_query_latency_ms) : '—'}
          unit={perf?.median_query_latency_ms ? 'ms P50' : undefined}
          subtitle={
            perf?.p95_query_latency_ms
              ? `P95: ${Math.round(perf.p95_query_latency_ms)} ms (${perf.percentile_method || 'linear_interpolation'})`
              : 'No measured queries yet'
          }
          icon={<Activity className="w-4 h-4" />}
          accent="amber"
          badge={{ text: `${perf?.measured_query_traces ?? 0} samples`, variant: 'neutral' }}
        />
      </div>

      {/* Provisioned Grafana Dashboards Directory */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <Layers className="w-4 h-4 text-emerald-400" />
              <span>Provisioned Grafana Dashboards (infra/monitoring/grafana/dashboards)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Launch after starting <code className="text-slate-300 font-mono">docker compose up -d</code> inside{' '}
              <code className="text-slate-300 font-mono">infra/monitoring</code>.
            </p>
          </div>
          <a
            href={`${grafanaBase}/dashboards`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-emerald-400 hover:text-emerald-300 flex items-center space-x-1"
          >
            <span>Browse All in Grafana</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {GRAFANA_DASHBOARDS.map((dash) => (
            <a
              key={dash.uid}
              href={`${grafanaBase}/d/${dash.uid}/${dash.slug}`}
              target="_blank"
              rel="noopener noreferrer"
              className="group bg-slate-950/70 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/40 rounded-lg p-3.5 transition flex flex-col justify-between space-y-2"
            >
              <div>
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200 group-hover:text-emerald-300 transition">
                    {dash.title}
                  </span>
                  <ExternalLink className="w-3 h-3 text-slate-500 group-hover:text-emerald-400" />
                </div>
                <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">{dash.focus}</p>
              </div>
              <div className="flex flex-wrap gap-1 pt-1">
                {dash.metrics.map((m) => (
                  <span
                    key={m}
                    className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </a>
          ))}
        </div>
      </div>

      {/* Live /metrics Exposition Inspector */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-200 flex items-center space-x-2">
              <Terminal className="w-4 h-4 text-sky-400" />
              <span>Live Prometheus /metrics Sample Stream</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Real-time bounded-cardinality counters, histograms, and gauges exposed for Prometheus scraping.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            {['llmops_sentinel_', 'http_', 'retrieval_', 'llm_', 'eval_', 'process_'].map((preset) => (
              <button
                key={preset}
                onClick={() => setFilterText(preset)}
                className={`px-2 py-1 rounded text-[10px] font-mono border transition ${
                  filterText === preset
                    ? 'bg-sky-500/20 text-sky-300 border-sky-500/40'
                    : 'bg-slate-950 text-slate-400 border-slate-800 hover:text-slate-200'
                }`}
              >
                {preset}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-slate-950 border border-slate-800/90 rounded-lg overflow-hidden">
          <div className="max-h-64 overflow-y-auto divide-y divide-slate-800/60 font-mono text-[11px]">
            {parsedSamples.length === 0 ? (
              <div className="p-4 text-slate-500 italic text-center">
                No metric lines match filter "{filterText}".
              </div>
            ) : (
              parsedSamples.map((s, i) => (
                <div
                  key={`${s.name}-${s.labels}-${i}`}
                  className="px-3 py-1.5 flex items-center justify-between gap-3 hover:bg-slate-900/50"
                >
                  <div className="truncate">
                    <span className="text-sky-300">{s.name}</span>
                    {s.labels && <span className="text-slate-500">{s.labels}</span>}
                  </div>
                  <span className="text-emerald-400 font-semibold shrink-0">{s.value}</span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-500 pt-1">
          <span className="flex items-center space-x-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>High-cardinality guard active: user queries, filenames, trace IDs &amp; secrets never exported as labels.</span>
          </span>
          <span className="flex items-center space-x-1 font-mono">
            <Server className="w-3 h-3" />
            <span>Showing {parsedSamples.length} series</span>
            <CheckCircle2 className="w-3 h-3 text-emerald-400 ml-1" />
          </span>
        </div>
      </div>
    </div>
  );
};

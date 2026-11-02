import React, { useState, useEffect } from 'react';
import {
  Activity,
  Layers,
  ScrollText,
  Clock,
  FlaskConical,
  Database,
  Cpu,
  RefreshCw,
  ExternalLink,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';
import { getEvaluationTraces, listBenchmarks } from '../api';
import { ActiveTab, BenchmarkSummary, HealthResponse, TraceAuditItem } from '../types';
import { KpiCard } from './common/KpiCard';
import { StatusBadge } from './common/StatusBadge';
import { InteractiveLineChart, LineDataPoint } from './charts/InteractiveLineChart';
import { DonutChart, DonutSegment } from './charts/DonutChart';
import { LatencyDistributionPlot } from './charts/LatencyDistributionPlot';
import { InteractiveBarChart, BarDataPoint } from './charts/InteractiveBarChart';

interface OverviewViewProps {
  apiKey: string;
  health: HealthResponse | null;
  healthLoading: boolean;
  onNavigateTab: (tab: ActiveTab) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  apiKey,
  health,
  healthLoading,
  onNavigateTab,
}) => {
  const [traces, setTraces] = useState<TraceAuditItem[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkSummary[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());
  const [fetchError, setFetchError] = useState<string | null>(null);

  const loadTelemetry = async () => {
    setIsLoading(true);
    setFetchError(null);
    try {
      const [traceData, benchData] = await Promise.all([
        getEvaluationTraces(apiKey, 50).catch(() => [] as TraceAuditItem[]),
        listBenchmarks(apiKey).catch(() => [] as BenchmarkSummary[]),
      ]);
      setTraces(traceData);
      setBenchmarks(benchData);
      setLastRefreshed(new Date());
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Telemetry sync failed';
      setFetchError(msg);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadTelemetry();
  }, [apiKey]);

  // Aggregate Real Metrics
  const totalQueries = traces.length;
  const latencies = traces.map((t) => t.execution_time_ms).filter((ms) => ms > 0);
  const meanLatency =
    latencies.length > 0
      ? Math.round(latencies.reduce((acc, v) => acc + v, 0) / latencies.length)
      : null;

  // Evaluation assessments breakdown
  let passCount = 0;
  let failCount = 0;
  let skippedCount = 0;

  traces.forEach((t) => {
    Object.values(t.assessments).forEach((a) => {
      const val = (a.value || '').toLowerCase();
      if (val === 'yes') passCount++;
      else if (val === 'no') failCount++;
      else skippedCount++;
    });
  });

  const totalAssessments = passCount + failCount + skippedCount;
  const reliabilityRate =
    totalAssessments > 0 ? Math.round((passCount / totalAssessments) * 100) : null;

  // Latency time-series data points (sorted chronologically)
  const sortedTraces = [...traces].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  const latencyChartData: LineDataPoint[] = sortedTraces.map((t) => {
    const d = new Date(t.timestamp_ms);
    const timeLabel = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return {
      label: timeLabel,
      value: Math.round(t.execution_time_ms),
      secondaryText: t.question || t.request_id,
    };
  });

  // Donut chart: Outcome distribution
  const outcomeData: DonutSegment[] = [
    { label: 'Passed', value: passCount, color: '#10b981' }, // emerald-500
    { label: 'Failed', value: failCount, color: '#f43f5e' }, // rose-500
    { label: 'Pending / Skipped', value: skippedCount, color: '#64748b' }, // slate-500
  ];

  // Benchmark history summary
  const latestBenchmark = benchmarks.length > 0 ? benchmarks[0] : null;

  // Bar chart: Model Comparison / Usage
  const modelBarData: BarDataPoint[] = [
    {
      label: 'Llama 3.2',
      value: traces.filter((t) => !t.model || t.model.includes('llama')).length,
      color: '#38bdf8',
      subtext: '3.2B Primary QA',
    },
    {
      label: 'Mistral',
      value: traces.filter((t) => t.model && t.model.includes('mistral')).length,
      color: '#fbbf24',
      subtext: '7B Judge & Arena',
    },
  ];

  return (
    <div className="flex-1 min-h-0 flex flex-col space-y-5 overflow-y-auto pr-1">
      {/* Top Banner / Sentinel Telemetry Header */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-lg font-bold text-slate-100 tracking-tight">
              Sentinel Command Centre
            </h2>
            <StatusBadge
              status={health?.status === 'ok' ? 'System Live' : 'Degraded'}
              variant={health?.status === 'ok' ? 'healthy' : 'degraded'}
              pulse={health?.status === 'ok'}
              size="sm"
            />
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Real-time telemetry, model reliability metrics, and RAG diagnostic streams from MLflow
            and ChromaDB.
          </p>
        </div>

        <div className="flex items-center space-x-3">
          <span className="text-[11px] text-slate-500 font-mono hidden sm:inline">
            Synced: {lastRefreshed.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
          </span>

          <button
            onClick={loadTelemetry}
            disabled={isLoading}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition text-xs border border-slate-700"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin text-sky-400' : ''}`} />
            <span>Sync Telemetry</span>
          </button>

          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-1 px-3 py-1.5 rounded-lg bg-sky-500/10 hover:bg-sky-500/20 text-sky-300 border border-sky-500/30 transition text-xs"
          >
            <span>MLflow Traces</span>
            <ExternalLink className="w-3 h-3 ml-1" />
          </a>
        </div>
      </div>

      {fetchError && (
        <div className="p-3 bg-rose-950/40 border border-rose-800/60 rounded-xl text-xs text-rose-300 flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
          <span>Notice: {fetchError}</span>
        </div>
      )}

      {/* Top KPI Metrics Strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Audited Inferences"
          value={totalQueries}
          subtitle="Real SQLite traces stored"
          icon={<Layers className="w-4 h-4" />}
          accent="sky"
          badge={{ text: 'Live Feed', variant: 'info' }}
          onClick={() => onNavigateTab('eval')}
        />

        <KpiCard
          title="Mean Execution Latency"
          value={meanLatency !== null ? meanLatency : '—'}
          unit={meanLatency !== null ? 'ms' : undefined}
          subtitle="End-to-end FastAPI roundtrip"
          icon={<Clock className="w-4 h-4" />}
          accent="indigo"
          badge={
            meanLatency !== null && meanLatency < 1000
              ? { text: 'Optimal', variant: 'success' }
              : { text: 'Monitored', variant: 'neutral' }
          }
        />

        <KpiCard
          title="Judge Reliability Rate"
          value={reliabilityRate !== null ? `${reliabilityRate}%` : '—'}
          subtitle={`${passCount} passed of ${totalAssessments} checks`}
          icon={<ScrollText className="w-4 h-4" />}
          accent="emerald"
          badge={
            reliabilityRate !== null && reliabilityRate >= 70
              ? { text: 'Compliant', variant: 'success' }
              : { text: 'Audit Required', variant: 'warning' }
          }
          onClick={() => onNavigateTab('eval')}
        />

        <KpiCard
          title="Benchmark Pass Rate"
          value={
            latestBenchmark?.relevancy_pass_rate !== null &&
            latestBenchmark?.relevancy_pass_rate !== undefined
              ? `${latestBenchmark.relevancy_pass_rate}%`
              : '—'
          }
          subtitle={
            latestBenchmark
              ? `${latestBenchmark.completed_cases}/${latestBenchmark.total_cases} cases passed`
              : 'No test runs yet'
          }
          icon={<FlaskConical className="w-4 h-4" />}
          accent="amber"
          badge={{
            text: latestBenchmark?.status || 'idle',
            variant: latestBenchmark?.status === 'completed' ? 'success' : 'neutral',
          }}
          onClick={() => onNavigateTab('benchmarks')}
        />
      </div>

      {/* Middle Row: Interactive Time Series & Outcome Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: Trace Latency Time-Series (7 cols) */}
        <div className="lg:col-span-7 flex flex-col space-y-4">
          <InteractiveLineChart
            data={latencyChartData}
            title="Inference Latency Timeline"
            yAxisLabel="Latency (ms)"
            unit="ms"
            accentColor="#38bdf8"
            height={220}
            emptyMessage="No query traces recorded yet. Submit queries in Live Chat to plot real-time latency."
            onPointClick={() => onNavigateTab('eval')}
          />

          {/* Latency Percentiles */}
          <LatencyDistributionPlot latencies={latencies} />
        </div>

        {/* Right Column: Outcomes Donut & Model Usage (5 cols) */}
        <div className="lg:col-span-5 flex flex-col space-y-4">
          {/* Assessment Donut */}
          <DonutChart
            data={outcomeData}
            title="Judge Evaluation Outcome Distribution"
            centerLabel="Assessments"
            centerValue={totalAssessments}
            emptyMessage="No evaluation assessments recorded yet in MLflow."
            onSegmentClick={() => onNavigateTab('eval')}
          />

          {/* Model Workload Breakdown */}
          <InteractiveBarChart
            data={modelBarData}
            title="Local Model Workload"
            unit="queries"
            height={160}
            emptyMessage="No model inferences recorded yet."
            onBarClick={() => onNavigateTab('arena')}
          />
        </div>
      </div>

      {/* Bottom Row: Sentinel Architecture & Quick Navigation */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-200">
              Infrastructure & Pipeline Telemetry
            </h3>
            <p className="text-xs text-slate-400">
              Live status of offline vector database, local LLM inference engines, and tracking stores.
            </p>
          </div>
          <span className="text-[11px] text-slate-500 font-mono">Status Check: HTTP 200</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* ChromaDB Card */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-3.5 flex items-start space-x-3">
            <div className="w-8 h-8 rounded-lg bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400 shrink-0 mt-0.5">
              <Database className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-slate-200">ChromaDB</span>
                <span
                  className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded ${
                    health?.chromadb === 'ok'
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                  }`}
                >
                  {healthLoading ? 'Checking...' : health?.chromadb === 'ok' ? 'Online' : 'Error'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Embedded vector index using Cosine Distance & nomic-embed-text.
              </p>
            </div>
          </div>

          {/* Ollama Card */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-3.5 flex items-start space-x-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 shrink-0 mt-0.5">
              <Cpu className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-slate-200">Ollama Engine</span>
                <span
                  className={`text-[10px] font-mono font-medium px-1.5 py-0.5 rounded ${
                    health?.ollama === 'ok'
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      : 'bg-rose-500/15 text-rose-400 border border-rose-500/30'
                  }`}
                >
                  {healthLoading ? 'Checking...' : health?.ollama === 'ok' ? 'Online' : 'Error'}
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Local runtime serving Llama 3.2 (3B) and Mistral (7B).
              </p>
            </div>
          </div>

          {/* MLflow Card */}
          <div className="bg-slate-950/80 border border-slate-800/90 rounded-lg p-3.5 flex items-start space-x-3">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shrink-0 mt-0.5">
              <Activity className="w-4 h-4" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-xs text-slate-200">MLflow GenAI</span>
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                  SQLite
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-1">
                Persistent trace catalog tracking Relevancy, Grounding, and Safety.
              </p>
            </div>
          </div>
        </div>

        {/* Action jump links */}
        <div className="mt-4 pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs">
          <span className="text-slate-400">Jump directly to workspace:</span>
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => onNavigateTab('chat')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white transition flex items-center space-x-1"
            >
              <span>Live Chat</span>
              <ArrowRight className="w-3 h-3 text-sky-400" />
            </button>
            <button
              onClick={() => onNavigateTab('arena')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white transition flex items-center space-x-1"
            >
              <span>Model Arena</span>
              <ArrowRight className="w-3 h-3 text-sky-400" />
            </button>
            <button
              onClick={() => onNavigateTab('eval')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white transition flex items-center space-x-1"
            >
              <span>Evaluation & Traces</span>
              <ArrowRight className="w-3 h-3 text-sky-400" />
            </button>
            <button
              onClick={() => onNavigateTab('benchmarks')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 hover:text-white transition flex items-center space-x-1"
            >
              <span>Benchmark Lab</span>
              <ArrowRight className="w-3 h-3 text-sky-400" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

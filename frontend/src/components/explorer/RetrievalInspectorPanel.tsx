import React from 'react';
import {
  Search,
  Loader2,
  AlertTriangle,
  HelpCircle,
  Clock,
  FileText,
  LocateFixed,
  Sparkles,
  Filter,
} from 'lucide-react';
import { IndexedDocItem, RetrievalInspectResponse } from '../../types';

interface RetrievalInspectorPanelProps {
  documents: IndexedDocItem[];
  question: string;
  onQuestionChange: (q: string) => void;
  scopeDoc: string;
  onScopeDocChange: (doc: string) => void;
  topK: number;
  onTopKChange: (k: number) => void;
  useMmr: boolean;
  onUseMmrChange: (val: boolean) => void;
  isInspecting: boolean;
  error: string | null;
  result: RetrievalInspectResponse | null;
  onInspect: () => void;
  onLocateChunk: (documentName: string | null, chunkId: string, text: string) => void;
}

export const RetrievalInspectorPanel: React.FC<RetrievalInspectorPanelProps> = ({
  documents,
  question,
  onQuestionChange,
  scopeDoc,
  onScopeDocChange,
  topK,
  onTopKChange,
  useMmr,
  onUseMmrChange,
  isInspecting,
  error,
  result,
  onInspect,
  onLocateChunk,
}) => {
  const getDistanceBadge = (score: number | null) => {
    if (score === null || score === undefined) {
      return (
        <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 text-[10px] font-mono">
          Distance: N/A
        </span>
      );
    }
    if (score < 0.4) {
      return (
        <span className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 text-[10px] font-mono font-semibold">
          Cosine Dist: {score.toFixed(4)} (High Match)
        </span>
      );
    }
    if (score <= 0.75) {
      return (
        <span className="px-2 py-0.5 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30 text-[10px] font-mono font-semibold">
          Cosine Dist: {score.toFixed(4)} (Moderate Match)
        </span>
      );
    }
    return (
      <span className="px-2 py-0.5 rounded bg-rose-500/15 text-rose-300 border border-rose-500/30 text-[10px] font-mono font-semibold">
        Cosine Dist: {score.toFixed(4)} (Low Match)
      </span>
    );
  };

  return (
    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
            <Search className="w-4 h-4 text-sky-400" />
            <span>Interactive Retrieval Inspector (Deduplicated Top-K &amp; MMR)</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Inspect query vectorization, strict document isolation, cosine distances, and duplicate removal without waiting for LLM synthesis.
          </p>
        </div>
      </div>

      {/* Controls Bar */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 bg-slate-950/60 border border-slate-800 rounded-xl p-3.5">
        <div className="lg:col-span-5">
          <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
            Query Text
          </label>
          <input
            type="text"
            value={question}
            onChange={(e) => onQuestionChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && question.trim() && !isInspecting) {
                onInspect();
              }
            }}
            placeholder="Enter a question to inspect retrieved chunks..."
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500"
          />
        </div>

        <div className="lg:col-span-3">
          <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400 block mb-1">
            Document Scope
          </label>
          <select
            value={scopeDoc}
            onChange={(e) => onScopeDocChange(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs text-slate-200 font-mono focus:outline-none focus:border-sky-500"
          >
            <option value="__all__">All Indexed Documents</option>
            {documents.map((d) => (
              <option key={d.filename} value={d.filename}>
                {d.filename} ({d.chunks} chunks)
              </option>
            ))}
          </select>
        </div>

        <div className="lg:col-span-2">
          <label className="text-[10px] font-mono uppercase tracking-wider text-slate-400 flex justify-between mb-1">
            <span>Top-K</span>
            <span className="text-sky-400 font-bold">K = {topK}</span>
          </label>
          <input
            type="range"
            min="1"
            max="12"
            value={topK}
            onChange={(e) => onTopKChange(parseInt(e.target.value, 10))}
            className="w-full accent-sky-500 cursor-pointer mt-2"
          />
        </div>

        <div className="lg:col-span-2 flex flex-col justify-between">
          <label className="flex items-center space-x-2 text-[11px] text-slate-300 cursor-pointer mb-1.5">
            <input
              type="checkbox"
              checked={useMmr}
              onChange={(e) => onUseMmrChange(e.target.checked)}
              className="rounded border-slate-700 bg-slate-900 text-sky-500 focus:ring-0"
            />
            <span>MMR Diversity (λ=0.72)</span>
          </label>
          <button
            onClick={onInspect}
            disabled={!question.trim() || isInspecting}
            className="w-full flex items-center justify-center space-x-1.5 px-3 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-xs font-medium transition shadow-sm shadow-sky-600/20"
          >
            {isInspecting ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Sparkles className="w-3.5 h-3.5" />
            )}
            <span>{isInspecting ? 'Inspecting...' : 'Inspect Retrieval'}</span>
          </button>
        </div>
      </div>

      {/* Explicit Cosine Distance Legend */}
      <div className="p-3 rounded-lg bg-slate-950/70 border border-slate-800 text-[11px] text-slate-300 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-start space-x-2">
          <HelpCircle className="w-4 h-4 text-sky-400 shrink-0 mt-0.5" />
          <span>
            <strong className="text-sky-300">Metric: Cosine Distance</strong> (d = 1 - cos(θ), NOT similarity).{' '}
            <strong>Lower distance indicates a closer semantic match</strong> (0.0 = identical; &lt;0.40 = High; 0.40–0.75 = Moderate; &gt;0.75 = Weak).
          </span>
        </div>
      </div>

      {error && (
        <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center space-x-2">
          <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
          <span>{error}</span>
        </div>
      )}

      {/* Results View */}
      {result && (
        <div className="space-y-3">
          {/* Telemetry Summary Strip */}
          <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-300 flex items-center space-x-1">
              <Clock className="w-3 h-3 text-sky-400" />
              <span>{result.retrieval_diagnostics.retrieval_latency_ms} ms</span>
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-sky-300">
              Scope: {result.document_scope}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-slate-200">
              Embed: {result.embedding_model}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-emerald-500/10 border border-emerald-500/30 text-emerald-300">
              Unique Returned: {result.retrieved_chunks.length} / K={result.retrieval_diagnostics.requested_k}
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-amber-300 flex items-center space-x-1">
              <Filter className="w-3 h-3" />
              <span>Duplicates Filtered: {result.retrieval_diagnostics.duplicates_removed ?? 0}</span>
            </span>
            <span className="px-2.5 py-1 rounded-md bg-slate-950 border border-slate-800 text-indigo-300">
              Strategy: {result.retrieval_diagnostics.retrieval_strategy}
            </span>
          </div>

          {result.retrieval_diagnostics.insufficient_unique_reason && (
            <div className="p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/40 text-xs text-amber-200 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>{result.retrieval_diagnostics.insufficient_unique_reason}</span>
            </div>
          )}

          {result.insufficient_evidence && result.empty_reason && (
            <div className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-500/40 text-xs text-rose-200 flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{result.empty_reason}</span>
            </div>
          )}

          {/* Retrieved Chunks Ranked Cards */}
          {result.retrieved_chunks.length === 0 ? (
            <div className="p-6 text-center rounded-xl bg-slate-950/60 border border-slate-800 text-xs text-slate-400">
              No matching unique chunks retrieved for this query within scope "{result.document_scope}".
            </div>
          ) : (
            <div className="space-y-2.5">
              {result.retrieved_chunks.map((chunk, idx) => (
                <div
                  key={chunk.chunk_id || idx}
                  className="bg-slate-950/80 border border-slate-800 hover:border-sky-500/40 rounded-xl p-3.5 transition"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 pb-2 mb-2 border-b border-slate-800/80">
                    <div className="flex flex-wrap items-center gap-1.5 text-[11px] font-mono">
                      <span className="px-2 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30 font-bold">
                        Rank #{idx + 1}
                      </span>
                      <span className="text-slate-300 font-semibold">{chunk.chunk_id}</span>
                      {chunk.document_name && (
                        <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                          <FileText className="w-3 h-3 text-slate-400" />
                          <span>{chunk.document_name}</span>
                        </span>
                      )}
                      {chunk.page_number !== null && chunk.page_number !== undefined && (
                        <span className="bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                          Page {chunk.page_number}
                        </span>
                      )}
                      {getDistanceBadge(chunk.score)}
                    </div>

                    <button
                      onClick={() =>
                        onLocateChunk(chunk.document_name, chunk.chunk_id, chunk.text)
                      }
                      className="flex items-center space-x-1 text-[11px] px-2.5 py-1 rounded-lg bg-sky-500/15 hover:bg-sky-500/25 text-sky-300 border border-sky-500/30 font-medium transition"
                    >
                      <LocateFixed className="w-3 h-3" />
                      <span>Locate in Chunk Viewer</span>
                    </button>
                  </div>

                  <p className="text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed max-h-40 overflow-y-auto">
                    {chunk.text}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

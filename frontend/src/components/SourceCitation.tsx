import React, { useState } from 'react';
import {
  BookOpen,
  ChevronDown,
  ChevronUp,
  Copy,
  Check,
  AlertTriangle,
  FileText,
  Clock,
  HelpCircle,
  ShieldCheck,
} from 'lucide-react';
import { RetrievalDiagnostics, SourceChunk } from '../types';

interface SourceCitationProps {
  sources?: string[];
  source_chunks?: SourceChunk[];
  retrieval_diagnostics?: RetrievalDiagnostics;
}

export const SourceCitation: React.FC<SourceCitationProps> = ({
  sources = [],
  source_chunks = [],
  retrieval_diagnostics,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);
  const [showScoreInfo, setShowScoreInfo] = useState(false);

  const chunksCount = source_chunks.length || sources.length;
  if (chunksCount === 0) return null;

  const handleCopy = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  // Cosine distance relevance tier badge
  const getScoreBadge = (score: number | null) => {
    if (score === null || score === undefined) return null;
    if (score < 0.40) {
      return (
        <span
          className="px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 text-[10px] font-mono font-medium"
          title="Cosine distance < 0.40 indicates high semantic match in vector space"
        >
          Dist: {score.toFixed(3)} (High Match)
        </span>
      );
    }
    if (score <= 0.75) {
      return (
        <span
          className="px-2 py-0.5 rounded bg-sky-500/15 text-sky-400 border border-sky-500/30 text-[10px] font-mono font-medium"
          title="Cosine distance between 0.40 and 0.75"
        >
          Dist: {score.toFixed(3)} (Moderate Match)
        </span>
      );
    }
    return (
      <span
        className="px-2 py-0.5 rounded bg-rose-500/15 text-rose-400 border border-rose-500/30 text-[10px] font-mono font-medium"
        title="Cosine distance > 0.75 indicates weak semantic similarity"
      >
        Dist: {score.toFixed(3)} (Low Match)
      </span>
    );
  };

  return (
    <div className="mt-3 pt-3 border-t border-slate-800 text-xs">
      {/* Header Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center space-x-1.5 text-xs font-semibold text-sky-400 hover:text-sky-300 transition group"
        >
          <BookOpen className="w-3.5 h-3.5 text-sky-400" />
          <span>
            {chunksCount} Retrieved Source Chunk{chunksCount > 1 ? 's' : ''} (ChromaDB)
          </span>
          {isOpen ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-200" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400 group-hover:text-slate-200" />
          )}
        </button>

        {/* Diagnostic Metadata Pills */}
        <div className="flex items-center space-x-2 text-[11px] text-slate-400 font-mono">
          {retrieval_diagnostics && (
            <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-slate-300">
              <Clock className="w-3 h-3 text-sky-400" />
              <span>{retrieval_diagnostics.retrieval_latency_ms} ms</span>
            </span>
          )}

          {retrieval_diagnostics?.low_confidence ? (
            <span className="flex items-center space-x-1 bg-amber-500/15 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded font-medium">
              <AlertTriangle className="w-3 h-3 text-amber-400" />
              <span>Low Confidence Match</span>
            </span>
          ) : (
            <span className="hidden sm:flex items-center space-x-1 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-medium">
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
              <span>Grounded Context</span>
            </span>
          )}

          <button
            onClick={() => setShowScoreInfo(!showScoreInfo)}
            className="text-slate-500 hover:text-slate-300 transition"
            title="Cosine Distance Semantics Guide"
          >
            <HelpCircle className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Explanatory banner for score semantics */}
      {showScoreInfo && (
        <div className="mt-2.5 p-3 bg-slate-950 border border-slate-800 rounded-lg text-[11px] text-slate-400 space-y-1.5 leading-relaxed font-sans">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-slate-200 font-mono">
              Retrieval Metric: Cosine Distance
            </span>
            <span className="text-[10px] text-sky-400 font-mono">Chroma HNSW</span>
          </div>
          <p>
            Chroma computes cosine distance $d = 1 - \cos(\theta)$ where <strong>0.0</strong>{' '}
            represents identical embeddings and higher values represent greater semantic distance.
          </p>
          <div className="grid grid-cols-3 gap-2 pt-1 font-mono text-[10px]">
            <span className="text-emerald-400">d &lt; 0.40 : High Relevance</span>
            <span className="text-sky-400">0.40–0.75 : Moderate Context</span>
            <span className="text-rose-400">d &gt; 0.75 : Weak / Out of Domain</span>
          </div>
        </div>
      )}

      {/* Low confidence warning banner */}
      {isOpen && retrieval_diagnostics?.low_confidence && (
        <div className="mt-2.5 p-2.5 bg-amber-950/30 border border-amber-800/60 rounded-lg flex items-center space-x-2 text-amber-300 text-[11px]">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
          <span>
            Top cosine distance is {retrieval_diagnostics.top_score} (&gt; 0.75). The retrieved
            passages may have weak semantic alignment with your query.
          </span>
        </div>
      )}

      {/* Expanded Sources List */}
      {isOpen && (
        <div className="mt-2.5 space-y-2.5">
          {source_chunks.length > 0
            ? source_chunks.map((chunk, idx) => (
                <div
                  key={chunk.chunk_id || idx}
                  className="bg-slate-950/90 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 relative group transition hover:border-slate-700"
                >
                  {/* Top metadata strip */}
                  <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 mb-2 border-b border-slate-800 pb-1.5 gap-1.5">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="font-bold text-sky-400">
                        {chunk.chunk_id || `Chunk #${idx + 1}`}
                      </span>

                      {chunk.document_name && (
                        <span className="flex items-center space-x-1 bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                          <FileText className="w-3 h-3 text-slate-400" />
                          <span className="max-w-[140px] truncate" title={chunk.document_name}>
                            {chunk.document_name}
                          </span>
                        </span>
                      )}

                      {chunk.page_number !== null && chunk.page_number !== undefined && (
                        <span className="bg-slate-900 border border-slate-800 px-1.5 py-0.5 rounded text-slate-300">
                          p. {chunk.page_number}
                        </span>
                      )}

                      {getScoreBadge(chunk.score)}

                      {chunk.token_count_estimate && (
                        <span className="text-[10px] text-slate-500">
                          ~{chunk.token_count_estimate} tok
                        </span>
                      )}
                    </div>

                    <button
                      onClick={() => handleCopy(chunk.text, idx)}
                      className="flex items-center space-x-1 text-slate-400 hover:text-slate-200 transition"
                      title="Copy chunk text"
                    >
                      {copiedIndex === idx ? (
                        <>
                          <Check className="w-3 h-3 text-emerald-400" />
                          <span className="text-emerald-400">Copied</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3 h-3 text-slate-500 group-hover:text-slate-300" />
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>

                  {/* Chunk Text Body */}
                  <p className="whitespace-pre-wrap leading-relaxed text-slate-300 select-text text-[11px] max-h-48 overflow-y-auto">
                    {chunk.text}
                  </p>
                </div>
              ))
            : sources.map((source, idx) => (
                <div
                  key={idx}
                  className="bg-slate-950/70 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-300 relative group"
                >
                  <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5 border-b border-slate-800 pb-1">
                    <span className="font-semibold text-slate-400">Chunk #{idx + 1}</span>
                    <button
                      onClick={() => handleCopy(source, idx)}
                      className="flex items-center space-x-1 hover:text-slate-200 transition"
                    >
                      {copiedIndex === idx ? (
                        <span className="text-emerald-400">Copied</span>
                      ) : (
                        <span>Copy</span>
                      )}
                    </button>
                  </div>
                  <p className="whitespace-pre-wrap leading-relaxed text-slate-300 select-text text-[11px]">
                    {source}
                  </p>
                </div>
              ))}
        </div>
      )}
    </div>
  );
};

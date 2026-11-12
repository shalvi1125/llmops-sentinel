import React, { useEffect, useRef } from 'react';
import {
  ChevronLeft,
  ChevronRight,
  Sliders,
  AlertTriangle,
  RotateCcw,
  Layers,
  FileText,
  Sparkles,
  CheckCircle2,
  Loader2,
} from 'lucide-react';
import {
  ChunkDetail,
  ChunkPreviewResponse,
  DocumentChunksResponse,
} from '../../types';

interface ChunkingViewerProps {
  docData: DocumentChunksResponse | null;
  previewData: ChunkPreviewResponse | null;
  loadingDoc: boolean;
  loadingPreview: boolean;
  previewError: string | null;
  activeChunkIndex: number;
  onSelectChunkIndex: (idx: number) => void;
  previewChunkSize: number;
  previewChunkOverlap: number;
  onPreviewChunkSizeChange: (val: number) => void;
  onPreviewChunkOverlapChange: (val: number) => void;
  onRunPreview: () => void;
  onResetPreview: () => void;
  highlightedRetrievedChunkIds: string[];
}

export const ChunkingViewer: React.FC<ChunkingViewerProps> = ({
  docData,
  previewData,
  loadingDoc,
  loadingPreview,
  previewError,
  activeChunkIndex,
  onSelectChunkIndex,
  previewChunkSize,
  previewChunkOverlap,
  onPreviewChunkSizeChange,
  onPreviewChunkOverlapChange,
  onRunPreview,
  onResetPreview,
  highlightedRetrievedChunkIds,
}) => {
  const activeSpanRef = useRef<HTMLSpanElement>(null);

  const isPreview = previewData !== null;
  const activeChunks: ChunkDetail[] = isPreview
    ? previewData.chunks
    : docData?.chunks || [];
  const sourceText = isPreview
    ? previewData.source_text
    : docData?.reconstructed_text || '';

  const safeIdx =
    activeChunks.length > 0
      ? Math.min(Math.max(0, activeChunkIndex), activeChunks.length - 1)
      : 0;
  const currentChunk = activeChunks[safeIdx] || null;

  useEffect(() => {
    if (activeSpanRef.current) {
      activeSpanRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center',
      });
    }
  }, [safeIdx, isPreview, docData?.document_name]);

  if (loadingDoc) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-10 flex items-center justify-center text-xs text-slate-400 space-x-2">
        <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
        <span>Loading persisted chunks and reconstructing extracted document text...</span>
      </div>
    );
  }

  if (!docData) {
    return (
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-xs text-slate-400">
        Select an indexed document above (or ingest a PDF/TXT file in Live Chat) to explore its text chunks, overlap boundaries, and vector pipeline.
      </div>
    );
  }

  // Render original extracted document text with exact highlights for active chunk + overlap regions
  const renderHighlightedDocument = () => {
    if (!currentChunk || !sourceText) {
      return <span className="text-slate-400">{sourceText}</span>;
    }

    const start = Math.max(0, Math.min(currentChunk.start_char, sourceText.length));
    const end = Math.max(start, Math.min(currentChunk.end_char, sourceText.length));
    const prevOverlapEnd = Math.min(end, start + (currentChunk.overlap_prev_chars || 0));
    const nextOverlapStart = Math.max(
      prevOverlapEnd,
      end - (currentChunk.overlap_next_chars || 0)
    );

    const before = sourceText.slice(0, start);
    const overlapPrevPart = sourceText.slice(start, prevOverlapEnd);
    const corePart = sourceText.slice(prevOverlapEnd, nextOverlapStart);
    const overlapNextPart = sourceText.slice(nextOverlapStart, end);
    const after = sourceText.slice(end);

    return (
      <>
        <span className="text-slate-500">{before}</span>
        <span
          ref={activeSpanRef}
          className="rounded px-0.5 bg-sky-500/15 ring-1 ring-sky-400/50 text-slate-100"
        >
          {overlapPrevPart && (
            <span
              className="bg-amber-500/30 text-amber-100 border-b border-amber-400"
              title={`Overlap with previous chunk (${currentChunk.overlap_prev_chars} chars)`}
            >
              {overlapPrevPart}
            </span>
          )}
          <span>{corePart}</span>
          {overlapNextPart && (
            <span
              className="bg-indigo-500/35 text-indigo-100 border-b border-indigo-400"
              title={`Overlap with next chunk (${currentChunk.overlap_next_chars} chars)`}
            >
              {overlapNextPart}
            </span>
          )}
        </span>
        <span className="text-slate-500">{after}</span>
      </>
    );
  };

  return (
    <div className="space-y-4">
      {/* Interactive Chunking Configuration Sandbox Controls */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Sliders className="w-4 h-4 text-sky-400" />
              <span>Interactive Chunking Configuration &amp; Sandbox</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Production index uses <code className="text-sky-300">chunk_size={docData.chunk_size}</code> and{' '}
              <code className="text-sky-300">chunk_overlap={docData.chunk_overlap}</code>. Experiment below without altering ChromaDB.
            </p>
          </div>
          <div className="flex items-center space-x-2">
            {isPreview && (
              <button
                onClick={onResetPreview}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-xs transition"
              >
                <RotateCcw className="w-3.5 h-3.5 text-sky-400" />
                <span>View Production Chunks ({docData.total_chunks})</span>
              </button>
            )}
            <button
              onClick={onRunPreview}
              disabled={loadingPreview || previewChunkOverlap >= previewChunkSize}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-xs font-medium transition shadow-sm shadow-sky-600/20"
            >
              {loadingPreview ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Sparkles className="w-3.5 h-3.5" />
              )}
              <span>Simulate Preview</span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 bg-slate-950/60 border border-slate-800 rounded-lg p-3.5">
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-slate-300 font-medium">Preview Chunk Size (chars)</span>
              <span className="font-mono text-sky-400 font-bold">{previewChunkSize}</span>
            </div>
            <input
              type="range"
              min="200"
              max="3000"
              step="50"
              value={previewChunkSize}
              onChange={(e) => {
                const val = parseInt(e.target.value, 10);
                onPreviewChunkSizeChange(val);
                if (previewChunkOverlap >= val) {
                  onPreviewChunkOverlapChange(Math.max(0, val - 50));
                }
              }}
              className="w-full accent-sky-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
              <span>200</span>
              <span>Production: {docData.chunk_size}</span>
              <span>3000</span>
            </div>
          </div>

          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-slate-300 font-medium">Preview Chunk Overlap (chars)</span>
              <span className="font-mono text-amber-400 font-bold">{previewChunkOverlap}</span>
            </div>
            <input
              type="range"
              min="0"
              max={Math.min(600, Math.max(0, previewChunkSize - 50))}
              step="25"
              value={previewChunkOverlap}
              onChange={(e) => onPreviewChunkOverlapChange(parseInt(e.target.value, 10))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-0.5">
              <span>0</span>
              <span>Production: {docData.chunk_overlap}</span>
              <span>{Math.min(600, Math.max(0, previewChunkSize - 50))}</span>
            </div>
          </div>
        </div>

        {previewError && (
          <div className="mt-3 p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center space-x-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{previewError}</span>
          </div>
        )}

        {isPreview ? (
          <div className="mt-3 p-2.5 rounded-lg bg-amber-950/30 border border-amber-500/40 text-xs text-amber-200 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>
                <strong>TEMPORARY PREVIEW MODE:</strong> Simulated{' '}
                <strong>{previewData.total_chunks}</strong> chunks (avg{' '}
                {previewData.avg_chunk_chars} chars) with size={previewData.preview_chunk_size}, overlap=
                {previewData.preview_chunk_overlap}. Persistent ChromaDB ({docData.total_chunks} chunks) was NOT modified.
              </span>
            </div>
          </div>
        ) : (
          <div className="mt-3 p-2 rounded-lg bg-emerald-950/20 border border-emerald-500/20 text-[11px] text-emerald-300 flex items-center space-x-2">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            <span>
              Viewing actual indexed ChromaDB chunks ({docData.total_chunks} chunks stored for{' '}
              <code className="text-emerald-200">{docData.document_name}</code>).
            </span>
          </div>
        )}
      </div>

      {/* Side-by-Side Original Document Highlight + Interactive Chunk Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: Original Extracted Document Text */}
        <div className="lg:col-span-6 bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/50 flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-sky-400" />
              <span className="text-xs font-semibold text-slate-200">
                Original Extracted Document Text
              </span>
            </div>
            <div className="flex items-center space-x-2 text-[10px] font-mono">
              <span className="px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-300 border border-sky-500/30">
                Active Chunk
              </span>
              <span className="px-1.5 py-0.5 rounded bg-amber-500/25 text-amber-200 border border-amber-500/40">
                Prev Overlap
              </span>
              <span className="px-1.5 py-0.5 rounded bg-indigo-500/30 text-indigo-200 border border-indigo-500/40">
                Next Overlap
              </span>
            </div>
          </div>
          <div className="p-4 h-[460px] overflow-y-auto text-xs font-mono leading-relaxed whitespace-pre-wrap select-text bg-slate-950/60">
            {renderHighlightedDocument()}
          </div>
        </div>

        {/* Right: Chunk Navigator + Detailed Chunk Inspector */}
        <div className="lg:col-span-6 bg-slate-900/60 border border-slate-800 rounded-xl flex flex-col overflow-hidden">
          <div className="px-4 py-3 border-b border-slate-800 bg-slate-950/50 flex items-center justify-between gap-2">
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-sky-400" />
              <span className="text-xs font-semibold text-slate-200">
                Chunk Inspector ({activeChunks.length > 0 ? `${safeIdx + 1} of ${activeChunks.length}` : '0'})
              </span>
            </div>
            <div className="flex items-center space-x-1.5">
              <button
                onClick={() => onSelectChunkIndex(Math.max(0, safeIdx - 1))}
                disabled={safeIdx <= 0}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 border border-slate-700 transition"
                title="Previous Chunk"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() =>
                  onSelectChunkIndex(Math.min(activeChunks.length - 1, safeIdx + 1))
                }
                disabled={safeIdx >= activeChunks.length - 1}
                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-200 border border-slate-700 transition"
                title="Next Chunk"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          <div className="p-4 h-[460px] overflow-y-auto space-y-3">
            {/* Horizontal Scrollable Chunk Pills Strip */}
            <div className="flex items-center gap-1.5 overflow-x-auto pb-2 border-b border-slate-800">
              {activeChunks.map((chunk, idx) => {
                const selected = idx === safeIdx;
                const isRetrieved = highlightedRetrievedChunkIds.includes(chunk.chunk_id);
                return (
                  <React.Fragment key={chunk.chunk_id}>
                    <button
                      onClick={() => onSelectChunkIndex(idx)}
                      className={`shrink-0 px-2.5 py-1.5 rounded-lg border text-[11px] font-mono transition flex items-center space-x-1.5 ${
                        selected
                          ? 'bg-sky-500/20 border-sky-400 text-sky-200 font-bold'
                          : isRetrieved
                          ? 'bg-emerald-500/15 border-emerald-400/50 text-emerald-300'
                          : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                      }`}
                    >
                      <span>Chunk {idx + 1}</span>
                      {isRetrieved && (
                        <span className="px-1 py-0.2 rounded bg-emerald-500/30 text-[9px] text-emerald-200">
                          HIT
                        </span>
                      )}
                    </button>
                    {idx < activeChunks.length - 1 && chunk.overlap_next_chars > 0 && (
                      <span
                        className="shrink-0 text-[9px] font-mono text-amber-400/90 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded"
                        title={`Overlap between Chunk ${idx + 1} and Chunk ${idx + 2}: ${chunk.overlap_next_chars} chars`}
                      >
                        +{chunk.overlap_next_chars}c
                      </span>
                    )}
                  </React.Fragment>
                );
              })}
            </div>

            {currentChunk && (
              <div className="space-y-3">
                {/* Selected Chunk Metadata Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] font-mono">
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-2">
                    <span className="text-slate-500 block text-[10px]">Chunk ID</span>
                    <span className="text-sky-300 truncate block" title={currentChunk.chunk_id}>
                      {currentChunk.chunk_id}
                    </span>
                  </div>
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-2">
                    <span className="text-slate-500 block text-[10px]">Document / Page</span>
                    <span className="text-slate-200 truncate block">
                      {currentChunk.document_name}
                      {currentChunk.page_number ? ` · p.${currentChunk.page_number}` : ''}
                    </span>
                  </div>
                  <div className="bg-slate-950 border border-slate-800 rounded-lg p-2">
                    <span className="text-slate-500 block text-[10px]">Length / Tokens</span>
                    <span className="text-slate-200">
                      {currentChunk.char_count} chars · ~{currentChunk.token_count_estimate} tok
                    </span>
                  </div>
                </div>

                {/* Preceding Overlap Box */}
                {currentChunk.overlap_prev_chars > 0 && (
                  <div className="bg-amber-950/25 border border-amber-500/35 rounded-lg p-2.5">
                    <div className="flex items-center justify-between text-[10px] font-mono text-amber-300 mb-1">
                      <span>Overlap with Preceding Chunk ({currentChunk.overlap_prev_chars} chars)</span>
                    </div>
                    <p className="text-[11px] font-mono text-amber-100/90 whitespace-pre-wrap">
                      "{currentChunk.overlap_prev_text}"
                    </p>
                  </div>
                )}

                {/* Full Chunk Text Content */}
                <div className="bg-slate-950 border border-slate-800 rounded-lg p-3">
                  <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mb-1.5 border-b border-slate-800 pb-1">
                    <span>Complete Chunk Content</span>
                    <span>
                      Offsets [{currentChunk.start_char} .. {currentChunk.end_char}]
                    </span>
                  </div>
                  <p className="text-xs font-mono text-slate-200 whitespace-pre-wrap leading-relaxed select-text">
                    {currentChunk.text}
                  </p>
                </div>

                {/* Following Overlap Box */}
                {currentChunk.overlap_next_chars > 0 && (
                  <div className="bg-indigo-950/25 border border-indigo-500/35 rounded-lg p-2.5">
                    <div className="flex items-center justify-between text-[10px] font-mono text-indigo-300 mb-1">
                      <span>Overlap with Following Chunk ({currentChunk.overlap_next_chars} chars)</span>
                    </div>
                    <p className="text-[11px] font-mono text-indigo-100/90 whitespace-pre-wrap">
                      "{currentChunk.overlap_next_text}"
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

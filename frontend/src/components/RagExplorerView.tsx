import React, { useCallback, useEffect, useState } from 'react';
import {
  FileText,
  RefreshCw,
  AlertCircle,
  GitBranch,
  Database,
  Layers,
  Scissors,
  Binary,
  BookOpen,
  Search,
} from 'lucide-react';
import {
  getDocumentChunks,
  getIndexedDocuments,
  inspectRetrieval,
  previewChunking,
  queryRAG,
} from '../api';
import {
  ChunkPreviewResponse,
  DocumentChunksResponse,
  IndexedDocItem,
  QueryResponse,
  RetrievalInspectResponse,
} from '../types';
import { PipelineVisualizer } from './explorer/PipelineVisualizer';
import { ChunkingViewer } from './explorer/ChunkingViewer';
import { RetrievalInspectorPanel } from './explorer/RetrievalInspectorPanel';

interface RagExplorerViewProps {
  apiKey: string;
  selectedDocument: string | null;
  onSelectDocument: (filename: string | null) => void;
}

type ExplorerSubTab = 'chunking' | 'retrieval' | 'pipeline';

export const RagExplorerView: React.FC<RagExplorerViewProps> = ({
  apiKey,
  selectedDocument,
  onSelectDocument,
}) => {
  const [subTab, setSubTab] = useState<ExplorerSubTab>('chunking');
  const [documents, setDocuments] = useState<IndexedDocItem[]>([]);
  const [activeDocName, setActiveDocName] = useState<string>(selectedDocument || '');
  const [docData, setDocData] = useState<DocumentChunksResponse | null>(null);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [loadingDocDetail, setLoadingDocDetail] = useState(false);
  const [docError, setDocError] = useState<string | null>(null);

  // Chunking Explorer & Sandbox State
  const [activeChunkIndex, setActiveChunkIndex] = useState(0);
  const [previewChunkSize, setPreviewChunkSize] = useState(1200);
  const [previewChunkOverlap, setPreviewChunkOverlap] = useState(100);
  const [previewData, setPreviewData] = useState<ChunkPreviewResponse | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);

  // Retrieval Inspector State
  const [question, setQuestion] = useState('');
  const [scopeDoc, setScopeDoc] = useState<string>(selectedDocument || '__all__');
  const [topK, setTopK] = useState(4);
  const [useMmr, setUseMmr] = useState(true);
  const [isInspecting, setIsInspecting] = useState(false);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [inspectResult, setInspectResult] = useState<RetrievalInspectResponse | null>(null);

  // Optional Stage 10 Live Query Telemetry
  const [liveQueryResult, setLiveQueryResult] = useState<QueryResponse | null>(null);
  const [isRunningLiveQuery, setIsRunningLiveQuery] = useState(false);

  const loadDocuments = useCallback(async () => {
    if (!apiKey) return;
    setLoadingDocs(true);
    try {
      const list = await getIndexedDocuments(apiKey);
      setDocuments(list);
      if (list.length > 0) {
        const preferred =
          (selectedDocument && list.some((d) => d.filename === selectedDocument)
            ? selectedDocument
            : null) ||
          (activeDocName && list.some((d) => d.filename === activeDocName)
            ? activeDocName
            : null) ||
          list[0].filename;
        setActiveDocName(preferred);
        if (scopeDoc === '__all__' && list.length === 1) {
          setScopeDoc(preferred);
        }
      } else {
        setActiveDocName('');
        setDocData(null);
      }
    } catch (err: unknown) {
      setDocError(err instanceof Error ? err.message : 'Failed loading indexed documents.');
    } finally {
      setLoadingDocs(false);
    }
  }, [apiKey, selectedDocument, activeDocName, scopeDoc]);

  useEffect(() => {
    loadDocuments();
  }, [apiKey]);

  const loadDocChunks = useCallback(
    async (filename: string) => {
      if (!apiKey || !filename) return;
      setLoadingDocDetail(true);
      setDocError(null);
      setPreviewData(null);
      setActiveChunkIndex(0);
      try {
        const res = await getDocumentChunks(filename, apiKey);
        setDocData(res);
        setPreviewChunkSize(res.chunk_size);
        setPreviewChunkOverlap(res.chunk_overlap);
      } catch (err: unknown) {
        setDocError(err instanceof Error ? err.message : 'Failed loading document chunks.');
        setDocData(null);
      } finally {
        setLoadingDocDetail(false);
      }
    },
    [apiKey]
  );

  useEffect(() => {
    if (activeDocName) {
      loadDocChunks(activeDocName);
    }
  }, [activeDocName, loadDocChunks]);

  const handleSelectDoc = (filename: string) => {
    setActiveDocName(filename);
    setScopeDoc(filename);
    onSelectDocument(filename);
  };

  const handleRunPreview = async () => {
    if (!docData) return;
    setLoadingPreview(true);
    setPreviewError(null);
    setActiveChunkIndex(0);
    try {
      const res = await previewChunking(
        {
          document_name: docData.document_name,
          text: docData.reconstructed_text,
          chunk_size: previewChunkSize,
          chunk_overlap: previewChunkOverlap,
        },
        apiKey
      );
      setPreviewData(res);
    } catch (err: unknown) {
      setPreviewError(err instanceof Error ? err.message : 'Preview simulation failed.');
    } finally {
      setLoadingPreview(false);
    }
  };

  const handleResetPreview = () => {
    setPreviewData(null);
    setPreviewError(null);
    setActiveChunkIndex(0);
    if (docData) {
      setPreviewChunkSize(docData.chunk_size);
      setPreviewChunkOverlap(docData.chunk_overlap);
    }
  };

  const handleInspectRetrieval = async () => {
    const q = question.trim();
    if (!q) return;
    setIsInspecting(true);
    setInspectError(null);
    setLiveQueryResult(null);
    try {
      const res = await inspectRetrieval(
        {
          question: q,
          top_k: topK,
          document_name: scopeDoc === '__all__' ? null : scopeDoc,
          use_mmr: useMmr,
          mmr_lambda: 0.72,
        },
        apiKey
      );
      setInspectResult(res);
    } catch (err: unknown) {
      setInspectError(err instanceof Error ? err.message : 'Retrieval inspection failed.');
    } finally {
      setIsInspecting(false);
    }
  };

  const handleRunLiveStage10 = async () => {
    const q = (inspectResult?.question || question).trim();
    if (!q || isRunningLiveQuery) return;
    setIsRunningLiveQuery(true);
    try {
      const targetDoc =
        inspectResult?.document_scope && inspectResult.document_scope !== 'all'
          ? inspectResult.document_scope
          : scopeDoc === '__all__'
          ? null
          : scopeDoc;
      const res = await queryRAG(q, topK, apiKey, 'default_qa', undefined, targetDoc);
      setLiveQueryResult(res);
    } catch (err: unknown) {
      setInspectError(err instanceof Error ? err.message : 'Stage 10 generation failed.');
    } finally {
      setIsRunningLiveQuery(false);
    }
  };

  const handleLocateChunk = async (
    documentName: string | null,
    chunkId: string,
    chunkText: string
  ) => {
    setPreviewData(null);
    let targetDoc = docData;

    if (documentName && documentName !== activeDocName) {
      setActiveDocName(documentName);
      onSelectDocument(documentName);
      try {
        targetDoc = await getDocumentChunks(documentName, apiKey);
        setDocData(targetDoc);
      } catch {
        // Keep current view on error
      }
    }

    if (targetDoc && targetDoc.chunks.length > 0) {
      let foundIdx = targetDoc.chunks.findIndex((c) => c.chunk_id === chunkId);
      if (foundIdx === -1 && chunkText) {
        const snippet = chunkText.slice(0, 80);
        foundIdx = targetDoc.chunks.findIndex((c) => c.text.includes(snippet));
      }
      if (foundIdx >= 0) {
        setActiveChunkIndex(foundIdx);
      }
    }
    setSubTab('chunking');
  };

  const highlightedRetrievedIds =
    inspectResult?.retrieved_chunks.map((c) => c.chunk_id) || [];

  return (
    <div className="space-y-4 pb-6">
      {/* Top Document Selector & Architecture Summary Header */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center space-x-2.5">
            <div className="w-9 h-9 rounded-lg bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400">
              <GitBranch className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100">
                RAG Pipeline Explorer
              </h2>
              <p className="text-xs text-slate-400">
                Inspect text chunking, consecutive overlap regions, vector embeddings, and deduplicated Top-K / MMR retrieval.
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <select
              value={activeDocName}
              onChange={(e) => handleSelectDoc(e.target.value)}
              disabled={loadingDocs || documents.length === 0}
              className="bg-slate-950 border border-slate-700 text-slate-100 text-xs font-mono rounded-lg px-3 py-2 focus:outline-none focus:border-sky-500 max-w-xs"
            >
              {documents.length === 0 ? (
                <option value="">No documents indexed</option>
              ) : (
                documents.map((d) => (
                  <option key={d.filename} value={d.filename}>
                    {d.filename} ({d.chunks} chunks)
                  </option>
                ))
              )}
            </select>
            <button
              onClick={loadDocuments}
              disabled={loadingDocs}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 transition"
              title="Refresh indexed documents"
            >
              <RefreshCw className={`w-4 h-4 ${loadingDocs ? 'animate-spin text-sky-400' : ''}`} />
            </button>
          </div>
        </div>

        {/* Document & Splitter Telemetry Cards */}
        {docData && (
          <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2 text-xs font-mono">
            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 flex items-center space-x-1">
                <FileText className="w-3 h-3 text-sky-400" />
                <span>Document</span>
              </span>
              <span className="text-slate-100 font-semibold truncate block mt-0.5" title={docData.document_name}>
                {docData.document_name}
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 block">File Type</span>
              <span className="text-sky-300 font-semibold uppercase block mt-0.5">
                {docData.file_type}
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 block">Extracted Text</span>
              <span className="text-slate-100 font-semibold block mt-0.5">
                {docData.total_extracted_chars.toLocaleString()} ch
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 block">Total Pages</span>
              <span className="text-slate-100 font-semibold block mt-0.5">
                {docData.total_pages ?? 'N/A'}
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 flex items-center space-x-1">
                <Layers className="w-3 h-3 text-emerald-400" />
                <span>Total Chunks</span>
              </span>
              <span className="text-emerald-300 font-bold block mt-0.5">
                {docData.total_chunks}
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 flex items-center space-x-1">
                <Scissors className="w-3 h-3 text-sky-400" />
                <span>Chunk Size</span>
              </span>
              <span className="text-slate-100 font-semibold block mt-0.5">
                {docData.chunk_size} ch
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 block">Chunk Overlap</span>
              <span className="text-amber-300 font-semibold block mt-0.5">
                {docData.chunk_overlap} ch
              </span>
            </div>

            <div className="bg-slate-950/70 border border-slate-800 rounded-lg p-2.5">
              <span className="text-[10px] text-slate-500 flex items-center space-x-1">
                <Binary className="w-3 h-3 text-indigo-400" />
                <span>Embed Model</span>
              </span>
              <span className="text-indigo-300 font-semibold truncate block mt-0.5" title={docData.embedding_model}>
                {docData.embedding_model}
              </span>
            </div>
          </div>
        )}

        {docError && (
          <div className="p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{docError}</span>
          </div>
        )}

        {/* Explorer Sub-View Tabs */}
        <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/80">
          <button
            onClick={() => setSubTab('chunking')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition border ${
              subTab === 'chunking'
                ? 'bg-sky-500/20 border-sky-400 text-sky-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Chunking Explorer &amp; Overlap</span>
          </button>

          <button
            onClick={() => setSubTab('retrieval')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition border ${
              subTab === 'retrieval'
                ? 'bg-sky-500/20 border-sky-400 text-sky-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>Retrieval Inspector (Top-K &amp; MMR)</span>
          </button>

          <button
            onClick={() => setSubTab('pipeline')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition border ${
              subTab === 'pipeline'
                ? 'bg-sky-500/20 border-sky-400 text-sky-200'
                : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>10-Stage Pipeline &amp; Embeddings</span>
          </button>
        </div>
      </div>

      {/* Active Sub-View */}
      {subTab === 'chunking' && (
        <ChunkingViewer
          docData={docData}
          previewData={previewData}
          loadingDoc={loadingDocDetail}
          loadingPreview={loadingPreview}
          previewError={previewError}
          activeChunkIndex={activeChunkIndex}
          onSelectChunkIndex={setActiveChunkIndex}
          previewChunkSize={previewChunkSize}
          previewChunkOverlap={previewChunkOverlap}
          onPreviewChunkSizeChange={setPreviewChunkSize}
          onPreviewChunkOverlapChange={setPreviewChunkOverlap}
          onRunPreview={handleRunPreview}
          onResetPreview={handleResetPreview}
          highlightedRetrievedChunkIds={highlightedRetrievedIds}
        />
      )}

      {subTab === 'retrieval' && (
        <RetrievalInspectorPanel
          documents={documents}
          question={question}
          onQuestionChange={setQuestion}
          scopeDoc={scopeDoc}
          onScopeDocChange={setScopeDoc}
          topK={topK}
          onTopKChange={setTopK}
          useMmr={useMmr}
          onUseMmrChange={setUseMmr}
          isInspecting={isInspecting}
          error={inspectError}
          result={inspectResult}
          onInspect={handleInspectRetrieval}
          onLocateChunk={handleLocateChunk}
        />
      )}

      {subTab === 'pipeline' && (
        <PipelineVisualizer
          docData={docData}
          inspectData={inspectResult}
          liveQueryResult={liveQueryResult}
          isRunningLiveQuery={isRunningLiveQuery}
          onRunLiveQuery={handleRunLiveStage10}
        />
      )}
    </div>
  );
};

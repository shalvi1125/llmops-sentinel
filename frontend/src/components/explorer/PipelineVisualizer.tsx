import React, { useState } from 'react';
import {
  Upload,
  FileText,
  Scissors,
  Binary,
  Database,
  Search,
  Sparkles,
  ListFilter,
  Layers,
  Bot,
  ChevronRight,
  Info,
  Cpu,
  ArrowRight,
  Play,
  Loader2,
} from 'lucide-react';
import {
  DocumentChunksResponse,
  QueryResponse,
  RetrievalInspectResponse,
} from '../../types';

interface PipelineVisualizerProps {
  docData: DocumentChunksResponse | null;
  inspectData: RetrievalInspectResponse | null;
  liveQueryResult: QueryResponse | null;
  isRunningLiveQuery: boolean;
  onRunLiveQuery?: () => void;
}

interface StageSpec {
  id: number;
  title: string;
  shortLabel: string;
  component: string;
  description: string;
  inputText: string;
  outputText: string;
  icon: React.ReactNode;
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({
  docData,
  inspectData,
  liveQueryResult,
  isRunningLiveQuery,
  onRunLiveQuery,
}) => {
  const [activeStageId, setActiveStageId] = useState<number>(3);
  const [activeEmbedStep, setActiveEmbedStep] = useState<number>(0);

  const stages: StageSpec[] = [
    {
      id: 1,
      title: 'Document Upload',
      shortLabel: 'Upload',
      component: 'POST /ingest -> src/services/ingest.py',
      description:
        'Validates file extension (.pdf / .txt), MIME type, size limit (10 MB), and assigns a unique document_id before clearing stale chunks on re-upload.',
      inputText: docData
        ? `File '${docData.document_name}' (.${docData.file_type}) with API key verification`
        : 'Select or upload a PDF/TXT document to inspect real upload telemetry',
      outputText: docData
        ? `document_id: ${docData.document_id || 'legacy-unversioned'} | ingested: ${
            docData.ingestion_timestamp || 'pre-Phase-1 index'
          }`
        : 'Validated temporary file stream',
      icon: <Upload className="w-3.5 h-3.5" />,
    },
    {
      id: 2,
      title: 'Text Extraction',
      shortLabel: 'Extract',
      component: 'PyPDFLoader / TextLoader (LangChain)',
      description:
        'Extracts page-by-page text from PDFs (recording 1-based page_number) or UTF-8 text from TXT files while stripping raw temporary paths.',
      inputText: docData
        ? `Raw .${docData.file_type} binary payload (${docData.total_pages ?? 1} page(s))`
        : 'Uploaded PDF or TXT file',
      outputText: docData
        ? `${docData.total_extracted_chars.toLocaleString()} extracted characters (~${docData.total_estimated_tokens.toLocaleString()} est. tokens)`
        : 'Raw extracted page documents',
      icon: <FileText className="w-3.5 h-3.5" />,
    },
    {
      id: 3,
      title: 'Text Chunking',
      shortLabel: 'Chunk',
      component: 'RecursiveCharacterTextSplitter (ingest.py)',
      description:
        'Splits extracted text sequentially using configured chunk_size and chunk_overlap, deduplicates exact repeated text within the upload, and stamps chunk_id.',
      inputText: docData
        ? `${docData.total_extracted_chars.toLocaleString()} chars | Config: size=${docData.chunk_size}, overlap=${docData.chunk_overlap}`
        : 'Extracted text stream',
      outputText: docData
        ? `${docData.total_chunks} persisted chunk(s) with unique chunk_id & content_hash`
        : 'Overlapping text segments',
      icon: <Scissors className="w-3.5 h-3.5" />,
    },
    {
      id: 4,
      title: 'Embedding Generation',
      shortLabel: 'Embed Docs',
      component: `OllamaEmbeddings (${docData?.embedding_model || 'nomic-embed-text'})`,
      description:
        'Passes each unique chunk through the local Ollama embedding model to transform text semantics into dense numerical vectors.',
      inputText: docData
        ? `${docData.total_chunks} unique text chunk(s) queued for vectorization`
        : 'Chunk text strings',
      outputText: docData
        ? `${docData.total_chunks} embedding vector(s) via ${docData.embedding_model}`
        : 'Dense floating-point embedding vectors',
      icon: <Binary className="w-3.5 h-3.5" />,
    },
    {
      id: 5,
      title: 'ChromaDB Storage',
      shortLabel: 'Chroma Store',
      component: 'Chroma (ragscope_collection, .chroma)',
      description:
        'Persists vectors, chunk text, and isolated document metadata (document_id, document_name, page_number, chunk_id) using deterministic chunk IDs.',
      inputText: docData
        ? `${docData.total_chunks} vector(s) + isolated document metadata for '${docData.document_name}'`
        : 'Vectors + metadata payload',
      outputText: docData
        ? `Indexed cleanly in ragscope_collection (cosine distance HNSW index)`
        : 'Queryable persistent HNSW vector collection',
      icon: <Database className="w-3.5 h-3.5" />,
    },
    {
      id: 6,
      title: 'Query Embedding',
      shortLabel: 'Embed Query',
      component: `POST /query/inspect & /query (${inspectData?.embedding_model || 'nomic-embed-text'})`,
      description:
        'Embeds the user question into the exact same vector space as document chunks so semantic distance can be measured reliably.',
      inputText: inspectData
        ? `Question: "${inspectData.question}" | Scope: ${inspectData.document_scope}`
        : 'Run Retrieval Inspector below to populate real query stage data',
      outputText: inspectData
        ? `Single query vector via ${inspectData.embedding_model} ready for filtered search`
        : 'Query semantic vector',
      icon: <Sparkles className="w-3.5 h-3.5" />,
    },
    {
      id: 7,
      title: 'Similarity Search',
      shortLabel: 'Vector Search',
      component: 'Chroma similarity_search_with_score(..., filter=...)',
      description:
        'Applies Phase 1 document-level Chroma metadata filters and oversamples candidates ranked by cosine distance d = 1 - cos(theta).',
      inputText: inspectData
        ? `where filter scope='${inspectData.document_scope}', fetch_k=${inspectData.retrieval_diagnostics.candidates_fetched ?? '—'}`
        : 'Query vector + strict document metadata filter',
      outputText: inspectData
        ? `${inspectData.retrieval_diagnostics.candidates_fetched ?? inspectData.retrieved_chunks.length} candidate(s) scored in ${inspectData.retrieval_diagnostics.retrieval_latency_ms} ms`
        : 'Candidate chunks ordered by cosine distance',
      icon: <Search className="w-3.5 h-3.5" />,
    },
    {
      id: 8,
      title: 'Top-K Retrieval',
      shortLabel: 'Dedup & Top-K',
      component: 'retrieve_context_chunks (src/services/query.py)',
      description:
        'Removes exact and >=90% near-duplicate chunks, applies relevance-balanced MMR reranking (lambda=0.72), and returns up to K unique matches without padding.',
      inputText: inspectData
        ? `Requested K=${inspectData.retrieval_diagnostics.requested_k}, Candidates=${inspectData.retrieval_diagnostics.candidates_fetched}, Strategy=${inspectData.retrieval_diagnostics.retrieval_strategy}`
        : 'Oversampled scored candidate list',
      outputText: inspectData
        ? `${inspectData.retrieved_chunks.length} unique chunk(s) kept | ${
            inspectData.retrieval_diagnostics.duplicates_removed ?? 0
          } duplicate(s) removed | Top dist=${inspectData.retrieval_diagnostics.top_score ?? 'N/A'}`
        : 'Unique, ranked SourceChunk records',
      icon: <ListFilter className="w-3.5 h-3.5" />,
    },
    {
      id: 9,
      title: 'Context Construction',
      shortLabel: 'Context',
      component: 'PromptTemplate (src/prompts/catalog.py)',
      description:
        'Concatenates deduplicated chunks (bounded to <=3500 chars) into the strict grounding prompt template that refuses unsupported claims.',
      inputText: inspectData
        ? `${inspectData.retrieved_chunks.length} deduplicated chunk(s) from '${inspectData.document_scope}'`
        : 'Deduplicated Top-K chunks',
      outputText: inspectData
        ? `${inspectData.context_char_count.toLocaleString()} chars assembled (${
            inspectData.insufficient_evidence ? 'Weak alignment / refusal caution' : 'Grounded prompt context ready'
          })`
        : 'Bounded prompt context string',
      icon: <Layers className="w-3.5 h-3.5" />,
    },
    {
      id: 10,
      title: 'LLM Answer Generation',
      shortLabel: 'Generate',
      component: 'ChatOllama (llama3.2) + Background Mistral Judge',
      description:
        'Synthesizes a grounded answer strictly from assembled context and schedules asynchronous MLflow + Mistral judge evaluation.',
      inputText: liveQueryResult
        ? `Grounded prompt (${liveQueryResult.source_chunks?.length ?? liveQueryResult.sources.length} chunks, prompt=${liveQueryResult.prompt_id})`
        : inspectData
        ? 'Inspect completed — click "Run Live Stage 10 Generation" to execute real LLM synthesis on this query'
        : 'Grounded context + user question',
      outputText: liveQueryResult
        ? `${liveQueryResult.answer.slice(0, 160)}${liveQueryResult.answer.length > 160 ? '...' : ''} (${liveQueryResult.generation_latency_ms ?? '—'} ms, model=${liveQueryResult.model})`
        : 'Final cited response + MLflow trace',
      icon: <Bot className="w-3.5 h-3.5" />,
    },
  ];

  const activeStage = stages.find((s) => s.id === activeStageId) || stages[0];

  const embeddingSteps = [
    {
      title: '1. Text is Divided into Chunks',
      detail: 'Long documents are split into focused segments (1200 chars with 100-char overlap) so key sentences are not cut off mid-thought.',
      visual: 'Text -> [Chunk 1] [Chunk 2] [Chunk 3]',
    },
    {
      title: '2. Passed to Embedding Model',
      detail: `Each chunk is sent to Ollama (${docData?.embedding_model || 'nomic-embed-text'}), which reads semantic meaning rather than just exact keyword spelling.`,
      visual: `[Chunk Text] -> ${docData?.embedding_model || 'nomic-embed-text'}`,
    },
    {
      title: '3. Converted into a Numerical Vector',
      detail: 'The model outputs a list of numbers representing where the chunk sits in meaning space. Semantically related ideas point in similar directions.',
      visual: 'Semantic meaning -> Dense numerical direction vector',
    },
    {
      title: '4. Vectors Stored in ChromaDB',
      detail: 'ChromaDB stores each vector alongside its text, page_number, and document_name inside an HNSW similarity index on disk.',
      visual: '(vector_i, text_i, metadata_i) -> ChromaDB',
    },
    {
      title: "5. User's Question Converted to Vector",
      detail: 'When you ask a question, the exact same embedding model converts your question into a numerical query vector.',
      visual: `Question -> ${docData?.embedding_model || 'nomic-embed-text'} -> Query Vector`,
    },
    {
      title: '6. Closest Vectors Retrieved',
      detail: 'Chroma computes cosine distance d = 1 - cos(theta) within the selected document scope. Smaller distance = closer semantic match.',
      visual: 'min cosine_distance(Query, Chunk_i) -> Top-K Unique Chunks',
    },
  ];

  return (
    <div className="space-y-4">
      {/* 10-Stage Interactive RAG Pipeline Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Cpu className="w-4 h-4 text-sky-400" />
              <span>Interactive End-to-End RAG Pipeline</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Click any stage to inspect real backend inputs, outputs, and component bindings.
            </p>
          </div>
          {inspectData && onRunLiveQuery && (
            <button
              onClick={onRunLiveQuery}
              disabled={isRunningLiveQuery}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-xs font-medium transition shadow-sm shadow-sky-600/20"
            >
              {isRunningLiveQuery ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Play className="w-3.5 h-3.5" />
              )}
              <span>
                {isRunningLiveQuery ? 'Generating Stage 10...' : 'Run Stage 10 LLM on Query'}
              </span>
            </button>
          )}
        </div>

        {/* Clickable Pipeline Stages */}
        <div className="flex flex-wrap items-center gap-1.5 pb-3 border-b border-slate-800/80">
          {stages.map((stage, idx) => {
            const isSelected = stage.id === activeStageId;
            const hasLiveSignal =
              (stage.id <= 5 && docData !== null) ||
              (stage.id >= 6 && stage.id <= 9 && inspectData !== null) ||
              (stage.id === 10 && liveQueryResult !== null);

            return (
              <React.Fragment key={stage.id}>
                <button
                  onClick={() => setActiveStageId(stage.id)}
                  className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border text-[11px] font-medium transition ${
                    isSelected
                      ? 'bg-sky-500/20 border-sky-400 text-sky-200 shadow-sm shadow-sky-500/15'
                      : hasLiveSignal
                      ? 'bg-slate-950/90 border-emerald-500/30 text-slate-200 hover:border-sky-500/50'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <span
                    className={`w-4 h-4 rounded-full text-[9px] font-mono flex items-center justify-center ${
                      isSelected
                        ? 'bg-sky-400 text-slate-950 font-bold'
                        : hasLiveSignal
                        ? 'bg-emerald-500/20 text-emerald-300'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {stage.id}
                  </span>
                  <span className={isSelected ? 'text-sky-300' : 'text-slate-400'}>
                    {stage.icon}
                  </span>
                  <span className="whitespace-nowrap">{stage.title}</span>
                </button>
                {idx < stages.length - 1 && (
                  <ChevronRight className="w-3.5 h-3.5 text-slate-600 shrink-0 hidden sm:inline" />
                )}
              </React.Fragment>
            );
          })}
        </div>

        {/* Active Stage Inspector Panel */}
        <div className="mt-4 grid grid-cols-1 lg:grid-cols-12 gap-4 bg-slate-950/70 border border-slate-800 rounded-xl p-4">
          <div className="lg:col-span-5 space-y-2 border-b lg:border-b-0 lg:border-r border-slate-800 pb-3 lg:pb-0 lg:pr-4">
            <div className="flex items-center space-x-2">
              <span className="px-2 py-0.5 rounded bg-sky-500/15 text-sky-300 border border-sky-500/30 font-mono text-[10px]">
                Stage {activeStage.id} / 10
              </span>
              <h4 className="text-sm font-bold text-slate-100">{activeStage.title}</h4>
            </div>
            <p className="text-xs text-slate-300 leading-relaxed">{activeStage.description}</p>
            <div className="pt-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 font-mono block">
                Project Component
              </span>
              <code className="text-[11px] font-mono text-sky-300 break-all">
                {activeStage.component}
              </code>
            </div>
          </div>

          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-mono flex items-center space-x-1 mb-1.5">
                <ArrowRight className="w-3 h-3 text-sky-400" />
                <span>Stage Input</span>
              </span>
              <p className="text-xs text-slate-200 font-mono leading-relaxed break-words">
                {activeStage.inputText}
              </p>
            </div>
            <div className="bg-slate-900/70 border border-slate-800 rounded-lg p-3">
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-mono flex items-center space-x-1 mb-1.5">
                <Sparkles className="w-3 h-3 text-emerald-400" />
                <span>Stage Output</span>
              </span>
              <p className="text-xs text-emerald-200/90 font-mono leading-relaxed break-words">
                {activeStage.outputText}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Visual Embedding Education Panel */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4 sm:p-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
          <div>
            <h3 className="text-sm font-semibold text-slate-100 flex items-center space-x-2">
              <Binary className="w-4 h-4 text-sky-400" />
              <span>How Embeddings &amp; Vector Similarity Work</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Beginner walkthrough showing how text becomes searchable geometry in ChromaDB.
            </p>
          </div>
          <span className="text-[10px] font-mono px-2.5 py-1 rounded-full bg-amber-500/10 text-amber-300 border border-amber-500/25 flex items-center space-x-1">
            <Info className="w-3 h-3" />
            <span>Conceptual Diagram Only — Not Raw Model Coordinates</span>
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-2">
            {embeddingSteps.map((step, idx) => {
              const active = idx === activeEmbedStep;
              return (
                <button
                  key={step.title}
                  onClick={() => setActiveEmbedStep(idx)}
                  className={`text-left p-3 rounded-lg border transition ${
                    active
                      ? 'bg-sky-500/15 border-sky-500/50 text-slate-100'
                      : 'bg-slate-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  <div className="text-xs font-semibold text-sky-300 mb-1">{step.title}</div>
                  <p className="text-[11px] leading-relaxed">{step.detail}</p>
                </button>
              );
            })}
          </div>

          {/* Conceptual Vector Similarity Geometry Diagram */}
          <div className="lg:col-span-5 bg-slate-950/80 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-2">
                <span>Step {activeEmbedStep + 1}: Concept Flow</span>
                <span className="text-sky-400">{docData?.embedding_model || 'nomic-embed-text'}</span>
              </div>
              <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 font-mono text-[11px] text-sky-200 mb-3">
                {embeddingSteps[activeEmbedStep].visual}
              </div>

              {/* Clean SVG conceptual direction diagram (explicitly non-numeric / illustrative) */}
              <svg
                viewBox="0 0 280 145"
                className="w-full h-36 bg-slate-900/50 rounded-lg border border-slate-800/80"
                role="img"
                aria-label="Conceptual illustration of query and chunk vectors with cosine angle"
              >
                <line x1="30" y1="120" x2="255" y2="120" stroke="#334155" strokeWidth="1" />
                <line x1="30" y1="120" x2="30" y2="15" stroke="#334155" strokeWidth="1" />

                {/* Relevant Chunk Vector (small angle) */}
                <line
                  x1="30"
                  y1="120"
                  x2="215"
                  y2="45"
                  stroke="#38bdf8"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                />
                <circle cx="215" cy="45" r="4" fill="#38bdf8" />
                <text x="155" y="34" fill="#7dd3fc" fontSize="9" fontFamily="monospace">
                  Chunk A (Close Match)
                </text>

                {/* Query Vector */}
                <line
                  x1="30"
                  y1="120"
                  x2="230"
                  y2="68"
                  stroke="#34d399"
                  strokeWidth="2.5"
                  strokeDasharray="4 2"
                  strokeLinecap="round"
                />
                <circle cx="230" cy="68" r="4" fill="#34d399" />
                <text x="172" y="84" fill="#6ee7b7" fontSize="9" fontFamily="monospace">
                  Query Vector
                </text>

                {/* Unrelated Chunk Vector (wide angle) */}
                <line
                  x1="30"
                  y1="120"
                  x2="80"
                  y2="24"
                  stroke="#f43f5e"
                  strokeWidth="2"
                  strokeLinecap="round"
                  opacity="0.85"
                />
                <circle cx="80" cy="24" r="3.5" fill="#f43f5e" />
                <text x="86" y="24" fill="#fda4af" fontSize="9" fontFamily="monospace">
                  Unrelated (Large Dist)
                </text>

                <text x="95" y="136" fill="#94a3b8" fontSize="8.5" fontFamily="monospace">
                  Smaller angle =&gt; Lower cosine distance d = 1 - cos(theta)
                </text>
              </svg>
            </div>

            <p className="text-[11px] text-slate-400 mt-2.5 leading-relaxed">
              Vectors are stored with strict <code className="text-sky-300">document_name</code> &amp;{' '}
              <code className="text-sky-300">document_id</code> filters so unrelated uploads cannot leak into Top-K results.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

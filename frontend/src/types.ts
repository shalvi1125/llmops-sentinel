export interface HealthResponse {
  status: string;
  chromadb: string;
  ollama: string;
}

export interface IngestResponse {
  status: string;
  chunks_stored: number;
  filename: string;
}

export interface IndexedDocItem {
  filename: string;
  chunks: number;
}

export interface SourceChunk {
  chunk_id: string;
  text: string;
  score: number | null;
  document_name: string | null;
  page_number: number | null;
  token_count_estimate: number | null;
}

export interface RetrievalDiagnostics {
  retrieval_latency_ms: number;
  top_score: number | null;
  low_confidence: boolean;
  score_metric: string;
}

export interface QueryRequest {
  question: string;
  top_k: number;
  prompt_id?: string;
  model?: string;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
  source_chunks?: SourceChunk[];
  retrieval_diagnostics?: RetrievalDiagnostics;
  prompt_id?: string;
  prompt_version?: string;
  model?: string;
  generation_latency_ms?: number;
}

export interface PromptStrategy {
  id: string;
  name: string;
  version: string;
  description: string;
  template: string;
}

export interface ModelComparisonItem {
  model: string;
  answer: string;
  generation_latency_ms: number;
  token_count_estimate: number;
  tokens_per_second: number | null;
}

export interface CompareResponse {
  question: string;
  sources: string[];
  source_chunks: SourceChunk[];
  retrieval_diagnostics: RetrievalDiagnostics;
  prompt_id: string;
  prompt_version: string;
  models: ModelComparisonItem[];
}

export interface AssessmentDetail {
  value: string;
  rationale: string | null;
}

export interface TraceAuditItem {
  request_id: string;
  timestamp_ms: number;
  execution_time_ms: number;
  status: string;
  question: string | null;
  answer: string | null;
  assessments: Record<string, AssessmentDetail>;
  prompt_id?: string | null;
  prompt_version?: string | null;
  model?: string | null;
}

export interface BenchmarkCaseResult {
  case_id: string;
  question: string;
  reference_answer: string;
  generated_answer: string | null;
  latency_ms: number | null;
  relevancy_verdict: string | null;
  relevancy_rationale: string | null;
  hallucination_verdict: string | null;
  hallucination_rationale: string | null;
  safety_verdict: string | null;
  status: string;
}

export interface BenchmarkRunResponse {
  benchmark_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  total_cases: number;
  completed_cases: number;
  mean_latency_ms: number | null;
  relevancy_pass_rate: number | null;
  hallucination_pass_rate: number | null;
  safety_pass_rate: number | null;
  cases: BenchmarkCaseResult[];
  error?: string | null;
}

export interface BenchmarkSummary {
  benchmark_id: string;
  status: string;
  started_at: string;
  completed_at: string | null;
  total_cases: number;
  completed_cases: number;
  mean_latency_ms: number | null;
  relevancy_pass_rate: number | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: string[];
  source_chunks?: SourceChunk[];
  retrieval_diagnostics?: RetrievalDiagnostics;
  prompt_id?: string;
  prompt_version?: string;
  model?: string;
  generation_latency_ms?: number;
  timestamp: Date;
  isError?: boolean;
}

export interface UploadedDoc {
  filename: string;
  chunksStored: number;
  uploadedAt: Date;
}

export type ActiveTab = 'overview' | 'chat' | 'arena' | 'eval' | 'benchmarks';

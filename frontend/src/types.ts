export interface HealthResponse {
  status: string;
  chromadb: string;
  ollama: string;
}

export interface IngestResponse {
  status: string;
  chunks_stored: number;
  filename: string;
  document_id?: string | null;
  ingestion_timestamp?: string | null;
}

export interface IndexedDocItem {
  filename: string;
  chunks: number;
  document_id?: string | null;
  ingestion_timestamp?: string | null;
}

export interface SourceChunk {
  chunk_id: string;
  text: string;
  score: number | null;
  document_id?: string | null;
  document_name: string | null;
  page_number: number | null;
  source?: string | null;
  ingestion_timestamp?: string | null;
  token_count_estimate: number | null;
}

export interface RetrievalDiagnostics {
  retrieval_latency_ms: number;
  top_score: number | null;
  low_confidence: boolean;
  score_metric: string;
  requested_k?: number | null;
  candidates_fetched?: number | null;
  unique_chunks_returned?: number | null;
  duplicates_removed?: number;
  retrieval_strategy?: string;
  insufficient_unique_reason?: string | null;
}

export interface QueryRequest {
  question: string;
  top_k: number;
  prompt_id?: string;
  model?: string;
  document_id?: string | null;
  document_name?: string | null;
}

export interface QueryResponse {
  answer: string;
  sources: string[];
  source_chunks?: SourceChunk[];
  retrieval_diagnostics?: RetrievalDiagnostics;
  document_scope?: string | null;
  prompt_id?: string;
  prompt_version?: string;
  model?: string;
  generation_latency_ms?: number;
  total_latency_ms?: number | null;
  context_char_count?: number | null;
  token_count_estimate?: number | null;
  tokens_per_second?: number | null;
  stage_timings?: Record<string, number> | null;
  trace_id?: string | null;
  evaluation_job_id?: string | null;
  evaluation_status?: string | null;
  generation_settings?: Record<string, number | string> | null;
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
  document_scope?: string | null;
  prompt_id: string;
  prompt_version: string;
  retrieval_latency_ms?: number | null;
  context_char_count?: number | null;
  generation_settings?: Record<string, number | string> | null;
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
  document_scope?: string | null;
  retrieval_latency_ms?: number | null;
  generation_latency_ms?: number | null;
  candidates_fetched?: number | null;
  unique_chunks_returned?: number | null;
  duplicates_removed?: number | null;
  use_mmr?: boolean | null;
  context_char_count?: number | null;
  token_count_estimate?: number | null;
  tokens_per_second?: number | null;
  evaluation_job_id?: string | null;
  evaluation_status?: string | null;
  evaluation_duration_ms?: number | null;
}

export interface ModelPerformanceItem {
  model: string;
  query_count: number;
  avg_generation_latency_ms: number | null;
  avg_total_latency_ms: number | null;
  avg_tokens_per_second: number | null;
}

export interface LatencyTrendItem {
  request_id: string;
  timestamp_ms: number;
  total_latency_ms: number;
  retrieval_latency_ms: number | null;
  generation_latency_ms: number | null;
  evaluation_duration_ms: number | null;
  tokens_per_second: number | null;
  model: string | null;
  evaluation_status: string | null;
  question_preview: string | null;
}

export interface PerformanceSummaryResponse {
  total_traces: number;
  measured_query_traces: number;
  avg_query_latency_ms: number | null;
  min_query_latency_ms?: number | null;
  median_query_latency_ms: number | null;
  p90_query_latency_ms?: number | null;
  p95_query_latency_ms: number | null;
  max_query_latency_ms?: number | null;
  percentile_method?: string;
  sample_sufficient_for_p95?: boolean;
  measured_latencies_ms?: number[];
  avg_retrieval_latency_ms: number | null;
  avg_generation_latency_ms: number | null;
  avg_evaluation_duration_ms: number | null;
  avg_tokens_per_second: number | null;
  eval_counts: Record<string, number>;
  eval_completion_rate: number | null;
  model_breakdown: ModelPerformanceItem[];
  latency_trend: LatencyTrendItem[];
}

export interface ObservabilitySummaryResponse {
  metrics_endpoint: string;
  mlflow_tracking_uri: string;
  prometheus_url: string;
  grafana_url: string;
  dependencies: Record<string, boolean>;
  process_memory_rss_mb: number;
  process_cpu_percent: number;
  process_uptime_seconds: number;
  embedding_cache_entries: number;
  eval_jobs_active: number;
  chromadb_documents: number;
  chromadb_chunks: number;
  ollama_models_available: number;
  registered_metric_families: string[];
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
  total_latency_ms?: number | null;
  tokens_per_second?: number | null;
  evaluation_status?: string | null;
  timestamp: Date;
  isError?: boolean;
}

export interface UploadedDoc {
  filename: string;
  chunksStored: number;
  uploadedAt: Date;
}

export interface ChunkDetail {
  chunk_id: string;
  chunk_index: number;
  text: string;
  document_id?: string | null;
  document_name: string;
  page_number: number | null;
  source?: string | null;
  char_count: number;
  token_count_estimate: number;
  start_char: number;
  end_char: number;
  overlap_prev_chars: number;
  overlap_next_chars: number;
  overlap_prev_text: string;
  overlap_next_text: string;
  ingestion_timestamp?: string | null;
}

export interface DocumentChunksResponse {
  document_name: string;
  document_id?: string | null;
  file_type: string;
  total_pages: number | null;
  total_chunks: number;
  total_extracted_chars: number;
  total_estimated_tokens: number;
  chunk_size: number;
  chunk_overlap: number;
  splitter_name: string;
  embedding_model: string;
  ingestion_timestamp?: string | null;
  reconstructed_text: string;
  chunks: ChunkDetail[];
}

export interface ChunkPreviewResponse {
  is_preview: boolean;
  persisted: boolean;
  document_name?: string | null;
  configured_chunk_size: number;
  configured_chunk_overlap: number;
  preview_chunk_size: number;
  preview_chunk_overlap: number;
  total_chars: number;
  total_chunks: number;
  avg_chunk_chars: number;
  total_estimated_tokens: number;
  source_text: string;
  chunks: ChunkDetail[];
  notice: string;
}

export interface RetrievalInspectResponse {
  question: string;
  document_scope: string;
  embedding_model: string;
  score_metric: string;
  score_explanation: string;
  retrieval_diagnostics: RetrievalDiagnostics;
  retrieved_chunks: SourceChunk[];
  context_preview: string;
  context_char_count: number;
  insufficient_evidence: boolean;
  empty_reason?: string | null;
}

export type ActiveTab =
  | 'overview'
  | 'chat'
  | 'arena'
  | 'explorer'
  | 'eval'
  | 'benchmarks'
  | 'observability';

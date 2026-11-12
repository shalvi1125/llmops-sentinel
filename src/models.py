from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    filename: str
    document_id: str | None = None
    ingestion_timestamp: str | None = None


class IndexedDocument(BaseModel):
    filename: str
    chunks: int
    document_id: str | None = None
    ingestion_timestamp: str | None = None


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None
    document_id: str | None = None
    document_name: str | None = None
    page_number: int | None = None
    source: str | None = None
    ingestion_timestamp: str | None = None
    token_count_estimate: int | None = None


class RetrievalDiagnostics(BaseModel):
    retrieval_latency_ms: float
    top_score: float | None = None
    low_confidence: bool = False
    score_metric: str = "cosine_distance"
    requested_k: int | None = None
    candidates_fetched: int | None = None
    unique_chunks_returned: int | None = None
    duplicates_removed: int = 0
    retrieval_strategy: str = "mmr_deduplicated"
    insufficient_unique_reason: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=20)
    prompt_id: str = Field(default="default_qa")
    model: str | None = None
    document_id: str | None = None
    document_name: str | None = None
    use_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.72, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    source_chunks: list[SourceChunk] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics | None = None
    document_scope: str | None = None
    prompt_id: str = "default_qa"
    prompt_version: str = "1.0.0"
    model: str = "llama3.2"
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    context_char_count: int | None = None
    token_count_estimate: int | None = None
    tokens_per_second: float | None = None
    stage_timings: dict[str, float] | None = None
    trace_id: str | None = None
    evaluation_job_id: str | None = None
    evaluation_status: str | None = None
    generation_settings: dict[str, int | float | str] | None = None


class HealthResponse(BaseModel):
    status: str
    chromadb: str
    ollama: str


class PromptStrategyResponse(BaseModel):
    id: str
    name: str
    version: str
    description: str
    template: str


class CompareRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=20)
    prompt_id: str = Field(default="default_qa")
    document_id: str | None = None
    document_name: str | None = None
    use_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.72, ge=0.0, le=1.0)


class ModelComparisonItem(BaseModel):
    model: str
    answer: str
    generation_latency_ms: float
    token_count_estimate: int
    tokens_per_second: float | None = None


class CompareResponse(BaseModel):
    question: str
    sources: list[str]
    source_chunks: list[SourceChunk] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics | None = None
    document_scope: str | None = None
    prompt_id: str
    prompt_version: str
    retrieval_latency_ms: float | None = None
    context_char_count: int | None = None
    generation_settings: dict[str, int | float | str] | None = None
    models: list[ModelComparisonItem]


class AssessmentDetail(BaseModel):
    value: str
    rationale: str | None = None


class TraceAuditItem(BaseModel):
    request_id: str
    timestamp_ms: int
    execution_time_ms: float
    status: str
    question: str | None = None
    answer: str | None = None
    assessments: dict[str, AssessmentDetail] = Field(default_factory=dict)
    prompt_id: str | None = None
    prompt_version: str | None = None
    model: str | None = None
    document_scope: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    candidates_fetched: int | None = None
    unique_chunks_returned: int | None = None
    duplicates_removed: int | None = None
    use_mmr: bool | None = None
    context_char_count: int | None = None
    token_count_estimate: int | None = None
    tokens_per_second: float | None = None
    evaluation_job_id: str | None = None
    evaluation_status: str | None = None
    evaluation_duration_ms: float | None = None


class ModelPerformanceItem(BaseModel):
    model: str
    query_count: int
    avg_generation_latency_ms: float | None = None
    avg_total_latency_ms: float | None = None
    avg_tokens_per_second: float | None = None


class LatencyTrendItem(BaseModel):
    request_id: str
    timestamp_ms: int
    total_latency_ms: float
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    evaluation_duration_ms: float | None = None
    tokens_per_second: float | None = None
    model: str | None = None
    evaluation_status: str | None = None
    question_preview: str | None = None


class PerformanceSummaryResponse(BaseModel):
    total_traces: int
    measured_query_traces: int
    avg_query_latency_ms: float | None = None
    min_query_latency_ms: float | None = None
    median_query_latency_ms: float | None = None
    p90_query_latency_ms: float | None = None
    p95_query_latency_ms: float | None = None
    max_query_latency_ms: float | None = None
    percentile_method: str = "linear_interpolation"
    sample_sufficient_for_p95: bool = False
    measured_latencies_ms: list[float] = Field(default_factory=list)
    avg_retrieval_latency_ms: float | None = None
    avg_generation_latency_ms: float | None = None
    avg_evaluation_duration_ms: float | None = None
    avg_tokens_per_second: float | None = None
    eval_counts: dict[str, int] = Field(default_factory=dict)
    eval_completion_rate: float | None = None
    model_breakdown: list[ModelPerformanceItem] = Field(default_factory=list)
    latency_trend: list[LatencyTrendItem] = Field(default_factory=list)


class ObservabilitySummaryResponse(BaseModel):
    metrics_endpoint: str = "/metrics"
    mlflow_tracking_uri: str
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3000"
    dependencies: dict[str, bool] = Field(default_factory=dict)
    process_memory_rss_mb: float = 0.0
    process_cpu_percent: float = 0.0
    process_uptime_seconds: float = 0.0
    embedding_cache_entries: int = 0
    eval_jobs_active: int = 0
    chromadb_documents: int = 0
    chromadb_chunks: int = 0
    ollama_models_available: int = 0
    registered_metric_families: list[str] = Field(default_factory=list)


class BenchmarkCaseResult(BaseModel):
    case_id: str
    question: str
    reference_answer: str
    generated_answer: str | None = None
    latency_ms: float | None = None
    relevancy_verdict: str | None = None
    relevancy_rationale: str | None = None
    hallucination_verdict: str | None = None
    hallucination_rationale: str | None = None
    safety_verdict: str | None = None
    status: str = "pending"  # "pending", "running", "completed", "failed", "skipped"


class BenchmarkRunResponse(BaseModel):
    benchmark_id: str
    status: str  # "running", "completed", "failed", "canceled"
    started_at: str
    completed_at: str | None = None
    total_cases: int
    completed_cases: int
    mean_latency_ms: float | None = None
    relevancy_pass_rate: float | None = None
    hallucination_pass_rate: float | None = None
    safety_pass_rate: float | None = None
    cases: list[BenchmarkCaseResult] = Field(default_factory=list)
    error: str | None = None


class BenchmarkSummary(BaseModel):
    benchmark_id: str
    status: str
    started_at: str
    completed_at: str | None = None
    total_cases: int
    completed_cases: int
    mean_latency_ms: float | None = None
    relevancy_pass_rate: float | None = None


class ChunkDetail(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    document_id: str | None = None
    document_name: str
    page_number: int | None = None
    source: str | None = None
    char_count: int
    token_count_estimate: int
    start_char: int
    end_char: int
    overlap_prev_chars: int = 0
    overlap_next_chars: int = 0
    overlap_prev_text: str = ""
    overlap_next_text: str = ""
    ingestion_timestamp: str | None = None


class DocumentChunksResponse(BaseModel):
    document_name: str
    document_id: str | None = None
    file_type: str
    total_pages: int | None = None
    total_chunks: int
    total_extracted_chars: int
    total_estimated_tokens: int
    chunk_size: int
    chunk_overlap: int
    splitter_name: str = "RecursiveCharacterTextSplitter"
    embedding_model: str
    ingestion_timestamp: str | None = None
    reconstructed_text: str
    chunks: list[ChunkDetail] = Field(default_factory=list)


class ChunkPreviewRequest(BaseModel):
    document_name: str | None = None
    text: str | None = Field(default=None, max_length=250000)
    chunk_size: int = Field(default=1200, ge=100, le=8000)
    chunk_overlap: int = Field(default=100, ge=0, le=2000)


class ChunkPreviewResponse(BaseModel):
    is_preview: bool = True
    persisted: bool = False
    document_name: str | None = None
    configured_chunk_size: int
    configured_chunk_overlap: int
    preview_chunk_size: int
    preview_chunk_overlap: int
    total_chars: int
    total_chunks: int
    avg_chunk_chars: float
    total_estimated_tokens: int
    source_text: str
    chunks: list[ChunkDetail] = Field(default_factory=list)
    notice: str = "Preview computed in memory only. Persistent ChromaDB index was not modified."


class RetrievalInspectRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=20)
    document_id: str | None = None
    document_name: str | None = None
    use_mmr: bool = Field(default=True)
    mmr_lambda: float = Field(default=0.72, ge=0.0, le=1.0)


class RetrievalInspectResponse(BaseModel):
    question: str
    document_scope: str
    embedding_model: str
    score_metric: str = "cosine_distance"
    score_explanation: str
    retrieval_diagnostics: RetrievalDiagnostics
    retrieved_chunks: list[SourceChunk] = Field(default_factory=list)
    context_preview: str
    context_char_count: int
    insufficient_evidence: bool = False
    empty_reason: str | None = None

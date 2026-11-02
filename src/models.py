from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    status: str
    chunks_stored: int
    filename: str


class IndexedDocument(BaseModel):
    filename: str
    chunks: int


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float | None = None
    document_name: str | None = None
    page_number: int | None = None
    token_count_estimate: int | None = None


class RetrievalDiagnostics(BaseModel):
    retrieval_latency_ms: float
    top_score: float | None = None
    low_confidence: bool = False
    score_metric: str = "cosine_distance"


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=4, ge=1, le=20)
    prompt_id: str = Field(default="default_qa")
    model: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    source_chunks: list[SourceChunk] = Field(default_factory=list)
    retrieval_diagnostics: RetrievalDiagnostics | None = None
    prompt_id: str = "default_qa"
    prompt_version: str = "1.0.0"
    model: str = "llama3.2"
    generation_latency_ms: float | None = None


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
    prompt_id: str
    prompt_version: str
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

import asyncio
import os
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Response, UploadFile

from src.metrics import get_runtime_snapshot, list_registered_metric_names, render_prometheus_metrics
from src.models import (
    BenchmarkRunResponse,
    BenchmarkSummary,
    ChunkPreviewRequest,
    ChunkPreviewResponse,
    CompareRequest,
    CompareResponse,
    DocumentChunksResponse,
    HealthResponse,
    IndexedDocument,
    IngestResponse,
    ObservabilitySummaryResponse,
    PerformanceSummaryResponse,
    PromptStrategyResponse,
    QueryRequest,
    QueryResponse,
    RetrievalInspectRequest,
    RetrievalInspectResponse,
    TraceAuditItem,
)
from src.prompts.catalog import list_prompt_strategies
from src.security import verify_api_key
from src.services.audit import get_performance_summary, get_recent_traces
from src.services.benchmark import (
    get_benchmark_status,
    list_benchmarks,
    start_benchmark,
)
from src.services.compare import handle_compare
from src.services.explorer import (
    get_document_chunks,
    inspect_retrieval,
    preview_chunking,
)
from src.services.health import check_health, probe_mlflow_status
from src.utils.env import MLFLOW_TRACKING_URI
from src.services.ingest import (
    clear_all_documents,
    delete_document,
    get_indexed_documents,
    ingest_document,
)
from src.services.query import handle_query

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a PDF or text document into the vector store",
)
async def ingest(
    file: UploadFile = File(...),
    clear_existing: bool = False,
    _: None = Depends(verify_api_key),
):
    return await ingest_document(file, clear_existing=clear_existing)


@router.get(
    "/ingest/documents",
    response_model=list[IndexedDocument],
    summary="List all unique documents currently stored in the vector store",
)
async def list_documents(_: None = Depends(verify_api_key)):
    return get_indexed_documents()


@router.delete(
    "/ingest/documents",
    summary="Clear all documents from the vector store",
)
async def clear_documents(_: None = Depends(verify_api_key)):
    return clear_all_documents()


@router.delete(
    "/ingest/documents/{filename}",
    summary="Delete a specific document and its chunks from the vector store",
)
async def delete_single_document(filename: str, _: None = Depends(verify_api_key)):
    return delete_document(filename)


@router.get(
    "/ingest/documents/{filename}/chunks",
    response_model=DocumentChunksResponse,
    summary="Inspect persisted chunks, overlaps, and chunking metadata for a document",
)
async def get_doc_chunks(filename: str, _: None = Depends(verify_api_key)):
    return get_document_chunks(filename)


@router.post(
    "/ingest/chunks/preview",
    response_model=ChunkPreviewResponse,
    summary="Simulate non-persistent chunk size and overlap settings without modifying ChromaDB",
)
async def preview_doc_chunks(
    request: ChunkPreviewRequest, _: None = Depends(verify_api_key)
):
    return preview_chunking(request)


@router.post(
    "/query/inspect",
    response_model=RetrievalInspectResponse,
    summary="Inspect deduplicated Top-K / MMR retrieval rankings and cosine distances",
)
async def inspect_query_retrieval(
    request: RetrievalInspectRequest, _: None = Depends(verify_api_key)
):
    return await asyncio.to_thread(inspect_retrieval, request)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG pipeline with a question",
)
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    return await handle_query(request, background_tasks=background_tasks)


@router.post(
    "/query/compare",
    response_model=CompareResponse,
    summary="Compare llama3.2 and mistral sequentially on the same retrieved context",
)
async def query_compare(
    request: CompareRequest,
    _: None = Depends(verify_api_key),
):
    return await handle_compare(request)


@router.get(
    "/prompts",
    response_model=list[PromptStrategyResponse],
    summary="List available prompt strategies and versions",
)
async def get_prompts(_: None = Depends(verify_api_key)):
    strategies = list_prompt_strategies()
    return [
        PromptStrategyResponse(
            id=s.id,
            name=s.name,
            version=s.version,
            description=s.description,
            template=s.template,
        )
        for s in strategies
    ]


@router.get(
    "/eval/traces",
    response_model=list[TraceAuditItem],
    summary="Retrieve recent MLflow traces with assessment scores and rationales",
)
async def get_eval_traces(limit: int = 25, _: None = Depends(verify_api_key)):
    return get_recent_traces(limit=limit)


@router.get(
    "/eval/performance",
    response_model=PerformanceSummaryResponse,
    summary="Aggregate real query, retrieval, generation, throughput, and judge evaluation telemetry",
)
async def get_eval_performance(limit: int = 100, _: None = Depends(verify_api_key)):
    return get_performance_summary(limit=limit)


@router.post(
    "/eval/benchmark",
    response_model=BenchmarkRunResponse,
    summary="Trigger a new RAG Evaluation Laboratory benchmark run",
)
async def trigger_benchmark(_: None = Depends(verify_api_key)):
    return start_benchmark()


@router.get(
    "/eval/benchmark/{benchmark_id}",
    response_model=BenchmarkRunResponse,
    summary="Get status and case results for a specific benchmark run",
)
async def get_benchmark(benchmark_id: str, _: None = Depends(verify_api_key)):
    run_obj = get_benchmark_status(benchmark_id)
    if not run_obj:
        raise HTTPException(status_code=404, detail="Benchmark run not found.")
    return run_obj


@router.get(
    "/eval/benchmarks",
    response_model=list[BenchmarkSummary],
    summary="List historical RAG benchmark runs",
)
async def get_all_benchmarks(_: None = Depends(verify_api_key)):
    return list_benchmarks()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check for API dependencies",
)
async def health():
    return await check_health()


@router.get(
    "/metrics",
    summary="Prometheus metrics exposition endpoint",
    include_in_schema=False,
)
async def metrics():
    payload, content_type = render_prometheus_metrics()
    return Response(content=payload, media_type=content_type)


@router.get(
    "/observability/summary",
    response_model=ObservabilitySummaryResponse,
    summary="Live operational telemetry and MLflow/Prometheus/Grafana integration snapshot",
)
async def observability_summary(_: None = Depends(verify_api_key)):
    health_state = await check_health()
    mlflow_up = await asyncio.to_thread(probe_mlflow_status)
    snapshot = get_runtime_snapshot()
    return ObservabilitySummaryResponse(
        metrics_endpoint="/metrics",
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
        prometheus_url=os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
        grafana_url=os.getenv("GRAFANA_URL", "http://localhost:3000"),
        dependencies={
            "fastapi": True,
            "chromadb": health_state.chromadb == "ok",
            "ollama": health_state.ollama == "ok",
            "mlflow": mlflow_up,
        },
        process_memory_rss_mb=float(snapshot["process_memory_rss_mb"]),
        process_cpu_percent=float(snapshot["process_cpu_percent"]),
        process_uptime_seconds=float(snapshot["process_uptime_seconds"]),
        embedding_cache_entries=int(snapshot["embedding_cache_entries"]),
        eval_jobs_active=int(snapshot["eval_jobs_active"]),
        chromadb_documents=int(snapshot["chromadb_documents"]),
        chromadb_chunks=int(snapshot["chromadb_chunks"]),
        ollama_models_available=int(snapshot["ollama_models_available"]),
        registered_metric_families=list(list_registered_metric_names()),
    )

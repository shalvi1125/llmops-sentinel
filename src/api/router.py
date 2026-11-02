from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from src.models import (
    BenchmarkRunResponse,
    BenchmarkSummary,
    CompareRequest,
    CompareResponse,
    HealthResponse,
    IndexedDocument,
    IngestResponse,
    PromptStrategyResponse,
    QueryRequest,
    QueryResponse,
    TraceAuditItem,
)
from src.prompts.catalog import list_prompt_strategies
from src.security import verify_api_key
from src.services.audit import get_recent_traces
from src.services.benchmark import (
    get_benchmark_status,
    list_benchmarks,
    start_benchmark,
)
from src.services.compare import handle_compare
from src.services.health import check_health
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

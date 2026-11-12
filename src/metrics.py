"""Centralized Prometheus operational metrics registry and helpers for LLMOps Sentinel.

Strict label cardinality rules apply:
- Allowed labels: HTTP method, normalized route template, status_code/class,
  fixed model family/tag, file_type, retrieval strategy, eval state/outcome/metric,
  cache_hit, token_source, dependency service name.
- Forbidden labels (never pass): user question text, document filenames/IDs,
  trace IDs, API keys, timestamps, or free-form exception messages.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Iterable

import psutil
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

PROCESS_START_TIME = time.time()
_PROCESS = psutil.Process(os.getpid())
_RUNTIME_LOCK = threading.Lock()

REGISTRY = CollectorRegistry(auto_describe=True)

# ---------------------------------------------------------------------------
# 1. HTTP & API Operational Metrics
# ---------------------------------------------------------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "llmops_sentinel_http_requests_total",
    "Total HTTP requests handled by the FastAPI backend",
    labelnames=("method", "route", "status_code"),
    registry=REGISTRY,
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "llmops_sentinel_http_request_duration_seconds",
    "HTTP request execution latency in seconds",
    labelnames=("method", "route", "status_code"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 45.0, 90.0),
    registry=REGISTRY,
)

HTTP_REQUESTS_IN_FLIGHT = Gauge(
    "llmops_sentinel_http_requests_in_flight",
    "Number of HTTP requests currently executing",
    labelnames=("route",),
    registry=REGISTRY,
)

HTTP_ERRORS_TOTAL = Counter(
    "llmops_sentinel_http_errors_total",
    "Total HTTP error responses grouped by status class (4xx or 5xx)",
    labelnames=("method", "route", "status_class"),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 2. RAG Ingestion & Retrieval Metrics
# ---------------------------------------------------------------------------
INGESTIONS_TOTAL = Counter(
    "llmops_sentinel_ingestions_total",
    "Total document ingestion attempts by file type and status",
    labelnames=("file_type", "status"),
    registry=REGISTRY,
)

INGESTION_CHUNKS_CREATED_TOTAL = Counter(
    "llmops_sentinel_ingestion_chunks_created_total",
    "Total document chunks persisted to ChromaDB during ingestion",
    labelnames=("file_type",),
    registry=REGISTRY,
)

INGESTION_DURATION_SECONDS = Histogram(
    "llmops_sentinel_ingestion_duration_seconds",
    "End-to-end document ingestion and embedding latency in seconds",
    labelnames=("file_type", "status"),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
    registry=REGISTRY,
)

RETRIEVAL_DURATION_SECONDS = Histogram(
    "llmops_sentinel_retrieval_duration_seconds",
    "Total RAG vector retrieval and reranking duration in seconds",
    labelnames=("strategy",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

QUERY_EMBEDDING_DURATION_SECONDS = Histogram(
    "llmops_sentinel_query_embedding_duration_seconds",
    "Query embedding compute or cache lookup duration in seconds",
    labelnames=("cache_hit",),
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)

CHROMA_SEARCH_DURATION_SECONDS = Histogram(
    "llmops_sentinel_chroma_search_duration_seconds",
    "ChromaDB similarity search latency in seconds",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

MMR_RERANK_DURATION_SECONDS = Histogram(
    "llmops_sentinel_mmr_rerank_duration_seconds",
    "Candidate deduplication and MMR reranking latency in seconds",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
    registry=REGISTRY,
)

RETRIEVAL_CANDIDATES_FETCHED = Histogram(
    "llmops_sentinel_retrieval_candidates_fetched",
    "Number of raw candidate chunks oversampled from ChromaDB per retrieval",
    buckets=(1, 2, 4, 8, 12, 16, 24, 32, 60),
    registry=REGISTRY,
)

RETRIEVAL_UNIQUE_CHUNKS = Histogram(
    "llmops_sentinel_retrieval_unique_chunks",
    "Number of unique chunks retained after deduplication and MMR",
    buckets=(0, 1, 2, 3, 4, 6, 8, 12, 20),
    registry=REGISTRY,
)

RETRIEVAL_DUPLICATES_REMOVED_TOTAL = Counter(
    "llmops_sentinel_retrieval_duplicates_removed_total",
    "Cumulative near-duplicate chunks filtered out during Top-K retrieval",
    registry=REGISTRY,
)

CONTEXT_CHAR_COUNT = Histogram(
    "llmops_sentinel_context_char_count",
    "Character length of constructed LLM prompt context",
    buckets=(100, 300, 600, 1000, 1500, 2000, 2500, 3000, 3500, 5000),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 3. LLM Generation & Comparison Metrics
# ---------------------------------------------------------------------------
LLM_REQUESTS_TOTAL = Counter(
    "llmops_sentinel_llm_requests_total",
    "Total LLM generation calls by model, endpoint, and status",
    labelnames=("model", "endpoint", "status"),
    registry=REGISTRY,
)

LLM_GENERATION_DURATION_SECONDS = Histogram(
    "llmops_sentinel_llm_generation_duration_seconds",
    "Local Ollama LLM generation wall-clock latency in seconds",
    labelnames=("model", "endpoint"),
    buckets=(0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 20.0, 35.0, 60.0, 120.0),
    registry=REGISTRY,
)

LLM_TOKENS_GENERATED_TOTAL = Counter(
    "llmops_sentinel_llm_tokens_generated_total",
    "Total generated completion tokens broken down by model and token_source (actual vs estimated)",
    labelnames=("model", "token_source"),
    registry=REGISTRY,
)

LLM_TOKENS_PER_SECOND = Histogram(
    "llmops_sentinel_llm_tokens_per_second",
    "Measured token generation throughput in tokens/sec per model",
    labelnames=("model",),
    buckets=(1.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 35.0, 50.0, 80.0),
    registry=REGISTRY,
)

LLM_LAST_TOKENS_PER_SECOND = Gauge(
    "llmops_sentinel_llm_last_tokens_per_second",
    "Most recently observed token throughput in tokens/sec per model",
    labelnames=("model",),
    registry=REGISTRY,
)

LLM_ACTIVE_REQUESTS = Gauge(
    "llmops_sentinel_llm_active_requests",
    "Currently running interactive LLM inference calls by model",
    labelnames=("model",),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 4. Evaluation, Preemption & Benchmark Lifecycle Metrics
# ---------------------------------------------------------------------------
EVAL_JOBS_TOTAL = Counter(
    "llmops_sentinel_eval_jobs_total",
    "Evaluation job lifecycle transitions across queued/running/completed/cancelled/skipped/failed",
    labelnames=("state",),
    registry=REGISTRY,
)

EVAL_JOBS_ACTIVE = Gauge(
    "llmops_sentinel_eval_jobs_active",
    "1 when background LLM-as-a-Judge evaluation is running, else 0",
    registry=REGISTRY,
)

EVAL_PREEMPTIONS_TOTAL = Counter(
    "llmops_sentinel_eval_preemptions_total",
    "Count of background judge evaluation preemptions triggered by interactive traffic",
    labelnames=("trigger",),
    registry=REGISTRY,
)

EVAL_DURATION_SECONDS = Histogram(
    "llmops_sentinel_eval_duration_seconds",
    "Duration of background LLM-as-a-Judge runs in seconds by terminal outcome",
    labelnames=("outcome",),
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0, 150.0, 300.0),
    registry=REGISTRY,
)

EVAL_SCORER_DURATION_SECONDS = Histogram(
    "llmops_sentinel_eval_scorer_duration_seconds",
    "Individual judge scorer duration in seconds by metric and outcome",
    labelnames=("metric", "outcome"),
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 60.0, 90.0),
    registry=REGISTRY,
)

BENCHMARK_RUNS_TOTAL = Counter(
    "llmops_sentinel_benchmark_runs_total",
    "Total Benchmark Lab suite executions by outcome status",
    labelnames=("status",),
    registry=REGISTRY,
)

BENCHMARK_PASS_RATE_PERCENT = Gauge(
    "llmops_sentinel_benchmark_pass_rate_percent",
    "Most recent Benchmark Lab pass rate percentage by evaluation metric",
    labelnames=("metric",),
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# 5. Service Dependency Health & Process Runtime Metrics
# ---------------------------------------------------------------------------
SERVICE_UP = Gauge(
    "llmops_sentinel_service_up",
    "Health status of dependencies (1 = up/ok, 0 = unavailable)",
    labelnames=("service",),
    registry=REGISTRY,
)

DEPENDENCY_CHECK_DURATION_SECONDS = Histogram(
    "llmops_sentinel_dependency_check_duration_seconds",
    "Health check probe round-trip latency per service in seconds",
    labelnames=("service",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    registry=REGISTRY,
)

CHROMADB_DOCUMENTS = Gauge(
    "llmops_sentinel_chromadb_documents",
    "Unique document count currently indexed in ChromaDB",
    registry=REGISTRY,
)

CHROMADB_CHUNKS = Gauge(
    "llmops_sentinel_chromadb_chunks",
    "Total vector chunk count currently stored in ChromaDB",
    registry=REGISTRY,
)

OLLAMA_MODELS_AVAILABLE = Gauge(
    "llmops_sentinel_ollama_models_available",
    "Number of models reported by local Ollama /api/tags",
    registry=REGISTRY,
)

EMBEDDING_CACHE_ENTRIES = Gauge(
    "llmops_sentinel_embedding_cache_entries",
    "Current entry count in the in-memory TTL query embedding cache",
    registry=REGISTRY,
)

PROCESS_MEMORY_RSS_BYTES = Gauge(
    "llmops_sentinel_process_memory_rss_bytes",
    "FastAPI process resident set size (RSS) memory usage in bytes",
    registry=REGISTRY,
)

PROCESS_CPU_PERCENT = Gauge(
    "llmops_sentinel_process_cpu_percent",
    "FastAPI process CPU utilization percentage",
    registry=REGISTRY,
)

PROCESS_UPTIME_SECONDS = Gauge(
    "llmops_sentinel_process_uptime_seconds",
    "FastAPI backend uptime in seconds",
    registry=REGISTRY,
)

# Initialize baseline service gauge cleanly
SERVICE_UP.labels(service="fastapi").set(1.0)


def refresh_process_runtime_metrics() -> None:
    """Refresh lightweight process CPU, RSS RAM, uptime, embedding cache, and eval state gauges."""
    with _RUNTIME_LOCK:
        try:
            mem = _PROCESS.memory_info()
            PROCESS_MEMORY_RSS_BYTES.set(float(mem.rss))
            PROCESS_CPU_PERCENT.set(float(_PROCESS.cpu_percent(interval=None)))
        except Exception:
            pass
        PROCESS_UPTIME_SECONDS.set(max(0.0, time.time() - PROCESS_START_TIME))

    try:
        from src.services.query import get_embedding_cache_stats

        stats = get_embedding_cache_stats()
        EMBEDDING_CACHE_ENTRIES.set(float(stats.get("entries", 0)))
    except Exception:
        pass

    try:
        from src.services.evaluate import is_evaluation_running

        EVAL_JOBS_ACTIVE.set(1.0 if is_evaluation_running() else 0.0)
    except Exception:
        pass


def record_http_request(route: str, method: str, status_code: int, duration_seconds: float) -> None:
    """Record bounded-cardinality HTTP counter, histogram, and error counters."""
    status_str = str(status_code)
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status_code=status_str).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=method, route=route, status_code=status_str
    ).observe(max(0.0, duration_seconds))
    if status_code >= 400:
        status_class = "5xx" if status_code >= 500 else "4xx"
        HTTP_ERRORS_TOTAL.labels(
            method=method, route=route, status_class=status_class
        ).inc()


def render_prometheus_metrics() -> tuple[bytes, str]:
    """Refresh live process gauges and serialize Prometheus text exposition payload."""
    refresh_process_runtime_metrics()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def get_runtime_snapshot() -> dict[str, float | int]:
    """Return current in-process telemetry values as JSON-friendly numbers."""
    refresh_process_runtime_metrics()
    rss_bytes = float(PROCESS_MEMORY_RSS_BYTES._value.get())
    cpu_pct = float(PROCESS_CPU_PERCENT._value.get())
    uptime_sec = float(PROCESS_UPTIME_SECONDS._value.get())
    embed_entries = int(EMBEDDING_CACHE_ENTRIES._value.get())
    eval_active = int(EVAL_JOBS_ACTIVE._value.get())
    docs_count = int(CHROMADB_DOCUMENTS._value.get())
    chunks_count = int(CHROMADB_CHUNKS._value.get())
    ollama_models = int(OLLAMA_MODELS_AVAILABLE._value.get())
    return {
        "process_memory_rss_mb": round(rss_bytes / (1024.0 * 1024.0), 2),
        "process_cpu_percent": round(cpu_pct, 2),
        "process_uptime_seconds": round(uptime_sec, 1),
        "embedding_cache_entries": embed_entries,
        "eval_jobs_active": eval_active,
        "chromadb_documents": docs_count,
        "chromadb_chunks": chunks_count,
        "ollama_models_available": ollama_models,
    }


def list_registered_metric_names() -> Iterable[str]:
    """Return sorted Prometheus metric family names registered for LLMOps Sentinel."""
    return sorted({family.name for family in REGISTRY.collect()})

import asyncio
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request

from src.api.router import router
from src.metrics import (
    HTTP_REQUESTS_IN_FLIGHT,
    SERVICE_UP,
    record_http_request,
    refresh_process_runtime_metrics,
)
from src.services.evaluate import (
    reconcile_orphaned_evaluations,
    shutdown_evaluation_cleanup,
)
from src.tracking.setup import mlflow_autolog
from src.utils.env import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OLLAMA_JUDGE_MODEL,
    OLLAMA_MODEL,
)
from src.utils.log_manager import logger


async def pull_model(client: httpx.AsyncClient, model: str) -> None:
    logger.info(f"Pulling Ollama model: {model}")
    try:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": model},
            timeout=None,
        ) as response:
            response.raise_for_status()
            async for _ in response.aiter_lines():
                pass
    except httpx.TimeoutException as exc:
        logger.warning(f"Timed out while pulling Ollama model {model}: {exc}")
        raise
    logger.info(f"Model ready: {model}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Configuring MLflow autolog...")
    mlflow_autolog()

    try:
        reconciled = await asyncio.to_thread(reconcile_orphaned_evaluations)
        if reconciled:
            logger.info(
                f"Startup reconciliation transitioned {reconciled} interrupted evaluation trace(s) to cancelled"
            )
    except Exception as exc:
        logger.warning(f"Startup evaluation reconciliation skipped: {exc}")

    logger.info("Starting up — pulling required Ollama models...")
    try:
        async with httpx.AsyncClient() as client:
            for model in [OLLAMA_MODEL, OLLAMA_JUDGE_MODEL, OLLAMA_EMBED_MODEL]:
                await pull_model(client, model)
        logger.info("All models ready. API is now accepting requests.")
    except Exception as exc:
        logger.warning(
            "Could not pre-pull one or more Ollama models. "
            f"Continuing startup without warm-up: {exc}"
        )
    try:
        yield
    finally:
        try:
            cancelled = await asyncio.to_thread(shutdown_evaluation_cleanup)
            if cancelled:
                logger.info(
                    f"Shutdown cleanup transitioned {cancelled} active/pending evaluation job(s) to cancelled"
                )
        except Exception as exc:
            logger.warning(f"Shutdown evaluation cleanup error: {exc}")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API with MLflow evaluation dashboard",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _bound_inflight_bucket(path: str) -> str:
    if path in ("/health", "/metrics", "/prompts"):
        return path
    for prefix in ("/ingest", "/query", "/eval", "/observability"):
        if path.startswith(prefix):
            return prefix
    return "other"


@app.middleware("http")
async def prometheus_http_middleware(request: Request, call_next):
    inflight_bucket = _bound_inflight_bucket(request.url.path)
    HTTP_REQUESTS_IN_FLIGHT.labels(route=inflight_bucket).inc()
    t0 = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        elapsed_sec = max(0.0, time.perf_counter() - t0)
        route_obj = request.scope.get("route")
        matched_path = getattr(route_obj, "path", None)
        route_label = (
            str(matched_path)
            if matched_path
            else (
                request.url.path
                if request.url.path in ("/health", "/metrics")
                else "unmatched"
            )
        )
        record_http_request(
            route=route_label,
            method=request.method,
            status_code=status_code,
            duration_seconds=elapsed_sec,
        )
        HTTP_REQUESTS_IN_FLIGHT.labels(route=inflight_bucket).dec()
        SERVICE_UP.labels(service="fastapi").set(1.0)
        refresh_process_runtime_metrics()


app.include_router(router)

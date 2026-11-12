import os
import time
import httpx
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from src.metrics import (
    CHROMADB_CHUNKS,
    CHROMADB_DOCUMENTS,
    DEPENDENCY_CHECK_DURATION_SECONDS,
    OLLAMA_MODELS_AVAILABLE,
    SERVICE_UP,
)
from src.models import HealthResponse
from src.services.evaluate import MLFLOW_DB_PATH
from src.utils.env import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MLFLOW_TRACKING_URI,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)


def _count_unique_docs_from_collection(collection) -> int | None:
    try:
        res = collection.get(include=["metadatas"])
        metas = res.get("metadatas") if isinstance(res, dict) else None
        if not isinstance(metas, list):
            return None
        doc_keys: set[str] = set()
        for meta in metas:
            if not isinstance(meta, dict):
                continue
            key = meta.get("document_id") or meta.get("document_name") or meta.get("filename")
            if isinstance(key, str) and key:
                doc_keys.add(key)
        return len(doc_keys)
    except Exception:
        return None


def probe_mlflow_status() -> bool:
    """Quickly check MLflow tracking server or SQLite store reachability without blocking."""
    t0 = time.perf_counter()
    reachable = False
    try:
        resp = httpx.get(MLFLOW_TRACKING_URI, timeout=1.5)
        reachable = resp.status_code < 500
    except Exception:
        reachable = os.path.exists(MLFLOW_DB_PATH)
    DEPENDENCY_CHECK_DURATION_SECONDS.labels(service="mlflow").observe(
        max(0.0, time.perf_counter() - t0)
    )
    SERVICE_UP.labels(service="mlflow").set(1.0 if reachable else 0.0)
    return reachable


async def check_health() -> HealthResponse:
    ollama_status = "ok"
    chromadb_status = "ok"

    t_ollama = time.perf_counter()
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code != 200:
                ollama_status = "error"
            else:
                try:
                    payload = resp.json()
                    models = payload.get("models") if isinstance(payload, dict) else None
                    if isinstance(models, list):
                        OLLAMA_MODELS_AVAILABLE.set(float(len(models)))
                except Exception:
                    pass
        except Exception:
            ollama_status = "error"
    DEPENDENCY_CHECK_DURATION_SECONDS.labels(service="ollama").observe(
        max(0.0, time.perf_counter() - t_ollama)
    )
    SERVICE_UP.labels(service="ollama").set(1.0 if ollama_status == "ok" else 0.0)

    t_chroma = time.perf_counter()
    try:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL
        )
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        chunk_count = vectorstore._collection.count()
        if isinstance(chunk_count, int):
            CHROMADB_CHUNKS.set(float(chunk_count))
        doc_count = _count_unique_docs_from_collection(vectorstore._collection)
        if doc_count is not None:
            CHROMADB_DOCUMENTS.set(float(doc_count))
    except Exception:
        chromadb_status = "error"
    DEPENDENCY_CHECK_DURATION_SECONDS.labels(service="chromadb").observe(
        max(0.0, time.perf_counter() - t_chroma)
    )
    SERVICE_UP.labels(service="chromadb").set(1.0 if chromadb_status == "ok" else 0.0)
    SERVICE_UP.labels(service="fastapi").set(1.0)

    return HealthResponse(
        status="ok",
        chromadb=chromadb_status,
        ollama=ollama_status,
    )

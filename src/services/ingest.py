import hashlib
import os
import tempfile
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from fastapi import HTTPException, UploadFile
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.metrics import (
    CHROMADB_CHUNKS,
    CHROMADB_DOCUMENTS,
    INGESTION_CHUNKS_CREATED_TOTAL,
    INGESTION_DURATION_SECONDS,
    INGESTIONS_TOTAL,
)
from src.models import IngestResponse
from src.utils.env import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
)
from src.utils.log_manager import logger

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".txt": {"text/plain", "application/octet-stream"},
}

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 100
SPLITTER_NAME = "RecursiveCharacterTextSplitter"


def normalize_text_for_dedup(text: str) -> str:
    """Collapse whitespace and case for duplicate chunk detection."""
    return " ".join(text.split()).strip().lower()


def build_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )


def _get_ingest_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _clear_query_cache() -> None:
    try:
        from src.services.query import clear_vectorstore_cache

        clear_vectorstore_cache()
    except ImportError:
        pass


def _extract_doc_name(meta: dict) -> str:
    raw_name = meta.get("document_name") or meta.get("filename") or meta.get("source") or "unknown"
    if not isinstance(raw_name, str) or not raw_name:
        return "unknown"
    return os.path.basename(raw_name) if ("/" in raw_name or "\\" in raw_name) else raw_name


def _delete_matching_chunks(vector_store: Chroma, identifier: str) -> int:
    """Delete all chunks matching a document filename or document_id from the given Chroma store."""
    try:
        res = vector_store._collection.get(include=["metadatas"])
    except Exception as e:
        logger.warning(f"Failed inspecting collection before cleanup for '{identifier}': {e}")
        return 0

    if not isinstance(res, dict):
        return 0

    ids = res.get("ids")
    metas = res.get("metadatas")
    if not isinstance(ids, list) or not isinstance(metas, list):
        return 0

    to_delete: list[str] = []
    for chunk_id, meta in zip(ids, metas):
        if isinstance(meta, dict):
            clean_name = _extract_doc_name(meta)
            doc_id = meta.get("document_id")
            raw_filename = meta.get("filename")
            raw_doc_name = meta.get("document_name")
            if identifier in {clean_name, doc_id, raw_filename, raw_doc_name}:
                to_delete.append(chunk_id)

    if to_delete:
        vector_store._collection.delete(ids=to_delete)
        logger.info(f"Removed {len(to_delete)} existing chunk(s) for '{identifier}'.")
    return len(to_delete)


def get_indexed_documents() -> list[dict[str, str | int | None]]:
    """Return all distinct documents indexed in ChromaDB with chunk counts and metadata."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get(include=["metadatas"])
        if not isinstance(res, dict):
            return []
        metas = res.get("metadatas") or []
        if not isinstance(metas, list):
            return []

        counts: Counter[str] = Counter()
        doc_ids: dict[str, str | None] = {}
        timestamps: dict[str, str | None] = {}

        for m in metas:
            if m and isinstance(m, dict):
                clean_name = _extract_doc_name(m)
                counts[clean_name] += 1
                if m.get("document_id"):
                    doc_ids[clean_name] = str(m["document_id"])
                if m.get("ingestion_timestamp"):
                    timestamps[clean_name] = str(m["ingestion_timestamp"])

        return [
            {
                "filename": name,
                "chunks": count,
                "document_id": doc_ids.get(name),
                "ingestion_timestamp": timestamps.get(name),
            }
            for name, count in sorted(counts.items())
        ]
    except Exception as e:
        logger.error(f"Error listing indexed documents: {e}")
        return []


def _refresh_chroma_gauges(vector_store: Chroma) -> None:
    try:
        chunk_count = vector_store._collection.count()
        if isinstance(chunk_count, int):
            CHROMADB_CHUNKS.set(float(chunk_count))
        res = vector_store._collection.get(include=["metadatas"])
        metas = res.get("metadatas") if isinstance(res, dict) else None
        if isinstance(metas, list):
            doc_keys = {
                str(m.get("document_id") or m.get("document_name") or m.get("filename"))
                for m in metas
                if isinstance(m, dict)
                and (m.get("document_id") or m.get("document_name") or m.get("filename"))
            }
            CHROMADB_DOCUMENTS.set(float(len(doc_keys)))
    except Exception:
        pass


def clear_all_documents() -> dict[str, str | int]:
    """Clear all documents from ChromaDB collection."""
    vector_store = _get_ingest_vectorstore()
    try:
        res = vector_store._collection.get()
        all_ids = res.get("ids", []) or [] if isinstance(res, dict) else []
        if isinstance(all_ids, list) and all_ids:
            vector_store._collection.delete(ids=all_ids)
        _clear_query_cache()
        CHROMADB_CHUNKS.set(0.0)
        CHROMADB_DOCUMENTS.set(0.0)
        return {"status": "ok", "deleted_chunks": len(all_ids) if isinstance(all_ids, list) else 0}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {e}")


def delete_document(filename: str) -> dict[str, str | int]:
    """Delete all chunks for a specific document (by filename or document_id) from ChromaDB."""
    vector_store = _get_ingest_vectorstore()
    try:
        deleted_count = _delete_matching_chunks(vector_store, filename)
        _clear_query_cache()
        _refresh_chroma_gauges(vector_store)
        return {"status": "ok", "deleted_chunks": deleted_count, "filename": filename}
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {e}")


async def ingest_document(file: UploadFile, clear_existing: bool = False) -> IngestResponse:
    t_ingest0 = time.perf_counter()
    if clear_existing:
        clear_all_documents()

    raw_filename = file.filename or ""
    clean_filename = os.path.basename(raw_filename) if raw_filename else ""
    suffix = Path(clean_filename).suffix.lower()
    metric_file_type = suffix.lstrip(".") if suffix in ALLOWED_EXTENSIONS else "unsupported"
    if suffix not in ALLOWED_EXTENSIONS:
        INGESTIONS_TOTAL.labels(file_type=metric_file_type, status="error").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .txt are accepted.",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    ingest_status = "error"
    try:
        total_bytes = 0

        with open(tmp_path, "wb") as tmp:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file is too large. "
                            f"Maximum allowed size is {MAX_UPLOAD_SIZE_BYTES} bytes."
                        ),
                    )

                tmp.write(chunk)

        if total_bytes == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        content_type = (file.content_type or "").lower()
        if content_type and content_type not in ALLOWED_CONTENT_TYPES.get(suffix, set()):
            raise HTTPException(
                status_code=400,
                detail=(f"Unsupported content type '{content_type}' for '{suffix}' files."),
            )

        embeddings = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )
        raw_chunks = splitter.split_documents(pages)
        if not raw_chunks:
            raise HTTPException(
                status_code=400,
                detail="Document contains no extractable text.",
            )

        # Remove intra-document exact duplicate / empty text chunks on real string content
        chunks = []
        seen_hashes: set[str] = set()
        for chunk in raw_chunks:
            raw_content = getattr(chunk, "page_content", None)
            if isinstance(raw_content, str):
                norm = normalize_text_for_dedup(raw_content)
                if not norm:
                    continue
                content_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
            chunks.append(chunk)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="Document contains no extractable non-empty text.",
            )

        # Ensure re-uploading an existing document replaces stale chunks cleanly
        if not clear_existing and clean_filename:
            _delete_matching_chunks(vector_store, clean_filename)

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        ingested_at = datetime.now(timezone.utc).isoformat()
        file_type = suffix.lstrip(".")
        total_pages = len(pages) if suffix == ".pdf" and isinstance(pages, list) else 1
        chunk_ids: list[str] = []

        for idx, chunk in enumerate(chunks):
            cid = f"{doc_id}-chunk-{idx:04d}"
            raw_content = getattr(chunk, "page_content", None)
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                chunk.metadata["document_id"] = doc_id
                chunk.metadata["document_name"] = clean_filename
                chunk.metadata["filename"] = clean_filename
                chunk.metadata["source"] = clean_filename
                chunk.metadata["file_type"] = file_type
                chunk.metadata["chunk_id"] = cid
                chunk.metadata["chunk_index"] = idx
                chunk.metadata["total_pages"] = total_pages
                chunk.metadata["ingestion_timestamp"] = ingested_at
                if isinstance(raw_content, str):
                    norm = normalize_text_for_dedup(raw_content)
                    chunk.metadata["content_hash"] = hashlib.sha256(
                        norm.encode("utf-8")
                    ).hexdigest()[:16]
                    chunk.metadata["char_count"] = len(raw_content)
                    chunk.metadata["token_count_estimate"] = max(1, len(raw_content) // 4)
                chunk_ids.append(cid)

                raw_page = chunk.metadata.get("page")
                if raw_page is not None:
                    try:
                        chunk.metadata["page_number"] = int(raw_page) + 1
                    except (ValueError, TypeError):
                        pass

        if len(chunk_ids) == len(chunks) and chunk_ids:
            vector_store.add_documents(documents=chunks, ids=chunk_ids)
        else:
            vector_store.add_documents(documents=chunks)
        _clear_query_cache()
        _refresh_chroma_gauges(vector_store)

        ingest_status = "success"
        INGESTION_CHUNKS_CREATED_TOTAL.labels(file_type=metric_file_type).inc(len(chunks))

        return IngestResponse(
            status="ok",
            chunks_stored=len(chunks),
            filename=clean_filename,
            document_id=doc_id,
            ingestion_timestamp=ingested_at,
        )
    finally:
        elapsed_sec = max(0.0, time.perf_counter() - t_ingest0)
        INGESTIONS_TOTAL.labels(file_type=metric_file_type, status=ingest_status).inc()
        INGESTION_DURATION_SECONDS.labels(
            file_type=metric_file_type, status=ingest_status
        ).observe(elapsed_sec)
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
